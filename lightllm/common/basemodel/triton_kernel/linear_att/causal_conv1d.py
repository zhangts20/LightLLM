# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/causal_conv1d.py
# and Dao-AILab causal_conv1d_ref (MIT).

from typing import Optional

import torch
import torch.nn.functional as F

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_HAS_MACA_CAUSAL_CONV1D = False
_HAS_SGL_CAUSAL_CONV1D = False
_maca_causal_conv1d_fn = None
_maca_causal_conv1d_update = None
_sgl_causal_conv1d_fwd = None
_sgl_causal_conv1d_update = None


def _is_maca_platform() -> bool:
    try:
        from lightllm.utils.envs_utils import get_env_start_args

        return getattr(get_env_start_args(), "hardware_platform", "cuda") == "maca"
    except Exception:
        return "metax" in (getattr(torch, "__version__", "") or "").lower()


if _is_maca_platform():
    try:
        from causal_conv1d import causal_conv1d_fn as _maca_causal_conv1d_fn
        from causal_conv1d import causal_conv1d_update as _maca_causal_conv1d_update

        _HAS_MACA_CAUSAL_CONV1D = True
        logger.info("Using maca causal_conv1d CUDA kernels (causal_conv1d package).")
    except ImportError:
        _HAS_MACA_CAUSAL_CONV1D = False

if not _HAS_MACA_CAUSAL_CONV1D:
    try:
        from sgl_kernel import causal_conv1d_fwd as _sgl_causal_conv1d_fwd
        from sgl_kernel import causal_conv1d_update as _sgl_causal_conv1d_update

        _HAS_SGL_CAUSAL_CONV1D = True
        logger.info("Using sgl_kernel causal_conv1d CUDA kernels.")
    except ImportError:
        _HAS_SGL_CAUSAL_CONV1D = False
        if _is_maca_platform():
            logger.warning(
                "Neither causal_conv1d nor sgl_kernel is available; "
                "causal_conv1d will use the PyTorch fallback (slow on long prefill)."
            )
        else:
            logger.warning(
                "sgl_kernel is not available; causal_conv1d will use the PyTorch fallback (slow on long prefill)."
            )


def _apply_activation(out: torch.Tensor, activation: Optional[str]) -> torch.Tensor:
    if activation in ["silu", "swish"]:
        return F.silu(out)
    return out


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    if x.stride(-2) == 1:
        return x
    return x.transpose(-1, -2).contiguous().transpose(-1, -2)


def _has_init_one(has_initial_state) -> bool:
    if has_initial_state is None:
        return False
    if isinstance(has_initial_state, torch.Tensor):
        return bool(has_initial_state.view(-1)[0].item())
    return bool(has_initial_state)


def _host_index_list(cache_indices: Optional[torch.Tensor]):
    if cache_indices is None:
        return None
    return cache_indices.detach().to("cpu", dtype=torch.int64).tolist()


def _host_has_init_list(has_initial_state, batch: int):
    if has_initial_state is None:
        return [False] * batch
    if isinstance(has_initial_state, torch.Tensor):
        return has_initial_state.detach().to("cpu").view(-1).tolist()
    return [bool(has_initial_state)] * batch


def _apply_packed_varlen(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    has_initial_state,
    conv_states: Optional[torch.Tensor],
    pad_slot_id: int,
    state_len: int,
    run_one,
) -> torch.Tensor:
    batch = int(query_start_loc.numel() - 1)
    need_state = conv_states is not None

    def _apply_one(start: int, end: int, state_idx: int, has_init: bool):
        if end <= start or state_idx == pad_slot_id:
            return
        init = conv_states[state_idx, :, :state_len] if (need_state and has_init) else None
        out_i, new_state = run_one(x[:, start:end], init)
        x[:, start:end].copy_(out_i)
        if need_state and new_state is not None:
            conv_states[state_idx, :, :state_len].copy_(new_state)

    if batch == 1:
        # conc=1: skip .tolist() D2H; read start/end with .item() and run once.
        _apply_one(
            int(query_start_loc[0].item()),
            int(query_start_loc[1].item()),
            0 if cache_indices is None else int(cache_indices[0].item()),
            _has_init_one(has_initial_state),
        )
        return x

    starts = query_start_loc.detach().to("cpu", dtype=torch.int64).tolist()
    idxs = _host_index_list(cache_indices)
    inits = _host_has_init_list(has_initial_state, batch)
    for i in range(batch):
        _apply_one(starts[i], starts[i + 1], idxs[i] if idxs is not None else i, bool(inits[i]))
    return x


def _causal_conv1d_fn_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
):
    dtype_in = x.dtype
    weight_f = weight.to(dtype=x.dtype)
    bias_f = bias.to(dtype=x.dtype) if bias is not None else None
    dim, width = weight_f.shape
    state_len = width - 1
    varlen = query_start_loc is not None

    def _run_one(seq: torch.Tensor, init_state: Optional[torch.Tensor]):
        # seq: (dim, seqlen)
        seqlen = seq.shape[-1]
        if init_state is None:
            # Left-pad zeros to make the convolution causal.
            x_pad = F.pad(seq.to(weight_f.dtype), (state_len, 0))
        else:
            x_pad = torch.cat([init_state.to(weight_f.dtype), seq.to(weight_f.dtype)], dim=-1)
        # F.conv1d expects (N, C, L); depthwise via groups=dim.
        out = F.conv1d(
            x_pad.unsqueeze(0),
            weight_f.unsqueeze(1),
            bias_f,
            padding=0,
            groups=dim,
        ).squeeze(0)
        # With left pad / init of length state_len, out length == seqlen.
        assert out.shape[-1] == seqlen, (out.shape, seq.shape, width)
        out = _apply_activation(out, activation).to(dtype_in)
        new_state = x_pad[:, -state_len:].to(dtype_in)
        return out, new_state

    if varlen:
        assert x.dim() == 2 and x.shape[0] == dim, f"expected x (dim, cu_seq_len), got {tuple(x.shape)}"
        return _apply_packed_varlen(
            x,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_states,
            pad_slot_id,
            state_len,
            _run_one,
        )

    # Non-varlen: x is (batch, dim, seqlen)
    assert x.dim() == 3 and x.shape[1] == dim, f"expected x (batch, dim, seqlen), got {tuple(x.shape)}"
    batch = x.shape[0]
    cache_indices_cpu = _host_index_list(cache_indices)
    has_init_cpu = _host_has_init_list(has_initial_state, batch)

    for i in range(batch):
        if cache_indices_cpu is not None and cache_indices_cpu[i] == pad_slot_id:
            continue
        state_idx = cache_indices_cpu[i] if cache_indices_cpu is not None else i
        init = None
        if conv_states is not None and has_init_cpu[i]:
            init = conv_states[state_idx, :, :state_len]
        out_i, new_state = _run_one(x[i], init)
        x[i].copy_(out_i)
        if conv_states is not None:
            conv_states[state_idx, :, :state_len].copy_(new_state)
    return x


def _causal_conv1d_update_pytorch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
):
    if cache_seqlens is not None:
        raise NotImplementedError("PyTorch causal_conv1d_update does not support circular cache_seqlens")

    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    assert conv_state.shape[-1] >= width - 1
    assert seqlen == 1, "graph-safe update currently supports decode seqlen=1 only"

    if conv_state_indices is None:
        indices = torch.arange(batch, device=x.device, dtype=torch.int64)
    else:
        indices = conv_state_indices.to(dtype=torch.int64)

    pad_mask = indices == pad_slot_id
    safe_indices = torch.where(pad_mask, torch.zeros_like(indices), indices)

    # state: [B, dim, width-1], x: [B, dim, 1]
    state = conv_state[safe_indices, :, : width - 1].to(dtype=torch.float32)
    x_f = x[:, :, 0].to(dtype=torch.float32)
    w_f = weight.to(dtype=torch.float32)

    y = x_f * w_f[:, width - 1]
    for i in range(width - 1):
        y = y + state[:, :, i] * w_f[:, i]
    if bias is not None:
        y = y + bias.to(dtype=torch.float32)
    if activation in ["silu", "swish"]:
        y = y * torch.sigmoid(y)
    out = y.to(dtype=dtype_in).unsqueeze(-1)

    # Roll state: [s1,...,s_{w-2}, x]
    if width == 2:
        new_state = x_f.to(dtype=conv_state.dtype).unsqueeze(-1)
    else:
        new_state = torch.cat(
            [state[:, :, 1:].to(dtype=conv_state.dtype), x_f.to(dtype=conv_state.dtype).unsqueeze(-1)],
            dim=-1,
        )
    conv_state[:, :, : width - 1].index_copy_(0, safe_indices, new_state)

    out = torch.where(pad_mask.view(batch, 1, 1), x.to(dtype_in), out)
    if unsqueeze:
        out = out.squeeze(-1)
    return out


def _maca_run_one_varlen_seq(
    seq: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    init_state: Optional[torch.Tensor],
    activation: Optional[str],
    need_final_state: bool,
):
    x_b = _to_channel_last(seq.unsqueeze(0))
    if not need_final_state and init_state is None:
        out = _maca_causal_conv1d_fn(x_b, weight, bias, activation=activation)
        return out[0], None

    state_len = weight.shape[1] - 1
    if init_state is None:
        init_cl = _to_channel_last(
            torch.zeros(1, seq.shape[0], state_len, device=seq.device, dtype=seq.dtype)
        )
    else:
        init_cl = _to_channel_last(init_state.unsqueeze(0).contiguous())
    final = torch.empty_like(init_cl)
    out, final_states = _maca_causal_conv1d_fn(
        x_b,
        weight,
        bias,
        initial_states=init_cl,
        return_final_states=True,
        final_states_out=final,
        activation=activation,
    )
    return out[0], final_states[0]


def _causal_conv1d_fn_maca(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
):
    # maca kernel is (B, D, L). Packed varlen is split per sequence then run as B=1.
    if query_start_loc is None:
        return _causal_conv1d_fn_pytorch(
            x,
            weight,
            bias=bias,
            query_start_loc=None,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation=activation,
            pad_slot_id=pad_slot_id,
        )

    need_state = conv_states is not None

    def _run_one(seq: torch.Tensor, init):
        return _maca_run_one_varlen_seq(seq, weight, bias, init, activation, need_state)

    return _apply_packed_varlen(
        x,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
        pad_slot_id,
        weight.shape[1] - 1,
        _run_one,
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    **kwargs,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    if _HAS_MACA_CAUSAL_CONV1D:
        return _causal_conv1d_fn_maca(
            x,
            weight,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation=activation,
            pad_slot_id=pad_slot_id,
        )

    if _HAS_SGL_CAUSAL_CONV1D:
        _sgl_causal_conv1d_fwd(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            activation in ["silu", "swish"],
            pad_slot_id,
        )
        return x

    return _causal_conv1d_fn_pytorch(
        x,
        weight,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_states,
        activation=activation,
        pad_slot_id=pad_slot_id,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(f"activation must be None, silu, or swish, actual: {activation}")

    if _HAS_MACA_CAUSAL_CONV1D:
        return _maca_causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=conv_state_indices,
        )

    if _HAS_SGL_CAUSAL_CONV1D:
        activation_val = activation in ["silu", "swish"]
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)
        _sgl_causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation_val,
            cache_seqlens,
            conv_state_indices,
            pad_slot_id,
        )
        if unsqueeze:
            x = x.squeeze(-1)
        return x

    return _causal_conv1d_update_pytorch(
        x,
        conv_state,
        weight,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        pad_slot_id=pad_slot_id,
    )
