import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.linear.gdn as gdn
import lightllm.common.basemodel.attention.create_linear_utils as linear_create_utils
import lightllm.common.basemodel.triton_kernel.linear_att.fla.ops as fla_ops
from lightllm.common.basemodel.attention.linear.flashqla import FlashQlaLinearAttBackend
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.server.api_cli import make_argument_parser
import lightllm.utils.backend_validator as backend_validator


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_prefill_casts_final_state_to_cache_dtype(monkeypatch, cache_dtype):
    ssm_states = torch.zeros((1, 1, 2, 2), dtype=cache_dtype)
    final_state = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(gdn, "fused_gdn_gating", lambda _log, a, b, _bias: (a, b))
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda mixed, *args, **kwargs: mixed)
    qkv = torch.zeros((1, 3), dtype=cache_dtype)
    q = torch.zeros((1, 1, 1, 1), dtype=cache_dtype)
    backend = SimpleNamespace(
        mtp_step=0,
        activation="silu",
        ssm_state_dtype=cache_dtype,
        _rearrange_mixed_qkv=lambda mixed: (q, q, q),
        prefill_kernel=lambda *args, **kwargs: (None, final_state),
    )
    state = gdn.LinearAttPrefillAttState(
        backend=backend,
        infer_state=SimpleNamespace(
            b1_cu_q_seq_len=torch.tensor([0, 1], dtype=torch.int32),
            b_ready_cache_len=0,
        ),
    )
    state.b_conv_buffer_idx = torch.tensor([0], dtype=torch.int64)
    state.b_ssm_buffer_idx = torch.tensor([0], dtype=torch.int64)
    layer_weight = SimpleNamespace(
        linear_A_log=SimpleNamespace(weight=None),
        linear_dt_bias=SimpleNamespace(weight=None),
        linear_conv1d=SimpleNamespace(mm_param=SimpleNamespace(weight=None), bias=None),
    )

    state._gdn_prefill_kernel(
        qkv,
        torch.zeros((1, 3), dtype=cache_dtype),
        ssm_states,
        torch.zeros((1, 1), dtype=cache_dtype),
        torch.zeros((1, 1), dtype=cache_dtype),
        state.infer_state,
        layer_weight,
    )

    assert ssm_states.dtype == cache_dtype
    assert torch.equal(ssm_states, final_state.to(cache_dtype))


def _create_decode_state(*, draft_step, dynamic_layout, b_req_idx, req_to_mtp_state_index=None, capacity=None):
    model = SimpleNamespace(
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: draft_step),
    )
    infer_state = SimpleNamespace(
        batch_size=b_req_idx.shape[0],
        b_req_idx=b_req_idx,
        b_mtp_index=torch.zeros_like(b_req_idx),
        req_manager=SimpleNamespace(
            HOLD_REQUEST_ID=req_to_mtp_state_index.shape[0] - 1 if req_to_mtp_state_index is not None else -1,
            req_to_mtp_state_index=req_to_mtp_state_index,
        ),
    )
    return gdn.LinearAttDecodeAttState(
        backend=SimpleNamespace(
            model=model,
            mtp_step=draft_step if capacity is None else capacity,
            uses_dynamic_spec_verify_layout=lambda: dynamic_layout,
        ),
        infer_state=infer_state,
    )


def test_decode_state_initializes_normal_layout():
    b_req_idx = torch.tensor([2, 4], dtype=torch.int32)
    state = _create_decode_state(draft_step=0, dynamic_layout=False, b_req_idx=b_req_idx)

    state.init_state()

    assert state.b_conv_buffer_idx is b_req_idx
    assert state.b_ssm_buffer_idx is b_req_idx
    assert state.b1_mtp_cu_q_seq_len is None
    assert state.b_num_accepted_tokens is None


def test_decode_state_initializes_fixed_mtp_layout():
    b_req_idx = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.int32)
    req_to_mtp_state_index = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.int32)
    state = _create_decode_state(
        draft_step=2,
        dynamic_layout=False,
        b_req_idx=b_req_idx,
        req_to_mtp_state_index=req_to_mtp_state_index,
    )

    state.init_state()

    torch.testing.assert_close(state.b1_mtp_cu_q_seq_len, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(state.b_conv_buffer_idx, torch.tensor([2, 4], dtype=torch.int32))
    torch.testing.assert_close(state.b_num_accepted_tokens, torch.tensor([2, 3], dtype=torch.int32))
    torch.testing.assert_close(
        state.b_ssm_buffer_idx,
        torch.tensor([[6, 7, 8], [12, 13, 14]], dtype=torch.int32),
    )


def test_decode_state_initializes_dynamic_mtp_layout(monkeypatch):
    b_req_idx = torch.tensor([2, 2, 4, 4, 4], dtype=torch.int32)
    req_to_mtp_state_index = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.int32)
    expected_cu_q_seq_len = torch.tensor([0, 2, 5, 5, 5, 5], dtype=torch.int32)
    expected_conv_buffer_idx = torch.tensor([2, 4, 5, 5, 5], dtype=torch.int32)
    expected_num_accepted_tokens = torch.tensor([2, 3, 1, 1, 1], dtype=torch.int32)
    build_calls = []

    def build_params(**kwargs):
        build_calls.append(kwargs)
        return expected_cu_q_seq_len, expected_conv_buffer_idx, expected_num_accepted_tokens

    monkeypatch.setattr(gdn, "build_dynamic_mtp_linear_att_state_params", build_params)
    state = _create_decode_state(
        draft_step=2,
        dynamic_layout=True,
        b_req_idx=b_req_idx,
        req_to_mtp_state_index=req_to_mtp_state_index,
    )

    state.init_state()

    assert state.b1_mtp_cu_q_seq_len is expected_cu_q_seq_len
    assert state.b_conv_buffer_idx is expected_conv_buffer_idx
    assert state.b_num_accepted_tokens is expected_num_accepted_tokens
    torch.testing.assert_close(
        state.b_ssm_buffer_idx,
        torch.tensor(
            [[6, 7, 8], [12, 13, 14], [15, 16, 17], [15, 16, 17], [15, 16, 17]],
            dtype=torch.int32,
        ),
    )
    assert len(build_calls) == 1
    assert build_calls[0]["b_req_idx"] is state.infer_state.b_req_idx
    assert build_calls[0]["b_mtp_index"] is state.infer_state.b_mtp_index
    assert build_calls[0]["req_to_mtp_state_index"] is req_to_mtp_state_index
    assert build_calls[0]["hold_req_id"] == 5


def test_linear_backend_is_abstract_and_triton_supplies_chunk_kernel():
    assert inspect.isabstract(gdn.LinearAttBackend)
    backend = object.__new__(TritonLinearAttBackend)
    assert backend.get_prefill_kernel() is fla_ops.chunk_gated_delta_rule


@pytest.fixture
def auto_linear_backend_args(monkeypatch):
    monkeypatch.setattr(
        linear_create_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            hardware_platform="cuda",
            llm_prefill_att_backend=["fa3"],
            llm_decode_att_backend=["flashinfer"],
        ),
    )


def test_missing_linear_backend_arg_uses_auto_selection(monkeypatch, auto_linear_backend_args):
    validate_calls = []
    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = lambda **kwargs: ("flashqla", kwargs)
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name: validate_calls.append(name) or True,
    )

    backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(index=1)

    assert backend_class is FlashQlaLinearAttBackend
    backend = object.__new__(backend_class)
    assert backend.get_prefill_kernel()(q="q")[0] == "flashqla"
    assert validate_calls == ["flashqla"]


def test_flashqla_validation_loads_linear_cache_config(monkeypatch):
    linear_config = SimpleNamespace(
        num_linear_k_heads=2,
        num_linear_v_heads=4,
        head_linear_k_dim=8,
        head_linear_v_dim=16,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
    )
    load_calls = []
    monkeypatch.setattr(
        LinearAttCacheConfig,
        "load_from_args",
        staticmethod(lambda: load_calls.append(True) or linear_config),
    )

    kernel_calls = []

    def kernel(**kwargs):
        kernel_calls.append(kwargs)
        return kwargs["q"], kwargs["initial_state"]

    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = kernel
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    monkeypatch.setattr(fla_ops, "chunk_gated_delta_rule", kernel)

    original_randn = torch.randn
    original_rand = torch.rand
    original_tensor = torch.tensor

    def cpu_randn(*args, **kwargs):
        kwargs.pop("device", None)
        return original_randn(*args, **kwargs)

    def cpu_rand(*args, **kwargs):
        kwargs.pop("device", None)
        return original_rand(*args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(backend_validator.torch, "randn", cpu_randn)
    monkeypatch.setattr(backend_validator.torch, "rand", cpu_rand)
    monkeypatch.setattr(backend_validator.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(backend_validator.torch.cuda, "synchronize", lambda: None)

    success, error = backend_validator._validate_flashqla()

    assert success is True
    assert error is None
    assert load_calls == [True]
    assert len(kernel_calls) == 2
    assert kernel_calls[0]["q"].shape == (1, 64, 2, 8)
    assert kernel_calls[0]["v"].shape == (1, 64, 4, 16)
    assert kernel_calls[0]["q"].dtype == torch.bfloat16
    assert kernel_calls[0]["initial_state"].shape == (1, 4, 8, 16)
    assert kernel_calls[0]["initial_state"].dtype == torch.float32


def test_explicit_linear_backend_arg_skips_auto_selection(monkeypatch):
    monkeypatch.setattr(
        linear_create_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            llm_prefill_att_backend=["fa3", "triton"],
            llm_decode_att_backend=["flashinfer", "triton"],
        ),
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda *args: pytest.fail("an explicit linear backend must not be auto-validated"),
    )

    prefill_backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(index=1)
    decode_backend_class = linear_create_utils.get_qwen35_linear_decode_att_backend_class(index=1)

    assert prefill_backend_class is TritonLinearAttBackend
    assert decode_backend_class is TritonLinearAttBackend


def test_cli_accepts_linear_backend_at_index_one():
    args = make_argument_parser().parse_args(
        [
            "--llm_prefill_att_backend",
            "fa3",
            "flashqla",
            "--llm_decode_att_backend",
            "flashinfer",
            "triton",
        ]
    )

    assert args.llm_prefill_att_backend == ["fa3", "flashqla"]
    assert args.llm_decode_att_backend == ["flashinfer", "triton"]


@pytest.mark.parametrize(
    "get_backend_class",
    [
        linear_create_utils.get_qwen35_linear_prefill_att_backend_class,
        linear_create_utils.get_qwen35_linear_decode_att_backend_class,
    ],
)
def test_linear_backend_rejects_non_linear_index(auto_linear_backend_args, get_backend_class):
    with pytest.raises(AssertionError, match="index must be 1"):
        get_backend_class(index=0)


def test_missing_linear_decode_backend_arg_defaults_to_triton(auto_linear_backend_args):
    backend_class = linear_create_utils.get_qwen35_linear_decode_att_backend_class(index=1)

    assert backend_class is TritonLinearAttBackend


def test_gdn_decode_backend_uses_priority_list(monkeypatch, auto_linear_backend_args):
    validate_calls = []
    optimized_backend = type("OptimizedLinearDecodeAttBackend", (), {})
    monkeypatch.setattr(
        linear_create_utils,
        "linear_decode_att_backend_classes",
        {"optimized": optimized_backend, "triton": TritonLinearAttBackend},
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name: validate_calls.append(name) or True,
    )

    backend_class = linear_create_utils.get_qwen35_linear_decode_att_backend_class(
        index=1, priority_list=("optimized", "triton")
    )

    assert backend_class is optimized_backend
    assert validate_calls == ["optimized"]


def test_gdn_prefill_backend_falls_back_when_flashqla_validation_fails(monkeypatch, auto_linear_backend_args):
    monkeypatch.setattr(linear_create_utils, "validate", lambda *args: False)

    backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(index=1)

    assert backend_class is TritonLinearAttBackend


def test_gdn_prefill_backend_tries_candidates_in_order(monkeypatch, auto_linear_backend_args):
    validate_calls = []
    flashqla2_backend = type("FlashQla2LinearAttBackend", (), {})
    flashqla3_backend = type("FlashQla3LinearAttBackend", (), {})
    monkeypatch.setattr(
        linear_create_utils,
        "linear_prefill_att_backend_classes",
        {"flashqla2": flashqla2_backend, "flashqla3": flashqla3_backend},
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name, *args: validate_calls.append(name) or name == "flashqla3",
    )

    backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(
        index=1, priority_list=("flashqla2", "flashqla3")
    )

    assert backend_class is flashqla3_backend
    assert validate_calls == ["flashqla2", "flashqla3"]


@pytest.mark.parametrize("draft_step", [0, 1, 2])
def test_decode_width_preserves_allocated_state_stride(draft_step):
    state = _create_decode_state(
        draft_step=draft_step,
        capacity=3,
        dynamic_layout=False,
        b_req_idx=torch.tensor([2] * (draft_step + 1) + [4] * (draft_step + 1), dtype=torch.int32),
        req_to_mtp_state_index=torch.tensor([0, 0, 3, 0, 2, 0], dtype=torch.int32),
    )
    state.init_state()
    torch.testing.assert_close(
        state.b_ssm_buffer_idx,
        torch.tensor([[8, 9, 10, 11], [16, 17, 18, 19]], dtype=torch.int32),
    )
    torch.testing.assert_close(state.b_accepted_ssm_state_indices, torch.tensor([11, 18], dtype=torch.int32))
    torch.testing.assert_close(
        state.b1_mtp_cu_q_seq_len,
        torch.tensor([0, draft_step + 1, 2 * (draft_step + 1)], dtype=torch.int32),
    )


def test_fixed_layout_rejects_partial_request():
    state = _create_decode_state(
        draft_step=2, dynamic_layout=False, b_req_idx=torch.tensor([2, 2], dtype=torch.int32)
    )
    with pytest.raises(AssertionError, match="divisible"):
        state.init_state()


def test_dynamic_npu_layout_fails_before_launching_fixed_width_kernels(monkeypatch):
    state = _create_decode_state(
        draft_step=2, dynamic_layout=True, b_req_idx=torch.tensor([2, 2], dtype=torch.int32)
    )
    state.infer_state.b_req_idx = SimpleNamespace(device=SimpleNamespace(type="npu"))
    monkeypatch.setattr(
        gdn, "build_dynamic_mtp_linear_att_state_params",
        lambda **kwargs: pytest.fail("must reject before launching the CUDA-only layout builder"),
    )
    with pytest.raises(NotImplementedError, match="variable-length"):
        state.init_state()


@pytest.mark.parametrize("draft_step", [0, 1, 2])
def test_npu_mtp_reads_accepted_state_and_writes_current_width(monkeypatch, draft_step):
    import sys
    from types import ModuleType

    state = _create_decode_state(
        draft_step=draft_step, capacity=3, dynamic_layout=False,
        b_req_idx=torch.tensor([2] * (draft_step + 1), dtype=torch.int32),
        req_to_mtp_state_index=torch.tensor([0, 0, 3, 0], dtype=torch.int32),
    )
    state.init_state()
    conv_calls, recurrent_calls = [], []
    conv_module = ModuleType("lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_spec_sgl_npu")

    def conv(**kwargs):
        conv_calls.append(kwargs)
        return kwargs["x"]

    conv_module.causal_conv1d_update_sgl_npu = conv
    monkeypatch.setitem(sys.modules, conv_module.__name__, conv_module)

    def recurrent(**kwargs):
        recurrent_calls.append(kwargs)
        return kwargs["v"], None

    monkeypatch.setattr(gdn, "mtp_fused_recurrent_gated_delta_rule", recurrent)
    state.backend.activation = "silu"
    state.backend._rearrange_mixed_qkv = lambda mixed, decode: (mixed, mixed, mixed)
    layer_weight = SimpleNamespace(
        linear_conv1d_mtp_weight_t=torch.zeros((4, 3)),
        linear_conv1d=SimpleNamespace(bias=None),
        linear_A_log=SimpleNamespace(weight=None),
        linear_dt_bias=SimpleNamespace(weight=None),
    )
    mixed = torch.zeros((draft_step + 1, 3))
    ssm = torch.zeros((16, 1, 1, 1))
    state._gdn_mtp_kernel_npu(mixed, None, ssm, None, None, state.infer_state, layer_weight)
    assert conv_calls[0]["mtp_step"] == draft_step
    call = recurrent_calls[0]
    torch.testing.assert_close(call["ssm_state_indices"], torch.tensor([11], dtype=torch.int32))
    torch.testing.assert_close(
        call["ssm_state_write_indices"], torch.arange(8, 9 + draft_step, dtype=torch.int32).view(1, -1)
    )
    assert call["initial_state"] is ssm
    assert call["final_state"] is ssm
