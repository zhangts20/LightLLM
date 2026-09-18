"""CPU contract tests for paged block attention; accelerator calls are recorded.

Load the backend in isolation so these tests need neither Triton nor a device
runtime. The actual backend state and dispatch methods run on CPU tensors.
"""
import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Optional

import pytest
import torch


ROOT = Path(__file__).resolve().parents[5]


def load_definitions(path, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def harness():
    args = SimpleNamespace(mtp_mode="dspark", mtp_step=5, mtp_dynamic_verify=False)
    base = load_definitions(ROOT / "lightllm/common/basemodel/attention/base_att.py", {
        "torch": torch, "dataclass": dataclasses.dataclass,
        "ABC": object, "abstractmethod": lambda f: f,
        "threading": __import__("threading"), "Optional": Optional,
        "Tuple": __import__("typing").Tuple, "Union": __import__("typing").Union,
        "Dict": dict, "get_env_start_args": lambda: args,
    })
    manager = load_definitions(ROOT / "lightllm/common/basemodel/mtp_manager.py", {
        "ClassVar": ClassVar, "Optional": Optional, "get_env_start_args": lambda: args,
        "HiddenCollector": object,
    }).MtpManager()
    calls = []

    def attention(**kw):
        calls.append(kw)
        return kw["q"].clone()

    def page_copy(**kw):
        table = kw["page_table"]
        ids = kw["b_req_idx"]
        table.copy_(kw["req_to_token_indexs"][ids, :table.shape[1] * kw["page_size"]:kw["page_size"]] // kw["page_size"])

    class GatherKernel:
        def __getitem__(self, grid):
            def run(k, v, out_k, out_v, mapping, req_ids, lengths, cu_k, *strides):
                for i, req in enumerate(req_ids.tolist()):
                    n, start = int(lengths[i]), int(cu_k[i])
                    ids = mapping[req, :n].long()
                    out_k[start:start + n].copy_(k[ids])
                    out_v[start:start + n].copy_(v[ids])
            return run

    fp = load_definitions(ROOT / "lightllm/common/basemodel/attention/paged_fa3/fp.py", {
        "torch": torch, "dataclasses": dataclasses, "Any": Any,
        "BaseAttBackend": base.BaseAttBackend, "BasePrefillAttState": base.BasePrefillAttState,
        "BaseDecodeAttState": base.BaseDecodeAttState, "AttControl": base.AttControl,
        "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y, jit=lambda f: GatherKernel()),
        "tl": SimpleNamespace(constexpr=int),
        "register_att_backend": lambda **kw: lambda cls: cls,
        "get_env_start_args": lambda: args, "get_current_device_id": lambda: "cpu",
        "page_table_copy": page_copy,
        "gen_cumsum_pad0_tensor": lambda q, k: tuple(torch.cat((torch.zeros(1, dtype=torch.int32), x.cumsum(0))) for x in (q, k)),
        "flash_attn_with_kvcache": attention,
        "maca_flash_attn_with_kvcache": attention, "maca_flash_attn_varlen_func": attention,
    })

    def make(*, draft=True, maca=True, groups=2, padding=0, graph_max=32, state_type=None, npu=False):
        width = manager.get_decode_batch_multiplier(draft)
        rows = groups * width + padding
        req_ids = torch.cat((torch.arange(1, groups + 1).repeat_interleave(width), torch.zeros(padding, dtype=torch.long)))
        lengths = torch.cat((torch.arange(17, 17 + width).repeat(groups), torch.ones(padding, dtype=torch.long))).int()
        model = type("Model", (), {})()
        model.__dict__.update(is_mtp_draft_model=draft, mtp_manager=manager,
                                graph_max_batch_size=graph_max, graph_max_len_in_batch=64,
                                req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(4 * 64).reshape(4, 64)))
        backend = fp.PagedFa3AttBackend(model, page_size=16)
        backend.is_maca = maca
        infer = SimpleNamespace(batch_size=rows, b_req_idx=req_ids, b_seq_len=lengths,
                                input_ids=torch.zeros(rows, dtype=torch.long), max_kv_seq_len=32,
                                microbatch_index=0, b1_cu_q_seq_len=torch.arange(rows + 1).int(),
                                b1_cu_kv_seq_len=torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0))).int())
        if npu:
            infer.input_ids = SimpleNamespace(device=SimpleNamespace(type="npu"))
        state = (state_type or fp.PagedFa3DecodeAttState)(backend=backend, infer_state=infer)
        state.init_state()
        return state

    # The constructor also consults the platform before dispatch.
    args.hardware_platform = "cuda"
    return SimpleNamespace(args=args, make=make, calls=calls, fp=fp)


@pytest.mark.parametrize("maca", [True, False])
@pytest.mark.parametrize("draft,width,causal", [(True, 5, False), (False, 6, True)])
def test_fixed_block_layout_and_dispatch(harness, maca, draft, width, causal):
    s = harness.make(draft=draft, maca=maca)
    assert s.decode_max_q_seq_len == width
    assert s.cu_seqlens_q.tolist() == [0, width, 2 * width]
    assert s.b_att_seq_len.tolist() == [16 + width] * 2
    if draft:
        assert s.page_table is None
        assert s.b_block_req_idx.tolist() == [1, 2]
    else:
        assert s.page_table[:, :2].tolist() == [[4, 5], [8, 9]]
    q, k = torch.randn(2 * width, 2, 8), torch.randn(256, 1, 8)
    torch.testing.assert_close(s.decode_att(q, k, k), q)
    call = harness.calls[-1]
    assert call["causal"] is causal
    assert call["q"].shape == ((2, width, 2, 8) if maca and not draft else q.shape)
    if draft:
        assert s.block_max_kv_len == 64
    else:
        assert s.page_table.shape == (2, 4)  # graph-wide row stride


@pytest.mark.parametrize("maca", [True, False])
@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("padding", [1, 3, 5])
def test_capture_padding_and_graph_max_buffers(harness, maca, draft, padding):
    width = 5 if draft else 6
    rows = 2 * width + padding
    s = harness.make(maca=maca, draft=draft, padding=padding, graph_max=rows)
    assert s.cu_seqlens_q[-1] == rows
    if draft:
        assert s.b_block_req_idx.shape == ((rows + width - 1) // width,)
        assert s.block_max_kv_len == 64
    else:
        assert s.page_table.shape == ((rows + width - 1) // width, 4)
        assert s.page_table.untyped_storage().data_ptr() == s.backend.get_page_table_buffer()[0].untyped_storage().data_ptr()
    assert s.b_att_seq_len[-1] == 1
    q, k = torch.randn(rows, 2, 8), torch.randn(256, 1, 8)
    assert s.decode_att(q, k, k).shape == q.shape
    if maca and rows % width:
        assert "cu_seqlens_q" in harness.calls[-1]
        assert harness.calls[-1]["cu_seqlens_q"][-1] == rows


@pytest.mark.parametrize("mode,step", [(None, 0), (None, 5), ("vanilla_with_att", 5), ("vanilla_no_att", 5),
                              ("eagle_with_att", 5), ("eagle_no_att", 5), ("eagle3", 5)])
def test_existing_native_layout_is_preserved(harness, mode, step):
    harness.args.mtp_mode, harness.args.mtp_step = mode, step
    for draft in (True, False):
        s = harness.make(draft=draft)
        assert s.decode_max_q_seq_len == (step + 1 if mode is not None and not draft else 1)
        assert s.causal


def test_dynamic_main_verify_rejected_but_fixed_draft_allowed(harness):
    harness.args.mtp_dynamic_verify = True
    with pytest.raises(NotImplementedError, match="mtp_dynamic_verify=False"):
        harness.make(draft=False)
    assert harness.make(draft=True).decode_max_q_seq_len == 5


def test_causal_override_cannot_silently_inherit_noncausal_layout(harness):
    class Int8Override(harness.fp.PagedFa3DecodeAttState):
        def _normal_decode_att(self, *args, **kwargs):
            raise AssertionError("must not reach causal-only override")
    with pytest.raises(NotImplementedError, match="supported CUDA or MACA backend"):
        harness.make(state_type=Int8Override)
    assert harness.make(state_type=Int8Override, draft=False).causal


def test_eager_outside_graph_capacity(harness):
    s = harness.make(graph_max=1, draft=False)
    assert s.page_table.shape == (2, 2)
    assert s.page_table[:, :2].tolist() == [[4, 5], [8, 9]]


def test_ascend_native_mtp_retains_per_row_bnsd(harness):
    harness.args.mtp_mode = "vanilla_with_att"
    s = harness.make(npu=True, maca=False, draft=False)
    assert s.use_mtp_bnsd
    assert s.page_table.shape == (12, 4)
    assert s.page_table[:, 0].tolist() == [4] * 6 + [8] * 6
    assert s.causal


def test_ascend_noncausal_block_rejected(harness):
    with pytest.raises(NotImplementedError, match="supported CUDA or MACA backend"):
        harness.make(npu=True, maca=False)


def test_maca_rejects_wrong_query_count(harness):
    s = harness.make()
    with pytest.raises(ValueError, match="Unexpected MetaX decode query shape"):
        s.decode_att(torch.randn(9, 2, 8), torch.randn(256, 1, 8), torch.randn(256, 1, 8))


def test_padded_graph_replay_copies_runtime_lengths(harness):
    captured = harness.make(padding=3, graph_max=13)
    # Reuse the same backend/storage, as graph replay does, with new KV lengths.
    infer = SimpleNamespace(**vars(captured.infer_state))
    infer.b_seq_len = infer.b_seq_len + 2
    runtime = harness.fp.PagedFa3DecodeAttState(backend=captured.backend, infer_state=infer)
    runtime.init_state()
    captured.copy_for_decode_cuda_graph(runtime)
    assert captured.b_att_seq_len.tolist() == [23, 23, 3]
    assert captured.cu_seqlens_k.tolist() == [0, 23, 46, 49]
    assert captured.cu_seqlens_q.tolist() == [0, 5, 10, 13]


def test_microbatch_buffers_do_not_alias(harness):
    first = harness.make(draft=False)
    infer = SimpleNamespace(**vars(first.infer_state))
    infer.microbatch_index = 1
    second = harness.fp.PagedFa3DecodeAttState(backend=first.backend, infer_state=infer)
    second.init_state()
    assert first.page_table.data_ptr() != second.page_table.data_ptr()
    torch.testing.assert_close(first.page_table[:, :2], second.page_table[:, :2])


@pytest.mark.parametrize("maca", [True, False])
def test_fragmented_partial_page_uses_exact_prefix_and_scratch_slots(harness, maca):
    s = harness.make(maca=maca, groups=1)
    mapping = s.backend.model.req_manager.req_to_token_indexs
    # Prefix ends at logical offset 16; scratch starts in a fresh physical page
    # rather than continuing the retained prefix's physical page.
    slots = torch.cat((torch.arange(32, 48), torch.arange(192, 197)))
    mapping[1, :21] = slots
    mapping[1, 21:] = -1  # invalid unused suffix must never be dereferenced
    mapping_before = mapping.clone()
    k = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 1, 8)
    v = -k
    k_before, v_before = k.clone(), v.clone()
    s.decode_att(torch.zeros(5, 2, 8), k, v)
    call = harness.calls[-1]
    if maca:
        assert "block_table" not in call
        torch.testing.assert_close(call["k"][:21], k[slots])
        torch.testing.assert_close(call["v"][:21], v[slots])
        assert call["cu_seqlens_k"].tolist() == [0, 21]
    else:
        assert call["k_cache"].shape[1] == 1
        assert call["page_table"][0, :21].tolist() == slots.tolist()
    assert call["causal"] is False
    torch.testing.assert_close(mapping, mapping_before)
    torch.testing.assert_close(k, k_before)
    torch.testing.assert_close(v, v_before)


@pytest.mark.parametrize("mode", ["vanilla_with_att", "eagle_with_att", "eagle3"])
@pytest.mark.parametrize("platform", ["cuda", "maca", "ascend"])
def test_native_draft_unit_width_capture_and_replay(harness, mode, platform):
    harness.args.mtp_mode = mode
    s = harness.make(maca=platform == "maca", npu=platform == "ascend", groups=3, graph_max=3)
    assert s.decode_max_q_seq_len == 1
    assert not s.use_mtp_bnsd
    assert s.cu_seqlens_q.tolist() == [0, 1, 2, 3]
    assert s.page_table.shape == (3, 4)
    assert s.b_att_seq_len.tolist() == [17, 17, 17]
    if platform != "ascend":
        q, k = torch.randn(3, 2, 8), torch.randn(256, 1, 8)
        assert s.decode_att(q, k, k).shape == q.shape
        assert harness.calls[-1]["causal"]
    infer = SimpleNamespace(**vars(s.infer_state))
    infer.b_seq_len = infer.b_seq_len + 1
    replay = harness.fp.PagedFa3DecodeAttState(backend=s.backend, infer_state=infer)
    replay.init_state()
    s.copy_for_decode_cuda_graph(replay)
    assert s.b_att_seq_len.tolist() == [18, 18, 18]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/MACA Triton device runtime")
@pytest.mark.parametrize("page_size", [16, 64])
def test_device_gather_fragmented_block_and_padded_capacity(page_size):
    from lightllm.common.basemodel.attention.paged_fa3.fp import _gather_block_kv_kernel

    # Noncontiguous head views match K/V slices of a combined KV cache.
    cache = torch.randn(8 * page_size, 4, 16, device="cuda", dtype=torch.float16)
    k, v = cache[:, :2], cache[:, 2:]
    max_len = 2 * page_size
    mapping = torch.full((3, max_len), -1, dtype=torch.int32, device="cuda")
    prefix = page_size + 1
    slots = torch.cat((torch.arange(prefix), torch.arange(4 * page_size, 4 * page_size + 5))).cuda()
    mapping[1, :prefix + 5] = slots.int()
    mapping[2, :3] = torch.tensor([9, 7, 6], device="cuda")
    mapping[0, 0] = 0
    req_ids = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([prefix + 5, 3, 1], dtype=torch.int32, device="cuda")
    cu_k = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0).int()))
    out_k = torch.full((3 * max_len, 2, 16), float("nan"), device="cuda", dtype=k.dtype)
    out_v = torch.empty_like(out_k)
    _gather_block_kv_kernel[((max_len * 32 + 255) // 256, 3)](
        k, v, out_k, out_v, mapping, req_ids, lengths, cu_k,
        mapping.stride(0), k.stride(0), v.stride(0), k.stride(1), v.stride(1),
        k.stride(2), v.stride(2), 2, 16, 256,
    )
    expected_slots = torch.cat((slots, torch.tensor([9, 7, 6, 0], device="cuda")))
    torch.testing.assert_close(out_k[:prefix + 9], k[expected_slots])
    torch.testing.assert_close(out_v[:prefix + 9], v[expected_slots])
    assert torch.isnan(out_k[prefix + 9:]).all()


@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("padding", [0, 1, 3])
def test_int8_maca_block_token_mapping_and_partial_padding(harness, monkeypatch, draft, padding):
    import sys
    import types
    from typing import Callable

    path = ROOT / "lightllm/common/basemodel/attention/paged_fa3/int8kv.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PagedFa3Int8KVMacaDecodeAttState"]
    ns = {"PagedFa3DecodeAttState": harness.fp.PagedFa3DecodeAttState,
          "torch": torch, "AttControl": harness.fp.AttControl, "Callable": Callable}
    exec(compile(tree, str(path), "exec"), ns)
    calls = []
    def kernel(**kw):
        calls.append(kw)
        return kw["q"].clone()
    module = types.ModuleType("maca_int8kv_flash_decoding")
    module.int8kv_flash_decode = kernel
    monkeypatch.setitem(sys.modules, "lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.maca_int8kv_flash_decoding", module)
    state = harness.make(draft=draft, groups=1, padding=padding, state_type=ns["PagedFa3Int8KVMacaDecodeAttState"])
    state.backend.quant_group_size = 8
    width = 5 if draft else 6
    q = torch.randn(width + padding, 2, 8)
    k = torch.zeros(256, 1, 8, dtype=torch.int8)
    scales = torch.ones(256, 1, 1)
    torch.testing.assert_close(state.decode_att(q, (k, scales), (k, scales)), q)
    call = calls[-1]
    assert call["causal"] is (not draft)
    if draft:
        assert call["page_table"] is state.backend.model.req_manager.req_to_token_indexs
        assert call["token_req_indices"] is state.b_block_req_idx
    else:
        assert call["page_table"] is state.page_table
        assert call["token_req_indices"] is None
    assert call["q"].shape[1] == width


@pytest.mark.parametrize("prefix_lens", [[0, 0, 0], [256, 128, 0]])
def test_int8_prefill_prefix_count_uses_current_input_schema(monkeypatch, prefix_lens):
    import os
    path = ROOT / "lightllm/common/basemodel/attention/paged_fa3/int8kv.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PagedFa3Int8KVMacaPrefillAttState")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "init_state")
    ns = {"torch": torch, "os": os, "DEFAULT_MACA_DEQUANT_CHUNK_TOKENS": 16384, "DEFAULT_MACA_ONESHOT_MAX_BYTES": 1 << 30}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    # Third row models HOLD padding; its query rows increase total and input
    # equally, leaving the real requests' cached-prefix count unchanged.
    state = SimpleNamespace(infer_state=SimpleNamespace(
        total_token_num=sum(prefix_lens) + 10, input_ids=torch.empty(10, device="meta"),
        b1_cu_q_seq_len=torch.tensor([0, 5, 8, 10]),
        b_ready_cache_len=torch.tensor(prefix_lens), b_req_idx=torch.tensor([1, 2, 0])))
    ns["init_state"](state)
    assert state.prefix_total_token_num == sum(prefix_lens)
    assert state.max_prefix_len == max(prefix_lens)
    if sum(prefix_lens):
        assert state.request_slices == ((0, 5, 1, 256), (5, 8, 2, 128), (8, 10, 0, 0))
    else:
        assert state.request_slices is None


@pytest.mark.parametrize("groups", [1, 3])
def test_native_eagle_extension_groups_request_prefixes(harness, groups):
    harness.args.mtp_mode = "eagle_with_att"
    harness.args.mtp_step = 3
    state = harness.make(draft=True, groups=1)
    infer = state.infer_state
    infer.batch_size = groups * 4
    infer.input_ids = torch.zeros(groups * 4, dtype=torch.long)
    infer.b1_cu_q_seq_len = torch.arange(groups * 4 + 1).int()
    infer.decode_query_group_size = 4
    infer.b_req_idx = torch.arange(1, groups + 1).repeat_interleave(4)
    infer.b_seq_len = (torch.arange(17, 21).repeat(groups) +
                       torch.arange(groups).repeat_interleave(4) * 4).int()
    state.init_state()
    assert state.decode_max_q_seq_len == 4
    assert state.page_table.shape[0] == groups
    assert state.b_att_seq_len.tolist() == [20 + i * 4 for i in range(groups)]
    assert state.cu_seqlens_q.tolist() == list(range(0, (groups + 1) * 4, 4))
    grouped_table = state.page_table.clone()
    # The very same row count in a recurrent batch means independent queries.
    infer.decode_query_group_size = 1
    state.init_state()
    assert state.decode_max_q_seq_len == 1
    assert state.page_table.shape[0] == groups * 4
    torch.testing.assert_close(state.page_table[::4, :2], grouped_table[:, :2])


def test_dspark_cannot_accidentally_use_native_group_override(harness):
    state = harness.make(draft=True)
    state.infer_state.decode_query_group_size = 4
    with pytest.raises(AssertionError):
        state.init_state()
