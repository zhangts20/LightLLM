from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.graph.base.decode_graph as cuda_graph_module
from lightllm.common.basemodel.graph import CudaGraph, DecodeGraph


@pytest.fixture(autouse=True)
def _graph_args(monkeypatch):
    args = SimpleNamespace(
        mtp_step=7,
        graph_split_batch_size=4,
        graph_grow_step_size=2,
        enable_decode_microbatch_overlap=False,
        enable_tpsp_mix_mode=False,
        enable_torch_memory_saver=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    backend = SimpleNamespace(
        name="cuda",
        runtime=SimpleNamespace(target_device=lambda: torch.device("cpu")),
        graph=SimpleNamespace(graph_pool_handle=lambda: object()),
    )
    monkeypatch.setattr(cuda_graph_module, "get_backend", lambda: backend)
    return args


def _batch_sizes(max_batch_size, batch_stride=1):
    physical_max_batch_size = max_batch_size * batch_stride
    graph = CudaGraph(
        batch_step_size_before_split=batch_stride,
        split_batch_size=4 * batch_stride,
        batch_step_size_after_split=2 * batch_stride,
        max_batch_size=physical_max_batch_size,
    )
    return graph.cuda_graph_batch_sizes


def test_dynamic_schedule_uses_compacted_physical_rows(_graph_args):
    assert _batch_sizes(max_batch_size=128) == [1, 2, 3, 4, *range(6, 129, 2)]


def test_public_static_schedule_preserves_original_static_mtp_default(_graph_args):
    assert CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=32,
    ) == [
        8,
        16,
        24,
        32,
    ]


def test_instance_and_public_static_schedule_match(_graph_args):
    graph = CudaGraph(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=128,
    )

    assert graph.cuda_graph_batch_sizes == CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=graph.max_batch_size,
        tp_world_size=graph.tp_world_size,
    )


def test_batch_step_size_before_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=4, batch_stride=8) == [8, 16, 24, 32]


def test_batch_step_size_after_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=8, batch_stride=7) == [
        7,
        14,
        21,
        28,
        42,
        56,
    ]


@pytest.mark.parametrize("platform", ["cuda", "maca", "musa", "ascend"])
def test_platform_factory_accepts_dynamic_schedule(monkeypatch, platform):
    if platform == "ascend":
        from lightllm.common.basemodel.graph.acl_graph import AclGraph

        monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=lambda: object()), raising=False)
        expected_class = AclGraph
    else:
        expected_class = CudaGraph
    graph = DecodeGraph(
        16, 8192, 1, platform,
        batch_step_size_before_split=1,
        split_batch_size=4,
        batch_step_size_after_split=2,
        capture_infer_cost=True,
    )
    assert isinstance(graph, expected_class)
    assert graph.graph_batch_sizes == [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    assert graph.cuda_graph_batch_sizes is graph.graph_batch_sizes
    assert graph.capture_infer_cost
    if platform == "ascend":
        assert sorted(graph.attn_params.handles) == graph.graph_batch_sizes
        assert graph._warmup_dummy_seq_len() == 8192


def test_legacy_platform_constructor_and_static_schedule():
    graph = DecodeGraph(32, 8192, 1, "maca")
    assert graph.graph_batch_sizes == [8, 16, 24, 32]
    assert DecodeGraph.gen_cuda_graph_batch_sizes(32, 1) == graph.graph_batch_sizes


def test_dynamic_schedule_aligns_and_deduplicates_tpsp(_graph_args):
    _graph_args.enable_tpsp_mix_mode = True
    graph = CudaGraph(
        max_batch_size=16, tp_world_size=4,
        batch_step_size_before_split=1, split_batch_size=4, batch_step_size_after_split=2,
    )
    assert graph.graph_batch_sizes == [4, 8, 12, 16]


@pytest.mark.parametrize("is_draft,width,rows", [(True, 5, 10), (False, 6, 12), (False, 6, 7)])
@pytest.mark.parametrize("fail_forward", [False, True])
def test_block_warmup_owns_complete_pages_and_restores_target(
    monkeypatch, is_draft, width, rows, fail_forward
):
    graph = CudaGraph(max_batch_size=rows, max_len_in_batch=32,
                      batch_step_size_before_split=1, split_batch_size=4, batch_step_size_after_split=2)
    graph.platform_backend.runtime.synchronize = lambda: None
    table = torch.full((8, 32), 777, dtype=torch.int32)
    before = table.clone()
    free_reqs = [1, 2, 3, 4, 5, 6]
    allocated = []
    released = []

    def alloc_pages(seq_len):
        assert seq_len == width + 1
        tokens = torch.arange(16 * (len(allocated) + 1), 16 * (len(allocated) + 2), dtype=torch.int32)
        allocated.append(tokens)
        return tokens

    req_manager = SimpleNamespace(
        req_to_token_indexs=table,
        alloc=lambda: free_reqs.pop(0),
        free_req=lambda req_idx: free_reqs.append(req_idx),
        alloc_page_aligned_mem_indices=alloc_pages,
    )
    model = SimpleNamespace(
        is_mtp_draft_model=is_draft,
        mtp_manager=SimpleNamespace(get_decode_batch_multiplier=lambda _: width),
        req_manager=req_manager,
        mem_manager=SimpleNamespace(free=lambda tokens: released.append(tokens)),
        _gen_special_model_input=lambda _: {},
    )
    try:
        with graph._block_warmup_input(model, rows) as model_input:
            offsets = torch.arange(rows) % width
            assert torch.equal(model_input.b_mtp_index, offsets)
            assert torch.equal(model_input.b_seq_len, offsets + 2)
            assert model_input.b_req_idx.unique().numel() == (rows + width - 1) // width
            for row in range(rows):
                req = model_input.b_req_idx[row]
                pos = model_input.b_seq_len[row] - 1
                assert table[req, pos] == model_input.mem_indexes[row]
            assert torch.equal(table[0], before[0])
            if fail_forward:
                raise RuntimeError("simulated capture failure")
    except RuntimeError as error:
        assert fail_forward and str(error) == "simulated capture failure"
    assert torch.equal(table, before)
    assert sorted(free_reqs) == [1, 2, 3, 4, 5, 6]
    assert len(released) == len(allocated)
    assert all(tokens.numel() == 16 for tokens in released)


@pytest.mark.parametrize("draft,width", [(True, 5), (False, 6)])
@pytest.mark.parametrize("overlap", [False, True])
def test_block_graph_warmup_caps_capture_and_lookup_to_request_pool(monkeypatch, draft, width, overlap):
    from contextlib import contextmanager
    graph = CudaGraph(max_batch_size=16 * width, batch_step_size_before_split=width,
                      split_batch_size=4 * width, batch_step_size_after_split=2 * width)
    graph.platform_backend.runtime.empty_cache = lambda: None
    active = 0
    captured = []
    @contextmanager
    def input_context(model, rows):
        nonlocal active
        requests = (rows + width - 1) // width
        active += requests
        assert active <= 8
        try:
            yield rows
        finally:
            active -= requests
    monkeypatch.setattr(graph, "_block_warmup_input", input_context)
    monkeypatch.setattr(graph, "_after_capture_batch", captured.append)
    model = SimpleNamespace(is_mtp_draft_model=draft,
                            mtp_manager=SimpleNamespace(get_decode_batch_multiplier=lambda _: width),
                            req_manager=SimpleNamespace(max_request_num=8),
                            forward=lambda _: None, microbatch_overlap_decode=lambda a,b: None)
    graph._warmup_block_graphs(model, overlap)
    maximum = (4 if overlap else 8) * width
    assert max(captured) == maximum
    assert graph.max_batch_size == maximum
    assert graph.cuda_graph_batch_sizes is graph.graph_batch_sizes
    assert graph.find_closest_graph_batch_size(maximum) == maximum
    assert graph.find_closest_graph_batch_size(maximum + 1) is None
    assert active == 0
