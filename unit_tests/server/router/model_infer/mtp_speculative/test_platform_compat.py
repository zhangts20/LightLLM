"""CPU regression checks for paged speculative ownership and runtime routing.

Loads the relevant definitions without importing accelerator-only dependencies.
Run with pytest on a CPU host; these do not replace accelerator integration tests.
"""
import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List, Tuple

import numpy as np
import pytest
import torch

REPO = next(p for p in Path(__file__).resolve().parents if (p / "lightllm").is_dir())
ROOT = REPO / "lightllm"
INFER = ROOT / "server/router/model_infer"


def definitions(path, only=None, **bindings):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (only is None or n.name in only)]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    ns = dict(torch=torch, np=np, List=List, Tuple=Tuple, **bindings)
    exec(compile(tree, str(path), "exec"), ns)
    return ns


class Memory:
    def __init__(self):
        self.next = 64
        self.allocations = []
        self.freed = []

    def alloc(self, count):
        self.allocations.append(count)
        result = torch.arange(self.next, self.next + count, dtype=torch.int32)
        self.next += count
        return result

    def free(self, indexes):
        self.freed.extend(indexes.tolist())


def manager(page):
    # Exercise the actual platform allocator methods, without its GPU constructor.
    ns = definitions(ROOT / "common/req_manager.py", only=["ReqManager"], get_page_size=lambda: page)
    cls = ns["ReqManager"]
    obj = cls.__new__(cls)
    obj.mem_manager = Memory()
    obj.mem_manager.next = ((64 + page - 1) // page) * page
    return obj


@pytest.fixture
def utils(monkeypatch):
    module = types.ModuleType("lightllm.server.router.model_infer.mtp_speculative.proposers.base")
    module.MtpMemIndexesToFree = lambda **kw: NS(free_mask_cpu=None, **kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return definitions(INFER / "mtp_speculative/utils.py")


def install_context(monkeypatch, req_manager):
    context = NS(req_manager=req_manager, radix_cache=None)
    module = types.ModuleType("lightllm.server.router.model_infer.infer_batch")
    module.g_infer_context = context
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return context


@pytest.mark.parametrize("page", [1, 4, 16, 64, 128])
def test_temporary_allocation_frees_padding(monkeypatch, utils, page):
    req_manager = manager(page)
    context = install_context(monkeypatch, req_manager)
    evictions = []
    context.radix_cache = NS(free_radix_cache_to_get_enough_token=evictions.append)
    allocations = []
    indexes = utils["alloc_mem_indexes"](5, allocations=allocations)
    assert len(indexes) == 5
    full_indexes = allocations[0].mem_indexes_cpu
    assert len(full_indexes) == ((5 + page - 1) // page) * page
    assert full_indexes[0].item() % page == 0
    assert evictions == [len(full_indexes)]
    utils["free_mem_indexes"](NS(model=NS(req_manager=req_manager)), allocations)
    assert req_manager.mem_manager.freed == full_indexes.tolist()
    assert utils["alloc_mem_indexes"](0, allocations=allocations).numel() == 0
    assert len(req_manager.mem_manager.allocations) == 1


@pytest.mark.parametrize("page", [4, 16, 64, 128])
def test_rejected_partial_page_stays_owned(utils, page):
    req_manager = manager(page)
    # Accepted page base and tail; a rejected suffix spans a new page.
    base = req_manager.mem_manager.next
    indexes = torch.arange(base, base + page + 2, dtype=torch.int32)
    rejected = torch.arange(len(indexes)) >= 2
    utils["free_mem_indexes"](NS(model=NS(req_manager=req_manager)), [NS(mem_indexes_cpu=indexes, free_mask_cpu=rejected)])
    assert req_manager.mem_manager.freed == list(range(base + page, base + 2 * page))


def test_decode_empty_and_mixed_width(monkeypatch):
    req_manager = manager(4)
    context = install_context(monkeypatch, req_manager)
    ns = definitions(INFER / "mode_backend/generic_pre_process.py", g_infer_context=context, ModelInput=NS, INT64_MAX=2**63-1)
    empty, run = ns["prepare_decode_inputs"]([])
    assert empty.batch_size == 0 and run == []
    assert empty.mem_indexes_cpu.numel() == 0
    def req(index, length, last, step):
        return NS(req_idx=index, cur_kv_len=length-1, mtp_step=step,
                  last_kv_mem_index=last, get_cur_total_len=lambda: length,
                  get_radix_cache_shared_len=lambda: 0, shared_kv_node=None, multimodal_params={})
    requests = [req(1, 4, 10, 2), req(2, 8, 22, 1)]
    model_input, run = ns["prepare_decode_inputs"](requests)
    assert model_input.mem_indexes_cpu.tolist() == [11, 64, 65, 23, 68]
    assert [r.last_kv_mem_index for r in requests] == [11, 23]
    tree = ast.parse((INFER / "mode_backend/base_backend.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModeBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_update_mtp_last_kv_mem_index")
    ns = dict(torch=torch, List=List, InferReq=NS)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "last_kv_update", "exec"), ns)
    ns["_update_mtp_last_kv_mem_index"](None, run, model_input.mem_indexes_cpu, torch.tensor([1,1,0,1,1]))
    assert [r.last_kv_mem_index for r in requests] == [64, 68]


@pytest.mark.parametrize("page", [4, 16, 64, 128])
def test_compaction_keeps_per_request_slots(monkeypatch, utils, page):
    req_manager = manager(page)
    req_manager.req_sampling_params_manager = NS(req_to_next_token_scores=torch.tensor([[1., .1, .1], [1., .9, .9]]))
    class Copy:
        def __init__(self, value): self.tensor = value.clone()
        def wait(self): pass
    pin = types.ModuleType("lightllm.server.router.model_infer.pin_mem_manager")
    pin.g_pin_mem_manager = NS(async_copy_from_gpu_tensor_with_event=lambda **kw: Copy(kw["gpu_tensor"]))
    monkeypatch.setitem(sys.modules, pin.__name__, pin)
    model_input = NS(batch_size=6, b_req_idx=torch.tensor([0,0,0,1,1,1]),
        b_mtp_index=torch.tensor([0,1,2,0,1,2]), b_seq_len=torch.tensor([4,5,6,4,5,6]),
        mem_indexes_cpu=torch.tensor([page-1, 2*page, 2*page+1, 4*page-1, 4*page, 4*page+1]),
        mem_indexes=torch.tensor([page-1, 2*page, 2*page+1, 4*page-1, 4*page, 4*page+1]),
        multimodal_params=list(range(6)), to_device=lambda device: None)
    backend = NS(max_draft_step=2, model=NS(req_manager=req_manager), backend_runtime=NS(target_device=lambda: "cpu"))
    result, mask = utils["compact_decode_input"](backend, model_input, 2, NS(pre_draft_step=2,dynamic_batch_size=4))
    assert mask.tensor.tolist() == [True,False,False,True,True,True]
    assert result.mem_indexes_cpu.tolist() == [page-1, 4*page-1, 4*page, 4*page+1]
    assert result.mem_indexes.tolist() == [page-1, 4*page-1, 4*page, 4*page+1]
    assert result.multimodal_params == [0,3,4,5]
    assert req_manager.mem_manager.freed == list(range(2*page,3*page))
    assert torch.equal(req_manager.req_sampling_params_manager.req_to_next_token_scores, torch.tensor([[1.,.1,.1],[1.,.9,.9]]))


def test_portable_chained_tokens():
    ns = definitions(INFER / "mtp_speculative/row_ops.py")
    draft = torch.tensor([90,91,92,93,94])
    result = ns["build_chained_mtp_decode_input_inplace"](torch.tensor([10,11,12,20,21]), draft, torch.tensor([0,3]), torch.tensor([3,1]))
    assert result is draft
    assert result.tolist() == [11,12,92,93,94]


@pytest.mark.parametrize("page", [16, 64, 128])
def test_dspark_scratch_shapes_and_full_page_cleanup(monkeypatch, utils, page):
    import copy
    req_manager = manager(page)
    install_context(monkeypatch, req_manager)
    forwards = []
    def forward(value):
        if len(forwards):
            assert value.batch_size == 6
            assert value.mem_indexes.shape == (6,)
        forwards.append(value)
        return NS(mtp_collector=NS(draft_token_ids=torch.arange(6), confidence_logits=None))
    backend = NS(draft_models=[NS(block_size=3, mask_token_id=99, forward=forward)])
    class Base:
        def __init__(self, **kw): self.__dict__.update(kw)
    ns = definitions(INFER / "mtp_speculative/proposers/dspark.py", copy=copy,
        BaseSpecProposer=Base, DSparkSpecProposal=NS, mtp_utils=NS(**utils),
        g_pin_mem_manager=NS(get_const_gpu_tensor=lambda **kw: torch.full(kw["shape"], kw["fill_value"], dtype=kw["dtype"])))
    proposer = ns["DSparkProposer"](backend=backend, enable_dynmaic_mtp=False)
    inp = NS(is_prefill=False, batch_size=4, max_kv_seq_len=10,
        b_req_idx=torch.tensor([0,0,1,1]), b_seq_len=torch.tensor([9,10,9,10]),
        b_mtp_index=torch.tensor([0,1,0,1]), b_position_delta=torch.zeros(4,dtype=torch.int32),
        b_shared_seq_len=torch.zeros(4,dtype=torch.int32), b_shared_radix_node_id=torch.zeros(4,dtype=torch.int64))
    proposal = proposer.propose_next(inp, NS(mtp_collector=NS(spec_hidden=torch.zeros(4,8))),
        torch.tensor([10,11,20,21]), torch.tensor([0,2]), 2, torch.tensor([1,2]))
    assert proposal.token_ids.shape == (2,2)
    assert proposal.extra_mem_indexes_cpu[0].mem_indexes_cpu.numel() == page
    utils["free_mem_indexes"](NS(model=NS(req_manager=req_manager)), proposal.extra_mem_indexes_cpu)
    base = ((64 + page - 1) // page) * page
    assert req_manager.mem_manager.freed == list(range(base,base+page))


def test_pin_manager_routes_events_and_device_cache(monkeypatch):
    import threading
    import collections
    from dataclasses import dataclass
    from typing import Dict, Union, Sequence, Any
    events = []
    class Event:
        def record(self): events.append("record")
        def synchronize(self): events.append("wait")
    runtime = NS(create_event=Event, target_device=lambda: torch.device("cpu"))
    ns = definitions(INFER / "pin_mem_manager.py", threading=threading, collections=collections,
        dataclass=dataclass, Dict=Dict, Union=Union, Sequence=Sequence, Any=Any,
        get_backend=lambda: NS(runtime=runtime))
    pin = ns["PinMemTensorManager"]()
    pin.async_copy_from_gpu_tensor = lambda **kw: kw["gpu_tensor"].clone()
    copied = pin.async_copy_from_gpu_tensor_with_event("test", torch.tensor([1]))
    copied.wait()
    assert events == ["record", "wait"]
    first = pin.get_const_gpu_tensor("same", (2,), 0, torch.int32)
    other = pin.get_const_gpu_tensor("same", (2,), 1, torch.int64)
    assert first.tolist() == [0,0] and other.tolist() == [1,1]
    assert first.dtype == torch.int32 and other.dtype == torch.int64
    runtime.target_device = lambda: torch.device("meta")
    assert pin.get_const_gpu_tensor("same", (2,), 0, torch.int32).device.type == "meta"


def test_argmax_handlers_require_only_logits():
    tree = ast.parse((INFER / "mode_backend/base_backend.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModeBackend")
    names = {"_gen_argmax_token_ids", "_gen_argmax_token_ids_and_prob"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(torch=torch, ModelOutput=NS)
    exec(compile(ast.Module(body=methods, type_ignores=[]), "argmax_handlers", "exec"), ns)
    logits = torch.tensor([[0., 3., 1.], [5., 1., 2.]])
    output = NS(logits=logits)
    # Neither handler may assume model, draft registry, vocabulary map, or collector state.
    assert ns["_gen_argmax_token_ids"](NS(), output).tolist() == [1, 0]
    tokens, probs = ns["_gen_argmax_token_ids_and_prob"](NS(), output)
    assert tokens.tolist() == [1, 0]
    torch.testing.assert_close(probs, logits.softmax(-1).amax(-1))


@pytest.mark.parametrize("page", [16, 64, 128])
def test_eagle_logical_pages_and_repeated_cleanup(monkeypatch, utils, page):
    req_manager = manager(page)
    req_manager.mem_manager.next = 10 * page
    context = install_context(monkeypatch, req_manager)
    evictions = []
    context.radix_cache = NS(free_radix_cache_to_get_enough_token=evictions.append)
    seq_lens = torch.tensor([page-1, page, 2*page-2], dtype=torch.int32)
    target_tails = torch.tensor([3*page-2, 5*page-1, 7*page-3], dtype=torch.int32)
    original_tails = target_tails.clone()
    backend = NS(model=NS(req_manager=req_manager))
    for iteration in range(2):
        start = req_manager.mem_manager.next
        allocations = []
        scratch = utils["alloc_eagle_mem_indexes"](
            seq_lens, target_tails, 3, allocations=allocations
        ).view(3, 3)
        expected = torch.tensor([
            [3*page-1, start, 7*page-2],
            [start+page, start+1, 7*page-1],
            [start+page+1, start+2, start+2*page],
        ], dtype=torch.int32)
        assert torch.equal(scratch, expected)
        for step in range(3):
            assert torch.equal(scratch[step] % page, (seq_lens + step) % page)
        assert torch.equal(target_tails, original_tails)
        assert len(allocations) == 1
        assert allocations[0].mem_indexes_cpu.numel() == 3 * page
        utils["free_mem_indexes"](backend, allocations)
        assert req_manager.mem_manager.freed[iteration*3*page:] == list(range(start, start+3*page))
    assert req_manager.mem_manager.allocations == [page] * 6
    assert all(need >= page for need in evictions)
    # Neither borrowed target pages nor pending rejected target pages were freed.
    assert min(req_manager.mem_manager.freed) >= 10 * page


@pytest.mark.parametrize("req_num, steps", [(0, 3), (2, 0)])
def test_eagle_empty_scratch_never_touches_allocator(utils, req_num, steps):
    allocations = []
    assert utils["alloc_eagle_mem_indexes"](
        torch.zeros(req_num, dtype=torch.int32), torch.zeros(req_num, dtype=torch.int32),
        steps, allocations=allocations,
    ).numel() == 0
    assert allocations == []


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_eagle_unpaged_scratch_owns_every_slot_without_host_copy(monkeypatch, utils, device):
    req_manager = manager(1)
    context = install_context(monkeypatch, req_manager)
    evictions = []
    context.radix_cache = NS(free_radix_cache_to_get_enough_token=evictions.append)
    allocations = []
    # Meta tensors cannot be copied to CPU: this also guards against accidental
    # host reads of either tensor on the unpaged fast path.
    scratch = utils["alloc_eagle_mem_indexes"](
        torch.tensor([10,20], device=device), torch.tensor([8,19], device=device),
        3, allocations=allocations,
    )
    assert scratch.tolist() == list(range(64,70))
    assert req_manager.mem_manager.allocations == [6]
    assert evictions == [6]
    assert len(allocations) == 1
    assert allocations[0].mem_indexes_cpu.tolist() == scratch.tolist()
    utils["free_mem_indexes"](NS(model=NS(req_manager=req_manager)), allocations)
    assert req_manager.mem_manager.freed == list(range(64,70))


@pytest.mark.parametrize("page", [16, 64, 128])
def test_eagle_within_retained_page_never_frees_it(monkeypatch, utils, page):
    req_manager = manager(page)
    install_context(monkeypatch, req_manager)
    allocations = []
    # Both requests have room for three more positions in their retained pages.
    tails = torch.tensor([2*page+2, 4*page+4], dtype=torch.int32)
    scratch = utils["alloc_eagle_mem_indexes"](
        torch.tensor([3,5], dtype=torch.int32), tails, 3, allocations=allocations
    ).view(3,2)
    assert torch.equal(scratch, tails[None, :] + torch.arange(1,4)[:,None])
    assert allocations == []
    assert req_manager.mem_manager.allocations == []
    utils["free_mem_indexes"](NS(model=NS(req_manager=req_manager)), allocations)
    assert req_manager.mem_manager.freed == []


def test_pin_constant_cache_uses_injected_runtime(pin_mem_test_runtime):
    from lightllm.server.router.model_infer.pin_mem_manager import PinMemTensorManager

    manager = PinMemTensorManager()
    first = manager.get_const_gpu_tensor("test_constant", (4,), False, torch.bool)
    again = manager.get_const_gpu_tensor("test_constant", (2,), False, torch.bool)
    assert first.device == pin_mem_test_runtime.target_device()
    assert not first.any().item()
    assert first.data_ptr() == again.data_ptr()
