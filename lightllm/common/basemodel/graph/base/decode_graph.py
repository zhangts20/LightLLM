import bisect
from contextlib import contextmanager, ExitStack
import copy
import triton
import torch
import torch.distributed as dist
from typing import Optional
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.torch_memory_saver_utils import TorchMemorySaverWrapper
from lightllm.platform import get_backend


logger = init_logger(__name__)

_DECODE_GRAPH_REGISTRY: dict[str, type["DecodeGraph"]] = {}


def register_decode_graph(*platforms: str):
    """Register a DecodeGraph subclass for one or more hardware platforms."""

    def decorator(cls: type["DecodeGraph"]) -> type["DecodeGraph"]:
        for platform in platforms:
            if platform in _DECODE_GRAPH_REGISTRY:
                existing = _DECODE_GRAPH_REGISTRY[platform]
                raise ValueError(
                    f"DecodeGraph for platform {platform!r} already registered as "
                    f"{existing.__module__}.{existing.__qualname__}"
                )
            _DECODE_GRAPH_REGISTRY[platform] = cls
        return cls

    return decorator


class DecodeGraph:

    @staticmethod
    def gen_cuda_graph_batch_sizes(
        max_batch_size: int = 8,
        tp_world_size: int = 1,
        *,
        batch_step_size_before_split: Optional[int] = None,
        split_batch_size: Optional[int] = None,
        batch_step_size_after_split: Optional[int] = None,
    ):
        args = get_env_start_args()
        if batch_step_size_before_split is None:
            batch_step_size_before_split = args.mtp_step + 1
        if split_batch_size is None:
            split_batch_size = args.graph_split_batch_size * batch_step_size_before_split
        if batch_step_size_after_split is None:
            batch_step_size_after_split = args.graph_grow_step_size * batch_step_size_before_split

        # Generate CUDA Graph batch sizes in two phases with independent steps:
        # use batch_step_size_before_split up to split_batch_size, then use
        # batch_step_size_after_split above it. For example, given
        # batch_step_size_before_split=8, split_batch_size=32,
        # batch_step_size_after_split=16, and max_batch_size=80, the result is
        # [8, 16, 24, 32, 48, 64, 80]. max_batch_size is always included.

        batch_sizes = list(range(batch_step_size_before_split, split_batch_size + 1, batch_step_size_before_split))
        batch_sizes.extend(
            range(split_batch_size + batch_step_size_after_split, max_batch_size, batch_step_size_after_split)
        )
        batch_sizes = sorted({size for size in batch_sizes if size < max_batch_size} | {max_batch_size})

        if args.enable_tpsp_mix_mode:
            batch_sizes = sorted({triton.cdiv(size, tp_world_size) * tp_world_size for size in batch_sizes})
        assert batch_sizes[-1] == max_batch_size
        return batch_sizes

    def __new__(
        cls,
        max_batch_size: int = 8,
        max_len_in_batch: int = 8192,
        tp_world_size: int = 1,
        platform_backend: str = "cuda",
        *,
        batch_step_size_before_split: Optional[int] = None,
        split_batch_size: Optional[int] = None,
        batch_step_size_after_split: Optional[int] = None,
        capture_infer_cost: bool = False,
    ):
        if cls is not DecodeGraph:
            return object.__new__(cls)
        if platform_backend == "ascend" and "ascend" not in _DECODE_GRAPH_REGISTRY:
            import lightllm.common.basemodel.graph.acl_graph as _acl_graph  # noqa: F401

        impl_cls = _DECODE_GRAPH_REGISTRY.get(platform_backend)
        if impl_cls is None:
            raise RuntimeError(
                f"No DecodeGraph registered for platform {platform_backend!r}. "
                f"Registered: {sorted(_DECODE_GRAPH_REGISTRY)}"
            )
        return object.__new__(impl_cls)

    def __init__(
        self,
        max_batch_size: int = 8,
        max_len_in_batch: int = 8192,
        tp_world_size: int = 1,
        platform_backend: str = "cuda",
        *,
        batch_step_size_before_split: Optional[int] = None,
        split_batch_size: Optional[int] = None,
        batch_step_size_after_split: Optional[int] = None,
        capture_infer_cost: bool = False,
    ):
        self.args = get_env_start_args()
        self.platform_backend = get_backend()
        self.target_device = self.platform_backend.runtime.target_device()
        self.mempool = self.platform_backend.graph.graph_pool_handle()
        self.tp_world_size = tp_world_size
        self.capture_infer_cost = capture_infer_cost
        self.infer_cost_ms_by_batch_size = {}
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = self.args.enable_decode_microbatch_overlap
        self.torch_memory_saver = TorchMemorySaverWrapper(self.args.enable_torch_memory_saver)
        self.graph_batch_sizes = self.gen_cuda_graph_batch_sizes(
            batch_step_size_before_split=batch_step_size_before_split,
            split_batch_size=split_batch_size,
            batch_step_size_after_split=batch_step_size_after_split,
            max_batch_size=max_batch_size,
            tp_world_size=tp_world_size,
        )
        self.cuda_graph_batch_sizes = self.graph_batch_sizes
        self.graph: dict[int, tuple] = {}
        self._init_decode_graph_extra()
        logger.info(f"cuda graph batch_sizes: {self.graph_batch_sizes}")

    def _init_decode_graph_extra(self):
        pass

    def _after_capture_batch(self, batch_size: int) -> None:
        pass

    def _warmup_dummy_seq_len(self) -> int:
        # CUDA graph captures kernel launches; b_seq_len is a tensor and can vary at replay.
        # Dummy decode only needs a tiny KV length. Ascend ACL graphs override this.
        return 2

    def _reset_warmup_linear_states(self, model, batch_size: int) -> None:
        if self.args.mtp_step != 0:
            return
        req_manager = model.req_manager
        conv_cache = getattr(req_manager, "req_to_conv_state", None)
        ssm_cache = getattr(req_manager, "req_to_ssm_state", None)
        if conv_cache is not None:
            conv_cache.buffer[:, :batch_size].zero_()
        if ssm_cache is not None:
            ssm_cache.buffer[:, :batch_size].zero_()

    def can_run(self, batch_size: int, max_len_in_batch: int) -> bool:
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size: int) -> bool:
        find_batch_size = self.find_closest_graph_batch_size(batch_size)
        if find_batch_size is not None:
            return find_batch_size not in self.graph
        else:
            assert False, "dead code"

    def find_closest_graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.graph_batch_sizes, batch_size)
        if index < len(self.graph_batch_sizes):
            find_batch_size = self.graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _graph_capture(self, graph_obj):
        if self.args.enable_torch_memory_saver:
            return self.torch_memory_saver.cuda_graph(graph_obj, pool=self.mempool)
        return self.platform_backend.graph.graph(graph_obj, pool=self.mempool)

    def _capture_decode(self, decode_func, infer_state: InferStateInfo) -> ModelOutput:
        graph_obj = self.platform_backend.graph.create_graph()
        batch_size = infer_state.input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            pure_para_set = set(vars(infer_state).keys())
            self.platform_backend.runtime.synchronize()
            decode_func(copy.copy(infer_state))
            self.platform_backend.runtime.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)

        with self._graph_capture(graph_obj):
            model_output = decode_func(infer_state)
        self.graph[batch_size] = (graph_obj, infer_state, model_output)

        if self.platform_backend.name != "ascend":
            self.platform_backend.graph.replay_graph(graph_obj)

        self._measure_replay_cost(graph_obj=graph_obj, batch_size=batch_size)
        return model_output

    def _capture_decode_overlap(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ) -> tuple[ModelOutput, ModelOutput]:
        graph_obj = self.platform_backend.graph.create_graph()
        batch_size = infer_state.input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        infer_state1.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state1.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            pure_para_set = set(vars(infer_state).keys())
            pure_para_set1 = set(vars(infer_state1).keys())
            self.platform_backend.runtime.synchronize()
            decode_func(copy.copy(infer_state), copy.copy(infer_state1))
            self.platform_backend.runtime.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)
            for param_name in set(vars(infer_state1).keys()):
                if param_name not in pure_para_set1:
                    delattr(infer_state1, param_name)

        with self._graph_capture(graph_obj):
            model_output, model_output1 = decode_func(infer_state, infer_state1)
        self.graph[batch_size] = (graph_obj, infer_state, infer_state1, model_output, model_output1)

        if self.platform_backend.name != "ascend":
            self.platform_backend.graph.replay_graph(graph_obj)

        self._measure_replay_cost(graph_obj=graph_obj, batch_size=batch_size)
        return model_output, model_output1

    def _measure_replay_cost(self, graph_obj, batch_size: int) -> None:
        if not self.capture_infer_cost:
            return

        def replay():
            if self.platform_backend.name == "ascend":
                # ACL replay must also update attention task parameters through
                # the subclass hook; replaying the raw graph bypasses that work.
                state_count = 2 if self.enable_decode_microbatch_overlap else 1
                self.replay(*self.graph[batch_size][1 : 1 + state_count])
            else:
                self.platform_backend.graph.replay_graph(graph_obj)

        dist.barrier(group=dist.group.WORLD)
        start_event = self.platform_backend.runtime.create_event(enable_timing=True)
        end_event = self.platform_backend.runtime.create_event(enable_timing=True)
        replay()
        start_event.record()
        replay()
        end_event.record()
        end_event.synchronize()
        infer_cost_ms_tensor = torch.tensor(
            [start_event.elapsed_time(end_event)],
            dtype=torch.float32,
            device=self.target_device,
        )
        dist.all_reduce(infer_cost_ms_tensor, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
        if self.enable_decode_microbatch_overlap:
            # overlap graph 每次 replay 同时处理两个等容量 microbatch。
            batch_size *= 2
        self.infer_cost_ms_by_batch_size[batch_size] = float(infer_cost_ms_tensor.item())

    def capture_decode(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: Optional[InferStateInfo] = None,
    ) -> tuple[ModelOutput, ModelOutput]:
        if self.enable_decode_microbatch_overlap:
            return self._capture_decode_overlap(decode_func, infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._capture_decode(decode_func, infer_state)

    def _replay(self, infer_state: InferStateInfo) -> ModelOutput:
        batch_size = infer_state.input_ids.shape[0]
        graph_obj, graph_infer_state, graph_output = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        self.platform_backend.graph.replay_graph(graph_obj)

        return graph_output

    def _replay_overlap(self, infer_state: InferStateInfo, infer_state1: InferStateInfo):
        batch_size = infer_state.input_ids.shape[0]
        (
            graph_obj,
            graph_infer_state,
            graph_infer_state1,
            graph_model_output,
            graph_model_output1,
        ) = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        self.platform_backend.graph.replay_graph(graph_obj)

        return graph_model_output, graph_model_output1

    def replay(self, infer_state: InferStateInfo, infer_state1: Optional[InferStateInfo] = None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(infer_state, infer_state1)
        assert infer_state1 is None
        return self._replay(infer_state)

    @contextmanager
    def _block_warmup_input(self, model, batch_size: int):
        """Own temporary requests and a short dummy page block without resetting target state."""
        width = model.mtp_manager.get_decode_batch_multiplier(model.is_mtp_draft_model)
        group_count = triton.cdiv(batch_size, width)
        alloc_seq_len = width + 1
        if alloc_seq_len > self.graph_max_len_in_batch:
            raise ValueError("decode graph max sequence length is smaller than the speculative warmup block")
        max_kv_seq_len = max(self._warmup_dummy_seq_len(), alloc_seq_len)
        prefix_len = alloc_seq_len - width
        req_manager = model.req_manager
        requests, allocations, saved_rows = [], [], []
        try:
            for _ in range(group_count):
                req_idx = req_manager.alloc()
                if req_idx is None:
                    raise RuntimeError("not enough free requests for speculative graph warmup")
                requests.append(req_idx)
                saved_rows.append(req_manager.req_to_token_indexs[req_idx].clone())
                page_tokens = req_manager.alloc_page_aligned_mem_indices(alloc_seq_len)
                if page_tokens is None:
                    raise RuntimeError("not enough free KV pages for speculative graph warmup")
                allocations.append(page_tokens)
                req_manager.req_to_token_indexs[req_idx, :alloc_seq_len] = page_tokens[:alloc_seq_len].to(
                    self.target_device
                )

            offsets = torch.arange(batch_size, dtype=torch.int32, device=self.target_device) % width
            req_ids = torch.tensor(requests, dtype=torch.int32, device=self.target_device)
            b_req_idx = req_ids.repeat_interleave(width)[:batch_size]
            mem_indexes = torch.cat([tokens[prefix_len:alloc_seq_len] for tokens in allocations])[:batch_size]
            b_seq_len = prefix_len + offsets + 1
            yield ModelInput(
                batch_size=batch_size,
                total_token_num=int(b_seq_len.sum().item()),
                max_q_seq_len=1,
                max_kv_seq_len=max_kv_seq_len,
                input_ids=torch.ones(batch_size, dtype=torch.int64, device=self.target_device),
                mem_indexes=mem_indexes.to(self.target_device),
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mtp_index=offsets,
                b_shared_seq_len=torch.zeros_like(offsets),
                b_shared_radix_node_id=torch.full(
                    (batch_size,), -1, dtype=torch.int64, device=self.target_device
                ),
                b_position_delta=torch.zeros_like(offsets),
                is_prefill=False,
                multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                **model._gen_special_model_input(batch_size),
            )
        finally:
            # KV writes and capture replays must finish before returning pages
            # to the shared target allocator or restoring its request table.
            self.platform_backend.runtime.synchronize()
            for req_idx, saved_row in zip(requests, saved_rows):
                req_manager.req_to_token_indexs[req_idx].copy_(saved_row)
            for tokens in allocations:
                model.mem_manager.free(tokens)
            for req_idx in requests:
                req_manager.free_req(req_idx)

    def _warmup_block_graphs(self, model, overlap: bool):
        # Graph limits may exceed the serving request pool (e.g. graph=16,
        # running=8). Block capture owns distinct requests, unlike HOLD-only
        # native warmup. Capture only reachable batches; keep lookup consistent.
        width = model.mtp_manager.get_decode_batch_multiplier(model.is_mtp_draft_model)
        request_capacity = model.req_manager.max_request_num // (2 if overlap else 1)
        reachable_rows = request_capacity * width
        if reachable_rows < 1:
            raise ValueError("request pool is too small for speculative graph warmup")
        if self.max_batch_size > reachable_rows:
            logger.info(f"Limit speculative graph rows from {self.max_batch_size} to {reachable_rows} "
                        f"for request pool capacity {model.req_manager.max_request_num}")
            self.max_batch_size = reachable_rows
            self.graph_batch_sizes = sorted({n for n in self.graph_batch_sizes if n < reachable_rows} | {reachable_rows})
            self.cuda_graph_batch_sizes = self.graph_batch_sizes
        for batch_size in reversed(self.graph_batch_sizes):
            with ExitStack() as stack:
                model_input = stack.enter_context(self._block_warmup_input(model, batch_size))
                if overlap:
                    model_input1 = stack.enter_context(self._block_warmup_input(model, batch_size))
                    model.microbatch_overlap_decode(model_input, model_input1)
                else:
                    model.forward(model_input)
                self._after_capture_batch(batch_size)
            self.platform_backend.runtime.empty_cache()

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from lightllm.common.basemodel.basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        if self.args.mtp_mode in ("dspark", "dflash"):
            self._warmup_block_graphs(model, overlap=False)
            return
        # decode cuda graph init
        for batch_size in self.graph_batch_sizes[::-1]:
            self._reset_warmup_linear_states(model, batch_size)
            seq_len = self._warmup_dummy_seq_len()
            total_token_num = batch_size * seq_len
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int64, device=self.target_device)
            mem_indexes = model.mem_manager.alloc(len(input_ids)).to(self.target_device)
            if self.args.mtp_step == 0:
                b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=self.target_device)
            else:
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)],
                    dtype=torch.int32,
                    device=self.target_device,
                )
            b_seq_len = torch.empty(batch_size, dtype=torch.int32, device=self.target_device)
            b_seq_len.fill_(seq_len)
            b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=self.target_device)
            b_shared_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=self.target_device)
            b_shared_radix_node_id = torch.full((batch_size,), -1, dtype=torch.int64, device=self.target_device)

            model_input = ModelInput(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_q_seq_len=1,
                max_kv_seq_len=max_len_in_batch,
                input_ids=input_ids,
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
                b_mtp_index=b_mtp_index,
                b_shared_seq_len=b_shared_seq_len,
                b_shared_radix_node_id=b_shared_radix_node_id,
                b_position_delta=torch.zeros(batch_size, dtype=torch.int32, device=self.target_device),
                is_prefill=False,
                multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                **model._gen_special_model_input(batch_size),
            )
            model_output: ModelOutput = model.forward(model_input)
            del model_output
            del input_ids
            del mem_indexes
            del b_req_idx
            del b_seq_len

            model.mem_manager.free_all()
            model.req_manager.free_all()
            self._after_capture_batch(batch_size)
            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            self.platform_backend.runtime.empty_cache()

        self._reset_warmup_linear_states(model, self.max_batch_size)
        logger.info(
            f"Capture cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture overlap cudagraph, use the --disable_cudagraph to disable it.")
        # for typing easy
        from lightllm.common.basemodel.basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        if self.args.mtp_mode in ("dspark", "dflash"):
            self._warmup_block_graphs(model, overlap=True)
            return

        for batch_size in self.graph_batch_sizes[::-1]:
            decode_batches = []
            for micro_batch_index in [0, 1]:
                # dummy decoding, capture the cudagraph
                seq_len = self._warmup_dummy_seq_len()
                total_token_num = batch_size * seq_len
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([1 for _ in range(batch_size)], dtype=torch.int64, device=self.target_device)
                mem_indexes = model.mem_manager.alloc(len(input_ids)).to(self.target_device)
                b_req_idx = torch.tensor(
                    [model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)], dtype=torch.int32, device=self.target_device
                )
                b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device=self.target_device)
                b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=self.target_device)
                b_shared_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=self.target_device)
                b_shared_radix_node_id = torch.full((batch_size,), -1, dtype=torch.int64, device=self.target_device)

                micro_batch = ModelInput(
                    is_prefill=False,
                    batch_size=batch_size,
                    total_token_num=total_token_num,
                    max_q_seq_len=1,
                    max_kv_seq_len=max_len_in_batch,
                    input_ids=input_ids,
                    b_mtp_index=b_mtp_index,
                    mem_indexes=mem_indexes,
                    b_req_idx=b_req_idx,
                    b_seq_len=b_seq_len,
                    b_shared_seq_len=b_shared_seq_len,
                    b_shared_radix_node_id=b_shared_radix_node_id,
                    b_position_delta=torch.zeros(batch_size, dtype=torch.int32, device=self.target_device),
                    multimodal_params=[{"images": [], "audios": []} for _ in range(batch_size)],
                    **model._gen_special_model_input(batch_size),
                )
                decode_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                self.platform_backend.runtime.empty_cache()

            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            del decode_batches

            # release local tensors
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            self.platform_backend.runtime.empty_cache()

        logger.info(
            f"Capture overlap cudagraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with cudagraph."
        )
