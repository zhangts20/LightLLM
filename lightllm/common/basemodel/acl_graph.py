import bisect
import copy
import torch
from dataclasses import dataclass, field
from typing import Any, Optional

from lightllm.common.basemodel.attention.paged_fa3.fp import update_attn_params
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.distributed.communication_op import CustomProcessGroup, lightllm_capture_graph
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class AclGraph:

    def __init__(
        self,
        max_batch_size: int = 8,
        max_len_in_batch: int = 8192,
    ) -> None:
        self.graph = {}
        self.mempool = torch.npu.graph_pool_handle()

        args = get_env_start_args()
        self.mtp_step = args.mtp_step
        self.max_batch_size = max_batch_size
        self.graph_max_len_in_batch = max_len_in_batch
        self.enable_decode_microbatch_overlap = args.enable_decode_microbatch_overlap

        graph_split_batch_size = args.graph_split_batch_size * (self.mtp_step + 1)
        graph_grow_step_size = args.graph_grow_step_size * (self.mtp_step + 1)

        batch_sizes = [i * (self.mtp_step + 1) for i in range(1, graph_split_batch_size + 1)]
        for _batch_size in range(graph_split_batch_size + graph_grow_step_size, max_batch_size,
                                 graph_grow_step_size):
            batch_sizes.append(_batch_size)
        batch_sizes = list(set([e for e in batch_sizes if e < max_batch_size]))
        batch_sizes.append(max_batch_size)
        batch_sizes.sort()
        self.acl_graph_batch_sizes = batch_sizes
        assert batch_sizes[-1] == self.max_batch_size

        logger.info(f"acl graph batch_sizes: {self.acl_graph_batch_sizes}")
        init_attn_params(batch_sizes)

    def can_run(self, batch_size: int, max_len_in_batch: int) -> bool:
        return batch_size <= self.max_batch_size and max_len_in_batch <= self.graph_max_len_in_batch

    def need_capture(self, batch_size: int) -> bool:
        find_batch_size = self.find_closest_graph_batch_size(batch_size)
        if find_batch_size is not None:
            return find_batch_size not in self.graph
        else:
            assert False, "dead code"

    def find_closest_graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.acl_graph_batch_sizes, batch_size)
        if index < len(self.acl_graph_batch_sizes):
            find_batch_size = self.acl_graph_batch_sizes[index]
            return find_batch_size
        else:
            return None

    def _capture_decode(self, decode_func, infer_state: InferStateInfo) -> ModelOutput:
        dist_group: CustomProcessGroup = infer_state.dist_group
        graph_obj = torch.npu.NPUGraph()
        batch_size = infer_state.input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            pure_para_set = set(vars(infer_state).keys())
            torch.npu.synchronize()
            decode_func(copy.copy(infer_state))
            torch.npu.synchronize()
            for param_name in set(vars(infer_state).keys()):
                if param_name not in pure_para_set:
                    delattr(infer_state, param_name)
        with lightllm_capture_graph(dist_group):
            with torch.npu.graph(graph_obj, pool=self.mempool):
                model_output = decode_func(infer_state)
        self.graph[batch_size] = (graph_obj, infer_state, model_output)
        graph_obj.replay()
        return model_output

    def _capture_decode_overlap(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ):
        dist_group: CustomProcessGroup = infer_state.dist_group
        dist_group1 = infer_state1.dist_group
        graph_obj = torch.npu.NPUGraph()
        batch_size = infer_state.input_ids.shape[0]
        infer_state.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state.total_token_num = self.graph_max_len_in_batch * batch_size
        infer_state1.max_kv_seq_len = self.graph_max_len_in_batch
        infer_state1.total_token_num = self.graph_max_len_in_batch * batch_size
        # warmup
        for _ in range(1):
            # 记录原始存在的变量
            pure_para_set = set(vars(infer_state).keys())
            pure_para_set1 = set(vars(infer_state1).keys())
            torch.npu.synchronize()
            decode_func(copy.copy(infer_state), copy.copy(infer_state1))
            torch.npu.synchronize()
            for para_name in set(vars(infer_state).keys()):
                if para_name not in pure_para_set:
                    delattr(infer_state, para_name)
            for para_name in set(vars(infer_state1).keys()):
                if para_name not in pure_para_set1:
                    delattr(infer_state1, para_name)

        with lightllm_capture_graph(dist_group1):
            with lightllm_capture_graph(dist_group):
                with torch.npu.graph(graph_obj, pool=self.mempool):
                    model_output, model_output1 = decode_func(infer_state, infer_state1)
        self.graph[batch_size] = (
            graph_obj,
            infer_state,
            infer_state1,
            model_output,
            model_output1,
        )

        return model_output, model_output1

    def capture_decode(
        self,
        decode_func,
        infer_state: InferStateInfo,
        infer_state1: Optional[InferStateInfo] = None,
    ):
        if self.enable_decode_microbatch_overlap:
            return self._capture_decode_overlap(decode_func, infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._capture_decode(decode_func, infer_state)

    def _replay(self, infer_state: InferStateInfo):
        batch_size = infer_state.input_ids.shape[0]
        graph_obj, graph_infer_state, graph_output = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        update_attn_params(
            batch_size, infer_state.b1_cu_q_seq_len_cpu, infer_state.b_cu_kv_seq_len_cpu
        )
        graph_obj.replay()

        return graph_output

    def _replay_overlap(
        self,
        infer_state: InferStateInfo,
        infer_state1: InferStateInfo,
    ):
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
        graph_obj.replay()

        return graph_model_output, graph_model_output1

    def replay(self, infer_state, infer_state1=None):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(infer_state, infer_state1)
        else:
            assert infer_state1 is None
            return self._replay(infer_state)

    @torch.no_grad()
    def warmup(self, model):
        logger.info("Begin capture aclgraph, use the --disable_cudagraph to disable it.")
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        device = model.device
        for batch_size in self.acl_graph_batch_sizes[::-1]:
            seq_len = self.graph_max_len_in_batch
            total_token_num = batch_size * seq_len
            max_len_in_batch = self.graph_max_len_in_batch
            input_ids = torch.tensor([1 for _ in range(batch_size)],
                                     dtype=torch.int32,
                                     device=device)
            mem_indexes = model.mem_manager.alloc(len(input_ids)).to(device)
            b_req_idx = torch.tensor([model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)],
                                     dtype=torch.int32,
                                     device=device)
            b_seq_len = torch.empty(batch_size, dtype=torch.int32, device=device)
            b_seq_len.fill_(seq_len)
            b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=device)

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
                is_prefill=False,
                multimodal_params=[{
                    "images": [],
                    "audios": []
                } for _ in range(batch_size)],
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
            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.npu.empty_cache()

        logger.info(
            f"Capture aclgraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with aclgraph."
        )

    @torch.no_grad()
    def warmup_overlap(self, model):
        logger.info("Begin capture overlap aclgraph, use the --disable_cudagraph to disable it.")
        from .basemodel import TpPartBaseModel

        model: TpPartBaseModel = model
        device = model.device

        for batch_size in self.acl_graph_batch_sizes[::-1]:
            decode_batches = []
            for micro_batch_index in [0, 1]:
                seq_len = self.graph_max_len_in_batch
                total_token_num = batch_size * seq_len
                max_len_in_batch = self.graph_max_len_in_batch
                input_ids = torch.tensor([1 for _ in range(batch_size)],
                                         dtype=torch.int32,
                                         device=device)
                mem_indexes = model.mem_manager.alloc(len(input_ids)).to(device)
                b_req_idx = torch.tensor([
                    model.req_manager.HOLD_REQUEST_ID for _ in range(batch_size)
                ],
                                         dtype=torch.int32,
                                         device=device)
                b_seq_len = torch.empty(batch_size, dtype=torch.int32, device=device)
                b_seq_len.fill_(seq_len)
                b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device=device)

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
                    multimodal_params=[{
                        "images": [],
                        "audios": []
                    } for _ in range(batch_size)],
                    **model._gen_special_model_input(batch_size),
                )
                decode_batches.append(micro_batch)
                del micro_batch

                for var_name, var_value in list(locals().items()):
                    if isinstance(var_value, torch.Tensor):
                        del locals()[var_name]
                torch.npu.empty_cache()

            _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])

            model.mem_manager.free_all()
            model.req_manager.free_all()

            del decode_batches

            for var_name, var_value in list(locals().items()):
                if isinstance(var_value, torch.Tensor):
                    del locals()[var_name]
            torch.npu.empty_cache()

        logger.info(
            f"Capture overlap aclgraph success, batch_size <={self.max_batch_size} "
            f"and max_len_in_batch <= {self.graph_max_len_in_batch} will infer with aclgraph."
        )


# Adapted from: https://github.com/vllm-project/vllm-ascend/blob/v0.11.0/vllm_ascend/compilation/acl_graph.py
@dataclass
class AclGraphParams:
    handles: dict[int, list[Any]] = field(default_factory=dict)
    workspaces: dict[int, Any] = field(default_factory=dict)
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)


ATTN_PARAMS: Optional[AclGraphParams] = None


def init_attn_params(batch_sizes: list[int]):
    global ATTN_PARAMS
    ATTN_PARAMS = AclGraphParams(
        handles={bs: [] for bs in batch_sizes},
        workspaces={bs: None for bs in batch_sizes},
        attn_params={bs: [] for bs in batch_sizes},
    )


def get_attn_params():
    return ATTN_PARAMS


def add_attn_params(batch_size: int, handle: Any, attn_params: tuple):
    global ATTN_PARAMS
    if ATTN_PARAMS is not None:
        ATTN_PARAMS.handles[batch_size].append(handle)
        ATTN_PARAMS.attn_params[batch_size].append(attn_params)
