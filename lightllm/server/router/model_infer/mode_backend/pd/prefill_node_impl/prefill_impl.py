import pickle
import torch.multiprocessing as mp
import random
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTask, PDDecodeNodeInfo
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import is_maca, kv_trans_use_p2p
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.impl import ChunkedPrefillBackend

logger = init_logger(__name__)


class PDChunkedPrefillForPrefillNode(ChunkedPrefillBackend):
    def __init__(self, info_queue: mp.Queue) -> None:
        super().__init__()
        self.support_overlap = False
        self.info_queue: mp.Queue = info_queue
        self.classed_req_no_decode = True
        self.pd_prefill_chunked_handle_func = self._prefill_chuncked_handle_func

    def init_custom(self):
        assert kv_trans_use_p2p()
        return

    def _filter_not_ready_reqs(self, req_ids: List[int]) -> List[InferReq]:
        """
        将错误请求从 req_ids 中过滤出来, 然后让 _get_classed_reqs 进行处理。 该函数
        主要用于在 nixl pd 分离模式下, 由子类继承重载, prefill 和 decode 节点过滤 kv 传输错误，或者 kv
        传输没有完成的请求。
        """
        ans_list: List[InferReq] = []
        for request_id in req_ids:
            req_obj: InferReq = g_infer_context.requests_mapping[request_id]

            # 推理已经 abort 时，主动通知 KV 传输层停掉还没完成的页，而不是干等到超时。
            pd_abort_req_send_count = getattr(req_obj, "pd_abort_req_send_count", 0)
            if (
                self.is_master_in_dp
                and req_obj.infer_aborted
                and req_obj.pd_task_num != 0
                and pd_abort_req_send_count < 6
            ):
                self.info_queue.put(PDAbortReq(request_id=req_obj.req_id, device_id=req_obj.pd_trans_device_id))
                req_obj.pd_abort_req_send_count = pd_abort_req_send_count + 1

            if (
                is_maca()
                and getattr(req_obj, "pd_wait_decode_node", False)
                and not req_obj.infer_aborted
            ):
                if not self._maca_attach_decode_node(req_obj):
                    continue
                token = getattr(req_obj, "pd_deferred_token", None)
                if token is None:
                    continue
                self._prefill_chuncked_handle_func(req_obj, token[0], token[1], token[2])
                if not getattr(req_obj, "pd_skip_token_output", False):
                    self._maca_publish_held_token(req_obj)
                self._maca_run_trans_builder(req_obj)

            prefill_finished = req_obj.shm_req.input_len <= req_obj.cur_kv_len
            if prefill_finished:
                # 等待所有传输任务都已经完成。
                if req_obj.pd_task_num == (req_obj.pd_task_failed_num + req_obj.pd_task_success_num):
                    ans_list.append(req_obj)
            else:
                if req_obj.infer_aborted:
                    if req_obj.pd_task_num == (req_obj.pd_task_failed_num + req_obj.pd_task_success_num):
                        ans_list.append(req_obj)
                    else:
                        continue
                else:
                    ans_list.append(req_obj)
        return ans_list

    def _prefill_chuncked_handle_func(
        self, req_obj: InferReq, next_token_id: int, next_token_prob: float, output_len: int
    ):
        """
        在每一步chuncked prefill 后，尝试生成chuncked 传输任务，发个 kv_move_manager 进行处理。
        """

        assert req_obj.cur_kv_len <= req_obj.shm_req.input_len
        if is_maca():
            if not self._maca_attach_decode_node(req_obj):
                req_obj.pd_wait_decode_node = True
                req_obj.pd_hold_token_output = True
                req_obj.pd_deferred_token = (next_token_id, next_token_prob, output_len)
                return
            req_obj.pd_wait_decode_node = False
            req_obj.pd_hold_token_output = False
            if getattr(req_obj, "pd_skip_token_output", False):
                return
            # 先让 token 写进 shm，再组传输任务。
            req_obj._maca_trans_builder = lambda req_obj=req_obj, next_token_id=next_token_id, next_token_prob=next_token_prob, output_len=output_len: self._enqueue_prefill_trans_tasks(
                req_obj, next_token_id, next_token_prob, output_len
            )
            return
        self._enqueue_prefill_trans_tasks(req_obj, next_token_id, next_token_prob, output_len)

    def _maca_run_trans_builder(self, req_obj: InferReq) -> None:
        builder = getattr(req_obj, "_maca_trans_builder", None)
        if builder is None:
            return
        req_obj._maca_trans_builder = None
        builder()

    def _enqueue_prefill_trans_tasks(
        self, req_obj: InferReq, next_token_id: int, next_token_prob: float, output_len: int
    ) -> None:
        input_len = req_obj.shm_req.input_len
        page_size = self.args.pd_kv_page_size
        prefill_finished = req_obj.cur_kv_len == input_len
        trans_task_list: List[PDChunckedTransTask] = []
        while req_obj.pd_trans_kv_start_index < req_obj.cur_kv_len:
            cur_page_size = min(page_size, req_obj.cur_kv_len - req_obj.pd_trans_kv_start_index)
            # 生成页面传输任务， 放入kv move manager 的处理队列中
            if cur_page_size == page_size or prefill_finished:
                trans_task = self._create_pd_trans_task(
                    req_obj=req_obj,
                    kv_start_index=req_obj.pd_trans_kv_start_index,
                    kv_end_index=req_obj.pd_trans_kv_start_index + cur_page_size,
                )
                req_obj.pd_trans_kv_start_index += cur_page_size
                trans_task_list.append(trans_task)
            else:
                break

        if prefill_finished and len(trans_task_list) != 0 and output_len == 1:
            if g_infer_context.is_linear_att_mixed_model:
                trans_task_list.append(
                    self._create_pd_trans_task(
                        req_obj=req_obj,
                        kv_start_index=input_len,
                        kv_end_index=input_len,
                        page_kind="linear_att_state",
                    )
                )
            trans_task_list[-1].first_gen_token_id = next_token_id
            trans_task_list[-1].first_gen_token_logprob = next_token_prob

        if self.is_master_in_dp:
            for trans_task in trans_task_list:
                self.info_queue.put(trans_task)
        return

    def _maca_publish_held_token(self, req_obj: InferReq):
        pending = getattr(req_obj, "pd_deferred_publish", None)
        req_obj.pd_deferred_publish = None
        if not pending:
            return
        next_token_id, next_token_logprob, next_token_rank, output_len, extra_post = pending
        if output_len <= 0:
            return
        shm_req = req_obj.shm_req
        finish_status = req_obj.finish_status
        req_obj.set_next_gen_token_id(next_token_id, next_token_logprob, output_len, rank=next_token_rank)
        if finish_status.is_finished():
            return
        req_obj.update_finish_status(eos_ids=self.eos_id, output_len=output_len)
        if extra_post is not None:
            extra_post(req_obj, next_token_id, next_token_logprob)
        if self.is_master_in_dp:
            shm_req.shm_cur_output_len = output_len
            if finish_status.is_finished():
                shm_req.finish_token_index = shm_req.input_len + output_len - 1
                shm_req.finish_status = req_obj.finish_status
            shm_req.candetoken_out_len = output_len

    def _maca_attach_decode_node(self, req_obj: InferReq) -> bool:
        if getattr(req_obj, "_maca_decode_attached", False):
            return True
        node: PDDecodeNodeInfo = req_obj.sampling_param.pd_decode_node
        blob = req_obj.sampling_param.shm_param.pd_kv_trans_params
        if node is None:
            if blob.data_len <= 0:
                return False
            node = pickle.loads(blob.get())
            req_obj.sampling_param.pd_decode_node = node
        if node.ready_kv_len > req_obj.pd_trans_kv_start_index:
            req_obj.pd_trans_kv_start_index = node.ready_kv_len
        req_obj._maca_decode_attached = True
        if req_obj.shm_req.input_len > 1 and node.ready_kv_len >= req_obj.shm_req.input_len - 1:
            req_obj.pd_skip_token_output = True
            req_obj.infer_aborted = True
            req_obj.shm_req.is_aborted = True
            if not req_obj.finish_status.is_finished():
                req_obj.finish_status.set_status(FinishStatus.FINISHED_ABORTED)
        return True

    def _cached_decode_peers(self, req_obj: InferReq, decode_agent_metadata):
        if getattr(req_obj, "_maca_decode_peers_ready", False):
            return getattr(req_obj, "_maca_decode_peers", None)
        try:
            node_meta = pickle.loads(decode_agent_metadata)
            peers = getattr(node_meta, "peers", None) or None
        except Exception:
            peers = None
        if is_maca():
            req_obj._maca_decode_peers = peers
            req_obj._maca_decode_peers_ready = True
        return peers

    def _pd_kv_mem_indexes(self, req_obj: InferReq, kv_start_index: int, kv_end_index: int):
        cached = getattr(req_obj, "_maca_pd_mem_indexes", None)
        base = getattr(req_obj, "_maca_pd_mem_base", None)
        if cached is not None and base is not None:
            local_start = kv_start_index - base
            local_end = kv_end_index - base
            if 0 <= local_start <= local_end <= len(cached):
                return cached[local_start:local_end]
        return (
            self.model.req_manager.req_to_token_indexs[req_obj.req_idx, kv_start_index:kv_end_index]
            .detach()
            .cpu()
            .tolist()
        )

    def _create_pd_trans_task(
        self,
        req_obj: InferReq,
        kv_start_index: int,
        kv_end_index: int,
        page_kind: str = "kv",
    ) -> PDChunckedTransTask:
        # 确定传输设备
        if req_obj.pd_trans_device_id == -1:
            if not hasattr(self, "pd_iter_device_id"):
                self.pd_iter_device_id = 0
            req_obj.pd_trans_device_id = self.pd_iter_device_id
            self.pd_iter_device_id = (self.pd_iter_device_id + 1) % self.node_world_size

        pd_decode_node_info = req_obj.sampling_param.pd_decode_node
        decode_agent_metadata = pd_decode_node_info.agent_metadata
        decode_agents = self._cached_decode_peers(req_obj, decode_agent_metadata)
        if decode_agents:
            rank = int(pd_decode_node_info.agent_name.rsplit("_", 1)[1])
            decode_agent_metadata = decode_agents.get(rank, decode_agent_metadata)
        if page_kind == "kv":
            mem_indexes = self._pd_kv_mem_indexes(req_obj, kv_start_index, kv_end_index)
            req_idx = None
        elif page_kind == "linear_att_state":
            mem_indexes = []
            req_idx = req_obj.req_idx
        else:
            raise ValueError(f"unknown PD trans page kind {page_kind}")
        trans_task = PDChunckedTransTask(
            request_id=req_obj.req_id,
            start_kv_index=kv_start_index,
            end_kv_index=kv_end_index,
            time_out_secs=182,
            pd_master_node_id=req_obj.sampling_param.pd_master_node_id,
            prefill_dp_index=self.dp_rank_in_node,
            decode_dp_index=None,
            src_device_id=req_obj.pd_trans_device_id,
            dst_device_id=None,
            mem_indexes=mem_indexes,
            prefill_agent_name=None,
            prefill_agent_metadata=None,
            prefill_num_pages=None,
            prefill_page_reg_desc=None,
            decode_agent_name=pd_decode_node_info.agent_name,
            decode_agent_metadata=decode_agent_metadata,
            decode_agents=decode_agents,
            decode_num_pages=pd_decode_node_info.num_pages,
            decode_page_reg_desc=pd_decode_node_info.page_reg_desc,
            first_gen_token_id=None,
            first_gen_token_logprob=None,
            page_kind=page_kind,
            req_idx=req_idx,
        )
        req_obj.pd_task_num += 1
        return trans_task
