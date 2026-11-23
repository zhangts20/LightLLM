import copy
import torch
import time
import inspect
import threading
import setproctitle
import torch.multiprocessing as mp
import queue
import pickle
from typing import List, Dict, Optional, Union
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTask
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..kv_transporter import create_kv_transporter
from lightllm.utils.error_utils import log_exception
from lightllm.utils.device_utils import is_maca
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.process_check import install_fatal_thread_excepthook, start_parent_check_thread


logger = init_logger(__name__)


def start_prefill_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    up_status_in_queue: Optional[mp.SimpleQueue] = None,
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue))
    proc.start()
    assert proc.is_alive()
    logger.info(f"prefill trans kv process for device: {device_id} started!")
    return proc


def _init_env(
    args: StartArgs,
    device_id: int,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
):
    install_fatal_thread_excepthook()
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _

    import os

    # prefill source-side page copy and UCX progress are on the request critical path.
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] = "0"

    torch.backends.cudnn.enabled = False
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::prefill_trans:Device{device_id}")

    module_ready = False
    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)
        task_out_queue.put("proc_start")

        # 从共享内存读取所有rank的mem_manager
        node_world_size = args.tp // args.nnodes
        mem_managers: List[MemoryManager] = [
            MemoryManager.loads_from_shm(rank_in_node=rank) for rank in range(node_world_size)
        ]

        task_out_queue.put("get_mem_managers_ok")

        manager = _PrefillTransModule(
            args=args,
            device_id=device_id,
            task_in_queue=task_in_queue,
            task_out_queue=task_out_queue,
            mem_managers=mem_managers,
        )
        assert manager is not None
        task_out_queue.put("module_ready")
        module_ready = True

        while True:
            time.sleep(100)

    except Exception as e:
        if not module_ready:
            task_out_queue.put("init_failed")
        logger.exception(str(e))
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


class _PrefillTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        mem_managers: List[MemoryManager],
        copy_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.args = args
        self.dp_world_size = self.args.tp // self.args.dp
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.mem_managers = mem_managers

        cur_mem_manager: MemoryManager = self.mem_managers[device_id]
        kv_move_buffer = cur_mem_manager.alloc_paged_kv_move_buffer(
            page_num=self.args.pd_kv_page_num, page_size=self.args.pd_kv_page_size
        )
        # 跑在 model rank 里时由调用方传入 priority=0 的 stream。独立进程仍用原来的高优先级 stream。
        self.copy_cuda_stream = copy_stream if copy_stream is not None else torch.cuda.Stream(priority=-1)
        self.transporter = create_kv_transporter(
            args=self.args,
            node_id=self.args.pd_node_id,
            tp_idx=device_id,
            kv_move_buffer=kv_move_buffer,
        )
        from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import attach_maca_linear_pack

        attach_maca_linear_pack(self.transporter, cur_mem_manager)
        scale_pages = getattr(cur_mem_manager, "scale_move_buffer", None)
        if scale_pages is not None:
            self.transporter.scale_move_buffer = scale_pages
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, PDChunckedTransTask] = {}
        self._scatter_lock = threading.Lock()
        # 剩余发送份数。0 表示这一页已经回报过，后面的分片不再回报。
        self._scatter_left: Dict[str, int] = {}

        self.local_copy_kv_queue = queue.Queue()
        self.ready_transfer_queue = queue.Queue()
        self.write_peer_kv_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.pd_kv_page_num):
            self.page_index_queue.put(page_index)

        # warmup 预先执行一次 kv 写入 page buffer，避免第一次拷贝时出现卡顿。
        self._warmup()

        for func in [
            self.recv_task_loop,
            self.local_copy_kv_loop,
            self.ready_transfer_loop,
            self.accept_decode_write_task_loop,
            self.write_peer_kv_loop,
            self.update_task_status_loop,
            self.success_loop,
            self.fail_loop,
        ]:
            threading.Thread(target=func, daemon=True).start()
        return

    def _warmup(self):
        for dp_index in range(self.args.dp // self.args.nnodes):
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.write_mem_to_page_kv_move_buffer(
                    mem_indexes=[0],
                    page_index=0,
                    dp_index=dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                    target_tp_index=self.device_id if is_maca() else None,
                )
                torch.cuda.current_stream().synchronize()
        return

    def _abort(self, request_id: int, error_info: str = "aborted req"):
        aborted_tasks = []
        with self.waiting_dict_lock:
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.request_id == request_id and trans_task.xfer_handle is None:
                    # 已经交给 NCCL 的传输不能在这里直接失败，否则页面可能被提前回收。
                    aborted_tasks.append(self.waiting_dict.pop(key))

        for trans_task in aborted_tasks:
            trans_task.error_info = error_info
            self.failed_queue.put(trans_task)
        return

    @log_exception
    def recv_task_loop(self):
        torch.cuda.set_device(self.device_id)

        while True:
            obj: Union[PDChunckedTransTask, PDAbortReq] = self.task_in_queue.get()
            if isinstance(obj, PDAbortReq):
                self._abort(request_id=obj.request_id)
                continue
            page_index = self.page_index_queue.get()
            trans_task: PDChunckedTransTask = obj
            trans_task.src_page_index = page_index

            # 初次校验 time out
            if trans_task.time_out():
                trans_task.error_info = "time out in recv_task_loop"
                self.failed_queue.put(trans_task)
            else:
                self.local_copy_kv_queue.put(trans_task)

    @log_exception
    def local_copy_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.local_copy_kv_queue.get()

            # 将kv 数据拷贝到 page 上，然后传输给 decode node，让其进行读取。
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.write_mem_to_page_kv_move_buffer(
                    trans_task.mem_indexes,
                    page_index=trans_task.src_page_index,
                    dp_index=trans_task.prefill_dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                    page_kind=trans_task.page_kind,
                    req_idx=trans_task.req_idx,
                    target_tp_index=self.device_id if is_maca() else None,
                )
                sync_event = torch.cuda.Event()
                sync_event.record()

            self.ready_transfer_queue.put((sync_event, trans_task))
        return

    def _store_key(self, task: PDChunckedTransTask) -> str:
        base = task.get_key()
        with self._scatter_lock:
            scatter = base in self._scatter_left
        if scatter:
            return f"{base}@{task.decode_agent_name}"
        return base

    def _drop_waiting(self, base_key: str):
        for store_key in list(self.waiting_dict.keys()):
            if store_key == base_key or store_key.startswith(base_key + "@"):
                self.waiting_dict.pop(store_key, None)

    def _scatter_should_emit(self, base_key: str, failed: bool = False) -> bool:
        with self._scatter_lock:
            left = self._scatter_left.get(base_key)
            if left is None:
                return True
            if left <= 0:
                return False
            if failed or left == 1:
                self._scatter_left[base_key] = 0
                return True
            self._scatter_left[base_key] = left - 1
            return False

    def _expand_decode_targets(self, task: PDChunckedTransTask) -> List[PDChunckedTransTask]:
        if not is_maca() or not task.decode_agent_name:
            return [task]
        from ..nccl_kv_transporter import maca_decode_ranks_for_prefill, maca_node_world_size

        agents = task.decode_agents or {}
        prefill_tp = maca_node_world_size()
        decode_tp = len(agents)
        if decode_tp == 0 or decode_tp == prefill_tp:
            return [task]
        node_name = task.decode_agent_name.rsplit("_", 1)[0]
        shards = []
        for rank in maca_decode_ranks_for_prefill(self.device_id, prefill_tp, decode_tp):
            agent_name = f"{node_name}_{rank}"
            metadata = agents.get(rank)
            if metadata is None:
                raise RuntimeError(f"decode agent {agent_name} is not published")
            shard = copy.copy(task)
            shard.decode_agent_name = agent_name
            shard.decode_agent_metadata = metadata
            shards.append(shard)
        if self.args.pd_detail_log:
            logger.debug(
                f"maca scatter prefill_rank={self.device_id} prefill_tp={prefill_tp} "
                f"decode_tp={decode_tp} targets={[shard.decode_agent_name for shard in shards]} "
                f"request_id={task.request_id} page_kind={task.page_kind}"
            )
        return shards

    @log_exception
    def ready_transfer_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            sync_event, trans_task = self.ready_transfer_queue.get()
            trans_task: PDChunckedTransTask = trans_task
            sync_event: torch.cuda.Event = sync_event
            if not is_maca():
                sync_event.synchronize()
            key = trans_task.get_key()
            try:
                shards = self._expand_decode_targets(trans_task)
                if is_maca():
                    # 拷贝和后面的 ready RPC 重叠。事件不能挂在任务上，任务会被 pickle 发给 decode。
                    # 份数按 scatter 计，每一路 send 都 wait 同一次拷贝。
                    self.transporter.note_maca_copy_event(trans_task, sync_event, consumers=len(shards))
                if len(shards) > 1:
                    with self._scatter_lock:
                        self._scatter_left[key] = len(shards)
                with self.waiting_dict_lock:
                    for shard in shards:
                        self.waiting_dict[self._store_key(shard)] = shard
                for shard in shards:
                    self.transporter.send_write_request_task_to_decode_node(shard)
                    if self.args.pd_detail_log:
                        logger.debug(f"send WRITE request to decode: {self._store_key(shard)}")
            except BaseException as e:
                if is_maca():
                    self.transporter.note_maca_copy_event(trans_task, sync_event, consumers=1)
                with self.waiting_dict_lock:
                    self._drop_waiting(key)
                logger.error(f"send WRITE request to decode failed: {trans_task.to_str()}")
                logger.exception(str(e))
                trans_task.error_info = f"send WRITE request to decode failed: {str(e)}"
                self.transporter.remove_remote_agent(peer_name=trans_task.decode_agent_name)
                self.failed_queue.put(trans_task)
                continue
        return

    @log_exception
    def accept_decode_write_task_loop(self):
        while True:
            try:
                notifies_dict = self.transporter.get_new_notifs()
            except BaseException as e:
                logger.error(f"get new notifies failed: {str(e)}")
                logger.exception(str(e))
                notifies_dict = {}

            if notifies_dict:
                for _, _notify_list in notifies_dict.items():
                    for notify in _notify_list:
                        if isinstance(notify, PDChunckedTransTask):
                            notify_obj = notify
                        else:
                            try:
                                notify_obj = pickle.loads(notify)
                            except BaseException:
                                notify_obj = None

                        if not isinstance(notify_obj, PDChunckedTransTask):
                            continue

                        if notify_obj.error_info is not None:
                            logger.warning(f"recv WRITE error from decode: {notify_obj.to_str()}")
                            self._abort(request_id=notify_obj.request_id, error_info=notify_obj.error_info)
                            continue

                        if notify_obj.write_stage == "ready":
                            key = notify_obj.get_key()
                            store_key = f"{key}@{notify_obj.decode_agent_name}"
                            with self.waiting_dict_lock:
                                trans_task = self.waiting_dict.pop(store_key, None)
                                if trans_task is None:
                                    trans_task = self.waiting_dict.pop(key, None)
                            if trans_task is not None:
                                trans_task.dst_page_index = notify_obj.dst_page_index
                                self.write_peer_kv_queue.put(trans_task)
                                if self.args.pd_detail_log:
                                    logger.debug(
                                        f"recv WRITE ready from decode request_id={trans_task.request_id} "
                                        f"kv=[{trans_task.start_kv_index},{trans_task.end_kv_index}) "
                                        f"srcpage={trans_task.src_page_index} dstpage={trans_task.dst_page_index}"
                                    )
                            else:
                                logger.warning(
                                    f"can not find pending WRITE request for ready notify: {notify_obj.to_str()}"
                                )
                                # 发一个error信息回去给 decode 节点，让其可以知道这边有问题了，它可以选择其他清理掉请求。
                                notify_obj.error_info = "can not find pending WRITE request for ready notify"
                                self.transporter.send_error_info_to_decode_node(trans_task=notify_obj)
                            continue
                        else:
                            logger.error(f"ignore unknown WRITE notify stage: {notify_obj.to_str()}")
                            continue

            self._check_tasks_time_out()

            if not notifies_dict:
                time.sleep(0.001)
        return

    def _check_tasks_time_out(self):
        with self.waiting_dict_lock:
            timeout_tasks = []
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.time_out():
                    timeout_tasks.append(self.waiting_dict.pop(key))

        for trans_task in timeout_tasks:
            trans_task.error_info = "time out waiting decode WRITE ready"
            self.failed_queue.put(trans_task)
        return

    @log_exception
    def write_peer_kv_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task = self.write_peer_kv_queue.get()
            trans_task: PDChunckedTransTask = trans_task

            try:
                xfer_handle = self.transporter.write_blocks_paged(trans_task=trans_task)
                trans_task.xfer_handle = xfer_handle
                if is_maca():
                    trans_task.transfer_time_out_secs = 180
                trans_task.start_trans_time = time.time()
                # Scatter shards share get_key(); keep each decode target under its own key.
                with self.waiting_dict_lock:
                    self.waiting_dict[self._store_key(trans_task)] = trans_task
                if self.args.pd_detail_log:
                    logger.debug(f"start WRITE to decode node: {trans_task.to_str()}")
                continue
            except BaseException as e:
                logger.error(f"write_blocks_paged failed: {trans_task.to_str()}")
                logger.exception(str(e))
                self.transporter.remove_remote_agent(peer_name=trans_task.decode_agent_name)
                trans_task.error_info = f"write_blocks_paged failed: {str(e)}"
                self.failed_queue.put(trans_task)
                continue
        return

    @log_exception
    def update_task_status_loop(
        self,
    ):
        while True:
            if len(self.waiting_dict) == 0:
                time.sleep(0.001)
                continue

            self._update_task_status_once()
            time.sleep(0.001)

    def _update_task_status_once(self):
        with self.waiting_dict_lock:
            tasks = list(self.waiting_dict.values())
            for trans_task in tasks:
                if trans_task.xfer_handle is None:
                    continue

                # A native NIXL error must fail this task without terminating the
                # status thread and stranding every transfer queued behind it.
                try:
                    ret = self.transporter.check_task_status(trans_task=trans_task)
                except BaseException as e:
                    failed_task = self.waiting_dict.pop(self._store_key(trans_task), None)
                    if failed_task is not None:
                        logger.exception(f"check transfer status failed: {failed_task.to_str()}")
                        failed_task.error_info = f"check transfer status failed: {str(e)}"
                        self.failed_queue.put(failed_task)
                    continue

                if ret == "DONE":
                    completed_task = self.waiting_dict.pop(self._store_key(trans_task), None)
                    if completed_task is None:
                        continue

                    try:
                        self.transporter.send_write_done_task_to_decode_node(completed_task)
                    except BaseException as e:
                        logger.exception(f"send WRITE done notify failed: {completed_task.to_str()}")
                        completed_task.error_info = f"send WRITE done notify failed: {str(e)}"
                        self.failed_queue.put(completed_task)
                        continue

                    if self.args.pd_detail_log:
                        logger.debug(
                            f"send WRITE done notify "
                            f"request_id={completed_task.request_id} "
                            f"kv=[{completed_task.start_kv_index},{completed_task.end_kv_index}) "
                            f"src_page={completed_task.src_page_index} dst_page={completed_task.dst_page_index}"
                        )
                    if self._scatter_should_emit(completed_task.get_key()):
                        self.success_queue.put(completed_task)
                    else:
                        if completed_task.xfer_handle is not None:
                            self.transporter.release_xfer_handle(completed_task.xfer_handle)
                            completed_task.xfer_handle = None
                        completed_task.src_page_index = None
                elif ret == "ERR":
                    failed_task = self.waiting_dict.pop(self._store_key(trans_task), None)
                    if failed_task is not None:
                        detail = getattr(trans_task.xfer_handle, "error_info", None)
                        failed_task.error_info = detail or "xfer error"
                        self.failed_queue.put(failed_task)
                elif trans_task.time_out():
                    failed_task = self.waiting_dict.pop(self._store_key(trans_task), None)
                    if failed_task is not None:
                        failed_task.error_info = "time out in update_task_status_loop"
                        self.failed_queue.put(failed_task)

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.success_queue.get()
            # 写回后，回收页面
            if trans_task.src_page_index is not None:
                self.page_index_queue.put(trans_task.src_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            ret.first_gen_token_id = None
            ret.first_gen_token_logprob = None
            self.task_out_queue.put(ret)

            if self.args.pd_detail_log:
                if trans_task.start_trans_time is not None:
                    logger.debug(f"trans task ret success:{ret} cost time: {trans_task.transfer_time()}s")
                else:
                    logger.debug(f"trans task ret success:{ret}")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.failed_queue.get()
            # 一张 prefill 发给多张 decode 时，分片共用同一个 src page。只让上报的那一片归还。
            if not self._scatter_should_emit(trans_task.get_key(), failed=True):
                if trans_task.xfer_handle is not None:
                    self.transporter.release_xfer_handle(trans_task.xfer_handle)
                    trans_task.xfer_handle = None
                trans_task.src_page_index = None
                continue

            # 拷贝事件还没被发送取完时，先等拷贝结束再归还源页。
            if is_maca():
                finish_copy = getattr(self.transporter, "finish_maca_copy", None)
                if finish_copy is not None:
                    finish_copy(trans_task)
            if trans_task.src_page_index is not None:
                self.page_index_queue.put(trans_task.src_page_index)
            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret fail:{ret}")

            if trans_task.error_info is not None:
                self._abort(request_id=trans_task.request_id, error_info=trans_task.error_info)
                self.transporter.send_error_info_to_decode_node(trans_task=trans_task)
