import copy
import torch
import time
import inspect
import threading
import setproctitle
import torch.multiprocessing as mp
import queue
import pickle
from typing import List, Dict, Union, Optional
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.server.pd_io_struct import (
    PDChunckedTransTask,
    PDChunckedTransTaskGroup,
    PDUpKVStatus,
    PDAbortReq,
)
from lightllm.server.pd_io_struct import PDDecodeNodeInfo
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..kv_transporter import create_kv_transporter
from lightllm.utils.error_utils import log_exception
from lightllm.utils.device_utils import is_maca
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.process_check import install_fatal_thread_excepthook, start_parent_check_thread

logger = init_logger(__name__)


def start_decode_trans_process(
    args,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    up_status_in_queue: Optional[mp.SimpleQueue],
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue, up_status_in_queue))
    proc.start()
    assert proc.is_alive()
    logger.info(f"prefill trans kv process for device: {device_id} started!")
    return proc


def _init_env(
    args: StartArgs,
    device_id: int,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    up_status_in_queue: Optional[mp.SimpleQueue],
):
    install_fatal_thread_excepthook()
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _

    import os

    # -------------------------------------------------------------------------
    # 问题背景（PD NIXL + 同卡多进程）：
    #   decode 物理 GPU 上至少有两个独立 CUDA 进程：model_infer（解码推理）与
    #   decode_trans（把 prefill 侧 KV page 拷入 decode KV cache）。
    #   lm_eval batch=64 时会在短时间内并发大量 read_page；拷贝在 copy_cuda_stream
    #   上排队，而推理在另一进程的 stream 上执行，彼此无法 cudaStreamWaitEvent
    #   协调。日志里的 read_page_gpu_time（event 差值）会把「等 GPU 时间片 /
    #   与推理争抢 SM」算进去，出现数十秒级毛刺，但并不代表单次 memcpy 真那么慢。
    #
    # 解决思路：依赖 NVIDIA MPS（Multi-Process Service）在同一 GPU 上多进程
    #   共享上下文并做客户端级调度；在子进程 import torch / 创建 CUDA 上下文
    #   **之前**设置下列环境变量（故必须放在本函数最前）。
    #
    # CUDA_MPS_CLIENT_PRIORITY="0"：
    #   MPS 下数值越小优先级越高。decode 侧 KV 拷贝处于 decode 关键路径（须先
    #   落盘 KV 才能出首 token），故给 trans 进程最高优先级，减轻被同卡推理
    #   饿死导致的排队放大。须集群已启动 nvidia-cuda-mps-control / mps-server，
    #   否则该变量不生效。 启动 mps 的命令为 nvidia-cuda-mps-control -d
    # -------------------------------------------------------------------------
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] = "0"

    torch.backends.cudnn.enabled = False
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::decode_trans:Device{device_id}")

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

        manager = _DecodeTransModule(
            args=args,
            device_id=device_id,
            task_in_queue=task_in_queue,
            task_out_queue=task_out_queue,
            mem_managers=mem_managers,
            up_status_in_queue=up_status_in_queue,
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


class _DecodeTransModule:
    def __init__(
        self,
        args: StartArgs,
        device_id: int,
        task_in_queue: mp.Queue,
        task_out_queue: mp.Queue,
        mem_managers: List[MemoryManager],
        up_status_in_queue: Optional[mp.SimpleQueue],
        copy_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.args = args
        self.dp_world_size = self.args.tp // self.args.dp
        self.device_id = device_id
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.mem_managers = mem_managers
        self.up_status_in_queue = up_status_in_queue
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
        self.recv_task_group_queue = queue.Queue()
        self.waiting_dict_lock = threading.Lock()
        self.waiting_dict: Dict[str, PDChunckedTransTask] = {}
        # prefill 卡更多时，一张 decode 卡要等齐多张 prefill 页再拼 head。
        self._gather: Dict[str, dict] = {}
        self._gather_closed = set()
        self._prefill_error_notes: List[PDChunckedTransTask] = []
        # 写请求可能比本卡把任务放进 waiting_dict 更早到。先放下，等任务来了再配。
        self._early_writes: Dict[str, List[tuple]] = {}
        self._timeout_marks: Dict[int, "_SharedTimeOutMark"] = {}
        self.request_page_task_queue = queue.Queue()
        self.ready_page_task_queue = queue.Queue()
        self.success_queue = queue.Queue()
        self.failed_queue = queue.Queue()
        # 接收还没结束的页。值是挂起的时间，用来打一次超时日志。
        self._held_pages: Dict[int, float] = {}
        self._held_lock = threading.Lock()

        self.page_index_queue = queue.Queue()
        for page_index in range(self.args.pd_kv_page_num):
            self.page_index_queue.put(page_index)

        # warmup 预先加载一次kv 数据到 mem manager，避免第一次拷贝时出现卡顿。
        self._warmup()

        funcs = [
            self.recv_task_loop,
            self.dispatch_task_loop,
            self.accept_peer_task_loop,
            self.request_page_loop,
            self.read_page_to_mems_loop,
            self.success_loop,
            self.fail_loop,
        ]
        if is_maca():
            funcs.append(self._release_held_pages_loop)
        for func in funcs:
            threading.Thread(target=func, daemon=True).start()
        return

    def _warmup(self):
        for dp_index in range(self.args.dp // self.args.nnodes):
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.read_page_kv_move_buffer_to_mem(
                    mem_indexes=[0],
                    page_index=0,
                    dp_index=dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                    target_tp_index=self.device_id if is_maca() else None,
                )
                torch.cuda.current_stream().synchronize()
        return

    @log_exception
    def recv_task_loop(self):
        while True:
            obj: Union[PDChunckedTransTaskGroup, PDAbortReq] = self.task_in_queue.get()
            if isinstance(obj, PDChunckedTransTaskGroup):
                self.recv_task_group_queue.put(obj)
            elif isinstance(obj, PDAbortReq):
                self._abort(request_id=obj.request_id)
            else:
                assert False, f"recv error obj {obj}"

    def _abort(self, request_id: int, error_info: str = "aborted req"):
        aborted_tasks = []
        with self.waiting_dict_lock:
            for key, trans_task in list(self.waiting_dict.items()):
                if trans_task.request_id == request_id and trans_task.dst_page_index is None:
                    # 对于 已经分配了page index 的任务，不能直接失败，需要两边走完正常流程再失败，不然可能
                    # 出现复杂的异步协同问题。
                    aborted_tasks.append(self.waiting_dict.pop(key))

        for trans_task in aborted_tasks:
            trans_task.error_info = error_info
            self.failed_queue.put(trans_task)
        return

    def _source_rank(self, agent_name: str) -> int:
        return int(agent_name.rsplit("_", 1)[1])

    def _copy_prefill_endpoint(self, dst: PDChunckedTransTask, src: PDChunckedTransTask) -> None:
        dst.prefill_agent_name = src.prefill_agent_name
        dst.prefill_agent_metadata = src.prefill_agent_metadata
        dst.prefill_num_pages = src.prefill_num_pages
        dst.prefill_page_reg_desc = src.prefill_page_reg_desc

    def _gather_source_ranks(self, prefill_world: int) -> List[int]:
        from ..nccl_kv_transporter import maca_node_world_size, maca_prefill_ranks_for_decode

        my_tp = maca_node_world_size()
        other_tp = prefill_world or my_tp
        return maca_prefill_ranks_for_decode(self.device_id, other_tp, my_tp)

    def _prefill_world(self, task: PDChunckedTransTask) -> int:
        name = task.prefill_agent_name or ""
        cache = getattr(self, "_prefill_world_cache", None)
        if cache is None:
            cache = {}
            self._prefill_world_cache = cache
        if name in cache:
            return cache[name]
        if not task.prefill_agent_metadata:
            cache[name] = 0
            return 0
        metadata = pickle.loads(task.prefill_agent_metadata)
        world = int(getattr(metadata, "world_size", 0) or 0)
        cache[name] = world
        return world

    def _waiting_key(self, task: PDChunckedTransTask) -> str:
        base = task.get_key()
        if base in self._gather and task.prefill_agent_name:
            return f"{base}@{task.prefill_agent_name}"
        return base

    def _close_gather(self, base: str, error_info: str) -> None:
        group = self._gather.pop(base, None)
        if group is None:
            return
        self._gather_closed.add(base)
        pages = []
        prefix = base + "@"
        told = set()
        for key, task in list(self.waiting_dict.items()):
            if key == base or key.startswith(prefix):
                self.waiting_dict.pop(key, None)
                self._note_prefill_error(task, error_info, told)
                if task.dst_page_index is not None:
                    pages.append(task.dst_page_index)
        for shard in group.get("shards", {}).values():
            self._note_prefill_error(shard, error_info, told)
        for page in group.get("pages", {}).values():
            if page is not None:
                pages.append(page)
        for page in set(pages):
            self._recycle_dst_page(page)
        template: PDChunckedTransTask = group["template"]
        template.dst_page_index = None
        template.error_info = error_info
        self.failed_queue.put(template)

    def _note_prefill_error(self, task: PDChunckedTransTask, error_info: str, told: set) -> None:
        name = task.prefill_agent_name
        if not name or name in told:
            return
        told.add(name)
        note = copy.copy(task)
        note.dst_page_index = None
        note.error_info = error_info
        self._prefill_error_notes.append(note)

    def _flush_prefill_errors(self) -> None:
        with self.waiting_dict_lock:
            notes = self._prefill_error_notes
            self._prefill_error_notes = []
        for note in notes:
            try:
                self.transporter.send_error_info_to_prefill_node(trans_task=note)
            except BaseException:
                logger.exception(f"send gather error to prefill failed request_id={note.request_id}")

    def _expire_early_writes(self) -> None:
        now = time.time()
        expired: List[PDChunckedTransTask] = []
        with self.waiting_dict_lock:
            for key, items in list(self._early_writes.items()):
                keep = []
                for ts, task in items:
                    # 和任务创建超时同一档。2 秒挡不住 dispatch 被堵住的情况。
                    if now - ts > task.time_out_secs:
                        expired.append(task)
                    else:
                        keep.append((ts, task))
                if keep:
                    self._early_writes[key] = keep
                else:
                    self._early_writes.pop(key, None)
        for task in expired:
            self._missing_write_task(task, "request")

    def _gather_is_closed(self, base: str) -> bool:
        with self.waiting_dict_lock:
            return base in self._gather_closed

    def _stitch_gather(self, template: PDChunckedTransTask, pages: Dict[int, int], src_world: int) -> int:
        acc_rank = min(pages)
        acc_page = pages[acc_rank]
        mem = self.mem_managers[self.device_id]
        with torch.cuda.stream(stream=self.copy_cuda_stream):
            for page in pages.values():
                if not self._wait_maca_recv(
                    page, self.copy_cuda_stream, timeout=self._maca_wait_timeout(template)
                ):
                    raise TimeoutError(f"maca recv not ready for gather page {page}")
            for rank, page in pages.items():
                if page == acc_page:
                    continue
                if template.page_kind == "kv":
                    from lightllm.common.kv_trans_kernel.nixl_kv_trans import copy_page_head_slice

                    copy_page_head_slice(
                        mem.kv_move_buffer,
                        acc_page,
                        page,
                        rank,
                        src_world,
                        template.transfer_kv_num(),
                    )
                    scale_pages = getattr(mem, "scale_move_buffer", None)
                    if scale_pages is not None:
                        copy_page_head_slice(
                            scale_pages,
                            acc_page,
                            page,
                            rank,
                            src_world,
                            template.transfer_kv_num(),
                        )
                else:
                    from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
                        Qwen3NextLinearAttPageHelper,
                    )

                    Qwen3NextLinearAttPageHelper(mem).copy_rank_slice(acc_page, page, rank, src_world)
        self.copy_cuda_stream.synchronize()
        for page in pages.values():
            if page != acc_page:
                self._release_synced_dst_page(page)
        if self.args.pd_detail_log:
            logger.debug(
                f"maca gather decode_rank={self.device_id} request_id={template.request_id} "
                f"page_kind={template.page_kind} sources={sorted(pages)} acc_page={acc_page}"
            )
        return acc_page

    @log_exception
    def dispatch_task_loop(self):
        while True:
            trans_task_group: PDChunckedTransTaskGroup = self.recv_task_group_queue.get()

            replay: List[PDChunckedTransTask] = []
            with self.waiting_dict_lock:
                for task in trans_task_group.task_list:
                    self._touch_request(task.request_id)
                    if task.need_transfer_page():
                        self.waiting_dict[task.get_key()] = task
                        replay.extend(item[1] for item in self._early_writes.pop(task.get_key(), []))
                    else:
                        task.start_trans_time = time.time()
                        self.success_queue.put((None, None, task))
            for remote in replay:
                self._accept_write_request(remote)

            # up status
            task = trans_task_group.task_list[0]

            # 同号 rank 都会等这份 KV。只有最初被分配的那张卡向 master 上报，避免一条请求被登记四次。
            if self.device_id != task.dst_device_id:
                continue

            pack_metadata = getattr(self.transporter, "agent_metadata_for_prefill", None)
            decode_node_info = PDDecodeNodeInfo(
                decode_node_id=self.args.pd_node_id,
                pd_master_node_id=task.pd_master_node_id,
                agent_name=self.transporter.agent_name,
                agent_metadata=pack_metadata() if pack_metadata is not None else self.transporter.agent_metadata,
                num_pages=self.transporter.num_pages,
                page_reg_desc=self.transporter.local_page_mem_desc,
                request_id=task.request_id,
                ready_kv_len=task.start_kv_index,
            )

            up_status = PDUpKVStatus(
                group_request_id=task.request_id,
                pd_master_node_id=task.pd_master_node_id,
                pd_kv_trans_params=pickle.dumps(decode_node_info),
            )

            self.up_status_in_queue.put(up_status)

    @log_exception
    def accept_peer_task_loop(
        self,
    ):
        torch.cuda.set_device(self.device_id)
        while True:
            # notify update
            try:
                notifies_dict = self.transporter.get_new_notifs()
            except BaseException as e:
                logger.error(f"get new notifies failed: {str(e)}")
                logger.exception(str(e))
                notifies_dict = {}

            if notifies_dict:
                for remote_agent_name, _notify_list in notifies_dict.items():
                    for notify in _notify_list:
                        if isinstance(notify, PDChunckedTransTask):
                            notify_obj = notify
                        else:
                            try:
                                notify_obj = pickle.loads(notify)
                            except Exception:
                                notify_obj = None

                        if not isinstance(notify_obj, PDChunckedTransTask):
                            continue

                        # 请求有错误
                        if notify_obj.error_info is not None:
                            # 直接清理掉所有的相关请求。
                            with self.waiting_dict_lock:
                                base = notify_obj.get_key()
                                if base in self._gather:
                                    self._close_gather(base, notify_obj.error_info)
                                else:
                                    local_trans_task = self.waiting_dict.pop(base, None)
                                    if local_trans_task is not None:
                                        local_trans_task.error_info = notify_obj.error_info
                                        # 页在 fail_loop 里归还。已经 start_recv 的页要等接收结束。
                                        self.failed_queue.put(local_trans_task)

                            self._abort(
                                request_id=notify_obj.request_id,
                                error_info=notify_obj.error_info,
                            )
                            continue

                        # 到了请求页面的阶段
                        remote_trans_task = notify_obj
                        if remote_trans_task.write_stage == "request":
                            self._accept_write_request(remote_trans_task)
                            continue

                        # prefill 写完数据到了 done 阶段
                        if remote_trans_task.write_stage == "done":
                            if self._accept_write_done(remote_trans_task):
                                continue
                            else:
                                # Same race as the WRITE request stage: decode may have cleaned the
                                # waiting task because the request was aborted, then a late done notify
                                # arrives from prefill. Preserve the original error path, but make the
                                # diagnostic tell future readers this can be abort-related noise.
                                logger.warning(
                                    "can not find waiting WRITE task for done notify, "
                                    "possibly because request was already aborted and cleaned on decode side: "
                                    f"{remote_trans_task.to_str()}"
                                )
                                # 发一个error信息回去给 prefill 节点，让其可以知道这边有问题了，它可以选择其他清理掉请求。
                                remote_trans_task.error_info = "can not find waiting WRITE task for done notify"
                                self.transporter.send_error_info_to_prefill_node(trans_task=remote_trans_task)
                            continue

            self._check_tasks_time_out()
            self._expire_early_writes()
            self._flush_prefill_errors()
            if not notifies_dict:
                time.sleep(0.001)

    def _accept_write_request(self, remote_trans_task: PDChunckedTransTask):
        base = remote_trans_task.get_key()
        source = remote_trans_task.prefill_agent_name
        with self.waiting_dict_lock:
            group = self._gather.get(base)
            if group is None:
                local_trans_task = self.waiting_dict.pop(base, None)
                if local_trans_task is None:
                    # 另一张卡的 dispatch 可能还没把任务放进来。先挂住，任务一到就配上。
                    self._early_writes.setdefault(base, []).append((time.time(), remote_trans_task))
                    return
                self._touch_request(local_trans_task.request_id)
                try:
                    world = self._prefill_world(remote_trans_task)
                    sources = self._gather_source_ranks(world)
                except RuntimeError as e:
                    local_trans_task.error_info = str(e)
                    self.failed_queue.put(local_trans_task)
                    return
                if len(sources) <= 1:
                    self._copy_prefill_endpoint(local_trans_task, remote_trans_task)
                    self.request_page_task_queue.put(local_trans_task)
                    if self.args.pd_detail_log:
                        logger.debug(f"recv WRITE request from prefill: {remote_trans_task.get_key()}")
                    return
                group = {
                    "sources": set(sources),
                    "shards": {},
                    "pages": {},
                    "template": local_trans_task,
                    "world": world,
                }
                self._gather[base] = group
            self._touch_request(remote_trans_task.request_id)
            if source is None:
                self._close_gather(base, "write request missing prefill agent")
                return
            rank = self._source_rank(source)
            if rank not in group["sources"] or rank in group["shards"]:
                self._close_gather(base, f"unexpected prefill source {source}")
                return
            shard = copy.copy(group["template"])
            self._copy_prefill_endpoint(shard, remote_trans_task)
            group["shards"][rank] = shard
            # 两路 prefill 都到了再占页面。先到的那一路把 16 张页面占满后，
            # 另一路永远拿不到页面，这一组也拼不完，请求会卡到 180 秒超时。
            if set(group["shards"]) == group["sources"]:
                ordered = [group["shards"][item] for item in sorted(group["shards"])]
                self._touch_request(remote_trans_task.request_id)
                self.request_page_task_queue.put(ordered)
        if self.args.pd_detail_log:
            logger.debug(f"recv WRITE request from prefill: {remote_trans_task.get_key()}")

    def _accept_write_done(self, remote_trans_task: PDChunckedTransTask) -> bool:
        base = remote_trans_task.get_key()
        source = remote_trans_task.prefill_agent_name
        store_key = f"{base}@{source}" if source else base
        stitch = None
        missing = False
        with self.waiting_dict_lock:
            group = self._gather.get(base)
            if group is None:
                local_trans_task = self.waiting_dict.pop(base, None)
                if local_trans_task is None:
                    return False
                self._touch_request(local_trans_task.request_id)
                local_trans_task.first_gen_token_id = remote_trans_task.first_gen_token_id
                local_trans_task.first_gen_token_logprob = remote_trans_task.first_gen_token_logprob
                self.ready_page_task_queue.put(local_trans_task)
                if self.args.pd_detail_log:
                    logger.debug(f"recv WRITE done from prefill: {remote_trans_task.get_key()}")
                return True
            local_trans_task = self.waiting_dict.pop(store_key, None)
            if local_trans_task is None:
                missing = True
            else:
                rank = self._source_rank(source)
                self._touch_request(local_trans_task.request_id)
                group["pages"][rank] = local_trans_task.dst_page_index
                group["template"].first_gen_token_id = remote_trans_task.first_gen_token_id
                group["template"].first_gen_token_logprob = remote_trans_task.first_gen_token_logprob
                if self.args.pd_detail_log:
                    logger.debug(f"recv WRITE done from prefill: {remote_trans_task.get_key()}")
                if set(group["pages"]) == group["sources"]:
                    self._gather.pop(base, None)
                    stitch = (group["template"], dict(group["pages"]), group["world"])
        if missing:
            self._missing_write_task(remote_trans_task, "done")
            return True
        if stitch is None:
            return True
        template, pages, world = stitch
        try:
            template.dst_page_index = self._stitch_gather(template, pages, world)
        except BaseException as e:
            logger.exception(f"gather stitch failed: {template.to_str()}")
            for page in pages.values():
                self._recycle_dst_page(page)
            template.dst_page_index = None
            template.error_info = f"gather stitch failed: {e}"
            self.failed_queue.put(template)
            return True
        self.ready_page_task_queue.put(template)
        return True

    def _missing_write_task(self, remote_trans_task: PDChunckedTransTask, stage: str):
        logger.warning(
            f"can not find waiting WRITE task for {stage} notify, "
            "possibly because request was already aborted and cleaned on decode side: "
            f"{remote_trans_task.to_str()}"
        )
        remote_trans_task.error_info = f"can not find waiting WRITE task for {stage} notify"
        self.transporter.send_error_info_to_prefill_node(trans_task=remote_trans_task)

    def _maca_wait_timeout(self, task: PDChunckedTransTask) -> float:
        return float(max(task.transfer_time_out_secs or 0, 180))

    def _wait_maca_recv(self, page_index, stream, timeout: float = None) -> bool:
        wait = getattr(self.transporter, "wait_maca_recv", None)
        if wait is None or page_index is None:
            return True
        return bool(wait(page_index, stream, timeout=timeout))

    def _recycle_dst_page(self, page_index) -> None:
        if page_index is None:
            return
        page = int(page_index)
        # 还没进入 ncclRecv 的预约不要占着页。NCCL 建连卡住时 event 一直是空的，
        # 16 页会被扣光，后面的块拿不到页，也就发不出 WRITE ready。
        disarm = getattr(self.transporter, "disarm_maca_recv", None)
        if is_maca() and disarm is not None and disarm(page):
            self.page_index_queue.put(page)
            return
        busy = getattr(self.transporter, "maca_page_recv_busy", None)
        if is_maca() and busy is not None and busy(page):
            with self._held_lock:
                self._held_pages.setdefault(page, time.monotonic())
            return
        self.page_index_queue.put(page)

    def _release_synced_dst_page(self, page_index) -> None:
        settle = getattr(self.transporter, "settle_maca_recv", None)
        if settle is not None:
            settle(page_index)
        self._recycle_dst_page(page_index)

    @log_exception
    def _release_held_pages_loop(self):
        torch.cuda.set_device(self.device_id)
        logged = set()
        while True:
            with self._held_lock:
                pages = list(self._held_pages.items())
            now = time.monotonic()
            for page, since in pages:
                if not self.transporter.poll_maca_page_idle(page):
                    if now - since > 30 and page not in logged:
                        logged.add(page)
                        logger.error(
                            f"maca decode page {page} still receiving after {now - since:.0f}s, "
                            "holding it out of the pool"
                        )
                    continue
                with self._held_lock:
                    self._held_pages.pop(page, None)
                logged.discard(page)
                self.page_index_queue.put(page)
            time.sleep(0.01)

    def _check_tasks_time_out(self):
        with self.waiting_dict_lock:
            timeout_tasks = []
            gather_bases = []
            for key, trans_task in list(self.waiting_dict.items()):
                if not self._request_timed_out(trans_task):
                    continue
                base = trans_task.get_key()
                if base in self._gather:
                    gather_bases.append(base)
                    continue
                timeout_tasks.append(self.waiting_dict.pop(key))
            for base, group in list(self._gather.items()):
                template: PDChunckedTransTask = group["template"]
                if self._request_timed_out(template) or base in gather_bases:
                    gather_bases.append(base)
            for base in set(gather_bases):
                self._close_gather(base, "time out in accept_peer_task_loop")
            self._drop_idle_timeout_marks()

        for trans_task in timeout_tasks:
            trans_task.error_info = "time out in accept_peer_task_loop"
            self.failed_queue.put(trans_task)
        return

    def _touch_request(self, request_id: int) -> None:
        mark = self._timeout_marks.get(request_id)
        if mark is None:
            mark = _SharedTimeOutMark()
            self._timeout_marks[request_id] = mark
        mark.update()

    def _request_timed_out(self, task: PDChunckedTransTask) -> bool:
        mark = self._timeout_marks.get(task.request_id)
        if mark is None:
            return task.time_out()
        return mark.time_out(task)

    def _drop_idle_timeout_marks(self) -> None:
        live = {task.request_id for task in self.waiting_dict.values()}
        for group in self._gather.values():
            template = group.get("template")
            if template is not None:
                live.add(template.request_id)
        for request_id in list(self._timeout_marks):
            if request_id not in live:
                self._timeout_marks.pop(request_id, None)

    def _assign_decode_page(self, trans_task: PDChunckedTransTask, dst_page_index: int) -> None:
        trans_task.dst_page_index = dst_page_index
        if is_maca():
            # 页面分出去之后，NCCL 还要排队。66 秒只够一对一，2p4d 一批页会超过这个时间。
            trans_task.transfer_time_out_secs = 180
        trans_task.start_trans_time = time.time()
        try:
            with self.waiting_dict_lock:
                base = trans_task.get_key()
                group = self._gather.get(base)
                if base in self._gather_closed:
                    self.page_index_queue.put(dst_page_index)
                    trans_task.dst_page_index = None
                    return
                self.waiting_dict[self._waiting_key(trans_task)] = trans_task
                self._touch_request(trans_task.request_id)
            self.transporter.send_write_ready_task_to_prefill_node(trans_task=trans_task)
        except BaseException as e:
            logger.error(f"send write ready task to prefill node failed: {trans_task.to_str()}")
            logger.exception(str(e))
            self.transporter.remove_remote_agent(peer_name=trans_task.prefill_agent_name)
            with self.waiting_dict_lock:
                self.waiting_dict.pop(self._waiting_key(trans_task), None)
                if trans_task.get_key() in self._gather and trans_task.prefill_agent_name:
                    rank = self._source_rank(trans_task.prefill_agent_name)
                    self._gather[trans_task.get_key()]["pages"][rank] = trans_task.dst_page_index
                    self._close_gather(trans_task.get_key(), f"send write ready task to prefill node failed: {e}")
                    return
            trans_task.error_info = f"send write ready task to prefill node failed: {str(e)}"
            self.failed_queue.put(trans_task)

    @log_exception
    def request_page_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            item = self.request_page_task_queue.get()
            tasks: List[PDChunckedTransTask] = item if isinstance(item, list) else [item]
            if tasks and self._gather_is_closed(tasks[0].get_key()):
                continue
            # 同一组 gather 的页面一起拿。只拿一张就停住等下一张时，在途的组还能把页面还回来。
            pages = [self.page_index_queue.get() for _ in tasks]
            if tasks and self._gather_is_closed(tasks[0].get_key()):
                for page in pages:
                    self.page_index_queue.put(page)
                continue
            for trans_task, dst_page_index in zip(tasks, pages):
                self._assign_decode_page(trans_task, dst_page_index)
            self._flush_prefill_errors()
        return

    @log_exception
    def read_page_to_mems_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.ready_page_task_queue.get()
            if is_maca() and not self._wait_maca_recv(
                trans_task.dst_page_index,
                self.copy_cuda_stream,
                timeout=self._maca_wait_timeout(trans_task),
            ):
                trans_task.error_info = "maca recv not finished before read"
                self.failed_queue.put(trans_task)
                continue
            copy_start_event = torch.cuda.Event(enable_timing=True)
            copy_end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream=self.copy_cuda_stream):
                copy_start_event.record(self.copy_cuda_stream)
                cur_mem = self.mem_managers[self.device_id]
                cur_mem.read_page_kv_move_buffer_to_mem(
                    trans_task.mem_indexes,
                    page_index=trans_task.dst_page_index,
                    dp_index=trans_task.decode_dp_index,
                    mem_managers=self.mem_managers,
                    dp_world_size=self.dp_world_size,
                    page_kind=trans_task.page_kind,
                    req_idx=trans_task.req_idx,
                    target_tp_index=self.device_id if is_maca() else None,
                )
                copy_end_event.record(self.copy_cuda_stream)
            self.success_queue.put((copy_end_event, copy_start_event, trans_task))

    @log_exception
    def success_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            copy_end_event, copy_start_event, trans_task = self.success_queue.get()
            trans_task: PDChunckedTransTask = trans_task
            copy_end_event: Optional[torch.cuda.Event] = copy_end_event
            copy_start_event: Optional[torch.cuda.Event] = copy_start_event
            read_page_gpu_time_ms = -1.0
            if copy_end_event is not None:
                copy_end_event.synchronize()
                read_page_gpu_time_ms = copy_start_event.elapsed_time(copy_end_event)

            if trans_task.dst_page_index is not None:
                self._release_synced_dst_page(trans_task.dst_page_index)

            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)

            if self.args.pd_detail_log:
                if trans_task.start_trans_time is not None:
                    logger.debug(
                        f"trans task ret success:{ret} cost time: {trans_task.transfer_time()} s "
                        f"read_page_gpu_time: {read_page_gpu_time_ms:.3f} ms"
                    )
                else:
                    logger.debug(f"trans task ret success:{ret}")

    @log_exception
    def fail_loop(self):
        torch.cuda.set_device(self.device_id)
        while True:
            trans_task: PDChunckedTransTask = self.failed_queue.get()

            # 已经 start_recv 的页先挂住，等接收结束再入池。
            if trans_task.dst_page_index is not None:
                self._recycle_dst_page(trans_task.dst_page_index)

            if trans_task.xfer_handle is not None:
                self.transporter.release_xfer_handle(trans_task.xfer_handle)

            ret = trans_task.createRetObj()
            self.task_out_queue.put(ret)
            logger.info(f"trans task ret fail:{ret}")

            if trans_task.error_info is not None:
                # 提前终结所有有问题的属于同一个请求的任务。
                self._abort(
                    request_id=trans_task.request_id,
                    error_info=trans_task.error_info,
                )
                self.transporter.send_error_info_to_prefill_node(trans_task=trans_task)


class _SharedTimeOutMark:
    """还没开始传的页共用请求进度。已经在传的页仍用自己的截止时间。"""

    def __init__(self):
        self.update()

    def update(self):
        self.time_mark = time.time()

    def time_out(self, task: PDChunckedTransTask) -> bool:
        if task.start_trans_time is not None:
            return task.time_out()
        return time.time() - self.time_mark > task.time_out_secs
