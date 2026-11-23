import time
import os
import signal
import threading
import queue
import torch.multiprocessing as mp
from typing import List, Dict, Union, Callable, Optional
from lightllm.utils.device_utils import is_maca
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import PDChunckedTransTaskRet
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs.shm_objs_io_buffer import ShmObjsIOBuffer
from .trans_process_obj import KVTransProcess
from lightllm.utils.error_utils import log_exception

logger = init_logger(__name__)


class BaseKVMoveManager:
    def __init__(
        self,
        args: StartArgs,
        info_queue: mp.Queue,
        start_trans_process_func: Callable,
        up_status_in_queue: Optional[mp.SimpleQueue] = None,
    ):
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.ret_obj_queue = queue.Queue()
        self._shard_lock = threading.Lock()
        self._shard_rets: Dict[tuple, List[PDChunckedTransTaskRet]] = {}

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            trans_process = KVTransProcess()
            self.kv_trans_processes[device_id] = trans_process
            assert trans_process.init_all(
                device_id=device_id,
                manager=self,
                start_func=start_trans_process_func,
                up_status_in_queue=up_status_in_queue,
            )

        for trans_process in self.kv_trans_processes:
            if not trans_process.wait_until_ready():
                raise RuntimeError(f"KV trans module for device {trans_process.device_id} failed to initialize")

        for trans_process in self.kv_trans_processes:
            threading.Thread(target=self.task_ret_handle_loop, args=(trans_process,), daemon=True).start()

        # 通过 io buffer 将命令写入到推理进程中
        self.shm_pd_trans_io_buffer = ShmObjsIOBuffer(tail_str="pd")

        for func in [self.task_dispatcher_loop, self.task_ret_upload_loop, self.check_trans_process_loop]:
            threading.Thread(target=func, daemon=True).start()
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================

    def task_dispatcher_loop(self):
        raise NotImplementedError()

    # ==================================================================================
    #  将收集到的传输返回信息，批量写回给推理进程，触发其进行相关管理信息的更新。
    # ==================================================================================
    @log_exception
    def task_ret_upload_loop(self):
        while True:
            ret_obj: PDChunckedTransTaskRet = self.ret_obj_queue.get()
            ret_objs: List[PDChunckedTransTaskRet] = [ret_obj]
            ret_objs.extend(self._collect_return_objects())

            while True:
                if self.shm_pd_trans_io_buffer.is_empty():
                    # to do, 这里写入的数量，可能会超过共享管道的大小。
                    self.shm_pd_trans_io_buffer.write_obj(ret_objs)
                    self.shm_pd_trans_io_buffer.set_ready()
                    break
                else:
                    time.sleep(0.01)
                    ret_objs.extend(self._collect_return_objects())

    def _collect_return_objects(self):
        """从队列中收集所有返回对象"""
        ret_objs = []
        try:
            while True:
                ret_obj = self.ret_obj_queue.get_nowait()
                ret_objs.append(ret_obj)
        except queue.Empty:
            pass
        return ret_objs

    # ==================================================================================
    # 处理各个传输任务的返回的信息
    # ==================================================================================
    @log_exception
    def task_ret_handle_loop(self, trans_process: KVTransProcess):
        while True:
            ret_obj: PDChunckedTransTaskRet = trans_process.task_out_queue.get()
            self._merge_rank_ret(ret_obj)

    def _merge_rank_ret(self, ret_obj: PDChunckedTransTaskRet):
        # 沐曦上一条请求会在每个 rank 各传自己的那一份。推理侧仍按一条任务计数，这里收齐再回一条。
        need = self.node_world_size if is_maca() else 1
        if need <= 1:
            self.ret_obj_queue.put(ret_obj)
            return
        key = (ret_obj.request_id, ret_obj.page_kind, ret_obj.start_kv_index, ret_obj.end_kv_index)
        with self._shard_lock:
            bucket = self._shard_rets.setdefault(key, [])
            bucket.append(ret_obj)
            if len(bucket) < need:
                return
            done = self._shard_rets.pop(key)
        merged = done[0]
        failed = next((item for item in done if item.has_error), None)
        if failed is not None:
            merged.has_error = True
            merged.error_info = failed.error_info
        self.ret_obj_queue.put(merged)

    # ==================================================================================
    # 定时检测传输进程的健康状态，出现问题拉崩整个系统触发重启
    # ==================================================================================
    def check_trans_process_loop(self):
        try:
            while True:
                for device_id in range(self.node_world_size):
                    if not self.kv_trans_processes[device_id].is_trans_process_health():
                        raise Exception(f"device_id {device_id} kv process is unhealth")

                time.sleep(10.0)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))

            for device_id in range(self.node_world_size):
                self.kv_trans_processes[device_id].killself()

            # 杀掉当前进程的父进程（router), 触发全局崩溃
            os.kill(os.getppid(), signal.SIGKILL)
            os.kill(os.getpid(), signal.SIGKILL)
            raise e
