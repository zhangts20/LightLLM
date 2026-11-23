import queue
import threading
import psutil
import torch.multiprocessing as mp
from typing import List, Callable, Optional
from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class KVTransProcess:
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    device_id: int = None

    def init_all(
        self,
        device_id: int,
        manager,
        start_func: Callable,
        up_status_in_queue: Optional[mp.SimpleQueue],
    ):
        from .base_kv_move_manager import BaseKVMoveManager

        manager: BaseKVMoveManager = manager
        self.device_id = device_id
        from .maca_nccl_in_rank import connect_maca_nccl_rank, use_maca_nccl_in_model_rank

        if use_maca_nccl_in_model_rank(manager.args):
            try:
                self.task_in_queue, self.task_out_queue, self._maca_health = connect_maca_nccl_rank(
                    manager.args, device_id, up_status_in_queue
                )
                assert self.task_out_queue.get(timeout=60) == "proc_start"
                assert self.task_out_queue.get(timeout=120) == "get_mem_managers_ok"
                self._maca_in_rank = True
                return True
            except Exception as e:
                logger.warning(f"Failed connect maca nccl in-rank worker for device {device_id}: {e}")
                logger.exception(str(e))
                return False

        self._maca_in_rank = False
        self.task_in_queue = mp.Queue()
        self.task_out_queue = mp.Queue()

        try:
            self.process = start_func(
                manager.args,
                device_id,
                self.task_in_queue,
                self.task_out_queue,
                up_status_in_queue,
            )
            assert self.task_out_queue.get(timeout=30) == "proc_start"
            assert self.task_out_queue.get(timeout=60) == "get_mem_managers_ok"

            return True

        except Exception as e:
            logger.warning(f"Failed start kv trans process for device {device_id}: {e}")
            logger.exception(str(e))
            return False

    def wait_until_ready(self):
        # 进程内 maca 线程在发出 get_mem_managers_ok 之前已经建好传输模块。
        if getattr(self, "_maca_in_rank", False):
            return True
        for _ in range(600):
            try:
                status = self.task_out_queue.get(timeout=1)
            except queue.Empty:
                if self.process is None or not self.process.is_alive():
                    logger.error(f"KV trans process for device {self.device_id} exited during initialization")
                    return False
                continue
            if status != "module_ready":
                logger.error(f"KV trans module for device {self.device_id} failed to initialize: {status}")
                return False
            return True
        logger.error(f"KV trans module for device {self.device_id} initialization timed out")
        return False

    def is_trans_process_health(self):
        if getattr(self, "_maca_in_rank", False):
            alive = bool(self._maca_health.get("ok"))
            if not alive:
                logger.error(f"maca nccl in-rank worker for device: {self.device_id} dead!!!")
            return alive
        try:
            process = psutil.Process(self.process.pid)
            if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
                logger.error(f"kv trans process for device: {self.device_id} dead!!!")
                return False
            else:
                return True
        except:
            return False

    def killself(self):
        if getattr(self, "_maca_in_rank", False):
            if getattr(self, "_maca_health", None) is not None:
                self._maca_health["ok"] = False
            return
        if self.process is not None:
            self.process.kill()
