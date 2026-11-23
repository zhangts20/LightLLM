import copy
import inspect
import logging
import setproctitle
import torch.multiprocessing as mp
import time
from typing import List, Dict, Union, Callable
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTask
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.core.objs import StartArgs
from ..trans_process_obj import KVTransProcess
from ..base_kv_move_manager import BaseKVMoveManager
from lightllm.utils.error_utils import log_exception
from lightllm.utils.device_utils import is_maca
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.process_check import start_parent_check_thread

logger = init_logger(__name__)


def start_prefill_kv_move_manager_process(args, info_queue: mp.Queue):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return


def _init_env(args, info_queue: mp.Queue, event: mp.Event):
    start_parent_check_thread()
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::prefill_kv_move_manager")

    from .prefill_trans_process import start_prefill_trans_process

    manager = PrefillKVMoveManager(
        args=args, info_queue=info_queue, start_trans_process_func=start_prefill_trans_process
    )
    assert manager is not None
    event.set()
    while True:
        time.sleep(100)
    return


class PrefillKVMoveManager(BaseKVMoveManager):
    def __init__(self, args: StartArgs, info_queue: mp.Queue, start_trans_process_func: Callable):
        super().__init__(args=args, info_queue=info_queue, start_trans_process_func=start_trans_process_func)
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================
    @log_exception
    def task_dispatcher_loop(self):
        # 获取任务，并分发给相关卡的处理队列
        while True:
            task: Union[PDChunckedTransTask, PDAbortReq] = self.info_queue.get()

            try:
                if isinstance(task, PDAbortReq):
                    # 沐曦每个 rank 各自有传输任务，abort 要送到每一张卡。
                    device_ids = (
                        range(self.node_world_size)
                        if is_maca() and self.node_world_size > 1
                        else (task.device_id,)
                    )
                    for device_id in device_ids:
                        self.kv_trans_processes[device_id].task_in_queue.put(task)
                    continue

                # vLLM 是每个 TP rank 把自己的 KV 送给同号 decode rank。不能只挑一张卡把其它卡的显存搬过来。
                if is_maca() and self.node_world_size > 1:
                    node_name = task.decode_agent_name.rsplit("_", 1)[0]
                    agents = task.decode_agents or {}
                    for device_id in range(self.node_world_size):
                        shard = copy.copy(task)
                        shard.src_device_id = device_id
                        # decode 卡更少时，同号 agent 不存在。先留着 master 给的名字，后面按 head 区间改目标。
                        metadata = agents.get(device_id)
                        if metadata is not None:
                            shard.decode_agent_name = f"{node_name}_{device_id}"
                            shard.decode_agent_metadata = metadata
                        self.kv_trans_processes[device_id].task_in_queue.put(shard)
                else:
                    device_id = task.src_device_id
                    trans_process: KVTransProcess = self.kv_trans_processes[device_id]
                    trans_process.task_in_queue.put(task)
                    if self.args.pd_detail_log and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"kv move manager dispatch task {task.to_str()} to device {device_id}")
            except BaseException as e:
                logger.exception(str(e))
