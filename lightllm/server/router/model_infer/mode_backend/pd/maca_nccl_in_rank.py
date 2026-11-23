"""沐曦 NCCL KV 留在 model rank 进程里做。

graph 录完以后，新进程里的 ncclCommInitRank 会卡在
mxkwCreateQueueBlock type:21。同一个进程里再建一套 NCCL 是可以的。
所以 maca + pd_trans_mode=nccl 时，传输逻辑跑在 model rank 的线程里。
kv move manager 用 unix socket 把任务送进来。其他平台仍然走独立传输进程。
"""

import os
import queue
import threading
import time
from multiprocessing.connection import Client, Listener
from typing import Optional

import torch
import torch.multiprocessing as mp

from lightllm.server.core.objs import StartArgs
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def use_maca_nccl_in_model_rank(args: StartArgs) -> bool:
    return (
        getattr(args, "hardware_platform", None) == "maca"
        and getattr(args, "pd_trans_mode", None) == "nccl"
        and getattr(args, "run_mode", None) in ("prefill", "decode")
    )


def maca_nccl_socket_path(run_mode: str, device_id: int) -> str:
    return f"/tmp/lightllm_{get_unique_server_name()}_maca_nccl_{run_mode}_{device_id}.sock"


def start_maca_nccl_in_rank(backend) -> None:
    device_id = backend.rank_in_node
    path = maca_nccl_socket_path(backend.args.run_mode, device_id)
    threading.Thread(
        target=_serve,
        args=(backend, device_id, path),
        name=f"maca_nccl_rank{device_id}",
        daemon=True,
    ).start()
    logger.info("maca nccl in-rank service starting on %s", path)


def _serve(backend, device_id: int, path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
        listener = Listener(path, family="AF_UNIX")
        logger.info("maca nccl in-rank listening %s", path)
        conn = listener.accept()
        outgoing: "queue.Queue" = queue.Queue()
        outgoing.put(("ready", "proc_start"))

        torch.cuda.set_device(device_id)
        copy_stream = torch.cuda.Stream(priority=0)
        node_world_size = backend.node_world_size
        # 只保留本 rank 的 KV。别的 rank 通过 IPC 映射进来之后，.device 会被改成本卡，
        # 随后的 Triton 读会走到组内 MetaXLink。同号发送用不到那些映射。
        mem_managers = []
        for rank in range(node_world_size):
            if rank == device_id:
                mem_managers.append(backend.model.mem_manager)
            else:
                mem_managers.append(None)

        task_in: "queue.Queue" = queue.Queue()
        task_out: "queue.Queue" = queue.Queue()
        if backend.args.run_mode == "prefill":
            from .prefill_node_impl.prefill_trans_process import _PrefillTransModule

            _PrefillTransModule(
                args=backend.args,
                device_id=device_id,
                task_in_queue=task_in,
                task_out_queue=task_out,
                mem_managers=mem_managers,
                copy_stream=copy_stream,
            )
        else:
            from .decode_node_impl.decode_trans_process import _DecodeTransModule

            up_status_queue: "queue.Queue" = queue.Queue()

            def pump_up():
                while True:
                    outgoing.put(("up", up_status_queue.get()))

            threading.Thread(target=pump_up, daemon=True).start()
            _DecodeTransModule(
                args=backend.args,
                device_id=device_id,
                task_in_queue=task_in,
                task_out_queue=task_out,
                mem_managers=mem_managers,
                up_status_in_queue=up_status_queue,
                copy_stream=copy_stream,
            )

        def pump_ret():
            while True:
                outgoing.put(("ret", task_out.get()))

        threading.Thread(target=pump_ret, daemon=True).start()
        outgoing.put(("ready", "get_mem_managers_ok"))
        logger.info("maca nccl in-rank device %s ready", device_id)

        def on_message(message):
            kind, obj = message
            if kind != "task":
                logger.error("maca nccl in-rank ignore message kind %s", kind)
                return
            task_in.put(obj)

        # 同一条连接不能两个线程一边 send 一边 recv，batch 一大就会把帧拆乱。
        _pump_conn(conn, outgoing, on_message)
    except Exception:
        logger.exception("maca nccl in-rank service failed device %s", device_id)


class _SocketPutQueue:
    def __init__(self, outgoing: "queue.Queue"):
        self._outgoing = outgoing

    def put(self, obj):
        self._outgoing.put(("task", obj))


def _pump_conn(conn, outgoing: "queue.Queue", on_message) -> None:
    while True:
        try:
            item = outgoing.get(timeout=0.001)
        except queue.Empty:
            item = None
        if item is not None:
            conn.send(item)
            while True:
                try:
                    conn.send(outgoing.get_nowait())
                except queue.Empty:
                    break
        while conn.poll(0):
            on_message(conn.recv())


def connect_maca_nccl_rank(args: StartArgs, device_id: int, up_status_in_queue: Optional[mp.SimpleQueue]):
    path = maca_nccl_socket_path(args.run_mode, device_id)
    deadline = time.time() + 30
    last_error = None
    conn = None
    while time.time() < deadline:
        try:
            conn = Client(path, family="AF_UNIX")
            break
        except Exception as e:
            last_error = e
            time.sleep(0.2)
    if conn is None:
        raise RuntimeError(f"maca nccl socket not ready: {path}") from last_error
    task_out: "queue.Queue" = queue.Queue()
    outgoing: "queue.Queue" = queue.Queue()
    health = {"ok": True}

    def on_message(message):
        kind, obj = message
        if kind in ("ready", "ret"):
            task_out.put(obj)
        elif kind == "up":
            if up_status_in_queue is None:
                logger.error("maca nccl got up status on a prefill rank")
            else:
                up_status_in_queue.put(obj)
        else:
            logger.error("maca nccl client ignore kind %s", kind)

    def recv_loop():
        try:
            _pump_conn(conn, outgoing, on_message)
        except Exception:
            health["ok"] = False
            logger.exception("maca nccl client recv loop exit device %s", device_id)

    threading.Thread(target=recv_loop, daemon=True).start()
    return _SocketPutQueue(outgoing), task_out, health
