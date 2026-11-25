import copy
import errno
import fcntl
import json
import os
import pickle
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import rpyc
import torch
from torch import Tensor
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer

from lightllm.distributed.pynccl import PyNcclCommunicator, StatelessP2PProcessGroup
from lightllm.server.pd_io_struct import PDChunckedTransTask, PDAgentMetadata
from lightllm.utils.device_utils import is_maca
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip

logger = init_logger(__name__)


def maca_node_world_size() -> int:
    args = get_env_start_args()
    return args.tp // args.nnodes


def _maca_require_tp_multiple(prefill_tp: int, decode_tp: int) -> None:
    # head 按整段切。一边的卡数必须是另一边的整数倍，区间才能对齐。
    if decode_tp % prefill_tp != 0 and prefill_tp % decode_tp != 0:
        raise RuntimeError(
            f"maca pd wants one tp to be a multiple of the other, prefill {prefill_tp} decode {decode_tp}"
        )


def maca_decode_ranks_for_prefill(prefill_rank: int, prefill_tp: int, decode_tp: int) -> List[int]:
    # prefill 2、decode 4：rank 0 盖住 decode 0 和 1。prefill 4、decode 2：rank 0 和 1 都只发给 decode 0。
    _maca_require_tp_multiple(prefill_tp, decode_tp)
    if decode_tp % prefill_tp == 0:
        span = decode_tp // prefill_tp
        start = prefill_rank * span
        return list(range(start, start + span))
    span = prefill_tp // decode_tp
    return [prefill_rank // span]


def maca_prefill_ranks_for_decode(decode_rank: int, prefill_tp: int, decode_tp: int) -> List[int]:
    _maca_require_tp_multiple(prefill_tp, decode_tp)
    if prefill_tp % decode_tp == 0:
        span = prefill_tp // decode_tp
        start = decode_rank * span
        return list(range(start, start + span))
    return [decode_rank * prefill_tp // decode_tp]


def _maca_node_of(agent_name: str) -> str:
    return agent_name.rsplit("_", 1)[0]


def maca_kv_local_span(global_heads: int, world: int, rank: int):
    # 全局 head 不能整除时（比如 4 个 KV head、TP=8）一张卡会复制 head，不能按段打包。
    if world <= 1 or global_heads <= 0 or global_heads % world != 0:
        return None
    if rank < 0 or rank >= world:
        return None
    local = global_heads // world
    return local, rank * local


def _maca_dist_store():
    import torch.distributed as dist
    from torch.distributed import distributed_c10d

    if not dist.is_initialized():
        raise RuntimeError("maca decode agents need torch.distributed to be initialized")
    return distributed_c10d._get_default_store()


def _store_set(store, key: str, payload: bytes) -> None:
    try:
        store.set(key, payload)
    except TypeError:
        store.set(key, payload.decode("latin1"))


def _store_get(store, key: str) -> bytes:
    deadline = time.time() + 30
    check = getattr(store, "check", None)
    if check is not None:
        while not check([key]):
            if time.time() >= deadline:
                raise TimeoutError(f"maca decode agent {key} was not published within 30s")
            time.sleep(0.02)
    value = store.get(key)
    if isinstance(value, str):
        return value.encode("latin1")
    return bytes(value)


@dataclass
class NcclAgentMetadata:
    agent_name: str
    host_ip: str
    control_port: int
    device_id: int
    world_size: int = 0
    # 只放在上报给 prefill 的那一份里。key 是本节点的 tp rank。
    peers: Optional[Dict[int, bytes]] = None


class NcclKVTransporter:
    """
    PD KV transporter backed by NCCL point-to-point operations.

    NCCL does not provide remote notifications or one-sided WRITE, so this class
    uses a small RPyC control channel for notifications and communicator bootstrap
    while preserving the same request/ready/done/error interface used by pd
    trans-process management.
    """

    def __init__(
        self,
        node_id: int,
        tp_idx: int,
        kv_move_buffer: Tensor,
        host_ip: Optional[str] = None,
        control_port_min: int = 30000,
        control_port_max: int = 40000,
    ):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.kv_move_buffer = kv_move_buffer
        args = get_env_start_args()
        self.args = args
        assert args.run_mode in ["prefill", "decode"], args.run_mode
        self.is_prefill_node = args.run_mode == "prefill"
        self.num_pages, self.page_size, self.num_layers, self.kv_head_num, self.head_dims = kv_move_buffer.shape

        self.host_ip = host_ip or get_hostname_ip()
        assert self.host_ip is not None, "can not get host ip for NcclKVTransporter"

        self.control_channel = _NcclControlChannel(
            host_ip=self.host_ip,
            port_min=control_port_min,
            port_max=control_port_max,
        )
        if is_maca() and not self.is_prefill_node:
            # 控制口写进本进程组的 TCPStore。上报的那张卡再收齐，跟请求一起带给 prefill。
            self._publish_maca_decode_agent()
        self.remote_agents: Dict[str, PDAgentMetadata] = {}
        self._peers: Dict[str, "_NcclPeer"] = {}
        self._peer_lock = threading.Lock()
        self._maca_send_q: Dict[str, deque] = {}
        self._maca_recv_q: Dict[str, deque] = {}
        # 页还在接收：(event, armed_at, generation)。
        # event 为空且 armed_at 有值：只是占了页，NCCL 还没开始收。
        # armed_at 为空：已经进入 ncclRecv，要等 event。
        self._maca_recv_events: Dict[int, tuple] = {}
        self._maca_recv_gen: Dict[int, int] = {}
        self._maca_copy_events: Dict[tuple, tuple] = {}
        self._maca_q_lock = threading.RLock()
        self._maca_match_cache = None
        # 同一台机器上的多卡仍由这一条线程用同一条通信器发送。
        # 不同节点各起一条线程，和 CUDA 的每个 peer 一条 recv 线程一样。
        if is_maca():
            threading.Thread(
                target=self._maca_nccl_loop, name=f"maca_nccl_{self.tp_idx}", daemon=True
            ).start()
        return

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self) -> bytes:
        cached = getattr(self, "_agent_metadata_bytes", None)
        if cached is None:
            cached = pickle.dumps(
                NcclAgentMetadata(
                    agent_name=self.agent_name,
                    host_ip=self.host_ip,
                    control_port=self.control_channel.port,
                    device_id=self.tp_idx,
                    world_size=maca_node_world_size(),
                )
            )
            self._agent_metadata_bytes = cached
        return cached

    @property
    def local_page_mem_desc(self) -> bytes:
        cached = getattr(self, "_local_page_mem_desc_bytes", None)
        if cached is None:
            cached = pickle.dumps(
                {
                    "num_pages": self.num_pages,
                    "page_size": self.page_size,
                    "num_layers": self.num_layers,
                    "kv_head_num": self.kv_head_num,
                    "head_dims": self.head_dims,
                    "dtype": str(self.kv_move_buffer.dtype),
                }
            )
            self._local_page_mem_desc_bytes = cached
        return cached

    def get_new_notifs(self) -> Dict[str, List[PDChunckedTransTask]]:
        notifs: Dict[str, List[PDChunckedTransTask]] = {}
        for notify in self.control_channel.get_notifs():
            notify_obj = self._load_notify(notify)
            if notify_obj is None:
                continue
            notifs.setdefault(self._notify_source_name(notify_obj), []).append(notify_obj)
        return notifs

    def connect_add_remote_agent(self, remote_agent: PDAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        metadata: NcclAgentMetadata = pickle.loads(remote_agent.agent_metadata)
        assert (
            metadata.agent_name == remote_agent.agent_name
        ), f"Peer name {metadata.agent_name} does not match remote name {remote_agent.agent_name}"

        self.remote_agents[remote_agent.agent_name] = remote_agent
        logger.info(f"Added NCCL remote agent {remote_agent.agent_name} at {metadata.host_ip}:{metadata.control_port}")
        if is_maca():
            self._share_maca_peer(metadata)
        return

    def remove_remote_agent(self, peer_name: str):
        if peer_name in self.remote_agents:
            self.remote_agents.pop(peer_name, None)
            with self._peer_lock:
                peer = self._peers.pop(peer_name, None)
            if peer is not None:
                peer.close()
        else:
            logger.warning(f"try to remove remote agent, but peer name {peer_name} agent did not exist")
        return

    def send_write_done_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        self._send_staged_notify(trans_task, "done", to_decode=True)
        return

    def send_write_request_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        self._send_staged_notify(trans_task, "request", to_decode=True)
        return

    def send_write_ready_task_to_prefill_node(self, trans_task: PDChunckedTransTask):
        if trans_task.prefill_agent_name not in self.remote_agents:
            self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())

        self._get_peer(trans_task.prefill_agent_name).start_recv(trans_task)
        self._send_staged_notify(trans_task, "ready", to_decode=False)
        return

    def send_error_info_to_prefill_node(self, trans_task: PDChunckedTransTask):
        if trans_task.prefill_agent_name is None:
            return
        self._send_staged_notify(trans_task, "error", to_decode=False)
        return

    def send_error_info_to_decode_node(self, trans_task: PDChunckedTransTask):
        self._send_staged_notify(trans_task, "error", to_decode=True)
        return

    def write_blocks_paged(self, trans_task: PDChunckedTransTask) -> "_NcclXferHandle":
        assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None
        decode_agent_name = trans_task.decode_agent_name
        if decode_agent_name not in self.remote_agents:
            self.connect_add_remote_agent(trans_task.create_decode_agent_obj())

        return self._get_peer(decode_agent_name).send_page(trans_task)

    def check_task_status(self, trans_task: PDChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        return trans_task.xfer_handle.check_status()

    def release_xfer_handle(self, handle):
        return

    def shutdown(self):
        with self._peer_lock:
            peers = list(self._peers.values())
            self._peers.clear()
        for peer in peers:
            peer.close()
        self.remote_agents.clear()
        self.control_channel.close()
        return

    def _get_peer(self, peer_name: str) -> "_NcclPeer":
        with self._peer_lock:
            peer = self._peers.get(peer_name)
            if peer is None:
                peer = _NcclPeer(self, peer_name)
                self._peers[peer_name] = peer
            return peer

    def _send_task_notif(self, remote_agent_name: str, trans_task: PDChunckedTransTask):
        if remote_agent_name not in self.remote_agents:
            if remote_agent_name == trans_task.decode_agent_name:
                self.connect_add_remote_agent(trans_task.create_decode_agent_obj())
            else:
                self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())

        remote_metadata = self._get_remote_metadata(remote_agent_name)
        self.control_channel.send_notif(
            remote_agent_name,
            remote_metadata.host_ip,
            remote_metadata.control_port,
            pickle.dumps(trans_task),
        )
        return

    def _get_remote_metadata(self, remote_agent_name: str) -> NcclAgentMetadata:
        cache = getattr(self, "_remote_meta_cache", None)
        if cache is None:
            cache = {}
            self._remote_meta_cache = cache
        cached = cache.get(remote_agent_name)
        if cached is None:
            remote_agent = self.remote_agents[remote_agent_name]
            cached = pickle.loads(remote_agent.agent_metadata)
            cache[remote_agent_name] = cached
        return cached

    def _copy_notify_task(self, trans_task: PDChunckedTransTask) -> PDChunckedTransTask:
        return trans_task.copy_for_notify()

    def _send_staged_notify(self, trans_task: PDChunckedTransTask, stage: str, to_decode: bool) -> None:
        # 发给 decode 的通知带 prefill 身份，发给 prefill 的通知带 decode 身份。
        # 接收端先看 error_info 再看阶段。错误通知没有 error_info 时，整批通知会被丢掉。
        if stage == "error":
            assert trans_task.error_info is not None
        note = self._copy_notify_task(trans_task)
        note.write_stage = stage
        if to_decode:
            note.prefill_agent_name = self.agent_name
            note.prefill_agent_metadata = self.agent_metadata
            note.prefill_num_pages = self.num_pages
            note.prefill_page_reg_desc = self.local_page_mem_desc
            self._send_task_notif(trans_task.decode_agent_name, note)
            return
        note.decode_agent_name = self.agent_name
        note.decode_agent_metadata = self.agent_metadata
        note.decode_num_pages = self.num_pages
        note.decode_page_reg_desc = self.local_page_mem_desc
        self._send_task_notif(trans_task.prefill_agent_name, note)

    def _load_notify(self, notify: bytes):
        notify_obj = pickle.loads(notify)
        if not isinstance(notify_obj, PDChunckedTransTask):
            return None
        return notify_obj

    def _notify_source_name(self, notify_obj: PDChunckedTransTask) -> str:

        if notify_obj.error_info is not None:
            if self.is_prefill_node:
                assert notify_obj.decode_agent_name is not None
                return notify_obj.decode_agent_name
            else:
                assert notify_obj.prefill_agent_name is not None
                return notify_obj.prefill_agent_name

        if notify_obj.write_stage == "request":
            assert notify_obj.prefill_agent_name is not None
            return notify_obj.prefill_agent_name

        if notify_obj.write_stage in ["ready", "done"]:
            assert notify_obj.decode_agent_name is not None
            return notify_obj.decode_agent_name

        raise AssertionError(f"unexpected notify stage: {notify_obj.write_stage}")

    def _publish_maca_decode_agent(self):
        import torch.distributed as dist

        rank = dist.get_rank()
        _store_set(_maca_dist_store(), f"lightllm_maca_decode_agent_{rank}", self.agent_metadata)

    def decode_agent_table(self) -> Dict[int, bytes]:
        cached = getattr(self, "_decode_agents", None)
        if cached is not None:
            return cached
        import torch.distributed as dist

        rank = dist.get_rank()
        world = maca_node_world_size()
        base = rank - self.tp_idx
        store = _maca_dist_store()
        table = {tp: _store_get(store, f"lightllm_maca_decode_agent_{base + tp}") for tp in range(world)}
        self._decode_agents = table
        return table

    def agent_metadata_for_prefill(self) -> bytes:
        meta: NcclAgentMetadata = pickle.loads(self.agent_metadata)
        meta.peers = self.decode_agent_table()
        return pickle.dumps(meta)

    def _peer_share_path(self) -> str:
        uid = os.environ.get("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "0")
        return f"/tmp/lightllm_{uid}_maca_nccl_peers.txt"

    def _share_maca_peer(self, metadata: NcclAgentMetadata):
        # 一条请求只落到一对卡上。把见过的对端写到本机文件，同节点的其它 rank 才能凑齐全连接。
        path = self._peer_share_path()
        line = (
            f"{metadata.agent_name}\t{metadata.host_ip}\t{metadata.control_port}\t"
            f"{metadata.device_id}\t{metadata.world_size}\n"
        )
        with open(path, "a+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            fp.seek(0)
            known = {ln.split("\t", 1)[0] for ln in fp.read().splitlines() if ln}
            if metadata.agent_name not in known:
                fp.write(line)
            fcntl.flock(fp, fcntl.LOCK_UN)

    def _load_shared_maca_peers(self):
        path = self._peer_share_path()
        if not os.path.exists(path):
            return
        with open(path, "r") as fp:
            fcntl.flock(fp, fcntl.LOCK_SH)
            lines = fp.readlines()
            fcntl.flock(fp, fcntl.LOCK_UN)
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            name, host_ip, port, device_id = parts[:4]
            world_size = int(parts[4]) if len(parts) >= 5 else 0
            if name in self.remote_agents:
                continue
            metadata = NcclAgentMetadata(
                agent_name=name,
                host_ip=host_ip,
                control_port=int(port),
                device_id=int(device_id),
                world_size=world_size,
            )
            self.remote_agents[name] = PDAgentMetadata(
                agent_name=name,
                agent_metadata=pickle.dumps(metadata),
                num_pages=self.num_pages,
            )

    def _sorted_peer_names(self) -> List[str]:
        # device_id 只在一台机器内部区分 rank。两台 decode 的 rank 0 都是 device_id 0，
        # 按 device_id 去重会把其中一台盖掉，那一台永远等不到 nccl_layout。
        # 同一个 node id 里 device_id 重复时只留最后登记的，避免重启后的旧名字还占着。
        latest: Dict[tuple, str] = {}
        for name in list(self.remote_agents.keys()):
            meta = self._get_remote_metadata(name)
            latest[(_maca_node_of(name), meta.device_id)] = name
        return [latest[key] for key in sorted(latest)]

    def _maca_queue(self, peer_name: str) -> deque:
        # 调用方已经拿着 _maca_q_lock。
        queues = self._maca_send_q if self.is_prefill_node else self._maca_recv_q
        return queues.setdefault(peer_name, deque())

    def _maca_enqueue_send(self, peer_name: str, trans_task: PDChunckedTransTask, handle: "_NcclXferHandle"):
        with self._maca_q_lock:
            self._maca_send_q.setdefault(peer_name, deque()).append((trans_task, handle))

    def _maca_enqueue_recv(self, peer_name: str, trans_task: PDChunckedTransTask):
        with self._maca_q_lock:
            self._maca_recv_q.setdefault(peer_name, deque()).append(trans_task)

    def _maca_queue_depth(self, peer_name: str) -> int:
        with self._maca_q_lock:
            pending = (self._maca_send_q if self.is_prefill_node else self._maca_recv_q).get(peer_name)
            if pending is None:
                return 0
            return len(pending)

    def _maca_take_tasks(self, peer_name: str, limit: int) -> list:
        with self._maca_q_lock:
            pending = self._maca_queue(peer_name)
            taken = []
            while pending and len(taken) < limit:
                taken.append(pending.popleft())
            return taken

    def _maca_requeue_front(self, peer_name: str, items: list) -> None:
        if not items:
            return
        with self._maca_q_lock:
            pending = self._maca_queue(peer_name)
            pending.extendleft(reversed(items))

    def _maca_copy_event_key(self, trans_task: PDChunckedTransTask):
        return (
            int(trans_task.request_id),
            str(trans_task.page_kind),
            int(trans_task.start_kv_index),
            int(trans_task.end_kv_index),
            int(trans_task.src_page_index),
        )

    def note_maca_copy_event(
        self, trans_task: PDChunckedTransTask, event: torch.cuda.Event, consumers: int = 1
    ) -> None:
        # 2p4d 一份拷贝要发给多张 decode。每一路 send 都要等到这次拷贝，不能被第一路取走。
        if trans_task.src_page_index is None:
            return
        key = self._maca_copy_event_key(trans_task)
        consumers = max(1, int(consumers))
        with self._maca_q_lock:
            current = self._maca_copy_events.get(key)
            if current is None:
                self._maca_copy_events[key] = (event, consumers)
                return
            kept, left = current
            self._maca_copy_events[key] = (kept, max(left, consumers))

    def take_maca_copy_event(self, trans_task: PDChunckedTransTask):
        if trans_task.src_page_index is None:
            return None
        key = self._maca_copy_event_key(trans_task)
        with self._maca_q_lock:
            current = self._maca_copy_events.get(key)
            if not current:
                return None
            event, left = current
            if left <= 1:
                self._maca_copy_events.pop(key, None)
            else:
                self._maca_copy_events[key] = (event, left - 1)
            return event

    def finish_maca_copy(self, trans_task: PDChunckedTransTask) -> None:
        # 发送没把拷贝事件取完就失败时，先等拷贝结束再归还源页。
        if trans_task.src_page_index is None:
            return
        key = self._maca_copy_event_key(trans_task)
        with self._maca_q_lock:
            current = self._maca_copy_events.pop(key, None)
        if current is None:
            return
        try:
            current[0].synchronize()
        except BaseException:
            logger.exception(f"maca copy event sync failed request_id={trans_task.request_id}")

    def arm_maca_recv(self, page_index: Optional[int]) -> None:
        if page_index is None:
            return
        with self._maca_q_lock:
            if page_index in self._maca_recv_events:
                return
            generation = self._maca_recv_gen.get(page_index, 0) + 1
            self._maca_recv_gen[page_index] = generation
            self._maca_recv_events[page_index] = (None, time.monotonic(), generation)

    def maca_recv_generation(self, page_index: Optional[int]) -> Optional[int]:
        if page_index is None:
            return None
        with self._maca_q_lock:
            slot = self._maca_recv_events.get(page_index)
            if slot is None:
                return None
            return slot[2]

    def begin_maca_recv(self, page_index: Optional[int], generation: Optional[int]) -> bool:
        # 进入 ncclRecv 之前调用。预约已经被取消，或页已经换了主人，就不要再写这块缓冲。
        if page_index is None or generation is None:
            return False
        with self._maca_q_lock:
            slot = self._maca_recv_events.get(page_index)
            if slot is None or slot[2] != generation:
                return False
            self._maca_recv_events[page_index] = (slot[0], None, slot[2])
            return True

    def disarm_maca_recv(self, page_index: Optional[int]) -> bool:
        # 只取消还没进入 ncclRecv 的预约。已经在收的页要等这次接收结束。
        # 取消成功时页可以回到池里。
        if page_index is None:
            return False
        with self._maca_q_lock:
            slot = self._maca_recv_events.get(page_index)
            if slot is None:
                return False
            event, armed_at, _generation = slot
            if event is not None or armed_at is None:
                return False
            self._maca_recv_events.pop(page_index, None)
            return True

    def maca_page_recv_busy(self, page_index: Optional[int]) -> bool:
        if page_index is None:
            return False
        with self._maca_q_lock:
            return page_index in self._maca_recv_events

    def settle_maca_recv(self, page_index: Optional[int]) -> None:
        # 调用方已经 synchronize 过等待这条接收的 stream。
        if page_index is None:
            return
        with self._maca_q_lock:
            self._maca_recv_events.pop(page_index, None)

    def poll_maca_page_idle(self, page_index: Optional[int]) -> bool:
        if page_index is None:
            return True
        with self._maca_q_lock:
            slot = self._maca_recv_events.get(page_index)
            if slot is None:
                return True
            event, _armed_at, generation = slot
        # 还没记下 event 的页不能在这里放回池子。任务失败时由 disarm_maca_recv 还页，
        # 避免和 fail_loop 各还一次。
        if event is None:
            return False
        try:
            done = event.query()
        except BaseException:
            done = True
        if not done:
            return False
        with self._maca_q_lock:
            current = self._maca_recv_events.get(page_index)
            if current is not None and current[2] == generation and current[0] is event:
                self._maca_recv_events.pop(page_index, None)
        return True

    def note_maca_recv(
        self, page_index: Optional[int], event: torch.cuda.Event, generation: Optional[int] = None
    ) -> None:
        if page_index is None:
            return
        with self._maca_q_lock:
            slot = self._maca_recv_events.get(page_index)
            if slot is None:
                return
            if generation is not None and slot[2] != generation:
                return
            self._maca_recv_events[page_index] = (event, None, slot[2])

    def wait_maca_recv(
        self, page_index: Optional[int], stream: torch.cuda.Stream, timeout: Optional[float] = None
    ) -> bool:
        # 事件还没记下就先等。传输线程仍然不为每一页做 stream.synchronize()。
        if page_index is None:
            return True
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            with self._maca_q_lock:
                slot = self._maca_recv_events.get(page_index)
                if slot is None:
                    return True
                event = slot[0]
            if event is not None:
                try:
                    if event.query():
                        return True
                except BaseException:
                    pass
                stream.wait_event(event)
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def _peer_share_stamp(self):
        path = self._peer_share_path()
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _maca_matching_peers(self):
        # 卡数相同就是同号。decode 是 prefill 的整数倍时，一张 prefill 卡对应连续的几张 decode 卡。
        # peer 文件和已登记的 agent 都没变时，沿用上次的配对，避免空转反复读文件、反序列化。
        agent_names = tuple(self.remote_agents.keys())
        cached = self._maca_match_cache
        if cached is not None and cached[1] == agent_names and time.time() < getattr(self, "_peer_stamp_until", 0):
            return cached[2], cached[3]
        stamp = self._peer_share_stamp()
        self._peer_stamp_until = time.time() + 0.2
        if cached is not None and cached[0] == stamp and cached[1] == agent_names:
            return cached[2], cached[3]

        self._load_shared_maca_peers()
        agent_names = tuple(self.remote_agents.keys())
        my_tp = maca_node_world_size()
        matched = []
        bad = []
        known_bad = {name for name, _err in cached[3]} if cached is not None else set()
        for name in self._sorted_peer_names():
            meta = self._get_remote_metadata(name)
            other_tp = meta.world_size or my_tp
            try:
                if self.is_prefill_node:
                    wanted = maca_decode_ranks_for_prefill(self.tp_idx, my_tp, other_tp)
                else:
                    wanted = maca_prefill_ranks_for_decode(self.tp_idx, other_tp, my_tp)
            except RuntimeError as e:
                bad.append((name, str(e)))
                if name not in known_bad:
                    logger.error(f"maca nccl reject peer={name} rank={self.tp_idx}: {e}")
                continue
            if meta.device_id in wanted:
                matched.append(name)
        bad = tuple(bad)
        self._maca_match_cache = (stamp, agent_names, matched, bad)
        return matched, bad

    def _maca_reject_peer(self, peer_name: str, error: str) -> None:
        # 卡数不是整数倍时不要空转到传输超时。队列里的任务直接失败，接收页也不再占着。
        taken = self._maca_take_tasks(peer_name, 1 << 20)
        for item in taken:
            if self.is_prefill_node:
                _trans_task, handle = item
                handle.status = "ERR"
                handle.error_info = error
                continue
            trans_task = item
            self.disarm_maca_recv(trans_task.dst_page_index)
            trans_task.error_info = error
            self.send_error_info_to_prefill_node(trans_task)

    def _maca_init_send_group(self, peers: List[str], stream: torch.cuda.Stream) -> None:
        # 一张 prefill 卡发给多张 decode 时只用一条通信器。两条 send 通信器来回切，
        # 第二路的 ncclSend 会停在驱动里，对端的 recv 也一起停。
        comm = getattr(self, "_maca_group_comm", None)
        if comm is None:
            from lightllm.distributed.pynccl_wrapper import NCCLLibrary

            nccl = NCCLLibrary()
            unique_id = nccl.ncclGetUniqueId()
            raw = PyNcclCommunicator.unique_id_to_bytes(unique_id)
            world = len(peers) + 1
            for index, name in enumerate(peers):
                store = _NcclControlStore(self, name)
                store.set("nccl_layout", json.dumps({"world": world, "rank": index + 1}).encode())
                store.set("nccl_uid", raw)
                peer = self._get_peer(name)
                peer.stream = stream
                peer.dst_rank = index + 1
            torch.cuda.set_stream(stream)
            comm = PyNcclCommunicator(None, self.tp_idx, rank=0, world_size=world, unique_id=unique_id)
            self._maca_group_comm = comm
            logger.info(f"maca nccl group ready rank={self.tp_idx} world={world} peers={peers}")
        for index, name in enumerate(peers):
            peer = self._get_peer(name)
            peer.stream = stream
            peer.dst_rank = index + 1
            peer.comm = comm

    def _maca_ensure_pair(self, peer_name: str, timeout: float = 120.0):
        peer = self._get_peer(peer_name)
        if peer.comm is not None:
            return
        stream = peer._get_stream()
        peer._use_stream_for_init(stream)
        peer._ensure_comm(is_server=self.is_prefill_node, timeout=timeout)
        logger.info(f"maca nccl pair ready peer={peer_name}")

    def _maca_peer_posted(self, peer_name: str, epoch: int) -> bool:
        return _NcclControlStore(self, peer_name).posted(f"work/{epoch}")

    @staticmethod
    def _maca_task_key(item) -> tuple:
        task = item[0] if isinstance(item, tuple) else item
        return (int(task.request_id), str(task.page_kind), int(task.start_kv_index), int(task.end_kv_index))

    def _maca_exchange_keys(self, peer_name: str, epoch: int, keys: list, timeout: float = 120.0) -> list:
        # 交换的是任务键，不是条数。两边按同一个排序取交集，NCCL 的字节数才能对上。
        self._maca_post_keys(peer_name, epoch, keys)
        return self._maca_wait_keys(peer_name, epoch, timeout=timeout)

    def _maca_post_keys(self, peer_name: str, epoch: int, keys: list) -> None:
        _NcclControlStore(self, peer_name).set(f"work/{epoch}", json.dumps(keys).encode())

    def _maca_wait_keys(self, peer_name: str, epoch: int, timeout: float = 120.0) -> list:
        # 不弹出。短超时后这一轮作废时，对端仍能读到同一个 key，两边的 epoch 才不会错开。
        raw = _NcclControlStore(self, peer_name).get(f"work/{epoch}", timeout=timeout, pop=False)
        return [tuple(item) for item in json.loads(raw.decode())]

    def _maca_match_items(self, taken: list, other_keys: list):
        local_keys = [self._maca_task_key(item) for item in taken]
        order = {key: index for index, key in enumerate(sorted(set(other_keys) & set(local_keys)))}
        left = {key: other_keys.count(key) for key in order}
        chosen = []
        rest = []
        for item in taken:
            key = self._maca_task_key(item)
            if left.get(key, 0) > 0:
                chosen.append(item)
                left[key] -= 1
            else:
                rest.append(item)
        chosen.sort(key=lambda item: order[self._maca_task_key(item)])
        return chosen, rest

    def _maca_inflight(self, peer_name: str) -> bool:
        peer = self._get_peer(peer_name)
        last_event = getattr(peer, "last_event", None)
        if last_event is None:
            return False
        try:
            inflight = not last_event.query()
        except BaseException:
            peer.last_event = None
            return False
        if inflight:
            return True
        peer.last_event = None
        return False

    def _maca_transfer_item(self, peer_name: str, item) -> None:
        peer = self._get_peer(peer_name)
        if self.is_prefill_node:
            trans_task, handle = item
            try:
                peer._send_page_now(trans_task, handle)
            except BaseException as e:
                handle.status = "ERR"
                handle.error_info = str(e)
                logger.exception(f"maca nccl send failed request_id={trans_task.request_id}")
            return
        peer._recv_page(item)

    def _maca_one_peer_step(self, peer_name: str, epoch: int, timeout: float = 120.0) -> bool:
        # 两边都空就跳过这次 RPC，epoch 不动。对端已经写入时仍要接上，避免有数据的一边空等。
        if self._maca_queue_depth(peer_name) == 0 and not self._maca_peer_posted(peer_name, epoch):
            return False
        taken = self._maca_take_tasks(peer_name, 32)
        try:
            self._maca_ensure_pair(peer_name, timeout)
            local_keys = [list(self._maca_task_key(item)) for item in taken]
            other_keys = self._maca_exchange_keys(peer_name, epoch, local_keys, timeout=timeout)
        except BaseException as exc:
            self._maca_requeue_front(peer_name, taken)
            # 同节点多卡的短轮询超时是正常让路，不打栈。
            if not isinstance(exc, TimeoutError) or timeout >= 120.0:
                logger.exception(f"maca nccl epoch sync failed peer={peer_name} epoch={epoch}")
                time.sleep(0.05)
            return False
        chosen, rest = self._maca_match_items(taken, other_keys)
        try:
            for item in chosen:
                self._maca_transfer_item(peer_name, item)
        except BaseException:
            logger.exception(f"maca nccl transfer failed peer={peer_name} epoch={epoch}")
            rest = chosen + rest
        self._maca_requeue_front(peer_name, rest)
        return True

    def _maca_group_turn(self, peers: List[str], epochs: Dict[str, int]) -> None:
        # 同一条通信器上一次只服务一路，并且要等这次发送结束再换下一路。
        # 两路同时在飞，或者某一路把线程堵住 120 秒，另一路的 recv 就永远对不上。
        if any(self._maca_inflight(name) for name in peers):
            return
        count = len(peers)
        start = getattr(self, "_group_rr", 0) % count
        for offset in range(count):
            name = peers[(start + offset) % count]
            epoch = epochs.get(name, 0)
            if self._maca_queue_depth(name) == 0 and not self._maca_peer_posted(name, epoch):
                continue
            if self._maca_one_peer_step(name, epoch, timeout=5.0):
                epochs[name] = epoch + 1
            self._group_rr = (start + offset + 1) % count
            return

    def _maca_pair_loop(self, peer_name: str) -> None:
        # 和 CUDA 一样：每个对端节点一条线程、一条 world=2 通信器。
        # 不和别的节点共用线程，就不会在两条通信器之间来回切。
        torch.cuda.set_device(self.tp_idx)
        epoch = 0
        while True:
            if self._maca_one_peer_step(peer_name, epoch, timeout=120.0):
                epoch += 1
            else:
                time.sleep(0.01)

    def _maca_nccl_loop(self):
        torch.cuda.set_device(self.tp_idx)
        epochs: Dict[str, int] = {}
        pair_threads: Dict[str, threading.Thread] = {}
        last_log = 0.0
        while True:
            try:
                peers, bad_peers = self._maca_matching_peers()
            except BaseException:
                logger.exception("maca nccl peer match failed")
                time.sleep(0.05)
                continue
            for peer_name, error in bad_peers:
                try:
                    self._maca_reject_peer(peer_name, error)
                except BaseException:
                    logger.exception(f"maca nccl reject peer failed peer={peer_name}")
            if not peers:
                if bad_peers:
                    time.sleep(0.05)
                    continue
                now = time.time()
                if now - last_log > 2:
                    logger.info(f"maca nccl wait peer rank={self.tp_idx} prefill={self.is_prefill_node}")
                    last_log = now
                time.sleep(0.01)
                continue
            by_node: Dict[str, List[str]] = {}
            for name in peers:
                by_node.setdefault(_maca_node_of(name), []).append(name)
            group_peers: List[str] = []
            for names in by_node.values():
                # 同一台机器上的连续几张卡仍共用一条通信器。不同节点走各自的线程。
                if self.is_prefill_node and len(names) > 1:
                    group_peers.extend(names)
                    continue
                for name in names:
                    if name in pair_threads:
                        continue
                    worker = threading.Thread(
                        target=self._maca_pair_loop,
                        args=(name,),
                        name=f"maca_nccl_{self.tp_idx}_{name[-6:]}",
                        daemon=True,
                    )
                    worker.start()
                    pair_threads[name] = worker
            if not group_peers:
                time.sleep(0.01)
                continue
            if any(self._maca_inflight(name) for name in group_peers):
                time.sleep(0.001)
                continue
            self._maca_group_turn(group_peers, epochs)
            time.sleep(0.001)


@dataclass
class _NcclXferHandle:
    peer_name: str
    event: Optional[torch.cuda.Event] = None
    status: str = "PROC"
    error_info: Optional[str] = None

    def check_status(self) -> str:
        if self.status != "PROC":
            return self.status
        # event 记在发送 stream 上，query 到完成后再回收页面。
        if self.event is None:
            return self.status

        try:
            if self.event.query():
                self.status = "DONE"
        except BaseException as e:
            self.status = "ERR"
            self.error_info = str(e)
        return self.status


class _NcclPeer:
    def __init__(self, transporter: NcclKVTransporter, peer_name: str):
        self.transporter = transporter
        self.peer_name = peer_name
        self.comm: Optional[PyNcclCommunicator] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.dst_rank = 1
        self.recv_queue: Optional["queue.Queue[Optional[PDChunckedTransTask]]"] = None
        self._lock = threading.Lock()

    def send_page(self, trans_task: PDChunckedTransTask) -> _NcclXferHandle:
        assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None
        if not is_maca():
            return self._send_page_now(trans_task)
        # 写线程只入队。真正的 ncclSend 在唯一的 maca 线程里，和 decode 的同一步对齐后执行。
        handle = _NcclXferHandle(peer_name=self.peer_name, event=None)
        self.transporter._maca_enqueue_send(self.peer_name, trans_task, handle)
        return handle

    def _send_page_now(self, trans_task: PDChunckedTransTask, handle: Optional[_NcclXferHandle] = None):
        copy_event = self.transporter.take_maca_copy_event(trans_task)
        page_tensor = self.transporter.kv_move_buffer[trans_task.src_page_index]
        stream = self._get_stream()
        self._use_stream_for_init(stream)
        if copy_event is not None:
            # 拷贝完成后再发送，但不把唯一的传输线程停在 synchronize 上。
            stream.wait_event(copy_event)
        with torch.cuda.stream(stream):
            payload = self._maca_pack_payload(page_tensor, trans_task, is_recv=False)
            scale_payload = self._scale_nccl_payload(trans_task, is_recv=False)
        comm = self._ensure_comm(is_server=True)
        comm.send(payload, dst=self.dst_rank, stream=stream)
        if scale_payload is not None:
            comm.send(scale_payload, dst=self.dst_rank, stream=stream)
        event = torch.cuda.Event()
        event.record(stream)
        if handle is None:
            handle = _NcclXferHandle(peer_name=self.peer_name, event=event)
        else:
            handle.event = event
        self.last_event = event
        if self.transporter.args.pd_detail_log:
            logger.debug(
                f"NCCL send page posted request_id={trans_task.request_id} "
                f"src_page={trans_task.src_page_index} dst_agent={self.peer_name}"
            )
        return handle

    def start_recv(self, trans_task: PDChunckedTransTask):
        trans_task = copy.copy(trans_task)
        if is_maca():
            self.transporter.arm_maca_recv(trans_task.dst_page_index)
            self.transporter._maca_enqueue_recv(self.peer_name, trans_task)
            return
        self._get_recv_queue().put(trans_task)
        return

    def close(self):
        with self._lock:
            recv_queue = self.recv_queue
            self.recv_queue = None
            comm = self.comm
            self.comm = None
            self.stream = None

        if recv_queue is not None:
            recv_queue.put(None)
        if comm is not None:
            comm.destroy()
        return

    def _use_stream_for_init(self, stream: torch.cuda.Stream):
        # 建通信器时的 warmup all_reduce 走 current_stream。沐曦上如果那是推理的
        # 默认 stream，batch 里会被 CUDA graph 堵住。
        if is_maca():
            torch.cuda.set_stream(stream)

    def _get_stream(self) -> torch.cuda.Stream:
        with self._lock:
            if self.stream is None:
                torch.cuda.set_device(self.transporter.tp_idx)
                if is_maca():
                    # 每路对端一条 stream。传输线程一轮只 synchronize 当前这一路。
                    self.stream = torch.cuda.Stream(priority=0)
                else:
                    self.stream = torch.cuda.Stream()
            return self.stream

    def _get_recv_queue(self) -> "queue.Queue[Optional[PDChunckedTransTask]]":
        with self._lock:
            if self.recv_queue is not None:
                return self.recv_queue

            self.recv_queue = queue.Queue()
            threading.Thread(target=self._recv_page_loop, args=(self.recv_queue,), daemon=True).start()
            return self.recv_queue

    def _recv_page_loop(self, recv_queue: "queue.Queue[Optional[PDChunckedTransTask]]"):
        torch.cuda.set_device(self.transporter.tp_idx)
        while True:
            trans_task = recv_queue.get()
            if trans_task is None:
                return
            self._recv_page(trans_task)

    def _recv_page(self, trans_task: PDChunckedTransTask):
        page = trans_task.dst_page_index
        generation = self.transporter.maca_recv_generation(page) if is_maca() else None
        if is_maca() and generation is None:
            return
        try:
            stream = self._get_stream()
            self._use_stream_for_init(stream)
            # 建连可能停在 ncclCommInitRank。这之前不碰页缓冲，超时还页后这里直接退出。
            comm = self._ensure_comm(is_server=False)
            if is_maca() and not self.transporter.begin_maca_recv(page, generation):
                return
            page_tensor = self.transporter.kv_move_buffer[page]
            self._kv_recv_packed = False
            self._linear_rank_packed = False
            payload = self._maca_pack_payload(page_tensor, trans_task, is_recv=True)
            kv_packed = self._kv_recv_packed
            linear_rank_packed = self._linear_rank_packed
            linear_rank = getattr(self, "_linear_rank", self.transporter.tp_idx)
            linear_world = getattr(self, "_linear_world", 0)
            self._kv_recv_packed = False
            self._linear_rank_packed = False
            comm.recv(payload, src=0, stream=stream)
            if kv_packed:
                with torch.cuda.stream(stream):
                    self._scatter_kv_pack(page_tensor, payload)
            scale_payload = self._scale_nccl_payload(trans_task, is_recv=True)
            if scale_payload is not None:
                comm.recv(scale_payload, src=0, stream=stream)
                scale_page = self.transporter.scale_move_buffer[trans_task.dst_page_index]
                if scale_payload.data_ptr() != scale_page.data_ptr():
                    with torch.cuda.stream(stream):
                        self._scatter_kv_pack(scale_page, scale_payload)
            unpack_rank = getattr(self.transporter, "maca_linear_unpack_rank", None)
            if linear_rank_packed and unpack_rank is not None:
                with torch.cuda.stream(stream):
                    unpack_rank(trans_task.dst_page_index, linear_rank, linear_world, payload)
            unpack = getattr(self.transporter, "maca_linear_unpack", None)
            if (
                is_maca()
                and trans_task.page_kind == "linear_att_state"
                and unpack is not None
                and not self._maca_full_linear_page()
            ):
                with torch.cuda.stream(stream):
                    unpack(trans_task.dst_page_index, self.transporter.tp_idx, payload)
            if is_maca():
                event = torch.cuda.Event()
                event.record(stream)
                self.last_event = event
                self.transporter.note_maca_recv(page, event, generation)
            if self.transporter.args.pd_detail_log:
                logger.debug(
                    f"NCCL recv page posted request_id={trans_task.request_id} "
                    f"dst_page={trans_task.dst_page_index}"
                )
        except BaseException as e:
            trans_task.error_info = str(e)
            logger.exception(str(e))
            self._drop_comm()
            self._note_recv_failure(trans_task, generation)
            self.transporter.send_error_info_to_prefill_node(trans_task)
        return

    def _note_recv_failure(self, trans_task: PDChunckedTransTask, generation: Optional[int] = None) -> None:
        page = trans_task.dst_page_index
        stream = self.stream
        if page is None or stream is None:
            self.transporter.disarm_maca_recv(page)
            return
        try:
            event = torch.cuda.Event()
            event.record(stream)
            self.transporter.note_maca_recv(page, event, generation)
        except BaseException:
            logger.exception(f"maca recv failure event record failed request_id={trans_task.request_id}")
            self.transporter.disarm_maca_recv(page)

    def _wait_nccl_layout(self, timeout: float = 120.0) -> dict:
        raw = _NcclControlStore(self.transporter, self.peer_name).get("nccl_layout", timeout=timeout)
        return json.loads(raw.decode())

    def _join_nccl_group(self, layout: dict) -> PyNcclCommunicator:
        raw = _NcclControlStore(self.transporter, self.peer_name).get("nccl_uid", timeout=120.0)
        unique_id = PyNcclCommunicator.unique_id_from_bytes(raw)
        stream = self._get_stream()
        self._use_stream_for_init(stream)
        comm = PyNcclCommunicator(
            None,
            self.transporter.tp_idx,
            rank=int(layout["rank"]),
            world_size=int(layout["world"]),
            unique_id=unique_id,
        )
        logger.info(
            f"maca nccl group joined peer={self.peer_name} rank={layout['rank']} world={layout['world']}"
        )
        return comm

    def _ensure_comm(self, is_server: bool, timeout: float = 120.0) -> PyNcclCommunicator:
        if self.comm is not None:
            return self.comm
        if is_maca():
            if is_server:
                peers, _bad = self.transporter._maca_matching_peers()
                same_node = [name for name in peers if _maca_node_of(name) == _maca_node_of(self.peer_name)]
                if len(same_node) > 1 and self.peer_name in same_node:
                    self.transporter._maca_init_send_group(same_node, self._get_stream())
                    return self.comm
                _NcclControlStore(self.transporter, self.peer_name).set(
                    "nccl_layout", json.dumps({"world": 2, "rank": 1}).encode()
                )
            else:
                layout = self._wait_nccl_layout(timeout)
                if int(layout["world"]) > 2:
                    self.comm = self._join_nccl_group(layout)
                    return self.comm
        with self._lock:
            if self.comm is not None:
                return self.comm

            if is_server:
                src_id = self.transporter.agent_name
                dest_id = self.peer_name
            else:
                src_id = self.peer_name
                dest_id = self.transporter.agent_name

            group = StatelessP2PProcessGroup.create(
                src_id=src_id,
                dest_id=dest_id,
                is_server=is_server,
                store=_NcclControlStore(self.transporter, self.peer_name),
            )
            self.comm = PyNcclCommunicator(group, self.transporter.tp_idx)
            logger.info(f"Created NCCL communicator with peer {self.peer_name}")
            return self.comm

    def _peer_tp(self) -> int:
        cached = getattr(self, "_cached_peer_tp", None)
        if cached is not None:
            return cached
        world = self.transporter._get_remote_metadata(self.peer_name).world_size or maca_node_world_size()
        self._cached_peer_tp = world
        return world

    def _kv_pack_span(self, page: Tensor):
        # 只看这一对的卡数。4 对 2、2 对 4 头的切分对不上，继续发整页。
        if self._peer_tp() != maca_node_world_size():
            return None
        return maca_kv_local_span(page.shape[2] // 2, maca_node_world_size(), self.transporter.tp_idx)

    def _transfer_rows(self, full_page: Tensor, trans_task: PDChunckedTransTask) -> Tensor:
        token_num = trans_task.transfer_kv_num()
        if 0 < token_num < full_page.shape[0]:
            return full_page[:token_num]
        return full_page

    def _pack_storage(self, full_page: Tensor, rows: int, local: int, storage_attr: str) -> Tensor:
        # 按整页容量分配一次。后面的页只取前缀，避免下一页换 buffer 时上一页的 NCCL 还在读。
        # KV 和 scale 各留一块，不能共用。
        cap_rows, layers, _heads, dim = full_page.shape
        need = cap_rows * layers * (2 * local) * dim
        buf = getattr(self, storage_attr, None)
        if buf is None or buf.numel() < need or buf.dtype != full_page.dtype or buf.device != full_page.device:
            buf = torch.empty(need, dtype=full_page.dtype, device=full_page.device)
            setattr(self, storage_attr, buf)
        use = rows * layers * (2 * local) * dim
        return buf[:use].view(rows, layers, 2 * local, dim)

    def _fill_kv_pack(self, page: Tensor, packed: Tensor, local: int, start: int) -> None:
        heads = page.shape[2] // 2
        packed[:, :, :local, :].copy_(page[:, :, start : start + local, :], non_blocking=True)
        packed[:, :, local:, :].copy_(page[:, :, heads + start : heads + start + local, :], non_blocking=True)

    def _scatter_kv_pack(self, page: Tensor, packed: Tensor) -> None:
        local = packed.shape[2] // 2
        heads = page.shape[2] // 2
        start = self.transporter.tp_idx * local
        dst = page[: packed.shape[0]]
        dst[:, :, start : start + local, :].copy_(packed[:, :, :local, :], non_blocking=True)
        dst[:, :, heads + start : heads + start + local, :].copy_(packed[:, :, local:, :], non_blocking=True)

    def _unequal_linear_payload(self, trans_task: PDChunckedTransTask, page_tensor: Tensor, is_recv: bool):
        pack = getattr(self.transporter, "maca_linear_pack_rank", None)
        buffer = getattr(self.transporter, "maca_linear_rank_buffer", None)
        if pack is None or buffer is None:
            return None
        if is_recv:
            rank = self.transporter.tp_idx
            world = maca_node_world_size()
        else:
            rank = int(self.transporter._get_remote_metadata(self.peer_name).device_id)
            world = self._peer_tp()
        page_index = trans_task.dst_page_index if is_recv else trans_task.src_page_index
        if is_recv:
            payload = buffer(page_index, rank, world, slot=self.peer_name)
            if payload.numel() == 0:
                return None
            self._linear_rank_packed = True
            self._linear_rank = rank
            self._linear_world = world
        else:
            payload = pack(page_index, rank, world)
            if payload.numel() == 0:
                return None
        self._log_pack_once("linear_rank", payload, page_tensor)
        return payload

    def _maca_full_linear_page(self) -> bool:
        # 只看正在通信的这一对。旧 agent 留在表里时，不会把 linear 打成整页。
        return self._peer_tp() != maca_node_world_size()

    def _scale_nccl_payload(self, trans_task: PDChunckedTransTask, is_recv: bool):
        if trans_task.page_kind != "kv":
            return None
        pages = getattr(self.transporter, "scale_move_buffer", None)
        if pages is None:
            return None
        page_index = trans_task.dst_page_index if is_recv else trans_task.src_page_index
        full = pages[page_index]
        page = self._transfer_rows(full, trans_task)
        span = self._kv_pack_span(page)
        if span is None:
            return page
        local, start = span
        packed = self._pack_storage(full, page.shape[0], local, "_scale_pack_storage")
        if not is_recv:
            self._fill_kv_pack(page, packed, local, start)
        return packed

    def _maca_pack_payload(self, page_tensor: Tensor, trans_task: PDChunckedTransTask, is_recv: bool) -> Tensor:
        if not is_maca():
            return page_tensor
        if trans_task.page_kind == "linear_att_state" and self._maca_full_linear_page():
            packed = self._unequal_linear_payload(trans_task, page_tensor, is_recv)
            if packed is not None:
                return packed
            self._log_pack_once(trans_task.page_kind, page_tensor, page_tensor)
            return page_tensor
        if trans_task.page_kind == "kv":
            self._kv_recv_packed = False
            page = self._transfer_rows(page_tensor, trans_task)
            span = self._kv_pack_span(page)
            if span is None:
                if page.shape[0] != page_tensor.shape[0]:
                    self._log_pack_once(trans_task.page_kind, page, page_tensor)
                return page
            local, start = span
            packed = self._pack_storage(page_tensor, page.shape[0], local, "_kv_pack_storage")
            if is_recv:
                self._kv_recv_packed = True
            else:
                self._fill_kv_pack(page, packed, local, start)
            self._log_pack_once(trans_task.page_kind, packed, page)
            return packed
        page_index = trans_task.dst_page_index if is_recv else trans_task.src_page_index
        if trans_task.page_kind == "linear_att_state" and is_recv:
            buffer = getattr(self.transporter, "maca_linear_buffer", None)
            if buffer is not None:
                payload = buffer(page_index, self.transporter.tp_idx)
                self._log_pack_once(trans_task.page_kind, payload, page_tensor)
                return payload
        pack = getattr(self.transporter, "maca_linear_pack", None)
        if trans_task.page_kind == "linear_att_state" and pack is not None:
            payload = pack(page_index, self.transporter.tp_idx)
            self._log_pack_once(trans_task.page_kind, payload, page_tensor)
            return payload
        return page_tensor

    def _log_pack_once(self, page_kind: str, payload: Tensor, page_tensor: Tensor):
        seen = getattr(self, "_pack_logged", None)
        if seen is None:
            seen = set()
            self._pack_logged = seen
        if page_kind in seen:
            return
        seen.add(page_kind)
        logger.info(
            f"maca nccl pack page_kind={page_kind} bytes={payload.numel() * payload.element_size()} "
            f"full_page_bytes={page_tensor.numel() * page_tensor.element_size()}"
        )

    def _drop_comm(self):
        with self._lock:
            comm = self.comm
            self.comm = None

        if comm is not None:
            comm.destroy()
            logger.warning(f"Dropped NCCL communicator with peer {self.peer_name}")
        return


class _NcclControlService(rpyc.Service):
    def __init__(self, channel: "_NcclControlChannel"):
        super().__init__()
        self.channel = channel

    def exposed_push_notif(self, payload: bytes):
        payload = obtain(payload)
        self.channel.notif_queue.put(payload)
        return

    def exposed_set_value(self, key: str, value: bytes):
        key = obtain(key)
        value = obtain(value)
        self.channel.add_store_value(key, value)
        return


class _NcclControlChannel:
    def __init__(
        self,
        host_ip: str,
        port_min: int,
        port_max: int,
    ):
        self.notif_queue: "queue.Queue[bytes]" = queue.Queue()
        self._store_values: Dict[str, bytes] = {}
        self._store_cond = threading.Condition()
        self._conn_lock = threading.Lock()
        self._conns: Dict[tuple[str, str, int], rpyc.Connection] = {}
        self._server, self.port = self._start_server(host_ip, port_min, port_max)

    def _start_server(self, host_ip: str, port_min: int, port_max: int) -> tuple[ThreadedServer, int]:
        last_error = None
        for cur_port in range(port_min, port_max + 1):
            server = None
            try:
                server = ThreadedServer(
                    _NcclControlService(self),
                    hostname=host_ip,
                    port=cur_port,
                    protocol_config={
                        "allow_pickle": True,
                        "allow_all_attrs": True,
                        "allow_getattr": True,
                        "allow_setattr": True,
                    },
                )
                # listen 必须在当前线程做完。rpyc 的 start() 把 listen 丢到后台，
                # 本机 prefill/decode 会同时 bind 同一端口，后 listen 的那个失败后把传输进程打死。
                server._listen()
                server._register()
            except OSError as e:
                last_error = e
                if server is not None:
                    try:
                        server.listener.close()
                    except Exception:
                        pass
                if e.errno == errno.EADDRINUSE:
                    logger.info(f"NCCL RPyC control port {host_ip}:{cur_port} is in use, try next port")
                else:
                    logger.warning(f"Create NCCL RPyC control channel on {host_ip}:{cur_port} failed: {e}")
                continue

            threading.Thread(target=self._accept_loop, args=(server,), daemon=True).start()
            logger.info(f"NCCL RPyC control channel listen on {host_ip}:{cur_port}")
            return server, cur_port
        raise RuntimeError(f"can not allocate NCCL control port in [{port_min}, {port_max}]") from last_error

    @staticmethod
    def _accept_loop(server: ThreadedServer) -> None:
        try:
            while server.active:
                server.accept()
        except EOFError:
            pass
        finally:
            server.close()

    def close(self):
        with self._conn_lock:
            for conn in self._conns.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()
        self._server.close()
        return

    def add_store_value(self, key: str, value: bytes):
        with self._store_cond:
            self._store_values[key] = value
            self._store_cond.notify_all()
        return

    def has_store_value(self, key: str) -> bool:
        with self._store_cond:
            return key in self._store_values

    def wait_store_value(self, key: str, timeout: float = 30.0, pop: bool = True) -> bytes:
        with self._store_cond:
            ok = self._store_cond.wait_for(lambda: key in self._store_values, timeout=timeout)
            if not ok:
                raise TimeoutError(f"wait timeout after {int(timeout * 1000)}ms, key: {key}")
            if pop:
                return self._store_values.pop(key)
            return self._store_values[key]

    def get_notifs(self) -> List[bytes]:
        notifs = []
        while True:
            try:
                notifs.append(self.notif_queue.get_nowait())
            except queue.Empty:
                break
        return notifs

    def send_notif(self, peer_name: str, host_ip: str, port: int, payload: bytes):
        self._call(peer_name, host_ip, port, "push_notif", payload)
        return

    def send_store_value(self, peer_name: str, host_ip: str, port: int, key: str, value: bytes):
        self._call(peer_name, host_ip, port, "set_value", key, value)
        return

    def _call(self, peer_name: str, host_ip: str, port: int, method: str, *args):
        conn_key = (peer_name, host_ip, port)
        with self._conn_lock:
            conn = self._conns.get(conn_key)
            if conn is None:
                conn = rpyc.connect(
                    host_ip,
                    port,
                    config={
                        "allow_pickle": True,
                        "allow_all_attrs": True,
                        "allow_getattr": True,
                        "allow_setattr": True,
                    },
                )
                self._conns[conn_key] = conn
            try:
                getattr(conn.root, method)(*args)
            except Exception as e:
                self._conns.pop(conn_key, None)
                try:
                    conn.close()
                except Exception:
                    pass
                raise RuntimeError(f"NCCL control RPC {method} to {peer_name} failed") from e
        return


class _NcclControlStore:
    def __init__(self, transporter: "NcclKVTransporter", remote_agent_name: str):
        self.transporter = transporter
        self.remote_agent_name = remote_agent_name

    def set(self, key: str, value: bytes):
        remote_metadata = self.transporter._get_remote_metadata(self.remote_agent_name)
        self.transporter.control_channel.send_store_value(
            self.remote_agent_name,
            remote_metadata.host_ip,
            remote_metadata.control_port,
            self._send_key(key),
            bytes(value),
        )
        return

    def posted(self, key: str) -> bool:
        return self.transporter.control_channel.has_store_value(self._recv_key(key))

    def get(self, key: str, timeout: float = 30.0, pop: bool = True) -> bytes:
        return self.transporter.control_channel.wait_store_value(self._recv_key(key), timeout=timeout, pop=pop)

    def _send_key(self, key: str) -> str:
        return f"{self.transporter.agent_name}->{self.remote_agent_name}:{key}"

    def _recv_key(self, key: str) -> str:
        return f"{self.remote_agent_name}->{self.transporter.agent_name}:{key}"
