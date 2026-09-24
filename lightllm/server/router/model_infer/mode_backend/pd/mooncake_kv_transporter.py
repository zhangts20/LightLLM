import copy
import os
import pickle
import queue
import threading
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from mooncake.engine import TransferEngine
except ImportError:
    raise ImportError("Mooncake is not installed. Please install it using `pip install mooncake-transfer-engine`")

import rpyc
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer

from lightllm.server.pd_io_struct import PDChunckedTransTask, PDAgentMetadata
from lightllm.utils.net_utils import get_hostname_ip


@dataclass
class MooncakeAgentMetadata:
    agent_name: str
    session_id: str
    host_ip: str
    control_port: int


class MooncakeKVTransporter:

    def __init__(
        self,
        node_id: int,
        tp_idx: int,
        kv_move_buffer: torch.Tensor,
        host_ip: Optional[str] = None,
        protocol: str = "rdma",
        control_port_min: int = 30000,
        control_port_max: int = 40000,
    ):
        self.node_id = node_id
        self.tp_idx = tp_idx
        self.host_ip = host_ip or get_hostname_ip()
        assert self.host_ip is not None, "Can not get host ip for MooncakeKVTransporter"
        self.num_pages, self.page_size, _, _, _ = kv_move_buffer.shape

        self.engine = TransferEngine()

        # https://kvcache-ai.github.io/Mooncake/getting_started/supported-protocols.html
        protocol = os.environ.get("MOONCAKE_INITIALIZE_PROTOCOL", protocol)
        # 预设值空自动发现或指定设备
        device_name = os.environ.get("MOONCAKE_INITIALIZE_DEVCIES", "auto-discovery")
        ret = self.engine.initialize(self.host_ip, "P2PHANDSHAKE", protocol, device_name)
        if ret != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake engine: {ret}, "
                f"protocol={protocol}, device_name={device_name}"
            )

        # The session id is used to identify the session between the two nodes
        self.session_id = f"{self.host_ip}:{self.engine.get_rpc_port()}"

        self.buffer_ptr = kv_move_buffer.data_ptr()
        self.buffer_len = kv_move_buffer.nbytes
        self.page_len = self.buffer_len // self.num_pages

        # Single buffer here; use batch_register_memory if registering multiple regions at once
        ret = self.engine.register_memory(self.buffer_ptr, self.buffer_len)
        if ret != 0:
            raise RuntimeError(f"Failed to register Mooncake memory: {ret}")

        self.control_channel = _MooncakeControlChannel(
            host_ip=self.host_ip,
            port_min=control_port_min,
            port_max=control_port_max,
        )

        self.capture_telemetry = False
        self.remote_agents: Dict[str, PDAgentMetadata] = {}

    @property
    def agent_name(self) -> str:
        return f"{self.node_id}_{self.tp_idx}"

    @property
    def agent_metadata(self) -> bytes:
        return pickle.dumps(
            MooncakeAgentMetadata(
                agent_name=self.agent_name,
                session_id=self.session_id,
                host_ip=self.host_ip,
                control_port=self.control_channel.port,
            )
        )

    @property
    def local_page_mem_desc(self) -> bytes:
        return pickle.dumps(
            {
                "ptr": self.buffer_ptr,
                "len": self.buffer_len,
                "num_pages": self.num_pages,
                "page_len": self.page_len,
            }
        )

    def get_new_notifs(self) -> Dict[str, List[bytes]]:
        notifs: Dict[str, List[bytes]] = {}
        for notify in self.control_channel.get_notifs():
            notifs.setdefault(self._get_notify_source_agent_name(notify), []).append(notify)
        return notifs

    def _get_notify_source_agent_name(self, notify: bytes) -> str:
        notify_obj = pickle.loads(notify)
        assert isinstance(notify_obj, PDChunckedTransTask), type(notify_obj)

        if notify_obj.error_info is not None:
            if notify_obj.decode_agent_name and notify_obj.decode_agent_name != self.agent_name:
                return notify_obj.decode_agent_name
            assert notify_obj.prefill_agent_name is not None
            return notify_obj.prefill_agent_name

        if notify_obj.write_stage == "request":
            assert notify_obj.prefill_agent_name is not None
            return notify_obj.prefill_agent_name

        if notify_obj.write_stage == "ready":
            assert notify_obj.decode_agent_name is not None
            return notify_obj.decode_agent_name

        if notify_obj.write_stage == "done":
            assert notify_obj.prefill_agent_name is not None
            return notify_obj.prefill_agent_name

        raise AssertionError(f"unexpected notify stage: {notify_obj.write_stage}")

    def connect_add_remote_agent(self, remote_agent: PDAgentMetadata):
        if remote_agent.agent_name in self.remote_agents:
            return

        metadata: MooncakeAgentMetadata = pickle.loads(remote_agent.agent_metadata)
        assert metadata.agent_name == remote_agent.agent_name

        self.remote_agents[remote_agent.agent_name] = remote_agent

    def _ensure_remote_agent(self, remote_agent_name: str, trans_task: PDChunckedTransTask):
        if remote_agent_name not in self.remote_agents:
            if remote_agent_name == trans_task.decode_agent_name:
                self.connect_add_remote_agent(trans_task.create_decode_agent_obj())
            else:
                self.connect_add_remote_agent(trans_task.create_prefill_agent_obj())

    def _get_remote_metadata(self, remote_agent_name: str) -> MooncakeAgentMetadata:
        remote_agent = self.remote_agents[remote_agent_name]
        return pickle.loads(remote_agent.agent_metadata)

    def _send_task_notif(self, remote_agent_name: str, trans_task: PDChunckedTransTask):
        self._ensure_remote_agent(remote_agent_name, trans_task)
        meta = self._get_remote_metadata(remote_agent_name)
        self.control_channel.send_notif(
            remote_agent_name,
            meta.host_ip,
            meta.control_port,
            pickle.dumps(trans_task),
        )

    def send_write_request_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        self._ensure_remote_agent(trans_task.decode_agent_name, trans_task)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "request"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc

        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)

    def send_write_ready_task_to_prefill_node(self, trans_task: PDChunckedTransTask):
        self._ensure_remote_agent(trans_task.prefill_agent_name, trans_task)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "ready"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc

        self._send_task_notif(trans_task.prefill_agent_name, new_trans_task)

    def send_write_done_task_to_decode_node(self, trans_task: PDChunckedTransTask):
        self._ensure_remote_agent(trans_task.decode_agent_name, trans_task)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "done"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc

        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)

    def send_error_info_to_prefill_node(self, trans_task: PDChunckedTransTask):
        if trans_task.prefill_agent_name is None:
            return
        self._ensure_remote_agent(trans_task.prefill_agent_name, trans_task)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "error"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.decode_agent_name = self.agent_name
        new_trans_task.decode_agent_metadata = self.agent_metadata
        new_trans_task.decode_num_pages = self.num_pages
        new_trans_task.decode_page_reg_desc = self.local_page_mem_desc

        self._send_task_notif(trans_task.prefill_agent_name, new_trans_task)

    def send_error_info_to_decode_node(self, trans_task: PDChunckedTransTask):
        self._ensure_remote_agent(trans_task.decode_agent_name, trans_task)

        new_trans_task: PDChunckedTransTask = copy.copy(trans_task)
        new_trans_task.write_stage = "error"
        new_trans_task.mem_indexes = None
        new_trans_task.xfer_handle = None
        new_trans_task.prefill_agent_name = self.agent_name
        new_trans_task.prefill_agent_metadata = self.agent_metadata
        new_trans_task.prefill_num_pages = self.num_pages
        new_trans_task.prefill_page_reg_desc = self.local_page_mem_desc

        self._send_task_notif(trans_task.decode_agent_name, new_trans_task)

    def write_blocks_paged(self, trans_task: PDChunckedTransTask) -> int:
        self._ensure_remote_agent(trans_task.decode_agent_name, trans_task)
        assert trans_task.src_page_index is not None and trans_task.dst_page_index is not None

        decode_meta: MooncakeAgentMetadata = pickle.loads(trans_task.decode_agent_metadata)
        dst_desc = pickle.loads(trans_task.decode_page_reg_desc)

        src_addr = self.buffer_ptr + trans_task.src_page_index * self.page_len
        dst_addr = dst_desc["ptr"] + trans_task.dst_page_index * dst_desc["page_len"]
        assert dst_desc["page_len"] == self.page_len

        # https://kvcache-ai.github.io/Mooncake/api-reference/python/transfer-engine.html#transfer-submit-write
        # Submit async write; check status in check_task_status().
        batch_id = self.engine.transfer_submit_write(
            decode_meta.session_id,
            src_addr,
            dst_addr,
            self.page_len,
        )

        # Batch ID for tracking this submit, or 0 on failure.
        # Older versions may also return other failure values, e.g. C++ uint64_t(-1) as (1<<64)-1 in Python.
        if batch_id in (0, (1 << 64) - 1) or (isinstance(batch_id, int) and batch_id < 0):
            raise RuntimeError(f"Failed to submit async write to decode node: {batch_id}")

        return batch_id

    def check_task_status(self, trans_task: PDChunckedTransTask) -> str:
        assert trans_task.xfer_handle is not None
        # https://kvcache-ai.github.io/Mooncake/api-reference/python/transfer-engine.html#transfer-check-status
        status = self.engine.transfer_check_status(int(trans_task.xfer_handle))

        # 1: Transfer completed successfully
        if status == 1:
            return "DONE"
        # -1: Transfer failed, -2: Transfer timed out
        if status in (-1, -2):
            return "ERR"
        # 0: Transfer still in progress
        return "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        return

    def remove_remote_agent(self, peer_name: str) -> None:
        if peer_name in self.remote_agents:
            self.remote_agents.pop(peer_name)
        else:
            pass

    def shutdown(self) -> None:
        self.remote_agents.clear()
        try:
            self.engine.unregister_memory(self.buffer_ptr, self.buffer_len)
        except Exception:
            pass

        self.control_channel.close()


class _MooncakeControlService(rpyc.Service):

    def __init__(self, channel: "_MooncakeControlChannel") -> None:
        super().__init__()

        self.channel = channel

    def exposed_push_notif(self, payload: bytes) -> None:
        payload = obtain(payload)
        self.channel.notif_queue.put(payload)


class _MooncakeControlChannel:

    def __init__(self, host_ip: str, port_min: int, port_max: int) -> None:
        self.notif_queue: "queue.Queue[bytes]" = queue.Queue()
        self._conn_lock = threading.Lock()
        self._conns: Dict[Tuple[str, str, int], rpyc.Connection] = {}
        self._server, self.port = self._start_server(host_ip, port_min, port_max)

    def _start_server(self, host_ip: str, port_min: int, port_max: int) -> Tuple[ThreadedServer, int]:
        last_error = None
        for cur_port in range(port_min, port_max + 1):
            try:
                server = ThreadedServer(
                    _MooncakeControlService(self),
                    hostname=host_ip,
                    port=cur_port,
                    protocol_config={
                        "allow_pickle": True,
                        "allow_all_attrs": True,
                        "allow_getattr": True,
                        "allow_setattr": True,
                    },
                )
                threading.Thread(target=server.start, daemon=True).start()
                return server, cur_port
            except OSError as e:
                last_error = e
        raise RuntimeError(
            f"can not allocate Mooncake control port in [{port_min}, {port_max}]"
        ) from last_error

    def close(self) -> None:
        with self._conn_lock:
            for conn in self._conns.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()
        self._server.close()

    def get_notifs(self) -> List[bytes]:
        notifs = []
        while True:
            try:
                notifs.append(self.notif_queue.get_nowait())
            except queue.Empty:
                break
        return notifs

    def send_notif(self, peer_name: str, host_ip: str, port: int, payload: bytes) -> None:
        self._call(peer_name, host_ip, port, "push_notif", payload)

    def _call(self, peer_name: str, host_ip: str, port: int, method: str, *args) -> None:
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
                raise RuntimeError(f"Mooncake control RPC {method} to {peer_name} failed") from e
