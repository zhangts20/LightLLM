"""多进程 PD master 共享前缀树和 prefill 在途负载。单进程时不连接。"""

import os
from threading import Lock

import rpyc
from rpyc.utils.classic import obtain

from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_pd_master_unix_socket_address

logger = init_logger(__name__)

_client = None
_client_checked = False


def shared_socket_address() -> bytes:
    return get_pd_master_unix_socket_address("pd_master_shared")


def shared_client():
    """只在多 worker 进程里连接共享状态。单进程返回 None。"""
    global _client, _client_checked
    if os.environ.get("LIGHTLLM_PD_MASTER_WORKER_INDEX") is None:
        return None
    if _client_checked:
        return _client
    _client_checked = True
    _client = rpyc.connect(socket_path=shared_socket_address(), config={"allow_pickle": True})
    return _client


@rpyc.service
class PDMasterSharedState(rpyc.Service):
    def __init__(self):
        from .pd_selector.prompt_cache_tree import PromptCacheTree

        self.tree = PromptCacheTree()
        self.loads = {}
        self.lock = Lock()

    @rpyc.exposed
    def prefix_match(self, text: str):
        text = obtain(text)
        result = self.tree.prefix_match(text)
        return result.prefill_node, result.matched_char_count, result.input_char_count, result.last_insert_time

    @rpyc.exposed
    def insert(self, text: str, prefill_node: str) -> None:
        text, prefill_node = obtain((text, prefill_node))
        self.tree.insert(text, prefill_node)

    @rpyc.exposed
    def add_load(self, client_ip_port: str, chars: int, req_num: int) -> None:
        client_ip_port, chars, req_num = obtain((client_ip_port, int(chars), int(req_num)))
        with self.lock:
            prompt_chars, request_num = self.loads.get(client_ip_port, (0, 0))
            self.loads[client_ip_port] = (max(0, prompt_chars + chars), max(0, request_num + req_num))

    @rpyc.exposed
    def get_loads(self, client_ip_ports):
        client_ip_ports = obtain(client_ip_ports)
        with self.lock:
            return tuple(self.loads.get(client_ip_port, (0, 0)) for client_ip_port in client_ip_ports)


def start_pd_master_shared_state(args, pipe_writer) -> None:
    import setproctitle
    from rpyc.utils.server import ThreadedServer

    from lightllm.utils.envs_utils import get_unique_server_name
    from lightllm.utils.graceful_utils import graceful_registry
    from lightllm.utils.process_check import start_parent_check_thread

    start_parent_check_thread()
    graceful_registry("start_pd_master_shared_state")
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::pd_master_shared")
    server = ThreadedServer(
        PDMasterSharedState(),
        socket_path=shared_socket_address(),
        protocol_config={"allow_pickle": True},
    )
    pipe_writer.send("init ok")
    server.start()
