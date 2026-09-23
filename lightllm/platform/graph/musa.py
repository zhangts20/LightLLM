import torch
from typing import Any, ContextManager, Optional

from lightllm.platform.base.graph import BackendGraph


class MusaGraphBackend(BackendGraph):

    def __init__(self) -> None:
        import torch_musa  # noqa: F401

        self._musa = torch.musa

    def create_graph(self) -> Any:
        return self._musa.MUSAGraph()

    def graph(self, graph_obj: Any, pool: Optional[Any] = None, stream: Optional[Any] = None) -> ContextManager:
        return self._musa.graph(graph_obj, pool=pool, stream=stream)

    def graph_pool_handle(self) -> Any:
        return self._musa.graph_pool_handle()

    def is_capturing(self) -> bool:
        return self._musa.is_current_stream_capturing()
