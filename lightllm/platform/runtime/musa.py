import torch
from typing import Any, ContextManager, Optional, Tuple, Union

from lightllm.platform.base.runtime import BackendRuntime


class MusaRuntime(BackendRuntime):

    def __init__(self) -> None:
        import torch_musa  # noqa: F401

        self._musa = torch.musa

    @property
    def device_type(self) -> str:
        return "musa"

    @property
    def dist_backend(self) -> str:
        return "mccl"

    @property
    def dist_init_passes_device_id(self) -> bool:
        return False

    def mem_get_info(self, device: Union[int, torch.device]) -> Tuple[int, int]:
        return self._musa.mem_get_info(device)

    def get_device_properties(self, device: Union[int, torch.device]) -> Any:
        return self._musa.get_device_properties(device)

    def device_count(self) -> int:
        return self._musa.device_count()

    def is_available(self) -> bool:
        return self._musa.is_available()

    def current_device(self) -> int:
        return self._musa.current_device()

    def get_device_name(self, device_id: Optional[int] = None) -> str:
        device_id = device_id if device_id is not None else self.current_device()
        return self._musa.get_device_name(device_id)

    def set_device(self, device: Union[int, str, torch.device]) -> None:
        self._musa.set_device(self._parse(device))

    def create_stream(self, **kwargs) -> Any:
        return self._musa.Stream(**kwargs)

    def stream(self, stream: Optional[Any] = None) -> ContextManager:
        return self._musa.stream(stream)

    def current_stream(self, device_id: Optional[int] = None) -> Any:
        device_id = device_id if device_id is not None else self.current_device()
        return self._musa.current_stream(device_id)

    def create_event(self, **kwargs) -> torch.Event:
        return self._musa.Event(**kwargs)

    def synchronize(self) -> None:
        self._musa.synchronize()

    def empty_cache(self) -> None:
        self._musa.empty_cache()

    def manual_seed_all(self, seed: int) -> None:
        self._musa.manual_seed_all(seed)
