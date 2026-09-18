from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def pin_mem_test_runtime(monkeypatch):
    """Use the test tensor device without configuring a serving process."""
    from lightllm.server.router.model_infer import pin_mem_manager

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    runtime = SimpleNamespace(target_device=lambda: device)
    monkeypatch.setattr(pin_mem_manager, "get_backend", lambda: SimpleNamespace(runtime=runtime))
    return runtime
