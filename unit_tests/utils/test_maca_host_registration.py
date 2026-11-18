from types import SimpleNamespace

import pytest

from lightllm.utils import cpu_cache_host_register as register


@pytest.mark.parametrize("version,whole_region", [("3.8.2.2", True), ("3.1.0.14", False)])
def test_maca_registration_selects_version_compatible_mapping(monkeypatch, version, whole_region):
    calls = []

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append(self.name)
            if self.name.endswith("GetDevicePointer"):
                args[0]._obj.value = 4096
            return 0

    library = SimpleNamespace(
        **{
            name: Function(name)
            for name in ("cudaHostRegister", "cudaHostGetDevicePointer", "mcHostRegister", "mcHostGetDevicePointer")
        }
    )
    monkeypatch.setattr(register.torch.version, "maca", version, raising=False)
    monkeypatch.setattr(register.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(register.os.path, "isfile", lambda path: path.endswith("libsymbol_cu.so"))
    monkeypatch.setattr(register, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(register.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(register, "patch_mc_host_register_gpu_free_check", lambda: calls.append("legacy_patch"))
    operations = register._metax_ops()
    size = 32 * 1024 ** 3
    assert operations.host_register_mappings_are_contiguous() is whole_region
    assert operations.registration_chunk_bytes(size) == (
        size if whole_region else register.MACA_HOST_REGISTER_LIMIT_BYTES
    )
    operations.register_segment(4096, 0, 1024)
    assert operations.get_device_ptr(4096) == 4096
    prefix = "cuda" if whole_region else "mc"
    assert prefix + "HostRegister" in calls
    assert ("legacy_patch" in calls) is not whole_region


@pytest.mark.parametrize("runtime_version,needs_patch", [(3080, False), (3100, False), (3010, True), (None, True)])
def test_maca_legacy_mapping_uses_runtime_version_before_patching(monkeypatch, runtime_version, needs_patch):
    calls = []

    class Function:
        def __call__(self, *args):
            return 0

    library = SimpleNamespace(mcHostRegister=Function(), mcHostGetDevicePointer=Function())
    # Keep the legacy registration path, even if Torch's release label differs
    # from the loaded runtime. A 3.8+ runtime must never receive the old patch.
    monkeypatch.setattr(register.torch.version, "maca", "3.1.0.14", raising=False)
    monkeypatch.setattr(register.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(register.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(register, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(register.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(register, "get_maca_runtime_version", lambda mc: runtime_version)
    monkeypatch.setattr(register, "patch_mc_host_register_gpu_free_check", lambda: calls.append("legacy_patch"))

    operations = register._metax_ops()

    assert operations.host_register_mappings_are_contiguous() is False
    assert operations.registration_chunk_bytes(32 * 1024 ** 3) == register.MACA_HOST_REGISTER_LIMIT_BYTES
    assert ("legacy_patch" in calls) is needs_patch
