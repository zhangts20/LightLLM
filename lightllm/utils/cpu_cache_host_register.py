import os
import ctypes
import torch
from typing import Callable, Dict
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.platform import get_backend
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class HostRegisterOps:

    def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
        raise NotImplementedError

    def get_device_ptr(self, shm_ptr: int) -> int:
        raise NotImplementedError


HostRegisterFactory = Callable[[], HostRegisterOps]

_REGISTRY: Dict[str, HostRegisterFactory] = {}


def register_host_register_worker(backend_name: str):
    def decorator(fn: HostRegisterFactory) -> HostRegisterFactory:
        if backend_name in _REGISTRY:
            raise ValueError(f"HostRegisterWorker {backend_name} already registered!")
        _REGISTRY[backend_name] = fn
        return fn

    return decorator


def get_host_register_worker() -> HostRegisterOps:
    backend_name = get_backend().name
    try:
        return _REGISTRY[backend_name]()
    except KeyError:
        raise RuntimeError(f"platform {backend_name} is not registered!")


@register_host_register_worker("cuda")
def _cuda_ops() -> HostRegisterOps:

    class CudaHostRegisterOps(HostRegisterOps):

        def __init__(self):
            cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
            cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            cuda.cudaHostRegister.restype = ctypes.c_int
            cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
            cuda.cudaHostGetDevicePointer.restype = ctypes.c_int
            self.cuda = cuda
            self.flag = 3
            self.device_id = get_current_device_id()
            torch.cuda.set_device(self.device_id)

        def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
            torch.cuda.set_device(self.device_id)
            ptr = ctypes.c_void_p(shm_ptr + offset)
            r = self.cuda.cudaHostRegister(ptr, ctypes.c_size_t(seg_len), self.flag)
            if r != 0:
                raise Exception(f"cudaHostRegister failed with error code {r}, prefer to use hugetlb")

        def get_device_ptr(self, shm_ptr: int) -> int:
            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = self.cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"cudaHostGetDevicePointer failed with error code {res}")
            logger.info(
                f"cudaHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}"
            )
            return device_ptr.value

    return CudaHostRegisterOps()


@register_host_register_worker("ascend")
def _npu_ops() -> HostRegisterOps:

    class AscendHostRegisterOps(HostRegisterOps):

        def __init__(self):
            import acl

            self.acl = acl
            acl.init()
            ret = acl.rt.set_device(get_current_device_id())
            assert ret == 0, f"acl.rt.set_device failed with error code {ret}"
            self.flag = 0  # ACL_HOST_REGISTER_MAPPED

        def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
            res = self.acl.rt.host_register(shm_ptr + offset, seg_len, self.flag)
            assert res[1] == 0, f"acl.rt.host_register failed with error code {res}"

        def get_device_ptr(self, shm_ptr: int) -> int:
            # Mapped host ptr is used directly as the tensor storage address.
            return shm_ptr

    return AscendHostRegisterOps()


@register_host_register_worker("maca")
def _metax_ops() -> HostRegisterOps:

    class MacaHostRegisterOps(HostRegisterOps):

        def __init__(self):
            mc = ctypes.CDLL(os.path.join(os.getenv("MACA_PATH", "/opt/maca"), "lib/libmcruntime.so"))
            mc.mcHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            mc.mcHostRegister.restype = ctypes.c_int
            mc.mcHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
            mc.mcHostGetDevicePointer.restype = ctypes.c_int
            self.mc = mc
            self.flag = 3
            self.device_id = get_current_device_id()
            torch.cuda.set_device(self.device_id)

        def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
            torch.cuda.set_device(self.device_id)
            ptr = ctypes.c_void_p(shm_ptr + offset)
            r = self.mc.mcHostRegister(ptr, ctypes.c_size_t(seg_len), self.flag)
            if r != 0:
                raise Exception(f"mcHostRegister failed with error code {r}, prefer to use hugetlb")

        def get_device_ptr(self, shm_ptr: int) -> int:
            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = self.mc.mcHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"mcHostGetDevicePointer failed with error code {res}")
            logger.info(
                f"mcHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}"
            )
            return device_ptr.value

    return MacaHostRegisterOps()
