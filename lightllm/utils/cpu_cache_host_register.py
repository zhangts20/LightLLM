import os
import ctypes
import ctypes.util
import threading
import re
import torch
from typing import Callable, Dict
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.platform import get_backend
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# 单次注册段长，单位 GB，默认 16。MACA 3.1.0.14 还会要求空闲显存大于这段长度，
# 各进程在第一次 mcHostRegister 前把这道比较跳过。主机 MemFree * 0.8 的检查保留。
MACA_HOST_REGISTER_LIMIT_BYTES = int(os.getenv("MACA_HOST_REGISTER_LIMIT_GB", 16)) * 1024 * 1024 * 1024

# cmp rbx, r14; jae fail; cmp [rsp+0x20], rbx; ja ok
_MACA_GPU_FREE_CHECK = bytes.fromhex("4c39f3730b48395c24200f87")
_MACA_GPU_FREE_PATCHED = bytes.fromhex("4c39f3730b48395c2420e9")
_maca_patch_lock = threading.Lock()


def patch_mc_host_register_gpu_free_check(soname: str = "libmcruntime.so") -> int:
    """跳过本进程 mcHostRegister 里「空闲显存必须大于注册长度」的比较。"""
    with _maca_patch_lock:
        regions = []
        with open("/proc/self/maps") as f:
            for line in f:
                if soname not in line:
                    continue
                span, perm = line.split()[:2]
                if "x" not in perm:
                    continue
                start, end = (int(x, 16) for x in span.split("-"))
                regions.append((start, end))
        if not regions:
            raise RuntimeError(f"{soname} is not loaded")

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        for start, end in regions:
            size = end - start
            blob = bytes((ctypes.c_char * size).from_address(start))
            done = blob.find(_MACA_GPU_FREE_PATCHED)
            if done != -1:
                return start + done

            hit = blob.find(_MACA_GPU_FREE_CHECK)
            if hit == -1:
                continue

            if blob.find(_MACA_GPU_FREE_CHECK, hit + 1) != -1:
                raise RuntimeError("mcHostRegister gpu-free check matched more than once")

            ja = start + hit + len(_MACA_GPU_FREE_CHECK) - 2
            rel = int.from_bytes((ctypes.c_char * 4).from_address(ja + 2), "little", signed=True)
            # ja 是 6 字节，jmp 是 5 字节，目标不变，位移加 1，末尾补 nop
            new = b"\xe9" + (rel + 1).to_bytes(4, "little", signed=True) + b"\x90"
            page = ja & ~0xFFF
            rc = libc.mprotect(ctypes.c_void_p(page), 0x2000, 7)
            if rc != 0:
                raise OSError(ctypes.get_errno(), "mprotect")
            ctypes.memmove(ja, new, 6)
            logger.info("patched mcHostRegister gpu-free check at %#x pid=%s", ja, os.getpid())
            return ja

        raise RuntimeError("mcHostRegister gpu-free check not found; MACA version differs")


class HostRegisterOps:
    def registration_chunk_bytes(self, size: int) -> int:
        # 在 CUDA 中分段注册后的设备地址仍然连续
        return 128 * 1024 * 1024

    def host_register_mappings_are_contiguous(self) -> bool:
        return True

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
            logger.info(f"cudaHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}")
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
            root = os.getenv("MACA_PATH", "/opt/maca")
            release = re.match(r"(\d+)\.(\d+)", str(getattr(torch.version, "maca", "")))
            self.whole_region = release is not None and tuple(map(int, release.groups())) >= (3, 8)
            compatibility_library = os.path.join(root, "lib64/libsymbol_cu.so")
            if not os.path.isfile(compatibility_library):
                compatibility_library = os.path.join(root, "lib/libsymbol_cu.so")
            if self.whole_region and os.path.isfile(compatibility_library):
                mc = ctypes.CDLL(compatibility_library)
                prefix = "cuda"
            else:
                mc = ctypes.CDLL(os.path.join(root, "lib/libmcruntime.so"))
                prefix = "mc"
            self._register = getattr(mc, prefix + "HostRegister")
            self._register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            self._register.restype = ctypes.c_int
            self._get_pointer = getattr(mc, prefix + "HostGetDevicePointer")
            self._get_pointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
            self._get_pointer.restype = ctypes.c_int
            self.mc = mc
            self.flag = 3
            self.device_id = get_current_device_id()
            torch.cuda.set_device(self.device_id)
            if not self.whole_region:
                patch_mc_host_register_gpu_free_check()

        def registration_chunk_bytes(self, size: int) -> int:
            # MACA 3.8 maps a whole registration; splitting it can alias windows.
            if self.whole_region:
                return size
            return min(size, MACA_HOST_REGISTER_LIMIT_BYTES)

        def host_register_mappings_are_contiguous(self) -> bool:
            return self.whole_region

        def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
            torch.cuda.set_device(self.device_id)
            ptr = ctypes.c_void_p(shm_ptr + offset)
            r = self._register(ptr, ctypes.c_size_t(seg_len), self.flag)
            if r != 0:
                raise Exception(f"mcHostRegister failed with error code {r}, prefer to use hugetlb")

        def get_device_ptr(self, shm_ptr: int) -> int:
            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = self._get_pointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"mcHostGetDevicePointer failed with error code {res}")
            logger.info(f"mcHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}")
            return device_ptr.value

    return MacaHostRegisterOps()


@register_host_register_worker("musa")
def _musa_ops() -> HostRegisterOps:
    class MusaHostRegisterOps(HostRegisterOps):
        def __init__(self):
            import torch_musa  # noqa: F401

            musa_home = os.getenv("MUSA_HOME", "/usr/local/musa")
            musa = ctypes.CDLL(os.path.join(musa_home, "lib", "libmusart.so"))
            musa.musaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            musa.musaHostRegister.restype = ctypes.c_int
            musa.musaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
            musa.musaHostGetDevicePointer.restype = ctypes.c_int
            self.musa = musa
            # /usr/local/musa-5.2/include/driver_types.h: musaHostRegisterPortable | musaHostRegisterMapped
            self.flag = 0x01 | 0x02
            self.device_id = get_current_device_id()
            torch.musa.set_device(self.device_id)

        def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
            torch.musa.set_device(self.device_id)
            ptr = ctypes.c_void_p(shm_ptr + offset)
            r = self.musa.musaHostRegister(ptr, ctypes.c_size_t(seg_len), self.flag)
            if r != 0:
                raise Exception(f"musaHostRegister failed with error code {r}, prefer to use hugetlb")

        def get_device_ptr(self, shm_ptr: int) -> int:
            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = self.musa.musaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"musaHostGetDevicePointer failed with error code {res}")
            logger.info(f"musaHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}")
            return device_ptr.value

    return MusaHostRegisterOps()


@register_host_register_worker("mlu")
def _mlu_ops() -> HostRegisterOps:
    class _CnrtPointerAttributes(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("device", ctypes.c_int),
            ("size", ctypes.c_size_t),
            ("devicePointer", ctypes.c_void_p),
            ("hostPointer", ctypes.c_void_p),
            ("cacheMode", ctypes.c_int),
            ("deviceBasePointer", ctypes.c_void_p),
        ]

    class MluHostRegisterOps(HostRegisterOps):
        def __init__(self):
            import torch_mlu  # noqa: F401

            neuware_home = os.getenv("NEUWARE_HOME", "/usr/local/neuware")
            cnrt = ctypes.CDLL(os.path.join(neuware_home, "lib64", "libcnrt.so"))
            cnrt.cnrtHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            cnrt.cnrtHostRegister.restype = ctypes.c_int
            cnrt.cnrtPointerGetAttributes.argtypes = [
                ctypes.POINTER(_CnrtPointerAttributes),
                ctypes.c_void_p,
            ]
            cnrt.cnrtPointerGetAttributes.restype = ctypes.c_int
            self.cnrt = cnrt
            # CNRT: DeviceMap=0x1, Portable=0x2. MLU590 reports DeviceMap unsupported
            # (attr 0x315=0); Portable pin works. CUDA's portable bit is the opposite 0x1.
            self.flag = 0x2
            self.device_id = get_current_device_id()
            self._register_supported = True
            torch.mlu.set_device(self.device_id)

        def register_segment(self, shm_ptr: int, offset: int, seg_len: int) -> None:
            if not self._register_supported:
                return
            torch.mlu.set_device(self.device_id)
            ptr = ctypes.c_void_p(shm_ptr + offset)
            r = self.cnrt.cnrtHostRegister(ptr, ctypes.c_size_t(seg_len), self.flag)
            if r == 100050:
                self._register_supported = False
                logger.warning("cnrtHostRegister is not supported, skip pinning cpu cache on mlu")
                return
            if r != 0:
                raise Exception(f"cnrtHostRegister failed with error code {r}, prefer to use hugetlb")

        def get_device_ptr(self, shm_ptr: int) -> int:
            if not self._register_supported:
                return shm_ptr
            attr = _CnrtPointerAttributes()
            res = self.cnrt.cnrtPointerGetAttributes(ctypes.byref(attr), ctypes.c_void_p(shm_ptr))
            if res != 0 or not attr.devicePointer:
                return shm_ptr
            logger.info(f"cnrtPointerGetAttributes success, host_ptr={shm_ptr}, device_ptr={attr.devicePointer}")
            return attr.devicePointer

    return MluHostRegisterOps()
