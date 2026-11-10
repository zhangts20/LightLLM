import ctypes
import os
from functools import lru_cache
from typing import Optional

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# mcRuntimeGetVersion encoding used by MetaX: 3080 == MACA 3.8.x
MACA_RUNTIME_VERSION_3_8 = 3080


def get_maca_runtime_version(mc=None) -> Optional[int]:
    try:
        if mc is None:
            lib = os.path.join(os.getenv("MACA_PATH", "/opt/maca"), "lib", "libmcruntime.so")
            mc = ctypes.CDLL(lib)
        version = ctypes.c_int()
        fn = mc.mcRuntimeGetVersion
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        if fn(ctypes.byref(version)) == 0:
            return version.value
    except Exception as e:
        logger.warning("failed to get MACA runtime version: %s", e)
    return None


@lru_cache(maxsize=None)
def maca_w8a8_need_partial_m_split() -> bool:
    if os.getenv("LIGHTLLM_MACA_W8A8_FORCE_PARTIAL_M_SPLIT", "False").upper() in ["ON", "TRUE", "1"]:
        logger.info("force maca w8a8 partial-M split by env")
        return True

    version = get_maca_runtime_version()
    if version is not None and version >= MACA_RUNTIME_VERSION_3_8:
        logger.info("skip maca w8a8 partial-M split for MACA runtime version %s", version)
        return False
    return True
