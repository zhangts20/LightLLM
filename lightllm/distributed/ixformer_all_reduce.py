import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from lightllm.utils.device_utils import is_corex
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    import ixformer.distributed as ixfd

    _IXF_OK = True
except ImportError:
    ixfd = None
    _IXF_OK = False

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class IxformerAllReduce:

    def __init__(self, group: ProcessGroup, device) -> None:
        self.disabled = True
        if not _IXF_OK or not is_corex():
            return
        self.group = group
        self.device = device if isinstance(device, torch.device) else torch.device("cuda", device)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size == 1:
            return
        try:
            # Fail fast at startup instead of on the first all_reduce. The
            # communicator is cached per group, so later calls reuse it.
            ixfd.init_comm_with_store(group)
        except Exception as e:
            logger.warning("IxformerAllReduce init failed: %s. Falling back to PyNccl.", e)
            return
        self.disabled = False
        logger.info("IxformerAllReduce enabled: world_size=%d, device=%s", self.world_size, self.device)

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled or not inp.is_cuda or not inp.is_contiguous():
            return False
        if inp.dtype not in _SUPPORTED_DTYPES:
            return False
        return inp.device == self.device

    def all_reduce(self, inp: torch.Tensor) -> None:
        ixfd.all_reduce(inp, group=self.group, async_op=True)
