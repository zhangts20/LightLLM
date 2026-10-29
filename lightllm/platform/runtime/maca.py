import torch

from lightllm.platform.runtime.cuda import CudaRuntime


class MacaRuntime(CudaRuntime):
    def copy_tensors(self, destinations: list[torch.Tensor], sources: list[torch.Tensor]) -> None:
        """Batch graph-input copies to reduce MACA host launch overhead."""
        if len(destinations) != len(sources):
            raise ValueError("Source and destination tensor counts must match")
        foreach_copy = getattr(torch, "_foreach_copy_", None)
        if foreach_copy is None:
            return super().copy_tensors(destinations, sources)

        groups = {}
        for destination, source in zip(destinations, sources):
            if destination.device.type != "cuda" or destination.device != source.device:
                destination.copy_(source, non_blocking=True)
                continue
            key = (destination.device, destination.dtype, source.dtype)
            group = groups.setdefault(key, ([], []))
            group[0].append(destination)
            group[1].append(source)
        for dst, src in groups.values():
            if len(dst) == 1:
                dst[0].copy_(src[0], non_blocking=True)
            else:
                foreach_copy(dst, src)
