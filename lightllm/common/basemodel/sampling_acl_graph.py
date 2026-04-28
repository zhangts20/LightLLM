import bisect
import torch
import torch_npu
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


from lightllm.common.basemodel.triton_kernel.apply_penalty_gpu_cache import apply_penalty_npu_cache
from lightllm.common.req_manager import ReqManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class _SamplingGraphStructure:
    graph: Optional["torch.npu.NPUGraph"]
    logits: torch.Tensor
    b_req_idx: torch.Tensor
    b_temperatures: torch.Tensor
    b_top_ps: torch.Tensor
    b_top_ks: torch.Tensor
    b_length_penalty_param: torch.Tensor
    b_mask_eos_reqs: torch.Tensor
    eos_ids: torch.Tensor
    rand_u: torch.Tensor
    u_clamped: torch.Tensor
    gumbel_scratch: torch.Tensor
    next_token_ids: torch.Tensor
    next_token_logprobs: torch.Tensor
    is_greedy: bool


def _gumbel_argmax_from_u(
    log_p: torch.Tensor,
    u: torch.Tensor,
    u_clamped: torch.Tensor,
    sum_out: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-12
    torch.clamp(u, min=eps, max=1.0 - eps, out=u_clamped)
    torch.log(u_clamped, out=sum_out)
    torch.neg(sum_out, out=sum_out)
    torch.log(sum_out, out=sum_out)
    torch.neg(sum_out, out=sum_out)
    torch.add(log_p, sum_out, out=sum_out)
    return sum_out.argmax(dim=-1, keepdim=True)


def _run_sample_body(
    logits: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_temperatures: torch.Tensor,
    b_top_ps: torch.Tensor,
    b_top_ks: torch.Tensor,
    b_length_penalty_param: torch.Tensor,
    b_mask_eos_reqs: torch.Tensor,
    eos_ids: torch.Tensor,
    sampling_params_manager,
    is_greedy: bool,
    rand_u: torch.Tensor,
    u_clamped: torch.Tensor,
    gumbel_scratch: torch.Tensor,
    next_token_ids: torch.Tensor,
    next_token_logprobs: torch.Tensor,
):
    apply_penalty_npu_cache(
        Logits=logits,
        b_req_idx=b_req_idx,
        b_length_penalty_param=b_length_penalty_param,
        b_mask_eos_reqs=b_mask_eos_reqs,
        eos_ids=eos_ids,
        sampling_params_manager=sampling_params_manager,
    )
    logits.div_(b_temperatures.view((-1, 1)))
    if is_greedy:
        torch.argmax(logits, dim=-1, out=next_token_ids)
        logsumexp = torch.logsumexp(logits, dim=-1)
        selected_logits = torch.gather(logits, 1, next_token_ids.view(-1, 1)).squeeze(-1)
        next_token_logprobs.copy_((selected_logits - logsumexp).to(dtype=torch.float32))
    else:
        filtered_logits = torch_npu.npu_top_k_top_p(logits, b_top_ps, b_top_ks)
        log_p = torch.log_softmax(filtered_logits, dim=-1, dtype=torch.float32)
        sampled_index = _gumbel_argmax_from_u(log_p, rand_u, u_clamped, gumbel_scratch)
        next_token_ids.copy_(sampled_index.view(-1))
        next_token_logprobs.copy_(log_p.gather(dim=1, index=sampled_index).squeeze(-1))


class SamplingAclGraph:
    def __init__(
        self,
        graph_mempool,
        acl_graph_batch_sizes: List[int],
        vocab_size: int,
        dtype: torch.dtype,
        req_manager: ReqManager,
    ) -> None:
        self.mempool = graph_mempool
        self.acl_graph_batch_sizes = sorted(acl_graph_batch_sizes)
        self.vocab_size = vocab_size
        self.dtype = dtype
        self._req_manager = req_manager
        self._graphs: Dict[Tuple[int, bool], _SamplingGraphStructure] = {}
        eos_list = get_env_start_args().eos_id
        if not eos_list:
            eos_list = [2]
        self._eos_cap_len = len(eos_list)
        self._eos_pin_buf = torch.zeros((self._eos_cap_len,), dtype=torch.int32, pin_memory=True)
        self._eos_tuple_cache: Optional[Tuple[int, ...]] = None
        self._warmup_done = True

    def find_closest_graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.acl_graph_batch_sizes, batch_size)
        if index < len(self.acl_graph_batch_sizes):
            return self.acl_graph_batch_sizes[index]
        return None

    def need_capture(self, padded_batch_size: int, is_greedy: bool) -> bool:
        return (padded_batch_size, is_greedy) not in self._graphs

    def warmup(self, device: Union[str, torch.device]) -> None:
        if not self.can_use_runtime():
            logger.info("SamplingAclGraph warmup skipped.")
            return
        dev = device if isinstance(device, torch.device) else torch.device(device)
        self._warmup_done = False
        for bs in self.acl_graph_batch_sizes[::-1]:
            self.ensure_captured(bs, True, dev)
            self.ensure_captured(bs, False, dev)
        torch.npu.synchronize()
        self._warmup_done = True
        missing = [
            (bs, g)
            for bs in self.acl_graph_batch_sizes
            for g in (True, False)
            if (bs, g) not in self._graphs
        ]
        if missing:
            logger.warning("SamplingAclGraph warmup incomplete, missing keys: %s", missing)
        logger.info(
            "SamplingAclGraph warmup finished; batch_sizes=%s -> %d graphs (padded_batch, is_greedy)=%s",
            self.acl_graph_batch_sizes,
            len(self._graphs),
            sorted(self._graphs.keys()),
        )

    def can_use_runtime(self) -> bool:
        args = get_env_start_args()
        if not getattr(args, "enable_sampling_acl_graph", False):
            return False
        if args.enable_decode_microbatch_overlap:
            return False
        if args.sampling_backend != "ascend":
            return False
        spm = self._req_manager.req_sampling_params_manager
        if spm.penalty_counter_mode == "cpu_counter":
            return False
        return True

    def _alloc(self, padded_bs: int, device: torch.device, is_greedy: bool) -> _SamplingGraphStructure:
        logits = torch.zeros((padded_bs, self.vocab_size), dtype=self.dtype, device=device)
        b_req_idx = torch.zeros((padded_bs,), dtype=torch.int32, device=device)
        b_temperatures = torch.ones((padded_bs,), dtype=self.dtype, device=device)
        b_top_ps = torch.ones((padded_bs,), dtype=self.dtype, device=device)
        b_top_ks = torch.ones((padded_bs,), dtype=torch.int32, device=device)
        b_length_penalty_param = torch.zeros((padded_bs,), dtype=torch.int32, device=device)
        b_mask_eos_reqs = torch.zeros((padded_bs,), dtype=torch.bool, device=device)
        eos_ids = torch.zeros((self._eos_cap_len,), dtype=torch.int32, device=device)
        rand_u = torch.empty((padded_bs, self.vocab_size), dtype=torch.float32, device=device)
        u_clamped = torch.empty((padded_bs, self.vocab_size), dtype=torch.float32, device=device)
        gumbel_scratch = torch.empty((padded_bs, self.vocab_size), dtype=torch.float32, device=device)
        next_token_ids = torch.empty((padded_bs,), dtype=torch.int64, device=device)
        next_token_logprobs = torch.empty((padded_bs,), dtype=torch.float32, device=device)
        return _SamplingGraphStructure(
            graph=None,
            logits=logits,
            b_req_idx=b_req_idx,
            b_temperatures=b_temperatures,
            b_top_ps=b_top_ps,
            b_top_ks=b_top_ks,
            b_length_penalty_param=b_length_penalty_param,
            b_mask_eos_reqs=b_mask_eos_reqs,
            eos_ids=eos_ids,
            rand_u=rand_u,
            u_clamped=u_clamped,
            gumbel_scratch=gumbel_scratch,
            next_token_ids=next_token_ids,
            next_token_logprobs=next_token_logprobs,
            is_greedy=is_greedy,
        )

    def _capture_one(self, padded_bs: int, is_greedy: bool, device: torch.device):
        spm = self._req_manager.req_sampling_params_manager
        b = self._alloc(padded_bs, device, is_greedy)
        eos_list = get_env_start_args().eos_id or [2]
        b.eos_ids.copy_(torch.tensor(eos_list, dtype=torch.int32, device=device))
        for _ in range(1):
            torch.npu.synchronize()
            _run_sample_body(
                b.logits,
                b.b_req_idx,
                b.b_temperatures,
                b.b_top_ps,
                b.b_top_ks,
                b.b_length_penalty_param,
                b.b_mask_eos_reqs,
                b.eos_ids,
                spm,
                is_greedy,
                b.rand_u,
                b.u_clamped,
                b.gumbel_scratch,
                b.next_token_ids,
                b.next_token_logprobs,
            )
            torch.npu.synchronize()
        graph_obj = torch.npu.NPUGraph()
        with torch.npu.graph(graph_obj, pool=self.mempool):
            _run_sample_body(
                b.logits,
                b.b_req_idx,
                b.b_temperatures,
                b.b_top_ps,
                b.b_top_ks,
                b.b_length_penalty_param,
                b.b_mask_eos_reqs,
                b.eos_ids,
                spm,
                is_greedy,
                b.rand_u,
                b.u_clamped,
                b.gumbel_scratch,
                b.next_token_ids,
                b.next_token_logprobs,
            )
        b.graph = graph_obj
        self._graphs[(padded_bs, is_greedy)] = b

    def ensure_captured(self, padded_batch_size: int, is_greedy: bool, device: torch.device):
        if not self.need_capture(padded_batch_size, is_greedy):
            return
        if self._warmup_done:
            logger.warning(
                "SamplingAclGraph on-demand capture padded_batch=%s is_greedy=%s — expect a latency spike; "
                "ensure decode padded sizes stay within acl_graph_batch_sizes and warmup ran on this device.",
                padded_batch_size,
                is_greedy,
            )
        self._capture_one(padded_batch_size, is_greedy, device)

    def replay(
        self,
        logits: torch.Tensor,
        origin_batch: int,
        padded_batch_size: int,
        eos_id: List[int],
        b_req_idx: torch.Tensor,
        b_temperatures: torch.Tensor,
        b_top_ps: torch.Tensor,
        b_top_ks: torch.Tensor,
        b_length_penalty_param: torch.Tensor,
        b_mask_eos_reqs: torch.Tensor,
        is_all_greedy: bool,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if origin_batch > padded_batch_size:
            return None
        if logits.shape[1] != self.vocab_size:
            return None
        if len(eos_id) != self._eos_cap_len:
            return None
        is_greedy = bool(is_all_greedy)
        if not self.can_use_runtime():
            return None
        self.ensure_captured(padded_batch_size, is_greedy, logits.device)
        b = self._graphs.get((padded_batch_size, is_greedy))
        if b is None:
            return None

        hold = self._req_manager.HOLD_REQUEST_ID
        b.logits[:origin_batch].copy_(logits)
        if padded_batch_size > origin_batch:
            b.logits[origin_batch:].zero_()
        b.b_req_idx[:origin_batch].copy_(b_req_idx)
        b.b_temperatures[:origin_batch].copy_(b_temperatures.to(dtype=self.dtype))
        b.b_top_ps[:origin_batch].copy_(b_top_ps.to(dtype=self.dtype))
        b.b_top_ks[:origin_batch].copy_(b_top_ks)
        b.b_length_penalty_param[:origin_batch].copy_(b_length_penalty_param)
        b.b_mask_eos_reqs[:origin_batch].copy_(b_mask_eos_reqs)
        if padded_batch_size > origin_batch:
            b.b_req_idx[origin_batch:].fill_(hold)
            b.b_temperatures[origin_batch:].fill_(1.0)
            b.b_top_ps[origin_batch:].fill_(1.0)
            b.b_top_ks[origin_batch:].fill_(1)
            b.b_length_penalty_param[origin_batch:].zero_()
            b.b_mask_eos_reqs[origin_batch:].fill_(False)

        eos_key = tuple(eos_id)
        if eos_key != self._eos_tuple_cache:
            self._eos_pin_buf.copy_(torch.tensor(eos_id, dtype=torch.int32))
            b.eos_ids.copy_(self._eos_pin_buf, non_blocking=True)
            self._eos_tuple_cache = eos_key

        if not is_greedy:
            b.rand_u[:origin_batch].uniform_(0.0, 1.0)

        b.graph.replay()
        return b.next_token_ids[:origin_batch].clone(), b.next_token_logprobs[:origin_batch].clone()
