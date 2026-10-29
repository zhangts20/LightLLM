import functools
from typing import List

import torch

from .impl import ChunkedPrefillBackend
from lightllm.common.basemodel.triton_kernel.post_process.apply_constraint_mask import (
    MASKED_LOGIT,
    apply_constraint_mask,
)
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class XgrammarBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.support_overlap = False
        self.prefill_mask_func = self._prefill_mask_callback
        self.decode_mask_func = self._decode_mask_callback
        self.extra_post_req_handle_func = self._update_xgrammar_fsm

    def init_custom(self):
        import xgrammar as xgr
        from transformers import PreTrainedTokenizerBase

        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        hf_tokenizer = self.tokenizer
        while not isinstance(hf_tokenizer, PreTrainedTokenizerBase) and hasattr(hf_tokenizer, "tokenizer"):
            hf_tokenizer = hf_tokenizer.tokenizer
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=8)
        max_mask_rows = self.args.running_max_req_size * (self.args.mtp_step + 1)
        self._xgrammar_bitmask_cpu = xgr.allocate_token_bitmask(
            max_mask_rows, self.tokenizer_info.vocab_size
        ).pin_memory()
        self._xgrammar_enabled_cpu = torch.zeros(max_mask_rows, dtype=torch.bool, pin_memory=True)
        self._xgrammar_bitmask_gpu = None
        self._xgrammar_enabled_gpu = None

        @functools.lru_cache(maxsize=200)
        def get_cached_grammar(constraint_type: str, grammar: str):
            logger.info(f"grammar cache miss for {constraint_type}: '{grammar}'")
            try:
                if constraint_type == "grammar":
                    if grammar == "json":
                        return self.xgrammar_compiler.compile_builtin_json_grammar()
                    return self.xgrammar_compiler.compile_grammar(grammar)
                if constraint_type == "schema":
                    return self.xgrammar_compiler.compile_json_schema(grammar)
                if constraint_type == "regex":
                    return self.xgrammar_compiler.compile_regex(grammar)
                raise ValueError(f"Unknown xgrammar type: {constraint_type}")
            except Exception as e:
                logger.error(f"Failed to compile {constraint_type}: {e}")
                raise

        self.get_cached_grammar = get_cached_grammar

    def _decode_mask_callback(
        self, run_reqs: List[InferReq], logits: torch.Tensor, b_mtp_index: torch.Tensor = None
    ):
        masked = False
        if self._has_output_constraints(run_reqs):
            self._init_req_xgrammar_matchers(run_reqs)
            masked = self._apply_xgrammar_masks(run_reqs, logits, b_mtp_index)
        if not masked:
            self._mask_padding_logits(logits)

    def _prefill_mask_callback(self, run_reqs: List[InferReq], logits: torch.Tensor):
        masked = False
        if self._has_output_constraints(run_reqs):
            self._init_req_xgrammar_matchers(run_reqs)
            masked = self._apply_xgrammar_masks(run_reqs, logits)
        if not masked:
            self._mask_padding_logits(logits)

    def _mask_padding_logits(self, logits: torch.Tensor) -> None:
        vocab_size = self.tokenizer_info.vocab_size
        if logits.shape[1] > vocab_size:
            logits[:, vocab_size:] = MASKED_LOGIT

    @staticmethod
    def _has_output_constraints(run_reqs: List[InferReq]) -> bool:
        return any(XgrammarBackend._has_output_constraint(req.sampling_param) for req in run_reqs)

    @staticmethod
    def _has_output_constraint(sample_params) -> bool:
        return (
            sample_params.regular_constraint is not None
            or sample_params.guided_grammar is not None
            or sample_params.guided_json is not None
        )

    @staticmethod
    def _record_xgrammar_error(sample_params, exc: BaseException) -> None:
        sample_params.xgrammar_error = str(exc)
        logger.exception("Failed to process output grammar: %s", sample_params.xgrammar_error)

    @staticmethod
    def _accept_xgrammar_token(sample_params, token_id: int) -> bool:
        if sample_params.xgrammar_in_reasoning:
            end_ids = sample_params.xgrammar_reasoning_end
            sample_params.xgrammar_reasoning_tail = (
                sample_params.xgrammar_reasoning_tail + (int(token_id),)
            )[-len(end_ids) :]
            if sample_params.xgrammar_reasoning_tail == end_ids:
                sample_params.xgrammar_in_reasoning = False
            return True
        return sample_params.xgrammar_matcher.accept_token(int(token_id))

    @staticmethod
    def _fill_xgrammar_mask(sample_params, bitmask: torch.Tensor, row: int) -> bool:
        if sample_params.xgrammar_error is not None or sample_params.xgrammar_in_reasoning:
            return False
        matcher = sample_params.xgrammar_matcher
        if matcher.is_terminated():
            return False
        matcher.fill_next_token_bitmask(bitmask, row)
        return True

    def _update_xgrammar_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        sample_params = req_obj.sampling_param
        if not getattr(sample_params, "xgrammar_initialized", False):
            return

        if sample_params.xgrammar_error is None:
            try:
                if not self._accept_xgrammar_token(sample_params, next_token_id):
                    raise RuntimeError(f"Grammar rejected sampled token {next_token_id}")
            except BaseException as exc:
                self._record_xgrammar_error(sample_params, exc)

        if sample_params.xgrammar_error is not None:
            req_obj.finish_status.set_status(FinishStatus.FINISHED_ERROR)
        elif not sample_params.xgrammar_in_reasoning:
            try:
                if sample_params.xgrammar_matcher.is_terminated():
                    req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)
            except BaseException as exc:
                self._record_xgrammar_error(sample_params, exc)
                req_obj.finish_status.set_status(FinishStatus.FINISHED_ERROR)

    def _apply_xgrammar_masks(
        self, run_reqs: List[InferReq], logits: torch.Tensor, b_mtp_index: torch.Tensor = None
    ) -> bool:
        row_count = len(run_reqs)
        bitmask = self._xgrammar_bitmask_cpu[:row_count]
        bitmask_enabled = self._xgrammar_enabled_cpu[:row_count]
        bitmask_enabled.zero_()
        enabled_flags = bitmask_enabled.numpy()
        has_enabled_mask = False

        if b_mtp_index is None:
            for i, run_obj in enumerate(run_reqs):
                if run_obj.get_chuncked_input_token_len() != run_obj.get_cur_total_len():
                    continue
                sample_params = run_obj.sampling_param
                if self._has_output_constraint(sample_params) and sample_params.xgrammar_error is None:
                    try:
                        if self._fill_xgrammar_mask(sample_params, bitmask, i):
                            enabled_flags[i] = True
                            has_enabled_mask = True
                    except BaseException as exc:
                        self._record_xgrammar_error(sample_params, exc)
        else:
            next_token_table = (
                g_infer_context.req_manager.req_sampling_params_manager.req_to_next_token_ids_cpu
            )
            group_start = 0
            while group_start < len(run_reqs):
                run_obj = run_reqs[group_start]
                group_end = group_start + 1
                while group_end < len(run_reqs) and run_reqs[group_end] is run_obj:
                    group_end += 1

                sample_params = run_obj.sampling_param
                if self._has_output_constraint(sample_params) and sample_params.xgrammar_error is None:
                    matcher = sample_params.xgrammar_matcher
                    group_size = group_end - group_start
                    draft_tokens = (
                        next_token_table[run_obj.req_idx, 1:group_size].tolist()
                        if group_size > 1
                        else []
                    )
                    accepted_grammar_count = 0
                    saved_in_reasoning = sample_params.xgrammar_in_reasoning
                    saved_reasoning_tail = sample_params.xgrammar_reasoning_tail
                    try:
                        for offset in range(group_size):
                            if not sample_params.xgrammar_in_reasoning and matcher.is_terminated():
                                break
                            if self._fill_xgrammar_mask(sample_params, bitmask, group_start + offset):
                                enabled_flags[group_start + offset] = True
                                has_enabled_mask = True
                            if offset < len(draft_tokens):
                                grammar_was_active = not sample_params.xgrammar_in_reasoning
                                if not self._accept_xgrammar_token(sample_params, draft_tokens[offset]):
                                    break
                                if grammar_was_active:
                                    accepted_grammar_count += 1
                    except BaseException as exc:
                        self._record_xgrammar_error(sample_params, exc)
                    finally:
                        try:
                            if accepted_grammar_count:
                                matcher.rollback(accepted_grammar_count)
                        except BaseException as exc:
                            self._record_xgrammar_error(sample_params, exc)
                        sample_params.xgrammar_in_reasoning = saved_in_reasoning
                        sample_params.xgrammar_reasoning_tail = saved_reasoning_tail

                group_start = group_end

        if has_enabled_mask:
            device = logits.device
            if self._xgrammar_bitmask_gpu is None:
                self._xgrammar_bitmask_gpu = torch.empty_like(self._xgrammar_bitmask_cpu, device=device)
                self._xgrammar_enabled_gpu = torch.empty_like(self._xgrammar_enabled_cpu, device=device)
            bitmask_gpu = self._xgrammar_bitmask_gpu[:row_count]
            enabled_gpu = self._xgrammar_enabled_gpu[:row_count]
            # Disabled rows hold stale words; the kernel skips them via enabled_gpu.
            enabled_gpu.copy_(bitmask_enabled, non_blocking=True)
            bitmask_gpu.copy_(bitmask, non_blocking=True)
            apply_constraint_mask(
                logits=logits,
                req_to_bitmask=bitmask_gpu,
                req_to_bitmask_enabled=enabled_gpu,
                vocab_size=self.tokenizer_info.vocab_size,
            )
        return has_enabled_mask

    def _init_req_xgrammar_matchers(self, run_reqs: List[InferReq]):
        import xgrammar as xgr

        for run_obj in run_reqs:
            sample_params = run_obj.sampling_param
            if not self._has_output_constraint(sample_params) or getattr(
                sample_params, "xgrammar_initialized", False
            ):
                continue
            sample_params.xgrammar_initialized = True
            sample_params.xgrammar_error = None
            sample_params.xgrammar_reasoning_end = tuple(sample_params.guided_reasoning_end)
            sample_params.xgrammar_reasoning_tail = ()
            sample_params.xgrammar_in_reasoning = bool(sample_params.xgrammar_reasoning_end)
            ctx = None
            try:
                if sample_params.regular_constraint is not None:
                    ctx = self.get_cached_grammar("regex", sample_params.regular_constraint)
                elif sample_params.guided_grammar is not None:
                    ctx = self.get_cached_grammar("grammar", sample_params.guided_grammar)
                elif sample_params.guided_json is not None:
                    ctx = self.get_cached_grammar("schema", sample_params.guided_json)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(ctx)
            except BaseException as exc:
                self._record_xgrammar_error(sample_params, exc)
