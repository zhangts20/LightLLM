"""Paged native prefill must match the causal reference with cached prefixes."""
import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.attention.mlu.sdpa import build_prefill_plan, paged_prefill_sdpa


@pytest.mark.parametrize("prefix", [0, 2])
@pytest.mark.parametrize("query_heads", [2, 4])
def test_paged_sdpa_handles_permuted_kv_and_cached_prefix(prefix, query_heads):
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(5 - prefix, query_heads, 8, generator=generator)
    key = torch.randn(12, 2, 8, generator=generator)
    value = torch.randn(12, 2, 8, generator=generator)
    # Endpoints alone look contiguous, but the middle entries are permuted.
    mapping = torch.tensor([[1, 3, 2, 4, 5]], dtype=torch.int32)
    plans = build_prefill_plan(torch.tensor([0]), torch.tensor([0]), torch.tensor([5]), torch.tensor([prefix]), mapping)
    assert plans[0].kv_slice is None
    for shift in [0, 1]:
        actual = torch.empty_like(query)
        paged_prefill_sdpa(query + shift, key, value, actual, plans)
        indices = mapping[0].long()
        current_key = key[indices].repeat_interleave(query_heads // 2, dim=1).transpose(0, 1).unsqueeze(0)
        current_value = value[indices].repeat_interleave(query_heads // 2, dim=1).transpose(0, 1).unsqueeze(0)
        mask = torch.arange(5)[None, :] <= (torch.arange(5 - prefix)[:, None] + prefix)
        expected = F.scaled_dot_product_attention(
            (query + shift).transpose(0, 1).unsqueeze(0), current_key, current_value, attn_mask=mask
        )[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected)


def test_prefill_plan_ignores_rows_with_no_new_tokens():
    plans = build_prefill_plan(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([3]), torch.tensor([3]), torch.tensor([[0, 1, 2]])
    )
    assert plans == []
