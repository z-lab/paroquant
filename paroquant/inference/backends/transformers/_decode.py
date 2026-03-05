"""Low-level decode primitives for torch.compile benchmarking.

These functions provide a manual decode loop compatible with torch.compile +
StaticCache + CUDA graph capture. Used by bench_model.py for measuring
raw decode throughput. For chat/interactive use, prefer Generator
which uses model.generate() + TextIteratorStreamer.
"""

from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    logits = logits[:, -1] / max(temperature, 1e-5)

    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < values.select(-1, -1).unsqueeze(-1), -float("inf"), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, mask)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def decode_one_token(model, cur_token, past_kv, cache_position, temperature, top_k, top_p):
    """Single decode step: one forward pass + sampling. Compatible with torch.compile."""
    logits = model(cur_token, past_key_values=past_kv, cache_position=cache_position)[0]
    return sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p), logits
