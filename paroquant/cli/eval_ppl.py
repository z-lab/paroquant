"""Evaluate perplexity of a model on wikitext2 and C4 datasets.

    python -m paroquant.cli.eval_ppl --model <path_or_hf_id> --seqlen 2048
"""

from __future__ import annotations

import argparse
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_wikitext2_tokens(seqlen: int, tokenizer) -> torch.Tensor:
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encoding = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    return encoding.input_ids


def _get_c4_tokens(seed: int, seqlen: int, tokenizer) -> torch.Tensor:
    data = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(seed)
    chunks = []
    for _ in range(256):
        while True:
            idx = random.randint(0, len(data) - 1)
            tokens = tokenizer(data[idx]["text"], return_tensors="pt")
            if tokens.input_ids.shape[1] >= seqlen:
                break
        start = random.randint(0, tokens.input_ids.shape[1] - seqlen - 1)
        chunks.append(tokens.input_ids[:, start : start + seqlen])
    return torch.hstack(chunks)


def _get_eval_tokens(name: str, seed: int, seqlen: int, tokenizer) -> torch.Tensor:
    if name == "wikitext2":
        return _get_wikitext2_tokens(seqlen, tokenizer)
    elif name == "c4":
        return _get_c4_tokens(seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset: {name}")


def _load_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if "AWQ" in path:
        model = model.model
    return model


def main(args: argparse.Namespace):
    model = _load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    for dataset_name in ["wikitext2", "c4"]:
        tokens = _get_eval_tokens(dataset_name, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
        n_samples = tokens.numel() // args.seqlen
        tokens = tokens[0, : args.seqlen * n_samples].view(n_samples, args.seqlen)

        total_loss = 0.0
        progress = tqdm(range(n_samples), desc=dataset_name)
        for i in progress:
            inp = tokens[i].cuda().view(1, -1)
            logits = model(inp, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:]
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            progress.set_description(f"{dataset_name} avg_loss={total_loss / (i + 1):.4f}")

        ppl = torch.exp(torch.tensor(total_loss / n_samples)).item()
        print(f"{dataset_name} perplexity: {ppl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity on wikitext2 and C4")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
