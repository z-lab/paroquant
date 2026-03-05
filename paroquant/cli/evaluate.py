from __future__ import annotations

import argparse
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import paroquant.inference.backends.transformers.quantizer  # noqa: F401


_DATASETS = {
    "wikitext2": ("wikitext", {"name": "wikitext-2-raw-v1", "split": "test"}),
    "c4": (
        "allenai/c4",
        {
            "data_files": {"validation": "en/c4-validation.00000-of-00008.json.gz"},
            "split": "validation",
        },
    ),
}


def _load_tokens(dataset_name: str, seq_len: int, tokenizer, seed: int = 0) -> torch.Tensor:
    name, kwargs = _DATASETS[dataset_name]
    data = load_dataset(name, **kwargs)

    if dataset_name == "wikitext2":
        return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids

    rng = random.Random(seed)
    chunks = []
    for _ in range(256):
        while True:
            text = data[rng.randint(0, len(data) - 1)]["text"]
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            if input_ids.shape[1] >= seq_len:
                break
        offset = rng.randint(0, input_ids.shape[1] - seq_len - 1)
        chunks.append(input_ids[:, offset : offset + seq_len])
    return torch.hstack(chunks)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    for dataset_name in ["wikitext2", "c4"]:
        tokens = _load_tokens(dataset_name, args.seq_len, tokenizer, args.seed)
        num_samples = tokens.numel() // args.seq_len
        tokens = tokens[0, : num_samples * args.seq_len].view(num_samples, args.seq_len)

        total_loss = 0.0
        for i in tqdm(range(num_samples), desc=dataset_name):
            input_ids = tokens[i].cuda().unsqueeze(0)
            logits = model(input_ids).logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:]
            total_loss += loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()

        ppl = torch.exp(torch.tensor(total_loss / num_samples)).item()
        print(f"{dataset_name}: {ppl:.2f}")


if __name__ == "__main__":
    main()
