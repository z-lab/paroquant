import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_engine.generation.transformers_backend import (
    decode_one_token,
    model_from_hf_path,
    sample_next_token,
)
from inference_engine.model_executor.models.cache_utils import StaticCache


torch.set_grad_enabled(False)


@torch.no_grad()
def benchmark(model, tokenizer, prefill_len, decode_len, past_kv):
    fixed_text = "The quick brown fox jumps over the lazy dog. "
    pieces = []
    while True:
        pieces.append(fixed_text)
        enc_full = tokenizer(
            "".join(pieces), return_tensors="pt", add_special_tokens=False
        )
        if enc_full.input_ids.shape[1] >= prefill_len:
            break

    input_ids = enc_full["input_ids"].to(0)
    batch_size, seq_len = input_ids.shape

    cache_position = torch.arange(seq_len, device=0)
    generated_ids = torch.empty(
        batch_size, seq_len + decode_len, dtype=torch.int, device=0
    )
    generated_ids[:, :seq_len] = input_ids

    model_inputs = {key: value.to(0) for key, value in enc_full.items()}
    out = model(
        **model_inputs,
        past_key_values=past_kv,
        cache_position=cache_position,
        use_cache=True,
    )
    logits = out[0]
    next_token = sample_next_token(logits)
    generated_ids[:, seq_len] = next_token

    decode_position = torch.tensor([seq_len + 1], device=0)
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(1, decode_len):
        with torch.nn.attention.sdpa_kernel(
            backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        ):
            next_token, _ = decode_one_token(
                model,
                next_token.clone(),
                past_kv,
                decode_position,
                1.0,
                None,
                1.0,
            )
        generated_ids[:, decode_position] = next_token.int()
        decode_position += 1

    torch.cuda.synchronize()
    elapsed = time.time() - start

    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_ids, texts, decode_len / elapsed


def run_benchmark(hf_path, prefill_len, decode_len, empty_model, max_new_tokens):
    model, model_str = model_from_hf_path(hf_path, empty_model=empty_model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    past_kv = StaticCache(
        model.config,
        1,
        2 * max_new_tokens,
        device=0,
        dtype=model.dtype,
    )

    benchmark(model, tokenizer, 2, 8, past_kv)

    print(
        "Capturing CUDA graphs, may take some time. If you are running a model over multiple GPUs, "
        "the first generation will be very slow due to compiling the model."
    )

    global decode_one_token
    decode_one_token = torch.compile(
        decode_one_token,
        mode="max-autotune",
        fullgraph=True,
    )

    benchmark(model, tokenizer, 16, 16, past_kv)
    _, _, decode_tps = benchmark(model, tokenizer, prefill_len, decode_len, past_kv)
    print(
        f"\nDecoding throughput: {decode_tps:.02f} tokens/sec. Includes tokens generated after the EOS token.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ParoQuant transformer backend decoding"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--empty-model",
        action="store_true",
        help="Load empty model by config for benchmark",
    )
    parser.add_argument("--prefill-len", type=int, default=256, help="Prefill length")
    parser.add_argument("--decode-len", type=int, default=512, help="Decode length")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Cache sizing hint for benchmark",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 for FP32 matmuls",
    )
    cli_args = parser.parse_args()

    if not cli_args.disable_tf32:
        torch.set_float32_matmul_precision("high")

    run_benchmark(
        cli_args.model,
        cli_args.prefill_len,
        cli_args.decode_len,
        cli_args.empty_model,
        cli_args.max_new_tokens,
    )
