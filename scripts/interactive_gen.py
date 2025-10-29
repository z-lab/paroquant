# This script is based off of the generation script in https://github.com/chu-tianxiang/QuIP-for-all
import time
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaConfig,
    AutoModelForCausalLM,
    Qwen2Config,
)
from inference_engine.model_executor.models.cache_utils import StaticCache
import os

torch.set_grad_enabled(False)

from inference_engine.model_executor.models.llama import LlamaForCausalLM
from inference_engine.model_executor.models.qwen3 import Qwen3ForCausalLM
from inference_engine.model_executor.models.qwen3_fp16 import (
    Qwen3ForCausalLMFP16,
)
from inference_engine.model_executor.models.llama_fp16 import (
    LlamaForCausalLMFP16,
)
import gc
from transformers.utils import CONFIG_NAME
import json
from huggingface_hub import hf_hub_download


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def _load_config_dict(path_or_repo: str):
    if os.path.isdir(path_or_repo):
        cfg_path = os.path.join(path_or_repo, CONFIG_NAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found under: {path_or_repo}")
        with open(cfg_path, "r") as f:
            return json.load(f)
    else:
        cfg_path = hf_hub_download(path_or_repo, CONFIG_NAME)
        with open(cfg_path, "r") as f:
            return json.load(f)


def model_from_hf_path(path, empty_model=False):
    try:
        bad_config = AutoConfig.from_pretrained(path)
        is_quantized = hasattr(bad_config, "paroquant_config")
    except ValueError as e:
        if "qwen3" in str(e).lower():
            d = _load_config_dict(path)
            bad_config = Qwen2Config(**d)
            is_quantized = hasattr(bad_config, "paroquant_config")

    model_type = bad_config.model_type
    if is_quantized or empty_model:
        if model_type == "llama":
            if not empty_model:
                model_str = LlamaConfig.from_pretrained(path).orig_model_name
            else:
                model_str = path
            model_cls = LlamaForCausalLM
        elif model_type == "qwen3":
            if not empty_model:
                model_str = Qwen2Config.from_pretrained(path).orig_model_name
            else:
                model_str = path
            model_cls = Qwen3ForCausalLM
        else:
            raise Exception
    else:
        if model_type == "llama":
            model_str = path
            model_cls = LlamaForCausalLMFP16
        elif model_type == "qwen3":
            model_str = path
            model_cls = Qwen3ForCausalLMFP16
    if empty_model:
        model = model_cls(bad_config)
        dtype = torch.float16
        model = model.to(dtype=dtype)
        model.to("cuda")
        model.eval()
    else:
        model = model_cls.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map="cuda",
        )

    return model, model_str


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.compile
def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.no_grad()
def decode_one_tokens(model, cur_token, past_kv, cache_position):
    logits = model(cur_token, past_key_values=past_kv, cache_position=cache_position)[0]
    new_token = sample(logits, temperature=0.6, top_k=5)[0]
    return new_token, logits


@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens, top_k, callback, past_kv):
    inputs = tokenizer(text, return_tensors="pt").to(0)
    batch_size, seq_length = inputs["input_ids"].shape
    cache_position = torch.arange(seq_length, device=0)
    generated_ids = torch.zeros(
        batch_size, seq_length + max_new_tokens, dtype=torch.int, device=0
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(0).int()
    logits = model(**inputs, past_key_values=past_kv, cache_position=cache_position)[0]

    next_token, _ = sample(logits, top_k=top_k)

    generated_ids[:, seq_length] = next_token
    callback(next_token)

    cache_position = torch.tensor([seq_length + 1], device=0)
    decode_time = time.time()
    for _ in range(1, max_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=True
        ):
            next_token, logits = decode_one_tokens(
                model, next_token.clone(), past_kv, cache_position
            )
        generated_ids[:, cache_position] = next_token.int()
        callback(next_token)
        cache_position += 1
    torch.cuda.synchronize()
    decode_time = time.time() - decode_time

    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_ids, text, max_new_tokens / decode_time


@torch.no_grad()
def benchmark(model, tokenizer, prefill_len, decode_len, past_kv):
    FIXED_TEXT = "The quick brown fox jumps over the lazy dog. "
    pieces = []
    while True:
        pieces.append(FIXED_TEXT)
        enc_full = tokenizer(
            "".join(pieces), return_tensors="pt", add_special_tokens=False
        )
        if enc_full.input_ids.shape[1] >= prefill_len:
            break

    inputs = enc_full.to(0)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    B, S = input_ids.shape

    cache_position = torch.arange(S, device=0)
    generated_ids = torch.empty(B, S + decode_len, dtype=torch.int, device=0)
    generated_ids[:, :S] = input_ids

    out = model(
        **enc_full,
        past_key_values=past_kv,
        cache_position=cache_position,
        use_cache=True,
    )
    logits = out[0]
    next_token, _ = sample(logits)
    generated_ids[:, S] = next_token

    cache_position = torch.tensor([S + 1], device=0)
    torch.cuda.synchronize()
    t0 = time.time()

    for t in range(1, decode_len):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=False, enable_math=True
        ):
            next_token, logits = decode_one_tokens(
                model, next_token.clone(), past_kv, cache_position
            )
        generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

    torch.cuda.synchronize()
    dt = time.time() - t0

    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_ids, texts, decode_len / dt


def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    return args[0]


def main(hf_path, compile, interactive, max_tokens, top_k):

    model, model_str = model_from_hf_path(hf_path)

    sharded = False

    tokenizer = AutoTokenizer.from_pretrained(model_str)

    tokenizer.pad_token = tokenizer.eos_token

    past_kv = StaticCache(
        model.config, 1, 2 * args.max_new_tokens, device=0, dtype=model.dtype
    )
    text = "This is a test of this large language model"
    callback = lambda x: x
    ids, text, _ = generate(model, tokenizer, text, 8, top_k, callback, past_kv)

    if compile:
        print(
            "Capturing CUDA graphs, may take some time. If you are running a model over multiple GPUs, the first generation will be very slow due to compiling the model."
        )
        global decode_one_tokens
        decode_one_tokens = torch.compile(
            decode_one_tokens, mode="max-autotune", fullgraph=True
        )

    text = "This is a test of this large language model"
    ids, text, _ = generate(model, tokenizer, text, 16, top_k, callback, past_kv)

    while True:
        prompt = input("What is your prompt? ")
        if prompt == "quit":
            exit()
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        buffer = []
        period_id = tokenizer.encode(".")[-1]
        done_generating = False

        def callback(x):
            nonlocal done_generating
            if done_generating:
                return
            buffer.append(tokenizer.decode([period_id] + x[0].tolist())[1:])
            if x[0].item() == tokenizer.eos_token_id:
                done_generating = True
            if len(buffer) == 4 or done_generating:
                print("".join(buffer), end="", flush=True)
                buffer.clear()

        if not interactive:
            callback = lambda x: x
        ids, text, decode_tps = generate(
            model, tokenizer, text, max_tokens, top_k, callback, past_kv
        )
        if not interactive:
            print(text)

        print(
            f"\nDecoding throughput: {decode_tps:.02f} tokens/sec. Includes tokens generated after the EOS token.\n\n"
        )


def bench_model(hf_path, prefill_len, decode_len, empty_model):

    model, model_str = model_from_hf_path(hf_path, empty_model=empty_model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    past_kv = StaticCache(
        model.config, 1, 2 * args.max_new_tokens, device=0, dtype=model.dtype
    )
    ids, text, _ = benchmark(model, tokenizer, 2, 8, past_kv)

    print(
        "Capturing CUDA graphs, may take some time. If you are running a model over multiple GPUs, the first generation will be very slow due to compiling the model."
    )

    global decode_one_tokens
    decode_one_tokens = torch.compile(
        decode_one_tokens, mode="max-autotune", fullgraph=True
    )
    ids, text, _ = benchmark(model, tokenizer, 16, 16, past_kv)
    ids, text, decode_tps = benchmark(
        model, tokenizer, prefill_len, decode_len, past_kv
    )
    print(
        f"\nDecoding throughput: {decode_tps:.02f} tokens/sec. Includes tokens generated after the EOS token.\n\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument("--hf_path", type=str, help="Path to checkpoint")
    parser.add_argument(
        "--streaming", action="store_true", help="Whether to launch in stream mode"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=32, help="Top-k for sampling.")
    parser.add_argument(
        "--no_compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help="Whether to disable TF32 for FP32 matmuls.",
    )
    parser.add_argument(
        "--bench_model",
        action="store_true",
        help="load pretrained model by config for benchmark",
    )
    parser.add_argument(
        "--empty_model",
        action="store_true",
        help="load empty model by config for benchmark",
    )
    parser.add_argument("--prefill_len", default=256, help="prefill len for benchmark")
    parser.add_argument("--decode_len", default=512, help="decode len for benchmark")

    args = parser.parse_args()

    if not args.disable_tf32:
        torch.set_float32_matmul_precision("high")
    if args.empty_model or args.bench_model:
        bench_model(args.hf_path, args.prefill_len, args.decode_len, args.empty_model)
    else:
        main(
            args.hf_path,
            not args.no_compile,
            args.streaming,
            args.max_new_tokens,
            args.top_k,
        )
