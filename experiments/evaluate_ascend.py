"""Generation, held-out perplexity and GSM8K for an actual checkpoint."""

import argparse
import json
import math
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

import paroquant.inference.backends.transformers.quantizer  # registers checkpoint format
from ascend_reference import dequantize, install_reference, rotation
from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear


def sync():
    torch.npu.synchronize()


def answer_number(text):
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.split("</think>")[-1])
    return str(Decimal(matches[-1].replace(",", "")).normalize()) if matches else None


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["optimized", "naive", "dense", "dense_fused", "bf16"], default="optimized")
    parser.add_argument("--task", choices=["generate", "perplexity", "gsm8k"], default="generate")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=971)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--layer-errors", type=Path)
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", help="Public document token counts for controlled generation timing"
    )
    parser.add_argument("--resume", action="store_true", help="Continue completed dataset batches")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resume and args.task == "generate":
        parser.error("--resume applies to dataset evaluation")
    previous = json.loads(args.output.read_text()) if args.resume and args.output.exists() else None
    if previous:
        for name in ("model", "mode", "task", "data", "context", "limit", "new_tokens", "batch", "start_index", "seed"):
            current = getattr(args, name)
            current = str(current) if isinstance(current, Path) else current
            if previous["configuration"][name] != current:
                raise ValueError(f"Resume configuration differs: {name}")
        if previous.get("complete"):
            print(f"Completed evaluation: {args.output}", flush=True)
            return
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.mode == "bf16" else torch.float16,
        device_map="npu",
        attn_implementation="sdpa",
    ).eval()
    if args.mode in ("naive", "dense", "dense_fused"):
        install_reference(model, cache_weights=args.mode != "naive", fuse_rotation=args.mode == "dense_fused")
    sync()
    report = previous or {
        "model": args.model,
        "mode": args.mode,
        "task": args.task,
        "seed": args.seed,
        "load_seconds": time.perf_counter() - start,
        "load_peak_bytes": torch.npu.max_memory_allocated(),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "records": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temporary.replace(args.output)

    layer_records, handles = {}, []
    if args.layer_errors:
        for name, layer in model.named_modules():
            if not isinstance(layer, RotateQuantizedLinear):
                continue

            def compare(module, inputs, output, name=name):
                if name in layer_records:
                    return
                weight = getattr(module, "_reference_weight", None)
                if weight is None:
                    weight = dequantize(module)
                expected = rotation(inputs[0], module) @ weight
                if module.bias is not None:
                    expected = expected + module.bias
                delta = output.float() - expected.float()
                relative = float(delta.square().sum() / expected.float().square().sum().clamp_min(1e-30)) ** 0.5
                layer_records[name] = {
                    "input_shape": list(inputs[0].shape),
                    "relative_l2": relative,
                    "max_error": float(delta.abs().max()),
                }
                if relative > 0.01:
                    raise AssertionError(f"Layer {name} relative L2 error: {relative}")

            handles.append(layer.register_forward_hook(compare))

    warmup_ids = tokenizer("Explain how rain forms.", return_tensors="pt").to(model.device)
    sync()
    torch.npu.reset_peak_memory_stats()
    before = time.perf_counter()
    model(**warmup_ids, use_cache=False)
    sync()
    report["warmup_seconds"] = time.perf_counter() - before
    report["first_forward_peak_bytes"] = torch.npu.max_memory_allocated()
    if args.layer_errors:
        args.layer_errors.parent.mkdir(parents=True, exist_ok=True)
        args.layer_errors.write_text(json.dumps(layer_records, indent=2))
        for handle in handles:
            handle.remove()

    if args.task == "perplexity":
        rows = [json.loads(line) for line in args.data.read_text().splitlines()]
        tokens = tokenizer("\n\n".join(row["text"] for row in rows), return_tensors="pt").input_ids[0]
        total_nll = sum(row["nll"] for row in report["records"])
        total_tokens = sum(row["tokens"] for row in report["records"])
        completed = {row["offset"] for row in report["records"]}
        starts = list(range(0, len(tokens) - 1, args.context))
        if args.limit:
            starts = starts[: args.limit]
        for start in starts:
            if start in completed:
                continue
            ids = tokens[start : start + args.context + 1].unsqueeze(0).to(model.device)
            sync()
            before = time.perf_counter()
            logits = model(ids[:, :-1], use_cache=False).logits.float()
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="sum")
            sync()
            nll = float(loss)
            total_nll += nll
            total_tokens += ids.shape[1] - 1
            report["records"].append(
                {"offset": start, "nll": nll, "tokens": ids.shape[1] - 1, "seconds": time.perf_counter() - before}
            )
            report.update(perplexity=math.exp(total_nll / total_tokens), evaluated_tokens=total_tokens)
            save()
    elif args.task == "gsm8k":
        rows = [json.loads(line) for line in args.data.read_text().splitlines()][args.start_index :]
        if args.limit:
            rows = rows[: args.limit]
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        completed = {row["index"] for row in report["records"]}
        for first in range(0, len(rows), args.batch):
            batch = rows[first : first + args.batch]
            if args.start_index + first in completed:
                continue
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": row["question"]
                            + "\nSolve step by step, then give the final numerical answer after ####.",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            ids = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            sync()
            before = time.perf_counter()
            generated = model.generate(
                **ids, do_sample=False, max_new_tokens=args.new_tokens, pad_token_id=tokenizer.pad_token_id
            )
            sync()
            elapsed = time.perf_counter() - before
            texts = tokenizer.batch_decode(generated[:, ids.input_ids.shape[1] :], skip_special_tokens=True)
            for offset, (row, text) in enumerate(zip(batch, texts)):
                expected, predicted = answer_number(row["answer"]), answer_number(text)
                report["records"].append(
                    {
                        "index": args.start_index + first + offset,
                        "text": text,
                        "expected": expected,
                        "predicted": predicted,
                        "correct": expected == predicted,
                    }
                )
            report["accuracy"] = sum(row["correct"] for row in report["records"]) / len(report["records"])
            report["evaluated_examples"] = len(report["records"])
            report.setdefault("batches", []).append(
                {"start_index": args.start_index + first, "examples": len(batch), "seconds": elapsed}
            )
            save()
            print(
                json.dumps(
                    {"examples": len(report["records"]), "accuracy": report["accuracy"], "batch_seconds": elapsed}
                ),
                flush=True,
            )
    else:
        if args.prompt_lengths:
            documents = [json.loads(line)["text"] for line in args.data.read_text().splitlines()]
            document_tokens = tokenizer("\n\n".join(documents), return_tensors="pt").input_ids[0]
            rows = [
                {
                    "question": "Summarize the following document in a short paragraph.\n\n"
                    + tokenizer.decode(document_tokens[:length]),
                    "document_tokens": length,
                }
                for length in args.prompt_lengths
            ]
        else:
            rows = [
                {"question": "Explain why the sky appears blue in three sentences."},
                {"question": "A shop has 24 apples. It sells 7 and receives 12. How many apples are left?"},
            ]
        if args.limit:
            rows = rows[: args.limit]
        for index, row in enumerate(rows):
            prompt = row["question"]
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids = tokenizer(rendered, return_tensors="pt").to(model.device)
            sync()
            before = time.perf_counter()
            prepared = model(**ids, use_cache=True, logits_to_keep=1)
            model(
                input_ids=prepared.logits[:, -1:].argmax(-1),
                past_key_values=prepared.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            sync()
            prompt_warmup = time.perf_counter() - before
            del prepared
            torch.npu.reset_peak_memory_stats()
            sync()
            before = time.perf_counter()
            output = model(**ids, use_cache=True, logits_to_keep=1)
            next_token = output.logits[:, -1:].argmax(-1)
            sync()
            ttft = time.perf_counter() - before
            generated = [next_token]
            cache = output.past_key_values
            decode_start = time.perf_counter()
            for _ in range(args.new_tokens - 1):
                output = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                next_token = output.logits[:, -1:].argmax(-1)
                generated.append(next_token)
            sync()
            decode_seconds = time.perf_counter() - decode_start
            text = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True)
            record = {
                "index": index,
                "prompt_tokens": ids.input_ids.numel(),
                "generated_tokens": len(generated),
                "document_tokens": row.get("document_tokens"),
                "decode_length": "fixed",
                "text": text,
                "ttft_seconds": ttft,
                "decode_seconds": decode_seconds,
                "prompt_warmup_seconds": prompt_warmup,
                "decode_tokens_per_second": (len(generated) - 1) / decode_seconds,
                "peak_bytes": torch.npu.max_memory_allocated(),
            }
            report["records"].append(record)
            save()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            del output, cache, generated, next_token
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
