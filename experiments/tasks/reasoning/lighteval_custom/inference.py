import experiments.tasks.reasoning.lighteval_custom.patch
import os
import json
import random
import argparse
from tqdm import tqdm

import torch
import transformers
from vllm import LLM
from vllm.engine.arg_utils import PoolerConfig

from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from experiments.tasks.reasoning.lighteval_custom.main_vllm import vllm


def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="whether to re-evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to save inference results."
    )
    # model
    parser.add_argument(
        "--model",
        type=str,
        default="./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B",
        help="Model to load.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="dtype to use")
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="AIME-2024",
        choices=[
            "AIME-2024",
            "AIME-2025",
            "AIME-90",
            "MATH-500",
            "NuminaMath-1.5",
            "GSM8K",
            "GPQA-Diamond",
            "MMLU-PRO",
        ],
        help="Dataset to load.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max #samples (for debug)"
    )
    # generation
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Generation temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Generation top_p")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate per output sequence.",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=32768,
        help="Maximum model input length.",
    )
    args = parser.parse_args()

    # force float16 for gptqmodel inference
    if "gptqmodel" in args.model:
        args.dtype = "float16"

    # output path
    args.model_name = args.model.strip("/").replace("/", "_")
    output_dir = (
        os.path.join("./outputs", "inference", f"{args.model_name}-seed{args.seed}")
        if args.output_dir is None
        else args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    args.output_path = os.path.join(output_dir, f"{args.dataset}.jsonl")

    # Distributed settings
    args.tensor_parallel_size = torch.cuda.device_count()

    return args


def main(args):
    if not args.debug and not args.overwrite and os.path.exists(args.output_path):
        print(f"Evaluation results found at {args.output_path}. Skip evaluation")
        return

    random.seed(args.seed)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    generation_parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=30 if "QwQ" in args.model else None,  # TODO. enable top_k only for QwQ?
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    model_config = VLLMModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        max_model_length=args.max_model_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        generation_parameters=generation_parameters,
    )

    if args.dataset == "AIME-2024":
        task_kwargs = {
            "tasks": "custom|aime24|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "AIME-2025":
        task_kwargs = {
            "tasks": "custom|aime25|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "AIME-90":
        task_kwargs = {
            "tasks": "custom|aime90|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "MATH-500":
        task_kwargs = {
            "tasks": "custom|math_500|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "NuminaMath-1.5":
        task_kwargs = {
            "tasks": "custom|numina_math|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "GSM8K":
        task_kwargs = {
            "tasks": "custom|gsm8k|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "GPQA-Diamond":
        task_kwargs = {
            "tasks": "custom|gpqa:diamond|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }
    elif args.dataset == "MMLU-PRO":
        task_kwargs = {
            "tasks": "custom|mmlu_pro|0|0",
            "custom_tasks": "experiments/tasks/reasoning/lighteval_custom/reasoning.py",
        }

    results, details = vllm(
        model_config=model_config,
        use_chat_template=True,
        # output_dir="./outputs/lighteval_outputs",
        max_samples=args.max_samples,
        **task_kwargs,
    )

    # save evaluation results
    eval_results = []
    task_name = list(details.keys())[0]
    for detail in details[task_name]:
        eval_results.append(
            {
                "full_prompt": detail["full_prompt"],
                "generated_text": detail["predictions"][0],
                "gold": detail["gold"],
                "metrics": detail["metrics"],
            }
        )
    with open(args.output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved at {args.output_path}.")


if __name__ == "__main__":
    args = parser_gen()
    main(args)
