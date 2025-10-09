import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
import sys
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import json
from matplotlib.ticker import FuncFormatter
import math

from quant.quantizer import UniformAffineQuantizer
from quant.util import (
    get_blocks,
    catch_first_layer_input,
    catch_linear_input_and_layer_output,
    get_calib_dataset,
    load_tokenizer,
    get_module_by_name,
    empty_cache,
    RotateIndependentFunction,
)
from quant.optimize import get_rotation_pairs, get_random_rotation_pairs

# Init plotting formats
sys.path.append(str(Path(__file__).resolve().parent))
from plot_init import *

device = "cuda"
BITS = 4
GROUP_SIZE = 128

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--layer", type=int, default=0)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--linear-name", type=str, default="self_attn.k_proj")
parser.add_argument("--output-dir", default="figures", type=str)
parser.add_argument("--no-cache", action="store_true")
parser.add_argument("--include-requant", action="store_true")
parser.add_argument("--no-custom-ticks", action="store_true")

args = parser.parse_args()

model_name = args.model.split("/")[-1]

file_stem = (
    f"loss_curves_{model_name.replace('-', '_')}_{args.layer}_{args.linear_name}"
)
cache_file = Path(args.output_dir) / "cache" / (file_stem + ".json")

if not args.no_cache and cache_file.exists():
    with open(cache_file, "r") as f:
        data = json.load(f)
        losses3, loss_re, loss_re_reverse, loss_re_random, loss_re_seq = (
            # data.get("losses1"),
            # data.get("losses2"),
            data.get("losses3"),
            data.get("loss_re"),
            data.get("loss_re_reverse"),
            data.get("loss_re_random"),
            data.get("loss_re_seq"),
        )
        og_loss = torch.tensor(data["og_loss"])
else:
    losses3, loss_re, loss_re_reverse, loss_re_random, loss_re_seq = (
        None,
        None,
        None,
        None,
        None,
    )

if losses3 is None or (args.include_requant and loss_re is None):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = load_tokenizer(args.model)

    blocks = get_blocks(model)
    samples = get_calib_dataset(
        "pileval",
        tokenizer=tokenizer,
        n_samples=128,
        block_size=512,
        seed=0,
        split="validation",
    )
    samples = torch.stack(samples, dim=0)
    with torch.no_grad():
        og_layer_input_batches, kwargs = catch_first_layer_input(model, blocks, samples)
    for i in range(args.layer):
        with torch.no_grad():
            og_layer_input_batches = blocks[i](og_layer_input_batches, **kwargs)

    linear_inputs, _ = catch_linear_input_and_layer_output(
        blocks[args.layer], [og_layer_input_batches], kwargs, output_only=False
    )

    x = linear_inputs[args.linear_name][0].float().to(device)
    linear = get_module_by_name(model.model.layers[0], args.linear_name)
    weight = linear.weight.float().to(device)
    y = x @ weight.T
    og_weight_q = UniformAffineQuantizer.pseudo_quantize(weight, BITS, GROUP_SIZE)
    yy = x @ og_weight_q.T
    og_loss = (yy - y).pow(2).mean()

    steps = args.steps

    def optimize_scaling(scales: torch.Tensor, lr: float) -> list[float]:
        optimizer = torch.optim.AdamW([scales], lr=lr)
        losses = []
        bar = tqdm(total=steps)
        for i in range(steps):
            optimizer.zero_grad()
            y = x @ weight.T
            weight_q = weight * scales
            weight_q = UniformAffineQuantizer.pseudo_quantize(
                weight_q, BITS, GROUP_SIZE
            )
            weight_q = weight_q / scales
            yy = x @ weight_q.T
            loss = (yy - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            bar.update(1)
            bar.set_postfix(loss=loss.item(), og_loss=og_loss.item())

        return losses

    if losses3 is None:
        scales = torch.nn.Parameter(torch.ones(1, weight.shape[1], device=device))
        losses3 = optimize_scaling(scales, lr=0.01)

    def optimize_requant(
        pairs: list[torch.Tensor],
        angles: list[torch.Tensor],
        scales: torch.Tensor,
        lr: float,
    ) -> list[float]:
        optimizer = torch.optim.AdamW([scales] + pairs + angles, lr=lr)
        losses = []
        bar = tqdm(total=steps, desc="ReQuant" + str(len(pairs)) + "R")
        for i in range(steps):
            optimizer.zero_grad()
            y = x @ weight.T
            weight_q = weight * scales.view(1, -1)
            for pair, angle in zip(pairs, angles):
                weight_q = RotateIndependentFunction.apply(weight_q, pair, angle)
            weight_q = UniformAffineQuantizer.pseudo_quantize(
                weight_q, BITS, GROUP_SIZE
            )
            for pair, angle in zip(reversed(pairs), reversed(angles)):
                weight_q = RotateIndependentFunction.apply(weight_q, pair, -angle)
            weight_q = weight_q / scales.view(1, -1)

            yy = x @ weight_q.T
            loss = (yy - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            bar.update(1)
            bar.set_postfix(loss=loss.item(), og_loss=og_loss.item())

        return losses

    if args.include_requant and loss_re is None:
        loss_re = []
        loss_re_random = []
        loss_re_reverse = []
        loss_re_seq = []
        for num_rotation in [1, 2, 4, 8, 16]:
            weight_grouped = weight.view(weight.shape[0], -1, GROUP_SIZE).permute(
                1, 0, 2
            )
            all_pairs = get_rotation_pairs(
                weight_grouped,
                group_size=GROUP_SIZE,
                num_rotations=num_rotation,
                num_pairs_factor=0.5,
            )
            all_pairs = [
                torch.tensor(pairs, device=device, dtype=torch.int32)
                for pairs in all_pairs
            ]
            angles = [
                torch.zeros(pairs.shape[0], device=device, requires_grad=True)
                for pairs in all_pairs
            ]
            scales = torch.ones(1, weight.shape[1], device=device, requires_grad=True)
            loss_re.append(optimize_requant(all_pairs, angles, scales, lr=0.02))

            all_pairs = get_random_rotation_pairs(
                weight_grouped,
                group_size=GROUP_SIZE,
                num_rotations=num_rotation,
                num_pairs_factor=0.5,
                seed=0 + num_rotation,
            )
            all_pairs = [
                torch.tensor(pairs, device=device, dtype=torch.int32)
                for pairs in all_pairs
            ]
            angles = [
                torch.zeros(pairs.shape[0], device=device, requires_grad=True)
                for pairs in all_pairs
            ]
            scales = torch.ones(1, weight.shape[1], device=device, requires_grad=True)
            loss_re_random.append(optimize_requant(all_pairs, angles, scales, lr=0.02))

            all_pairs = get_rotation_pairs(
                weight_grouped,
                group_size=GROUP_SIZE,
                num_rotations=num_rotation,
                num_pairs_factor=0.5,
                reverse=True,
            )
            all_pairs = [
                torch.tensor(pairs, device=device, dtype=torch.int32)
                for pairs in all_pairs
            ]
            angles = [
                torch.zeros(pairs.shape[0], device=device, requires_grad=True)
                for pairs in all_pairs
            ]
            scales = torch.ones(1, weight.shape[1], device=device, requires_grad=True)
            loss_re_reverse.append(optimize_requant(all_pairs, angles, scales, lr=0.02))

        transformed_w = weight.clone()
        all_pairs = []
        all_angles = []
        for _ in range(1, 9):
            weight_grouped = transformed_w.view(
                transformed_w.shape[0], -1, GROUP_SIZE
            ).permute(1, 0, 2)
            this_pairs = get_rotation_pairs(
                weight_grouped,
                group_size=GROUP_SIZE,
                num_rotations=1,
                num_pairs_factor=0.5,
            )[0]
            this_pairs = torch.tensor(this_pairs, device=device, dtype=torch.int32)
            all_pairs.append(this_pairs)
            angles = torch.zeros(this_pairs.shape[0], device=device, requires_grad=True)
            all_angles.append(angles)
            scales = torch.ones(1, weight.shape[1], device=device, requires_grad=True)
            loss_re_seq.append(optimize_requant(all_pairs, all_angles, scales, lr=0.02))

            for pair, angle in zip(all_pairs, all_angles):
                transformed_w = RotateIndependentFunction.apply(weight, pair, angle)

    if not args.no_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        output_dict = {
            "losses3": losses3,
            "og_loss": og_loss.item(),
        }
        if args.include_requant:
            output_dict["loss_re"] = loss_re
            output_dict["loss_re_random"] = loss_re_random
            output_dict["loss_re_reverse"] = loss_re_reverse
            output_dict["loss_re_seq"] = loss_re_seq
        with open(cache_file, "w") as f:
            json.dump(output_dict, f)


plt.figure(figsize=(3.6, 1.6), dpi=300)
plot_kwargs = {"linewidth": 1, "rasterized": True}
plt.plot(losses3, label="Scaling", color="C2", **plot_kwargs)
# plt.plot(losses1, label="Rotation", color="C0", **plot_kwargs)
# plt.plot(losses2, label="Rotation (10% pairs)", color="C1", **plot_kwargs)
if args.include_requant:
    for idx, (loss_re_curve, loss_re_random_curve) in enumerate(
        zip(loss_re, loss_re_random)
    ):
        plt.plot(
            loss_re_curve,
            label=f"{2**idx}R",
            color=f"C{3+idx}",
            **plot_kwargs,
        )
        plt.plot(
            loss_re_random_curve,
            label=f"{2**idx}R (random)",
            color=f"C{3+idx}",
            linestyle="dashed",
            **plot_kwargs,
        )
        # plt.plot(
        #     loss_re_reverse[idx],
        #     label=f"{2**idx}R (reverse)",
        #     color=f"C{3+idx}",
        #     linestyle="dotted",
        #     **plot_kwargs,
        # )
    for idx, n_rot in enumerate([1, 2, 4, 8]):
        plt.plot(
            loss_re_seq[idx],
            label=f"{n_rot}R (sequential)",
            color=f"C{3+idx}",
            linestyle="dotted",
            **plot_kwargs,
        )
plt.ylim(0, og_loss.item() * 1.1)

plt.rcParams.update({"legend.fontsize": 7})


plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.axhline(
    y=og_loss.item(), color="gray", linestyle="--", label="RTN", zorder=0, alpha=0.5
)
plt.xlabel("Steps")
plt.ylabel("Output Error")
plt.xticks([0, 100, 200])
# plt.legend()
plt.legend(loc="upper left", ncol=2, bbox_to_anchor=(1, 1))
plt.grid(False)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = get_file_path(output_dir, file_stem)

plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
plt.close()
