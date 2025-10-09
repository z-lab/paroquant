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
    RotateIndependentFunction,
)
from quant.optimize import get_rotation_pairs, _get_sorted_sensitive_channel_pairs

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
        losses1, losses2, losses3, loss_re = (
            data.get("losses1"),
            data.get("losses2"),
            data.get("losses3"),
            data.get("loss_re"),
        )
        og_loss = torch.tensor(data["og_loss"])
else:
    losses1, losses2, losses3, loss_re = None, None, None, None

if (
    losses1 is None
    or losses2 is None
    or losses3 is None
    or (args.include_requant and loss_re is None)
):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, torch_dtype=torch.float16
    )
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
    og_layer_input_batches, kwargs = catch_first_layer_input(
        model,
        blocks,
        samples,
    )
    for i in range(args.layer):
        og_layer_input_batches = blocks[i](og_layer_input_batches, kwargs)

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

    def optimize(
        angles: torch.nn.Parameter, idx_i: torch.Tensor, idx_j: torch.Tensor, lr: float
    ) -> list[float]:
        optimizer = torch.optim.AdamW([angles], lr=lr)
        losses = []
        bar = tqdm(total=steps)
        for i in range(steps):
            optimizer.zero_grad()
            y = x @ weight.T
            A = torch.zeros([weight.shape[1]] * 2, device=device)
            A[idx_i, idx_j] = angles
            R = (A - A.T).matrix_exp()
            weight_q = weight @ R
            weight_q = UniformAffineQuantizer.pseudo_quantize(
                weight_q, BITS, GROUP_SIZE
            )
            weight_q = weight_q @ R.T
            yy = x @ weight_q.T
            loss = (yy - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            bar.update(1)
            bar.set_postfix(loss=loss.item(), og_loss=og_loss.item())

        return losses

    # Full rotation
    if losses1 is None:
        pairs = []
        for i in range(0, weight.shape[1]):
            for j in range(i + 1, weight.shape[1]):
                pairs.append((i, j))

        angles = torch.nn.Parameter(torch.zeros(len(pairs), device=device))
        idx_i = torch.tensor([x[0] for x in pairs], device=device)
        idx_j = torch.tensor([x[1] for x in pairs], device=device)

        # losses1 = optimize(angles, idx_i, idx_j, lr=0.001)

    # Partial rotation
    if losses2 is None:
        weight_grouped = weight.view(weight.shape[0], -1, weight.shape[1]).permute(
            1, 0, 2
        )
        pairs = _get_sorted_sensitive_channel_pairs(weight_grouped)[0]
        num_pairs = 0.1 * weight.shape[1] * (weight.shape[1] - 1) / 2
        pairs = list(reversed(pairs))[: int(num_pairs)]
        angles = torch.nn.Parameter(torch.zeros(len(pairs), device=device))
        idx_i = torch.tensor([x[0] for x in pairs], device=device)
        idx_j = torch.tensor([x[1] for x in pairs], device=device)

        losses2 = optimize(angles, idx_i, idx_j, lr=0.01)

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
        bar = tqdm(total=steps)
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
        weight_grouped = weight.view(weight.shape[0], -1, GROUP_SIZE).permute(1, 0, 2)
        all_pairs = get_rotation_pairs(
            weight_grouped,
            group_size=GROUP_SIZE,
            num_rotations=8,
            num_pairs_factor=0.5,
        )
        all_pairs = [
            torch.tensor(pairs, device=device, dtype=torch.int32) for pairs in all_pairs
        ]
        angles = [
            torch.zeros(pairs.shape[0], device=device, requires_grad=True)
            for pairs in all_pairs
        ]
        scales = torch.ones(1, weight.shape[1], device=device, requires_grad=True)
        loss_re = optimize_requant(all_pairs, angles, scales, lr=0.02)

    if not args.no_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        output_dict = {
            "losses1": losses1,
            "losses2": losses2,
            "losses3": losses3,
            "og_loss": og_loss.item(),
        }
        if args.include_requant:
            output_dict["loss_re"] = loss_re
        with open(cache_file, "w") as f:
            json.dump(output_dict, f)


plt.figure(figsize=(3.6, 1.6), dpi=300)
plot_kwargs = {"linewidth": 1, "rasterized": True}
plt.plot(losses3, label="Channel-wise scaling", color="C2", **plot_kwargs)
plt.plot(losses1, label="Rotation", color="C0", **plot_kwargs)
plt.plot(losses2, label="Rotation (10% pairs)", color="C1", **plot_kwargs)
if args.include_requant:
    plt.plot(loss_re, label="ReQuant", color="C4", **plot_kwargs)
plt.ylim(0, og_loss.item() * 1.1)

plt.rcParams.update({"legend.fontsize": 7})


def sci_notation(x, pos):
    if x == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(x))))
    coeff = x / 10**exponent
    assert coeff == 1
    return r"$10^{{{}}}$".format(exponent)


if not args.no_custom_ticks:
    plt.gca().yaxis.set_major_formatter(FuncFormatter(sci_notation))
    plt.yticks([0, 1e-3])
else:
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

if not args.include_requant:
    plt.axhline(
        y=og_loss.item(), color="gray", linestyle="--", label="RTN", zorder=0, alpha=0.5
    )
plt.xlabel("Steps")
plt.ylabel("Output Error")
plt.xticks([0, 100, 200])
plt.legend(frameon=False)
plt.grid(False)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = get_file_path(output_dir, file_stem)

plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
plt.close()
