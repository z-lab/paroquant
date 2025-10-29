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
import torch.nn as nn
from copy import deepcopy

sys.path.extend(
    [Path(__file__).parent.as_posix(), Path(__file__).parents[2].as_posix()]
)


from paroquant.quantizer import UniformAffineQuantizer
from paroquant.util import (
    get_blocks,
    catch_first_layer_input,
    get_calib_dataset,
    load_tokenizer,
    get_module_by_name,
    empty_cache,
    get_named_linears,
)
from paroquant.optimize import get_random_rotation_pairs
from paroquant.module import PseudoQuantizedLinear
from paroquant.convert_utils import transform_to_kernel_data

try:
    from hadamard_utils import random_hadamard_matrix
except ImportError:
    random_hadamard_matrix = None

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
parser.add_argument("--include", type=str, default="all")
parser.add_argument("--no-custom-ticks", action="store_true")
parser.add_argument("--no-labels", action="store_true")
parser.add_argument("--grid", action="store_true")
parser.add_argument("--linewidth", type=float, default=1.0)
parser.add_argument("--file-name", type=str, default=None)
parser.add_argument("--figsize", type=str, default="3.6,1.6")
parser.add_argument("--legend", action="store_true")

args = parser.parse_args()


@torch.no_grad()
def catch_linear_input_and_layer_output(
    layer: nn.Module,
    input_batches: list[torch.Tensor],
    kwargs: dict,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    linear_inputs = {}
    layer_output_batches = []

    def _get_hook_fn(name: str, inputs_dict: dict) -> callable:
        def hook_fn(module, input, output):
            inputs_dict.setdefault(name, []).append(input[0].cpu())

        return hook_fn

    hooks = []
    for name, module in get_named_linears(layer).items():
        handle = module.register_forward_hook(_get_hook_fn(name, linear_inputs))
        hooks.append(handle)

    for input in input_batches:
        layer_output = layer(input, **kwargs)[0]
        layer_output_batches.append(layer_output.cpu())
        empty_cache()

    for hook in hooks:
        hook.remove()

    return linear_inputs, layer_output_batches


@torch.no_grad()
def _get_sorted_sensitive_channel_pairs(
    tensor: torch.Tensor,
) -> torch.Tensor:
    assert tensor.dim() == 3  # (batch, seq, channel)
    batch_size, _, num_channels = tensor.shape

    # Calculate the variance of each channel
    variances = tensor.var(dim=1)  # (batch, channel)
    epsilon = torch.finfo(variances.dtype).eps
    all_ratios = []

    for i in range(num_channels - 1):
        channel_var = variances[:, i].unsqueeze(1)  # (batch, 1)
        other_channels_vars = variances[:, i + 1 :]  # (batch, channel - 1)
        # Calculate the ratio of variances to other channels (sensitivity)
        ratios = other_channels_vars / (channel_var + epsilon)  # (batch, channel - 1)
        ratios = torch.maximum(ratios, 1.0 / (ratios + epsilon))  # (batch, channel - 1)

        # Stack the indices and ratios
        current_pair_count = ratios.shape[1]
        i_indices = torch.full(
            (batch_size, current_pair_count), i, device=tensor.device, dtype=torch.long
        )
        j_indices = (
            torch.arange(i + 1, num_channels, device=tensor.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        stacked = torch.stack(
            [i_indices.to(ratios.dtype), j_indices.to(ratios.dtype), ratios], dim=-1
        )  # (batch, current_pair_count, 3)
        all_ratios.append(stacked)

    all_ratios = torch.cat(all_ratios, dim=1)  # (batch, pair_num, 3)

    # Sort the pairs by sensitivity
    ratios_only = all_ratios[..., 2]  # (batch, pair_num)
    sorted_indices = torch.argsort(
        ratios_only, dim=1, descending=True
    )  # (batch, pair_num)
    expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)

    sorted_pairs = torch.gather(
        all_ratios, dim=1, index=expanded_indices
    )  # (batch, pair_num, 3)
    sorted_pairs = sorted_pairs[..., :2].long()

    return sorted_pairs


model_name = args.model.split("/")[-1]

file_stem = (
    f"loss_curves_{model_name.replace('-', '_')}_{args.layer}_{args.linear_name}"
)
file_stem = file_stem if args.file_name is None else args.file_name
cache_file = Path(args.output_dir) / "cache" / (file_stem + ".json")

if not args.no_cache and cache_file.exists():
    with open(cache_file, "r") as f:
        data = json.load(f)
else:
    data = {}

args.include = args.include.split(",")
if args.include[0] == "all":
    args.include = [
        "full_rotation",
        "partial_rotation",
        "scaling",
        "paroquant_no_scaling",
        "paroquant",
        "hadamard",
    ]

if "hadamard" in args.include and random_hadamard_matrix is None:
    print(
        "hadamard_utils not found, hadamard will not work. "
        "Please download the script from "
        "https://raw.githubusercontent.com/facebookresearch/SpinQuant/refs/heads/main/utils/hadamard_utils.py"
        " into this directory."
    )
    args.include.remove("hadamard")


results_exist = all(key in data for key in args.include)
if not results_exist:
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

    with torch.no_grad():
        og_layer_input_batches, kwargs = catch_first_layer_input(
            model,
            blocks,
            samples,
            batch_size=None,
        )
        for i in range(args.layer):
            og_layer_input_batches = blocks[i](og_layer_input_batches, **kwargs)

        linear_inputs, _ = catch_linear_input_and_layer_output(
            blocks[args.layer], [og_layer_input_batches], kwargs
        )

    x = linear_inputs[args.linear_name][0].float().to(device)
    linear = get_module_by_name(model.model.layers[0], args.linear_name).to(device)
    weight = linear.weight.float()
    y = x @ weight.T
    og_weight_q = UniformAffineQuantizer.pseudo_quantize(weight, BITS, GROUP_SIZE)
    yy = x @ og_weight_q.T
    og_loss = (yy - y).pow(2).mean()

    steps = args.steps

    def optimize_full(ratio: float, lr: float) -> list[float]:
        if ratio == 1.0:
            pairs = []
            for i in range(0, weight.shape[1]):
                for j in range(i + 1, weight.shape[1]):
                    pairs.append((i, j))
        else:
            weight_grouped = weight.view(weight.shape[0], -1, weight.shape[1]).permute(
                1, 0, 2
            )
            pairs_all = _get_sorted_sensitive_channel_pairs(weight_grouped)[0]
            num_pairs = ratio * weight.shape[1] * (weight.shape[1] - 1) / 2
            pairs = list(reversed(pairs_all.tolist()))[: int(num_pairs)]

        angles = torch.nn.Parameter(torch.zeros(len(pairs), device=device))
        idx_i = torch.tensor([x[0] for x in pairs], device=device)
        idx_j = torch.tensor([x[1] for x in pairs], device=device)

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

    def optimize_scaling(lr: float) -> list[float]:
        scales = torch.nn.Parameter(torch.ones(1, weight.shape[1], device=device))
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

    def optimize_paroquant(include_scaling: bool, lr: float) -> list[float]:
        weight_grouped = weight.view(weight.shape[0], -1, GROUP_SIZE).permute(1, 0, 2)
        all_pairs = get_random_rotation_pairs(
            weight_grouped,
            group_size=GROUP_SIZE,
            num_rotations=8,
            num_pairs_factor=0.5,
            seed=0,
        )
        all_pairs = [
            torch.tensor(pairs, device=device, dtype=torch.int32) for pairs in all_pairs
        ]
        angles = [
            torch.zeros(pairs.shape[0], device=device, requires_grad=True)
            for pairs in all_pairs
        ]
        if include_scaling:
            scales = torch.ones(1, weight.shape[1], device=device, requires_grad=True)
        else:
            scales = None

        npairs, angles, mask = transform_to_kernel_data(
            all_pairs,
            angles,
            group_size=GROUP_SIZE,
        )
        rotation_pairs = [npairs, angles, mask]

        linear_cp = deepcopy(linear).to(torch.float32)
        m = PseudoQuantizedLinear(
            linear_cp,
            rotation_pairs,
            scales,
            group_size=GROUP_SIZE,
            n_bits=BITS,
            num_rotations=8,
        )
        m.set_optim_enabled(angles=True, channel_scales=include_scaling)

        optim_params = m.get_optim_params("angles")
        if include_scaling:
            optim_params += m.get_optim_params("channel_scales")
        optimizer = torch.optim.AdamW(optim_params, lr=lr)
        losses = []
        bar = tqdm(total=steps)
        for i in range(steps):
            optimizer.zero_grad()
            y = linear_cp(x)
            yy = m(x)
            loss = (yy - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            bar.update(1)
            bar.set_postfix(loss=loss.item(), og_loss=og_loss.item())

        return losses

    def hadamard(num_seeds: int) -> float:
        losses = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            H = (
                random_hadamard_matrix(weight.shape[1], seed=seed, device=device)
                .to(device)
                .to(weight.dtype)
            )
            weight_h = weight @ H
            weight_q = UniformAffineQuantizer.pseudo_quantize(
                weight_h, BITS, GROUP_SIZE
            )
            weight_q = weight_q @ H.T
            yy = x @ weight_q.T
            loss = (yy - y).pow(2).mean()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        return avg_loss

    fn_map = {
        "full_rotation": [optimize_full, (1.0, 0.001)],
        "partial_rotation": [optimize_full, (0.1, 0.01)],
        "scaling": [optimize_scaling, (0.01,)],
        "paroquant": [optimize_paroquant, (True, 0.01)],
        "paroquant_no_scaling": [optimize_paroquant, (False, 0.01)],
        "hadamard": [hadamard, (100,)],
    }

    for key in args.include:
        if key not in data:
            fn, fn_args = fn_map[key]
            print(f"Optimizing {key}...")
            losses = fn(*fn_args)
            data[key] = losses
    data["og_loss"] = og_loss.item()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f)

name_map = {
    "full_rotation": "Full rotation",
    "partial_rotation": "Rotation (10% pairs)",
    "scaling": "Channel-wise scaling",
    "paroquant": "8 independent rotations + scaling",
    "paroquant_no_scaling": "8 independent rotations",
    "hadamard": "Hadamard",
    "og_loss": "RTN",
}

figsize = tuple(float(s) for s in args.figsize.split(","))
plt.figure(figsize=figsize, dpi=300)
plot_kwargs = {"linewidth": args.linewidth, "rasterized": True}
colors = ["C0", "C1", "C2", "C5", "C6"]

for key, color in zip(args.include, colors):
    result = data[key]
    if isinstance(result, list):
        plt.plot(
            result,
            label=name_map[key],
            **plot_kwargs,
            color=color,
        )

for key in args.include:
    result = data[key]
    if isinstance(result, float):
        plt.axhline(
            result,
            label=name_map[key],
            **plot_kwargs,
            linestyle="--",
            color="gray",
            alpha=0.5,
        )

plt.ylim(0, data["og_loss"] * 1.1)
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

if not args.no_labels:
    plt.xlabel("Steps")
    plt.ylabel("Output Error")
    plt.legend(frameon=False)

plt.xticks([0, 100, 200])
if args.grid:
    plt.grid(args.grid, alpha=0.3, linestyle="--", zorder=-2)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = get_file_path(output_dir, file_stem)

legend_handles_labels = None
if args.legend:
    legend_handles_labels = plt.gca().get_legend_handles_labels()

plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
plt.close()

print(f"Figure saved to {output_filename}")

if args.legend:
    handles, labels = legend_handles_labels
    legend_fig, legend_ax = plt.subplots(figsize=(0.1, 0.1))
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        ncol=3,
        borderaxespad=0,
        borderpad=0.0,
    )
    legend_filename = output_filename.with_name(
        f"{output_filename.stem}_legend{output_filename.suffix}"
    )
    legend_fig.tight_layout(pad=0)
    legend_fig.savefig(legend_filename, bbox_inches="tight", pad_inches=0)
    plt.close(legend_fig)
    print(f"Legend saved to {legend_filename}")
