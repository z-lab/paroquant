import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from pathlib import Path
from argparse import ArgumentParser
import sys

from fast_givens_transform import fast_givens_transform
from quant.model import KPseudoQuantizedLinear
from quant.optimize import _get_sorted_sensitive_channel_pairs

# Init plotting formats
sys.path.append(str(Path(__file__).resolve().parent))
from plot_init import *


parser = ArgumentParser()
parser.add_argument(
    "--optimize-result-path",
    required=True,
    type=str,
    help="Path to the optimized linear state dict.",
)
parser.add_argument(
    "--q_group_size",
    default=128,
    type=int,
    help="Group size for quantization.",
)
parser.add_argument(
    "--output-dir",
    default="figures",
    type=str,
    help="Directory to save the output figure.",
)
args = parser.parse_args()

print("Loading optimized results...")
results_sd = torch.load(args.optimize_result_path, map_location="cuda")
qlinear = KPseudoQuantizedLinear.from_state_dict(results_sd)
weight = qlinear.weight.detach()

print("Calculating original and rotated weights...")
weight_r = qlinear.weight.detach()
weight_r = weight_r * qlinear.channel_scales
weight_r = fast_givens_transform(
    weight_r, qlinear.pairs_grouped, qlinear.angles_grouped
)

print("Finding the most sensitive pair...")
weight_grouped = weight.view(weight.shape[0], -1, args.q_group_size).permute(1, 0, 2)
# all_pairs = _get_sorted_sensitive_channel_pairs(weight_grouped)
max_sensitivity = 0
max_idx = -1, -1
for rot_idx in range(qlinear.pairs_grouped.shape[0]):
    pairs = qlinear.pairs_grouped[rot_idx].reshape(-1, 2).cpu()
    for i, j in pairs:
        sensitivity = weight[:, i].var() / weight[:, j].var()
        sensitivity = max(sensitivity, 1 / sensitivity)
        if sensitivity > max_sensitivity:
            max_sensitivity = sensitivity
            max_idx = (i, j)

i, j = max_idx

channel_i = weight[:, i].clone()
channel_j = weight[:, j].clone()

channel_i_r = weight_r[:, i].clone()
channel_j_r = weight_r[:, j].clone()

all_data = torch.cat([channel_i, channel_j, channel_i_r, channel_j_r])
min_val = all_data.min().item()
max_val = all_data.max().item()

max_abs_val = max(abs(min_val), abs(max_val))
lim = max_abs_val * 1.1

print("Generating plots...")
output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(
    1, 2, figsize=(3.2, 1.6), dpi=300, gridspec_kw={"wspace": 0.04}
)
fig.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.9, wspace=0.1)

ax1 = axes[0]

min_i, max_i = channel_i.min().item(), channel_i.max().item()
min_j, max_j = channel_j.min().item(), channel_j.max().item()
width1 = max_i - min_i
height1 = max_j - min_j
padding = 0.05 * max_abs_val
rect1 = patches.Rectangle(
    (min_i - padding, min_j - padding),
    width1 + 2 * padding,
    height1 + 2 * padding,
    edgecolor="none",
    facecolor="gray",
    alpha=0.2,
)
ax1.add_patch(rect1)

ax1.scatter(
    channel_i.cpu().numpy(),
    channel_j.cpu().numpy(),
    s=0.5,
    alpha=0.5,
    rasterized=True,
)
ax1.set_title("Original")
ax1.set_ylabel(f"Channel {j}")

ax2 = axes[1]
min_i_r, max_i_r = channel_i_r.min().item(), channel_i_r.max().item()
min_j_r, max_j_r = channel_j_r.min().item(), channel_j_r.max().item()
width2 = max_i_r - min_i_r
height2 = max_j_r - min_j_r
rect2 = patches.Rectangle(
    (min_i_r - padding, min_j_r - padding),
    width2 + 2 * padding,
    height2 + 2 * padding,
    edgecolor="none",
    facecolor="gray",
    alpha=0.2,
)
ax2.add_patch(rect2)

ax2.scatter(
    channel_i_r.cpu().numpy(),
    channel_j_r.cpu().numpy(),
    s=0.5,
    alpha=0.5,
    color="C1",
    rasterized=True,
)
ax2.set_title("Transformed")

for ax in axes:
    ax.set_xlabel(f"Channel {i}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, zorder=0, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, zorder=0, alpha=0.5)

output_filename = get_file_path(output_path, "channel_pairs")
plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
print(f"Figure saved to: {output_filename}")
