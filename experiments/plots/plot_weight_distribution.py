import matplotlib.pyplot as plt
import torch
from pathlib import Path
from argparse import ArgumentParser
from matplotlib.colors import LogNorm
import sys

from paroquant_kernels import scaled_pairwise_rotation
from quant.model import KPseudoQuantizedLinear

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
    "--output-dir",
    default="figures",
    type=str,
    help="Directory to save the output figure.",
)
args = parser.parse_args()

print("Loading optimized results...")
results_sd = torch.load(args.optimize_result_path, map_location="cuda")
qlinear = KPseudoQuantizedLinear.from_state_dict(results_sd)

print("Calculating original and rotated weights...")
weight_r = qlinear.weight.detach()
weight_r = weight_r * qlinear.channel_scales
weight_r = scaled_pairwise_rotation(
    weight_r, qlinear.pairs_grouped, qlinear.angles_grouped
)
weight_r_abs = weight_r.T.cpu().abs().numpy()
og_weight_abs = qlinear.weight.detach().T.cpu().abs().numpy()

print("Generating plots...")
output_path = Path(args.output_dir)
output_path.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(
    1, 2, figsize=(3.2, 1.6), dpi=300, gridspec_kw={"wspace": 0.04}
)
fig.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.9, wspace=0.1)

vmin = min(og_weight_abs[og_weight_abs > 0].min(), weight_r_abs[weight_r_abs > 0].min())
vmax = max(og_weight_abs.max(), weight_r_abs.max())
norm = LogNorm(vmin=vmin, vmax=vmax)

ax1 = axes[0]
im1 = ax1.imshow(
    og_weight_abs,
    cmap="viridis",
    norm=norm,
    interpolation="nearest",
    aspect="auto",
)
ax1.set_title("Original")
ax1.set_ylabel("Channel")

ax2 = axes[1]
ax2.imshow(
    weight_r_abs,
    cmap="viridis",
    norm=norm,
    interpolation="nearest",
    aspect="auto",
)
ax2.set_title("Transformed")

for ax in axes:
    ax.set_xlabel("Token")
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

cax = fig.add_axes([0.82, 0.2, 0.04, 0.7])
fig.colorbar(im1, cax=cax)

output_filename = get_file_path(output_path, "weight_comparison")
plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
print(f"Figure saved to: {output_filename}")
