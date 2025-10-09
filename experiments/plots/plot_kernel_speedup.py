import matplotlib.pyplot as plt
from pathlib import Path
from plot_init import *

row_sizes = [4096, 8192, 16384, 32768]

speedup_data = {
    1: [1.36, 2.06, 3.30, 5.18],
}


fig, ax = plt.subplots(figsize=(1.8, 1.2))

for rcount, values in speedup_data.items():
    ax.plot(row_sizes, values, marker=".")


ax.set_ylabel("Speedup")
ax.set_xlabel("Channel dimension")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xticks(row_sizes)
ax.set_xscale("log", base=2)
ax.set_ylim(1, max(speedup_data[1]) * 1.1)
plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8})

output_path = Path("figures")
output_path.mkdir(parents=True, exist_ok=True)
output_filename = get_file_path(output_path, "kernel_speedup")
plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
