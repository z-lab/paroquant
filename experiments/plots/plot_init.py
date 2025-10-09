import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
import shutil
import os

plt.style.use("ggplot")

if shutil.which("latex") and not os.environ.get("NO_LATEX", False):
    print("Using LaTeX...")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
        }
    )

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 9,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    }
)


def get_file_path(dir: Path, name: str) -> Path:
    if plt.rcParams.get("text.usetex", False):
        return dir / f"{name}.pgf"
    else:
        return dir / f"{name}.pdf"


__all__ = ["get_file_path"]
