import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float, NDArray


def read_ratio_and_blinks(filename: str) -> Dict[str, Union[List[float],
                                                            List[Union[float, str]]]]:
    ratios: List[float] = []
    # non-blinks are marked with "-" to have indices matched with ratios
    blinks: List[Union[float, str]] = []
    thress: List[float] = []

    with open(filename, mode="r") as f:
        for line in f:
            if line.startswith("*"):
                line = line.lstrip("*")
                # rewrite, consider the one berfore the last one
                # the middle of blink
                blinks[-2] = ratios[-2]
            blinks.append("-")
            ratio, thres = map(float, line.split())
            ratios.append(ratio)
            thress.append(thres)
    return {"ratios": ratios, "blinks": blinks, "thress": thress}


def calculate_rolling_mean(
        seq: Sequence[float],
        win_size: int) -> NDArray[(Any,), Float[64]]:
    # means are constant if the sequence is shorter than the size of window
    if len(seq) <= win_size:
        return np.full(len(seq), np.mean(seq), dtype=np.float64)

    sum = math.fsum(seq[:win_size])
    means = np.full(len(seq), sum / win_size, dtype=np.float64)
    for i in range(win_size, len(seq)):
        sum -= seq[i - win_size]
        sum += seq[i]
        means[i] = sum / win_size
    return means


def plot_ratio_and_blinks(ratios: List[float], blinks: List[Union[float, str]],
                          thress: List[float], title: str = "",
                          show: bool = True) -> None:
    # standard deviation
    std = np.std(ratios)

    fig, ax = plt.subplots()
    ax.set_title(f"{title}; {std:.4f}")

    ratio_range = (0.1, 0.4)

    # plot ratios
    ax.plot(np.arange(1, len(ratios)+1), ratios, linewidth=1.2)

    # plot blinks
    pos = []
    blk = []
    for i, b in enumerate(blinks):
        if b != "-":
            pos.append(i + 1)
            blk.append(b)

    ax.scatter(pos, blk, color="r")

    # plot thress
    ax.plot(np.arange(1, len(thress)+1), thress, color=mcolors.CSS4_COLORS["aqua"], ls="--", linewidth=1)

    ax.set(xlim=(1, len(ratios)+1),
           ylim=ratio_range, yticks=np.arange(*ratio_range, 0.02))

    ax.legend(["ratio", "blink"])

    # ratio avg line
    ax.plot(np.arange(1, len(thress)+1), calculate_rolling_mean(ratios, 1500),
            color=mcolors.CSS4_COLORS["lime"], ls="--", linewidth=1)
    # # blink avg line
    # mean_of_non_blinks = (np.sum(ratios) - np.sum(blk)) / (len(ratios) - len(blk))
    # ax.axhline(y=mean_of_non_blinks, color="black", ls="--", linewidth=1)

    if show:
        plt.show()
    else:
        dir_path = Path.cwd() / "plots"
        dir_path.mkdir(exist_ok=True)
        plt.savefig(dir_path / f"{title}.png", dpi=150)


if __name__ == "__main__":
    if len(sys.argv) != 1 and sys.argv[1] == "show":
        path = (Path.cwd() / "ratio.txt").resolve()
        ratio_and_blinks = read_ratio_and_blinks(str(path))
        plot_ratio_and_blinks(**ratio_and_blinks, title=path.stem)  # type: ignore
    else:
        dir_path = Path.cwd() / "samples"
        for path in dir_path.iterdir():
            ratio_and_blinks = read_ratio_and_blinks(str(path))
            plot_ratio_and_blinks(**ratio_and_blinks, title=path.stem, show=False)  # type: ignore
