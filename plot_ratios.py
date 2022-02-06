import sys

import matplotlib.pyplot as plt
import numpy as np


def read_ratio_and_blinks(filename):
    ratios = []
    blinks = []

    with open(filename, mode="r") as f:
        for line in f:
            if line.startswith("*"):
                line = line.lstrip("*")
                # rewrite, consider the one berfore the last one
                # the middle of blink
                blinks[-2] = ratios[-2]
            blinks.append("-")
            ratios.append(float(line))
    return {"ratios": ratios, "blinks": blinks}


def plot_ratio_and_blinks(ratios, blinks):
    fig, ax = plt.subplots()
    ax.set_title("Ratio and Blinks")

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

    ax.set(xlim=(1, len(ratios)+1),
           ylim=ratio_range, yticks=np.arange(*ratio_range, 0.02))

    ax.legend(["ratio", "blink"])

    # average line
    ax.axhline(y=np.mean(blk), color="black", ls="--", linewidth=1)

    mean_of_non_blinks = (np.sum(ratios) - np.sum(blk)) / (len(ratios) - len(blk))
    ax.axhline(y=mean_of_non_blinks, color="black", ls="--", linewidth=1)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        file = "./ratio.txt"
    else:
        file = "./samples/" + sys.argv[1]
    ratio_and_blinks = read_ratio_and_blinks(file)
    plot_ratio_and_blinks(**ratio_and_blinks)
