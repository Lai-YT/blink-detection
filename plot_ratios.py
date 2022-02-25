import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, TextIO, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float, NDArray


class RatioPlotter:
    def __init__(self, output_dir: Path) -> None:
        self._ratios: List[float] = []
        # non-blinks are marked with "-" to have indices matched with ratios
        self._blinks: List[Union[float, str]] = []
        self._thress: List[float] = []
        self._output_dir = output_dir

    def read_ratio_and_thress_from(self, filename: str) -> None:
        self._filename = filename
        self._stem = Path(filename).stem

        self._get_win_size_from_filename(fallback=1500)
        self._clear_ratio_blink_and_thress()
        self._update_ratio_blink_and_thress()

    def _update_ratio_blink_and_thress(self) -> None:
        with open(self._filename, mode="r") as f:
            self._parse_ratio_blink_and_thress_from_fp(f)

    def _parse_ratio_blink_and_thress_from_fp(self, fp: TextIO) -> None:
        for line in fp:
            self._curr_line_to_parse = line

            if self._line_has_blink_marker():
                self._remove_marker_from_line()
                self._write_the_penultimate_ratio_to_blink()
            self._append_non_blink_marker()
            self._append_ratio_and_thres_from_line()

    def _append_non_blink_marker(self) -> None:
        self._blinks.append("-")

    def _append_ratio_and_thres_from_line(self) -> None:
        ratio, thres = map(float, self._curr_line_to_parse.split())
        self._ratios.append(ratio)
        self._thress.append(thres)

    def _write_the_penultimate_ratio_to_blink(self) -> None:
        # consider the one before the last one the middle of blink
        self._blinks[-2] = self._ratios[-2]

    def _line_has_blink_marker(self) -> bool:
        return self._curr_line_to_parse.startswith("*")

    def _remove_marker_from_line(self) -> None:
        self._curr_line_to_parse = self._curr_line_to_parse.lstrip("*")

    def _clear_ratio_blink_and_thress(self) -> None:
        self._ratios.clear()
        self._thress.clear()
        self._blinks.clear()

    def plot(self, show: bool = True) -> None:
        fig, self._ax = plt.subplots()
        self._ax.set_title(f"{self._stem}; {np.std(self._ratios):.4f}")

        self._plot_ratios()
        self._plot_blinks()
        self._plot_thress()
        self._set_limit_and_ticks()
        self._ax.legend(["ratio", "blink"])
        self._plot_ratio_avg_line()

        # # blink avg line
        # mean_of_non_blinks = (np.sum(ratios) - np.sum(blk)) / (len(ratios) - len(blk))
        # ax.axhline(y=mean_of_non_blinks, color="black", ls="--", linewidth=1)

        if show:
            plt.show()
        else:
            self._output_dir.mkdir(exist_ok=True)
            plt.savefig(self._output_dir / f"{self._stem}.png", dpi=150)
            plt.close()  # so can be garbage collected

    def _plot_ratios(self) -> None:
        self._ax.plot(np.arange(1, len(self._ratios)+1), self._ratios, linewidth=1.2)

    def _plot_thress(self) -> None:
        self._ax.plot(np.arange(1, len(self._thress)+1), self._thress,
                color=mcolors.CSS4_COLORS["aqua"], ls="--", linewidth=1)

    def _plot_blinks(self) -> None:
        pos = []
        blk = []
        for i, b in enumerate(self._blinks):
            if b != "-":
                pos.append(i + 1)
                blk.append(b)
        self._ax.scatter(pos, blk, color="r")

    def _set_limit_and_ticks(self) -> None:
        ratio_range = (0.1, 0.4)
        self._ax.set(xlim=(1, len(self._ratios)+1),
                     ylim=ratio_range, yticks=np.arange(*ratio_range, 0.02))

    def _plot_ratio_avg_line(self) -> None:
        self._ax.plot(np.arange(1, len(self._thress)+1),
                      self._calculate_rolling_mean(self._ratios),
                      color=mcolors.CSS4_COLORS["lime"], ls="--", linewidth=1)

    def _calculate_rolling_mean(
        self,
        seq: Sequence[float],
    ) -> NDArray[(Any,), Float[64]]:
        # means are constant if the sequence is shorter than the size of window
        if len(seq) <= self._win_size:
            return np.full(len(seq), np.mean(seq), dtype=np.float64)

        sum = math.fsum(seq[:self._win_size])
        means = np.full(len(seq), sum / self._win_size, dtype=np.float64)
        for i in range(self._win_size, len(seq)):
            sum -= seq[i - self._win_size]
            sum += seq[i]
            means[i] = sum / self._win_size
        return means

    def _get_win_size_from_filename(self, fallback: int) -> None:
        info = self._stem.split("_")
        win_size = fallback
        for p in info:
            if p.startswith("w"):
                win_size = int(p.strip("w"))
                break
        self._win_size = win_size


if __name__ == "__main__":
    if (len(sys.argv) not in (2, 3)
            or (len(sys.argv) == 3 and sys.argv[2] != "show")):
        raise RuntimeError(f"\n\t usage: python {__file__} ./$(file_path) [show]")

    file_path = Path.cwd() / sys.argv[1]
    plotter = RatioPlotter(output_dir=(Path.cwd() / "plots"))
    plotter.read_ratio_and_thress_from(str(file_path))
    show = (len(sys.argv) == 3)
    plotter.plot(show=show)
