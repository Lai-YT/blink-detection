import json
import math
import sys
from itertools import filterfalse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from nptyping import Float, NDArray


class RatioPlotter:
    # markers are from external file,
    # fillers are used for internal representation
    MISS_FACE_MARKER = "x"
    BLINK_MARKER = "*"
    # np.nan provides "skipping" effect when plotting
    NON_BLINK_FILLER = np.nan
    MISS_FACE_FILLER = np.nan

    def __init__(self, output_dir: Path) -> None:
        self._ratios: List[Union[float]] = []
        self._blinks: List[Union[float]] = []
        self._thress: List[Union[float]] = []
        self._means: List[Union[float]] = []
        self._output_dir = output_dir

    def read_samples_from(self, filename: str) -> None:
        self._filename = filename
        self._stem = Path(filename).stem

        self._get_win_size_from_filename(fallback=500)
        self._clear_samples()
        self._update_samples()
        self._sample_size = len(self._ratios)

    def _update_samples(self) -> None:
        with open(self._filename, mode="r") as f:
            self._parse_samples_from(f)

    def _parse_samples_from(self, fp: TextIO) -> None:
        for line in fp:
            self._curr_line_to_parse = line

            if self._line_has_miss_face_marker():
                self._append_miss_face_filler()
                continue

            if self._line_has_blink_marker():
                self._remove_marker_from_line()
                self._fill_ratio_back_to_blink()
            self._append_non_blink_filler()
            self._append_ratio_and_thres_from_line()

    def _append_non_blink_filler(self) -> None:
        self._blinks.append(self.NON_BLINK_FILLER)

    def _append_miss_face_filler(self) -> None:
        self._ratios.append(self.MISS_FACE_FILLER)
        self._thress.append(self.MISS_FACE_FILLER)
        self._blinks.append(self.MISS_FACE_FILLER)
        self._means.append(np.nan)

    def _append_ratio_and_thres_from_line(self) -> None:
        ratio, thres, *mean = map(float, self._curr_line_to_parse.split())
        self._ratios.append(ratio)
        self._thress.append(thres)
        if mean:
            self._means.append(mean[0])
        else:
            self._means.append(np.nan)

    def _fill_ratio_back_to_blink(self) -> None:
        # consider the one before the last one the middle of blink
        self._blinks[-2] = self._ratios[-2]

    def _line_has_miss_face_marker(self) -> bool:
        return self._curr_line_to_parse.startswith(self.MISS_FACE_MARKER)

    def _line_has_blink_marker(self) -> bool:
        return self._curr_line_to_parse.startswith(self.BLINK_MARKER)

    def _is_miss_face_filler(self, x) -> bool:
        return x == self.MISS_FACE_FILLER

    def _is_non_blink_filler(self, x) -> bool:
        return x == self.NON_BLINK_FILLER

    def _remove_marker_from_line(self) -> None:
        self._curr_line_to_parse = (
            self._curr_line_to_parse.lstrip(self.BLINK_MARKER)
        )

    def _clear_samples(self) -> None:
        self._ratios.clear()
        self._thress.clear()
        self._blinks.clear()
        self._means.clear()

    def plot(self, show: bool = True) -> None:
        fig, self._ax = plt.subplots()
        std = np.std(list(filterfalse(np.isnan, self._ratios)))
        self._ax.set_title(f"{self._stem}; {std:.4f}")

        self._plot_ratios()
        self._plot_rolling_stds()
        # self._plot_thress()
        # self._plot_means()
        # self._plot_blinks()
        self._plot_annotate_blinks_if_exist()
        self._set_limit_and_ticks()
        self._ax.legend()

        if show:
            plt.show()
        else:
            self._output_dir.mkdir(exist_ok=True)
            plt.savefig(self._output_dir / f"{self._stem}.png", dpi=150)
            plt.close()  # so can be garbage collected

    def _plot_rolling_stds(self) -> None:
        def _with_window_size(n: int):
            ratios = np.array(self._ratios)
            r_stds = [np.nan] * (n - 1)
            for i in range((n - 1), len(ratios)):
                r_stds.append(0.32 + np.std(ratios[i-(n-1):i+1]))
            return r_stds
        r_stds = _with_window_size(10)
        plt.axhline(0.32, color="black", alpha=0.5)
        for i in range(1, len(r_stds)):
            if (r_stds[i] - r_stds[i-1])*100 > 0.8:
                plt.axvline(i, color=mcolors.CSS4_COLORS["indigo"], alpha=0.3)
        self._r_stds_line, = self._ax.plot(np.arange(len(r_stds)),
                                           r_stds,
                                           color="r",
                                           linewidth=1,
                                           label="r_std")

    def _plot_ratios(self) -> None:
        self._ratios_line, = self._ax.plot(np.arange(self._sample_size),
                                           self._ratios,
                                           linewidth=1,
                                           label="ratio")

    def _plot_thress(self) -> None:
        self._thress_line, = self._ax.plot(np.arange(self._sample_size),
                                           self._thress,
                                           color=mcolors.CSS4_COLORS["aqua"],
                                           ls="--", linewidth=1.2,
                                           label="thres")

    def _plot_means(self) -> None:
        self._means_line, = self._ax.plot(np.arange(self._sample_size),
                                          self._means,
                                          color=mcolors.CSS4_COLORS["lime"],
                                          ls="--", linewidth=1.2,
                                          label="mean")
    def _plot_blinks(self) -> None:
        self._detect_dot = self._ax.scatter(np.arange(self._sample_size),
                                             self._blinks,
                                             color="g", alpha=0.5,
                                             label="detect")

    def _plot_annotate_blinks_if_exist(self) -> None:
        annotate_blink_path = self._get_annotate_blink_path()
        if not annotate_blink_path.exists():
            return

        annotate_blinks: List[int] = json.loads(
            annotate_blink_path.read_text()
        )
        self._real_dot = self._ax.scatter(annotate_blinks,
                                          [self._ratios[i] for i in annotate_blinks],
                                          color="r", alpha=0.5,
                                          label="real")

    def _set_limit_and_ticks(self) -> None:
        ratio_range = (0.1, 0.4)
        self._ax.set(xlim=(1, self._sample_size),
                     ylim=ratio_range, yticks=np.arange(*ratio_range, 0.02))

    def _get_annotate_blink_path(self) -> Path:
        # The annotate file is under the "video" folder.
        return Path.cwd() / "video" / self._get_annotate_blink_filename()

    def _get_annotate_blink_filename(self) -> str:
        # The annotate file has a suffix of "_no".
        return f"{self._get_stem_without_info_encoding()}_no.json"

    def _get_stem_without_info_encoding(self) -> str:
        # The original stem of filename encode info to indicate the
        # window size used in this specific detection.
        info = self._stem.split("_")
        # first 2 is the pure name
        return "_".join(info[:2])

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
    plotter.read_samples_from(str(file_path))
    show = (len(sys.argv) == 3)
    plotter.plot(show=show)
    # plotter = RatioPlotter(output_dir=(Path.cwd() / "plots"))
    # for path in (Path.cwd() / "video").iterdir():
    #     if path.suffix == ".txt":
    #         plotter.read_samples_from(str(path))
    #         plotter.plot()
