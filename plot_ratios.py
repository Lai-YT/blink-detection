import json
from itertools import filterfalse
from pathlib import Path
from typing import List, TextIO, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from detector import BlinkDetector


class RatioPlotter:
    # markers are from external file,
    # fillers are used for internal representation
    MISS_FACE_MARKER = "x"
    BLINK_MARKER = "*"
    # np.nan provides "skipping" effect when plotting
    MISS_FACE_FILLER = np.nan

    def __init__(self, output_dir: Path) -> None:
        self._ratios: List[Union[float, str]] = []
        # frame numbers of blinks
        self._blink_nos: List[Union[float, str]] = []
        self._output_dir = output_dir

    def read_samples_from(self, filename: str) -> None:
        self._filename = filename
        self._stem = Path(filename).stem

        self._clear_samples()
        self._update_samples()
        self._sample_size = len(self._ratios)

    def _update_samples(self) -> None:
        with open(self._filename, mode="r") as f:
            self._parse_samples_from(f)

    def _parse_samples_from(self, fp: TextIO) -> None:
        for i, line in enumerate(fp):
            self._line_to_parse: str = line

            if self._line_has_miss_face_marker():
                self._append_miss_face_filler()
                continue

            if self._line_has_blink_marker():
                self._remove_marker_from_line()
                self._blink_nos.append(i)

            # only ratio remaining
            ratio = float(self._line_to_parse)
            self._ratios.append(ratio)

    def _append_miss_face_filler(self) -> None:
        self._ratios.append(self.MISS_FACE_FILLER)
        self._blink_nos.append(self.MISS_FACE_FILLER)

    def _line_has_miss_face_marker(self) -> bool:
        return self._line_to_parse.startswith(self.MISS_FACE_MARKER)

    def _line_has_blink_marker(self) -> bool:
        return self._line_to_parse.startswith(self.BLINK_MARKER)

    def _remove_marker_from_line(self) -> None:
        self._line_to_parse = (
            self._line_to_parse.lstrip(self.BLINK_MARKER)
        )

    def _clear_samples(self) -> None:
        self._ratios.clear()
        self._blink_nos.clear()

    def plot(self, show: bool = True) -> None:
        _, self._ax = plt.subplots()
        std = np.std(list(filterfalse(np.isnan, self._ratios)))
        self._ax.set_title(f"{self._stem}; {std:.4f}")

        self._plot_ratios()
        self._plot_blinks()
        self._plot_rolling_stds()
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
        OFFSET_FOR_SEP = 0.32  # to show EAR and STD in the same plot but not overlapped

        def _roll_with_window_size(n: int) -> Tuple[List[float], List[float]]:
            r_stds = [np.nan] * (n - 1)
            r_means = [np.nan] * (n - 1)
            for i in range((n - 1), len(self._ratios)):
                # to find change point
                r_stds.append(np.std(self._ratios[i-(n-1):i+1]) + OFFSET_FOR_SEP)
                # to detet increasing/decreasing
                r_means.append(np.mean(self._ratios[i-(n-1):i+1]))
            return r_stds, r_means

        # Default to plot with the parameters used by BlinkDetector.
        # You may adjust these critical parameters in the following section to compare the
        # difference in sensitivity.
        window_size = BlinkDetector.WINDOW_SIZE
        dramatic_std_change = BlinkDetector.DRAMATIC_STD_CHANGE
        r_stds, r_means = _roll_with_window_size(window_size)
        self._ax.axhline(OFFSET_FOR_SEP, color="black", alpha=0.5)
        # for i in range(1, len(r_stds)):
        #     if (r_stds[i] - r_stds[i-1]) > dramatic_std_change:  # is change point
        #         if r_means[i] - r_means[i-1] < 0:  # is decreasing
        #             color = "yellow"
        #         else:
        #             color = "orange"
        #         self._ax.axvline(i, color=mcolors.CSS4_COLORS[color], alpha=0.5)

        self._ax.plot(
            np.arange(len(r_stds)), r_stds,
            color="r",
            linewidth=1,
            label="r_std"
        )

    def _plot_ratios(self) -> None:
        self._ax.plot(
            np.arange(self._sample_size), self._ratios,
            linewidth=1,
            label="ratio"
        )

    def _plot_blinks(self) -> None:
        self._ax.vlines(
            self._blink_nos,
            ymin=0, ymax=1,
            color=mcolors.CSS4_COLORS["indigo"],
            alpha=0.5
        )

    def _plot_annotate_blinks_if_exist(self) -> None:
        annotate_blink_path = self._get_annotate_blink_path()
        if not annotate_blink_path.exists():
            return

        annotate_blinks: List[int] = json.loads(
            annotate_blink_path.read_text()
        )
        self._ax.scatter(
            annotate_blinks, [self._ratios[i] for i in annotate_blinks],
            color="r", alpha=0.5,
            label="real"
        )

    def _set_limit_and_ticks(self) -> None:
        ratio_range = (0.1, 0.4)
        self._ax.set(xlim=(1, self._sample_size),
                     ylim=ratio_range, yticks=np.arange(*ratio_range, 0.02))

    def _get_annotate_blink_path(self) -> Path:
        # The annotate file is under the "video" folder.
        return Path(__file__).parent / "video" / self._get_annotate_blink_filename()

    def _get_annotate_blink_filename(self) -> str:
        # The annotate file has a suffix of "_no".
        return f"{self._stem}_no.json"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        help="the eye aspect ratio file generated by main.py",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--show",
        action="store_true",
        help="show the plot instead of storing it",
    )
    group.add_argument(
        "--output",
        default=Path(__file__).parent / "plots",
        help="the directory where the plot is stored to",
    )
    args = parser.parse_args()

    plotter = RatioPlotter(output_dir=Path(args.output))
    plotter.read_samples_from(args.file)
    plotter.plot(show=args.show)
