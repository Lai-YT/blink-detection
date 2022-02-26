from __future__ import annotations

import cv2
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import imutils

from util.image_type import ColorImage


class BlinkVideoAnnotator:
    def __init__(self, filename: str) -> None:
        self._file: Path = (Path.cwd() / filename).resolve()
        self._video = cv2.VideoCapture(
            str(self._file)
        )
        self._num_of_frames_that_blink: List[int] = []

    def start_annotating(self) -> None:
        for frame_no, frame in enumerate(self._frames_from_video()):
            cv2.destroyAllWindows()
            cv2.imshow(f"no. {frame_no}", imutils.resize(frame, width=900))
            self._annotate_frame_num_by_key(frame_no)

    def get_annotations(self) -> List[int]:
        return self._num_of_frames_that_blink.copy()

    def write_annotations(self) -> None:
        def add_trailing_timestamp_to(text: str) -> str:
            print(text)
            return f"{text}-{get_timestamp()}"

        def get_timestamp() -> str:
            return datetime.now().strftime("%Y%m%d-%H%M%S")

        stem = add_trailing_timestamp_to(f"ann-{self._file.stem}")
        json_file = (Path.cwd() / stem).with_suffix(".json")
        json_file.write_text(
            json.dumps(self._num_of_frames_that_blink, indent=4)
        )

    def __enter__(self) -> BlinkVideoAnnotator:
        return self

    def __exit__(self, *exc_info) -> None:
        self.write_annotations()

    def _annotate_frame_num_by_key(self, frame_no: int) -> None:
        while True:
            self._read_key()
            if self._is_blink_key():
                self._num_of_frames_that_blink.append(frame_no)
            if self._is_valid_key():
                break

    def _read_key(self) -> None:
        self._key = cv2.waitKey(1) & 0xFF

    def _is_valid_key(self) -> bool:
        return chr(self._key) in ("o", "x")

    def _is_blink_key(self) -> bool:
        return self._key == ord("o")

    def _is_non_blink_key(self) -> bool:
        return self._key == ord("x")

    def _frames_from_video(self) -> Iterator[ColorImage]:
        while self._video.isOpened():
            ret, frame = self._video.read()
            if not ret:
                raise StopIteration
            yield frame


def main(video_to_annotate: str) -> None:
    with BlinkVideoAnnotator(video_to_annotate) as annotator:
        annotator.start_annotating()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"\n\t usage: python {__file__} $(video_to_annotate)")

    main(sys.argv[1])
