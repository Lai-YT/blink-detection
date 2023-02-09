from __future__ import annotations

import cv2
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, List

import imutils

from util.image_type import ColorImage


class BlinkVideoAnnotator:
    def __init__(self, filename: str) -> None:
        self._video_file: Path = (Path.cwd() / filename).resolve()
        self._video = cv2.VideoCapture(
            str(self._video_file)
        )
        self._num_of_frames_that_blink: List[int] = []

    def start_annotating(self) -> None:
        print(f"Annotating blinks on {self._video_file.name}, which has "
              f"{self._video.get(cv2.CAP_PROP_FRAME_COUNT)} frames...")

        for frame_no, frame in enumerate(self._frames_from_video()):
            cv2.destroyAllWindows()
            self._show_frame_in_middle_of_screen(frame_no, frame)
            self._annotate_frame_num_by_key(frame_no)

    def get_annotations(self) -> List[int]:
        return self._num_of_frames_that_blink.copy()

    def write_annotations(self) -> None:
        self._generate_json_file_path()

        print(f"Writing the annotations into {self._json_file.name}...")

        self._json_file.write_text(
            json.dumps(self._num_of_frames_that_blink, indent=4)
        )

    def __enter__(self) -> BlinkVideoAnnotator:
        return self

    def __exit__(self, *exc_info) -> None:
        self.write_annotations()

    @staticmethod
    def _show_frame_in_middle_of_screen(frame_no: int, frame: ColorImage) -> None:
        win_name = f"no. {frame_no}"
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, 250, 80)
        cv2.imshow(win_name, imutils.resize(frame, width=900))

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
                break
            yield frame

    def _generate_json_file_path(self) -> None:
        def add_trailing_timestamp_to(text: str) -> str:
            return f"{text}-{get_timestamp()}"

        def get_timestamp() -> str:
            return datetime.now().strftime("%Y%m%d-%H%M%S")

        stem = add_trailing_timestamp_to(f"ann-{self._video_file.stem}")
        self._json_file = (Path(__file__).parent / stem).with_suffix(".json")


def main(video_to_annotate: str) -> None:
    with BlinkVideoAnnotator(video_to_annotate) as annotator:
        annotator.start_annotating()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="the video to perform annotation on")
    args = parser.parse_args()

    main(args.video)
