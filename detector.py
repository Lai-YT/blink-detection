import math
import statistics
from collections import deque
from typing import Deque, Tuple

from imutils import face_utils
from nptyping import Int, NDArray


class BlinkDetector:
    """Detects whether the eyes are blinking or not by calculating
    the eye aspect ratio (EAR).

    A window-based approach is used to detect the change points of EARs, which
    indicates a possible occurrence of blink.

    Attributes:
        LEFT_EYE_START_END_IDXS:
            The start and end index (end excluded) which represents the left
            eye in the 68 face landmarks.
        RIGHT_EYE_START_END_IDXS:
            The start and end index (end excluded) which represents the right
            eye in the 68 face landmarks.
    """

    LEFT_EYE_START_END_IDXS:  Tuple[int, int] = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    RIGHT_EYE_START_END_IDXS: Tuple[int, int] = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # critical parameters to fine-tune
    WINDOW_SIZE = 9
    DRAMATIC_STD_CHANGE = 0.008

    def __init__(self) -> None:
        self._is_blinking = False
        self._window: Deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._pre_mean: float = 0
        self._pre_std:  float = 0
        self._cool_down: int = -1

    @classmethod
    def get_average_eye_aspect_ratio(
            cls,
            landmarks: NDArray[(68, 2), Int[32]]) -> float:
        """Returns the averaged EAR of the two eyes."""
        # use the left and right eye coordinates to compute
        # the eye aspect ratio for both eyes
        left_ratio = BlinkDetector._get_eye_aspect_ratio(
            cls._extract_left_eye(landmarks)
        )
        right_ratio = BlinkDetector._get_eye_aspect_ratio(
            cls._extract_right_eye(landmarks)
        )

        # average the eye aspect ratio together for both eyes
        return statistics.mean((left_ratio, right_ratio))

    def detect_blink(self, landmarks: NDArray[(68, 2), Int[32]]) -> None:
        if not landmarks.any():
            raise ValueError("landmarks should represent a face")

        ratio = BlinkDetector.get_average_eye_aspect_ratio(landmarks)

        if self._is_initial_detection():
            self._pre_mean = ratio
            self._pre_std  = 0
            # dummy samples
            self._window.extend([ratio] * (self.WINDOW_SIZE - 1))

        self._window.append(ratio)
        cur_mean = statistics.mean(self._window)
        cur_std = statistics.stdev(self._window)

        # important details when implementing this approach
        self._is_blinking = (
            self._not_too_near()
                and self._dramatically_changed(cur_std)
                and self._ear_decreased(cur_mean)
        )
        if self._is_blinking:
            self._start_cooling_down()
        else:
            self._cool_down -= 1

        self._pre_mean = cur_mean
        self._pre_std  = cur_std

    def _not_too_near(self) -> bool:
        # a near blink is probably caused by noise
        return self._cool_down < 0

    def _dramatically_changed(self, cur_std: float) -> bool:
        return cur_std - self._pre_std > self.DRAMATIC_STD_CHANGE

    def _ear_decreased(self, cur_mean: float) -> bool:
        return cur_mean - self._pre_mean < 0

    def _start_cooling_down(self) -> None:
        # no frequent blinkings can happen within 3 slides
        self._cool_down = 3

    def is_blinking(self) -> bool:
        """Returns the result of the latest detection."""
        return self._is_blinking

    def _is_initial_detection(self) -> bool:
        # it's impossible for the mean to be 0
        return self._pre_mean == 0

    @staticmethod
    def _get_eye_aspect_ratio(eye: NDArray[(6, 2), Int[32]]) -> float:
        """Returns the EAR of eye.

        Eye aspect ratio is the ratio between height and width of the eye.
        EAR = (eye height) / (eye width)
        An opened eye has EAR between 0.2 and 0.4 normaly.
        """
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks
        vert = []
        vert.append(math.dist(eye[1], eye[5]))
        vert.append(math.dist(eye[2], eye[4]))

        # compute the euclidean distance between the horizontal
        # eye landmarks
        hor = []
        hor.append(math.dist(eye[0], eye[3]))

        return statistics.mean(vert) / statistics.mean(hor)

    @classmethod
    def _extract_left_eye(
            cls,
            landmarks: NDArray[(68, 2), Int[32]]) -> NDArray[(6, 2), Int[32]]:
        return landmarks[cls.LEFT_EYE_START_END_IDXS[0]
                         :cls.LEFT_EYE_START_END_IDXS[1]]

    @classmethod
    def _extract_right_eye(
            cls,
            landmarks: NDArray[(68, 2), Int[32]]) -> NDArray[(6, 2), Int[32]]:
        return landmarks[cls.RIGHT_EYE_START_END_IDXS[0]
                         :cls.RIGHT_EYE_START_END_IDXS[1]]
