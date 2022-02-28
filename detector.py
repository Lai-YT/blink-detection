import math
import statistics
from decimal import Decimal
from typing import Tuple, Union

import numpy as np
from imutils import face_utils
from nptyping import Int, NDArray


class BlinkDetector:
    """Detects whether the eyes are blinking or not by calculating
    the eye aspect ratio (EAR).

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

    def __init__(self, ratio_threshold: Union[Decimal, float] = Decimal("0.24")) -> None:
        """
        Arguments:
            ratio_threshold:
                Having ratio lower than the threshold is considered to be a blink.
        """
        self._ratio_threshold = Decimal(ratio_threshold)

    @property
    def ratio_threshold(self) -> Decimal:
        return self._ratio_threshold

    @ratio_threshold.setter
    def ratio_threshold(self, threshold: Decimal) -> None:
        self._ratio_threshold = threshold

    @classmethod
    def get_average_eye_aspect_ratio(
            cls,
            landmarks: NDArray[(68, 2), Int[32]]) -> Decimal:
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

    def detect_blink(self, landmarks: NDArray[(68, 2), Int[32]]) -> bool:
        """Returns whether the eyes in the face landmarks are blinking or not."""
        if not landmarks.any():
            raise ValueError("landmarks should represent a face")

        ratio = BlinkDetector.get_average_eye_aspect_ratio(landmarks)
        return ratio < self._ratio_threshold

    @staticmethod
    def _get_eye_aspect_ratio(eye: NDArray[(6, 2), Int[32]]) -> Decimal:
        """Returns the EAR of eye.

        Eye aspect ratio is the ratio between height and width of the eye.
        EAR = (eye height) / (eye width)
        An opened eye has EAR between 0.2 and 0.4 normaly.
        """
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks
        vert = []
        vert.append(Decimal(math.dist(eye[1], eye[5])))
        vert.append(Decimal(math.dist(eye[2], eye[4])))

        # compute the euclidean distance between the horizontal
        # eye landmarks
        hor = []
        hor.append(Decimal(math.dist(eye[0], eye[3])))

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


class AntiNoiseBlinkDetector:
    """Noise or face move may cause a false-positive "blink".
    AntiNoiseBlinkDetector uses a normal BlinkDetector as its underlayer but
    agrees a "blink" only if it continues for a sufficient number of frames.
    """

    def __init__(
            self,
            ratio_threshold: Union[Decimal, float] = Decimal("0.24"),
            consec_frame: int = 3) -> None:
        """
        Arguments:
            ratio_threshold: The eye aspect ratio to indicate blink.
            consec_frame:
                The number of consecutive frames the eye must be below the
                threshold to indicate an anti-noise blink.
        """
        super().__init__()
        # the underlaying BlinkDetector
        self._base_detector = BlinkDetector(ratio_threshold)
        self._consec_frame = consec_frame
        self._consec_count: int = 0

    @property
    def ratio_threshold(self) -> Decimal:
        return self._base_detector.ratio_threshold

    @ratio_threshold.setter
    def ratio_threshold(self, threshold: Decimal) -> None:
        self._base_detector.ratio_threshold = threshold

    def detect_blink(self, landmarks: NDArray[(68, 2), Int[32]]) -> bool:
        """Returns whether the eyes in the face landmarks are blinking or not.

        Notice that the return value is about a "delayed" state. Since it's
        anti-noised by the number of consecutive frames, we can only determine
        whether this is a blink or not after the consecutiveness ends.
        """
        # Uses the base detector with EYE_AR_CONSEC_FRAMES to determine whether
        # there's an anti-noise blink.
        blinked: bool = False
        if self._base_detector.detect_blink(landmarks):
           self._consec_count += 1
        else:
            # if the eyes were closed for a sufficient number of frames,
            # it's considered to be a real blink
            if self._consec_count >= self._consec_frame:
                blinked = True
            self._consec_count = 0
        return blinked
