import math
import statistics
from collections import deque
from decimal import Decimal
from enum import Enum, auto
from typing import Deque, Tuple, Union

import numpy as np
from imutils import face_utils
from nptyping import Int, NDArray


class EyeSide(Enum):
    LEFT  = auto()
    RIGHT = auto()


class DynamicThresholdMaker:
    """Take a certain number of landmarks that contains face to dynamically
    determine a threshold of eye aspect ratio.

    It can be used with a BlinkDetector to provide better detections on different
    users.
    "Decimal" module is used to reduce round-off errors, provide better precision.
    """
    def __init__(
            self,
            temp_thres: Union[Decimal, float] = Decimal("0.24"),
            num_thres: int = 100) -> None:
        """
        Arguments:
            temp_thres:
                The temporary threshold used before the dynamic threshold is ready,
                which prevents from getting an unreliable threshold due to low
                number of samples. Should be in range [0.15, 0.5].
            num_thres:
                Higher number (included) of samples than this is considered to
                be reliable.
        """
        if temp_thres < 0.15 or temp_thres > 0.5:
            raise ValueError("resonable threshold should >= 0.15 and <= 0.5")
        if num_thres < 100:
            raise ValueError("number of samples under 100 makes the normal eye"
                             "aspect ratio susceptible to extreme values")

        self._dyn_thres = Decimal(temp_thres)
        self._num_thres = num_thres
        # To calculate the mean and standard deviation without re-summing all the
        # samples, we store the real time sum and sum of squares.
        self._cur_sum = Decimal(0)
        self._cur_sum_of_sq = Decimal(0)
        # The only reason we stored the sample ratios is to remove the old
        # ratios in order.
        self._samp_ratios: Deque[Decimal] = deque()

    @property
    def threshold(self) -> Decimal:
        """The dynamic threshold; temporary threshold if the number of sample
        ratios is not yet reliable.
        """
        return self._dyn_thres

    def read_sample(self, landmarks: NDArray[(68, 2), Int[32]]) -> None:
        """Reads the EAR value from the landmarks.

        Notice that passing blinking landmarks makes the dynamic threshold
        deviate, so avoid them if possible.
        """
        # Empty landmarks is not counted.
        if not landmarks.any():
            return

        self.read_ratio(BlinkDetector.get_average_eye_aspect_ratio(landmarks))

    def read_ratio(self, ratio: Union[Decimal, float]) -> None:
        """Reads the EAR value directly.

        Calculating EAR from landmarks or not is the only difference with
        read_sample.

        Notice that passing blinking ratios makes the dynamic threshold
        deviate, so avoid them if possible.
        """
        self._update_sums(ratio)
        self._update_dynamic_threshold()

    def _update_sums(self, new_ratio: Union[Decimal, float]) -> None:
        # add new ratio
        new_ratio = Decimal(new_ratio)
        self._samp_ratios.append(new_ratio)
        self._cur_sum += new_ratio
        self._cur_sum_of_sq += new_ratio * new_ratio
        # remove old ratio
        if len(self._samp_ratios) > self._num_thres:
            old_ratio = self._samp_ratios.popleft()
            self._cur_sum -= old_ratio
            self._cur_sum_of_sq -= old_ratio * old_ratio

    def _update_dynamic_threshold(self) -> None:
        """Updates the dynamic threshold if the number of sample ratios
        is enough.

        Note that the threshold = MEAN(sample EARs) - 1.5 * STD(sample EARs).
        """
        if len(self._samp_ratios) == self._num_thres:
            mean = self._cur_sum / self._num_thres
            mean_of_sq = self._cur_sum_of_sq / self._num_thres
            std = (mean_of_sq - mean * mean).sqrt()
            self._dyn_thres = mean - Decimal("1.5") * std


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
    def get_average_eye_aspect_ratio(cls, landmarks: NDArray[(68, 2), Int[32]]) -> Decimal:
        """Returns the averaged EAR of the two eyes."""
        # use the left and right eye coordinates to compute
        # the eye aspect ratio for both eyes
        left_ratio = BlinkDetector._get_eye_aspect_ratio(cls._extract_eye(landmarks, EyeSide.LEFT))
        right_ratio = BlinkDetector._get_eye_aspect_ratio(cls._extract_eye(landmarks, EyeSide.RIGHT))

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
    def _extract_eye(cls, landmarks: NDArray[(68, 2), Int[32]], side: EyeSide) -> NDArray[(6, 2), Int[32]]:
        eye: NDArray[(6, 2), Int[32]]
        if side is EyeSide.LEFT:
            eye = landmarks[cls.LEFT_EYE_START_END_IDXS[0]:cls.LEFT_EYE_START_END_IDXS[1]]
        elif side is EyeSide.RIGHT:
            eye = landmarks[cls.RIGHT_EYE_START_END_IDXS[0]:cls.RIGHT_EYE_START_END_IDXS[1]]
        else:
            raise TypeError(f'type of argument "side" must be "EyeSide", not "{type(side).__name__}"')
        return eye


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
