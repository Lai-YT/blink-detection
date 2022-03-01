from abc import ABC, abstractmethod
from collections import deque
from decimal import Decimal
from typing import Deque, Union

from nptyping import Int, NDArray

from detector import BlinkDetector


class DynamicThresholdMaker(ABC):
    """Take a certain number of landmarks that contains face to dynamically
    determine a threshold of eye aspect ratio.

    It can be used with a BlinkDetector to provide better detections on
    different users.
    """

    @property
    @abstractmethod
    def threshold(self) -> Decimal:
        pass

    @abstractmethod
    def read_sample(self, landmarks: NDArray[(68, 2), Int[32]]) -> None:
        """Reads the EAR value from the landmarks."""

    @abstractmethod
    def read_ratio(self, ratio: Union[Decimal, float]) -> None:
        """Reads the EAR value directly."""


class StatisticalThresholdMaker(DynamicThresholdMaker):
    """Uses mean and standard deviation on numbers of samples to dynamically
    determine the threshold.
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
        super().__init__()

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

    # Override
    @property
    def threshold(self) -> Decimal:
        """The dynamic threshold; temporary threshold if the number of sample
        ratios is not yet reliable.
        """
        return self._dyn_thres

    # Override
    def read_sample(self, landmarks: NDArray[(68, 2), Int[32]]) -> None:
        """Reads the EAR value from the landmarks.

        Notice that passing blinking landmarks makes the dynamic threshold
        deviate, so avoid them if possible.
        """
        # Empty landmarks is not counted.
        if not landmarks.any():
            return

        self.read_ratio(BlinkDetector.get_average_eye_aspect_ratio(landmarks))

    # Override
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
        self._update_sums_by_adding_new_ratio(new_ratio)
        if len(self._samp_ratios) > self._num_thres:
            self._update_sums_by_removing_old_ratio()

    def _update_sums_by_adding_new_ratio(
            self,
            new_ratio: Union[Decimal, float]) -> None:
        new_ratio = Decimal(new_ratio)
        self._cur_sum += new_ratio
        self._cur_sum_of_sq += new_ratio * new_ratio
        self._samp_ratios.append(new_ratio)

    def _update_sums_by_removing_old_ratio(self) -> None:
        old_ratio = self._samp_ratios.popleft()
        self._cur_sum -= old_ratio
        self._cur_sum_of_sq -= old_ratio * old_ratio

    def _update_dynamic_threshold(self) -> None:
        """Updates the dynamic threshold if the number of sample ratios
        is enough.
        """
        if self._is_not_yet_reliable():
            return

        self._calculate_mean_and_std()
        print(self._get_factor())
        self._dyn_thres = self._mean - self._get_factor() * self._std

    def _is_not_yet_reliable(self) -> bool:
        return len(self._samp_ratios) < self._num_thres

    def _calculate_mean_and_std(self) -> None:
        self._mean = self._cur_sum / self._num_thres
        mean_of_sq = self._cur_sum_of_sq / self._num_thres
        self._std = (mean_of_sq - self._mean * self._mean).sqrt()

    def _get_factor(self) -> Decimal:
        # Line:
        #   when mean = 0.22, factor = 0.8;
        #        mean = 0.28, factor = 1.8
        # => y = (50 / 3) * (x - 0.22) + 0.8
        return Decimal("50") / Decimal("3") * (self._mean - Decimal("0.22")) + Decimal("0.8")
