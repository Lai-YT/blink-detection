import random
import unittest
from decimal import Decimal

import numpy as np

from blink_detector import DynamicThresholdMaker


class DynamicThresholdMakerTestCase(unittest.TestCase):
    TEMP_THRES = 0.24
    NUM_THRES  = 500

    def setUp(self):
        self.thres_maker = DynamicThresholdMaker(self.TEMP_THRES, self.NUM_THRES)

    def test_temporary_threshold(self):
        for _ in range(self.NUM_THRES - 1):
            self.thres_maker.read_ratio(0.3)
            self.assertAlmostEqual(self.thres_maker.threshold, self.TEMP_THRES)

    def test_dynamic_threshold(self):

        def gen_rand_ratio() -> Decimal:
            # radomly generate number between [0.25, 0.35] with step size 0.01
            return Decimal(random.randint(25, 35)) / 100

        def get_thres_should_be(ratios) -> Decimal:
            # mean - standard deviation
            return np.mean(ratios) - np.std(ratios)

        ratios = []
        # fill up the temporary region
        for _ in range(self.NUM_THRES - 1):
            ratio = gen_rand_ratio()
            ratios.append(ratio)
            self.thres_maker.read_ratio(ratio)

        for _ in range(10_000):
            ratio = gen_rand_ratio()
            ratios.append(ratio)
            self.thres_maker.read_ratio(ratio)
            # 2 decimal place seems to be the limit.
            self.assertAlmostEqual(self.thres_maker.threshold,
                                   get_thres_should_be(ratios), places=2)


if __name__ == "__main__":
    unittest.main()
