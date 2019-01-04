# -*- coding: utf-8 -*-

from cie.common import logger

import matplotlib.pyplot as plt
from cie.preprocessing import *
from cie.data import load_iris
from cie.data import CieDataFrame
import pytest


logger = logger.get_logger(name=logger.get_name(__file__))


def load_data():
    X, y = load_iris()
    return X, y


def test_transformer():
    power_transformer = SkewPowerTransformer(method='yeo-johnson')

    X, y = load_data()
    result = power_transformer.fit_transform(X)
    print(result[:5])

    # plt.subplot(2, 1, 1)
    # plt.hist(X, bins=30)
    # plt.subplot(2, 1, 2)
    # plt.hist(result, bins=30)
    # plt.show()


if __name__ == "__main__":
    print("program begins")
    test_transformer()
    # pytest.main([__file__])
    print("program ends")
