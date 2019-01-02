# -*- coding: utf-8 -*-
import numpy as np
from cie.data import CieDataFrame
from cie.common import logger

logger = logger.get_logger(name=logger.get_name(__file__))


def test_sgd():
    from cie.models.regression import SGDRegressor
    n_samples, n_features = 10, 5
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    X = CieDataFrame.to_cie_data(X)
    y = CieDataFrame.to_cie_data(y)
    regressor = SGDRegressor(max_iter=1000, tol=1e-3)
    regressor.fit(X, y)
    print(regressor.predict(X))


if __name__ == "__main__":
    print("program begins")
    test_sgd()
    print("program ends")
