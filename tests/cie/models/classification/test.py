# -*- coding: utf-8 -*-
from cie.data import *
from cie.common import logger
import pytest

logger = logger.get_logger(name=logger.get_name(__file__))


def test_sgd():
    from cie.models.classification import SGDClassifier
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    from cie.data import CieDataFrame
    X = CieDataFrame(X)
    y = CieDataFrame(y)
    # X = pd.DataFrame(X)
    # y = pd.DataFrame(y)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100, tol=1e-3)
    clf.fit(X, y)
    print(clf.predict([[2., 2.]]))


def test_lr():
    from cie.models.classification import LogisticRegression
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    X = CieDataFrame(X)
    y = CieDataFrame(y)
    clf = LogisticRegression(penalty="l2", solver='lbfgs')
    clf.fit(X, y)
    print(clf.predict([[2., 2.]]))


if __name__ == "__main__":
    print("program begins")
    test_sgd()
    # test_lr()
    # pytest.main([__file__])
    print("program ends")
