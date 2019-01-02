# -*- coding: utf-8 -*-
from cie.data import CieDataFrame
from cie.common import logger

logger = logger.get_logger(name=logger.get_name(__file__))


def test_sgd():
    from cie.models.classification import SGDClassifier
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    X = CieDataFrame.to_cie_data(X)
    y = CieDataFrame.to_cie_data(y)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100, tol=1e-3)
    clf.fit(X, y)
    print(clf.predict([[2., 2.]]))


def test_lr():
    from cie.models.classification import LogisticRegression
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    X = CieDataFrame.to_cie_data(X)
    y = CieDataFrame.to_cie_data(y)
    clf = LogisticRegression(penalty="l2", solver='lbfgs')
    clf.fit(X, y)
    print(clf.predict([[2., 2.]]))


if __name__ == "__main__":
    print("program begins")
    test_sgd()
    test_lr()
    print("program ends")
