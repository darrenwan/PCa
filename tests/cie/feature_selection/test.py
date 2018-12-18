from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
import numpy as np
from cie.feature_selection.feature_selection import FeatureSelectionLr12
import pytest


num_features = 3


def load_data():
    from sklearn.datasets import load_iris

    iris = load_iris()
    res = iris.data, iris.target
    print(iris.data.shape, iris.target.shape)
    return res


def test_var_threshold():
    X, y = load_data()
    selector = VarianceThreshold(threshold=0.1)
    result = selector.fit_transform(X)
    print(result[:5])


def test_corr():
    from scipy.stats import pearsonr

    X, y = load_data()

    def score_func(X, y):
        return list(zip(*map(lambda x: pearsonr(x, y), X.T)))
    result = SelectKBest(score_func=score_func, k=num_features).fit_transform(X, y)
    print(result[:5])


def test_chi2():
    from sklearn.feature_selection import chi2

    X, y = load_data()
    result = SelectKBest(chi2, k=num_features).fit_transform(X, y)
    print(result[:5])


def test_mine():
    from minepy import MINE

    X, y = load_data()

    def score_func(X, y):
        # 返回二元组（score, p-value）
        def score_p(x, y):
            m = MINE()
            m.compute_score(x, y)
            return m.mic(), 0.5
        # 如果有p-value，则返回list或者tuple
        # 否则，返回其他类型，比如ndarray
        # np.array(list(map(lambda x: mic(x, y), X.T)))
        return list(zip(*map(lambda x: score_p(x, y), X.T)))

    result = SelectKBest(score_func=score_func, k=num_features).fit_transform(X, y)
    print(result[:5])


def test_rfe():
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    X, y = load_data()
    print(np.unique(y))
    result = RFE(estimator=LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500),
                 n_features_to_select=5).fit_transform(X, y)
    print(result[:2])


def test_l1():
    from sklearn.linear_model import LogisticRegression

    X, y = load_data()

    # smaller C, stronger regularization
    model = SelectFromModel(LogisticRegression(multi_class='auto', penalty="l1", C=0.01)).fit(X, y)
    print(model._get_support_mask())
    result = model.transform(X)
    print(result)


def test_l1l2():
    X, y = load_data()

    model = SelectFromModel(FeatureSelectionLr12(multi_class='auto', threshold=0.5, C=0.1)).fit(X, y)
    print(getattr(model.estimator_, "coef_", None))
    result = model.transform(X)
    print(result[:5])


def test_gbdt():
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = load_data()
    # 在没有设置threshold的情况下：
    # gbdt： 按照feature_importances_的均值作为threshold;
    # Lasso/l1 penalty：按照0作为threshold
    model = SelectFromModel(GradientBoostingClassifier(), threshold=None).fit(X, y)
    result = model.transform(X)
    print(model.estimator_.feature_importances_)
    print(result[:5])


if "__main__" == __name__:
    # test_var_threshold()
    # test_corr()
    # test_chi2()
    # test_rfe()
    # test_mine()
    # test_l1()
    # test_l1l2()
    # test_gbdt()
    pytest.main([__file__])
