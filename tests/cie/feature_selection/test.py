from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from cie.feature_selection.feature_selection import FeatureSelectionLr12

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


def test_l1():
    from sklearn.linear_model import LogisticRegression

    X, y = load_data()

    # smaller C, stronger regularization
    model = SelectFromModel(LogisticRegression(multi_class='auto', penalty="l1", C=0.01)).fit(X, y)
    print(model.get_support())
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


def test_sfs_sbs_floating():
    # SFS, SBS, SFFS, SFBS
    from sklearn.ensemble import GradientBoostingClassifier
    from mlxtend.feature_selection import SequentialFeatureSelector

    X, y = load_data()
    estimator = GradientBoostingClassifier()
    forwards = [True, False]
    floatings = [False, True]
    for forward in forwards:
        for floating in floatings:
            sfs = SequentialFeatureSelector(estimator,
                                            k_features=2,
                                            forward=forward,
                                            floating=floating,
                                            verbose=0,
                                            scoring='accuracy',
                                            cv=5)
            model = sfs.fit(X, y)
            result = model.transform(X)
            print()
            print("(forward, floating)", (forward, floating))
            print("k_feature_idx_:", model.k_feature_idx_)
            print(result[:5])


def test_rfe():
    # similar to sequential step wise backward selection
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.linear_model import LogisticRegression

    X, y = load_data()
    print("shape: ", X.shape)
    estimator = GradientBoostingClassifier()
    model = RFE(estimator=estimator, n_features_to_select=4).fit(X, y)
    result = model.transform(X)
    print()
    print("get_support()", model.scores_)
    print(result[:2])


def test_sfs():
    from cie.feature_selection.sfs_alg import Sfs
    from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.linear_model import LogisticRegression

    X, y = load_data()
    print("shape: ", X.shape)
    estimator = GradientBoostingClassifier()
    model = Sfs(estimator,
                verbose=0, scoring=None,
                cv=5, n_jobs=1,
                persist_features=None,
                pre_dispatch='2*n_jobs',
                clone_estimator=True).fit(X, y)
    result = model.transform(X)
    print()
    print(model.selected)
    print(model.get_metric_dict())
    print(result[:2])


if "__main__" == __name__:
    # test_var_threshold()
    # test_corr()
    # test_chi2()
    # test_mine()
    # test_l1()
    # test_l1l2()
    # test_gbdt()
    # test_sfs_sbs_floating()
    # test_rfe()
    test_sfs()
    # pytest.main([__file__])
