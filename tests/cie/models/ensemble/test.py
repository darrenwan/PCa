# -*- coding: utf-8 -*-
from cie.data import CieDataFrame
from cie.common import logger
from cie.evaluate import *

logger = logger.get_logger(name=logger.get_name(__file__))


def test_stacking():
    from cie.data import load_boston
    from cie.model_selection import train_test_split
    from cie.evaluate.metrics import mean_absolute_error
    from cie.models.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from cie.models.ensemble import XGBRegressor
    from cie.models.ensemble import StackingTransformer

    # Load demo data
    boston = load_boston()
    X, y = boston.data, boston.target
    X = CieDataFrame.to_cie_data(X)
    y = CieDataFrame.to_cie_data(y)
    # Make train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0)


    # Init 1st level estimators
    estimators_L1 = [('ExtraTreesRegressor', ExtraTreesRegressor(random_state=0,
                                                n_jobs=-1,
                                                n_estimators=100,
                                                max_depth=3)),
                     ('RandomForestRegressor', RandomForestRegressor(random_state=0,
                                                  n_jobs=-1,
                                                  n_estimators=100,
                                                  max_depth=3)),
                     ('XGBRegressor', XGBRegressor(random_state=0,
                                          n_jobs=-1,
                                          learning_rate=0.1,
                                          n_estimators=100,
                                          max_depth=3))]
    # Stacking
    stack = StackingTransformer(estimators=estimators_L1,
                                regression=True,
                                shuffle=True,
                                random_state=0,
                                verbose=2)
    stack = stack.fit(X_train, y_train)
    S_train = stack.transform(X_train)
    S_test = stack.transform(X_test)

    estimator_L2 = XGBRegressor(random_state=0,
                                n_jobs=-1,
                                learning_rate=0.1,
                                n_estimators=100,
                                max_depth=3)
    estimator_L2 = estimator_L2.fit(S_train, y_train)
    y_pred = estimator_L2.predict(S_test)
    stacking_mae = mean_absolute_error(y_test, y_pred)

    for item in estimators_L1:
        single_model = item[1]
        single_model = single_model.fit(X_train, y_train)
        y_pred = single_model.predict(X_test)
        pre_mae = mean_absolute_error(y_test, y_pred)
        print('单模型%s MAE: [%.8f]' % (item[0], pre_mae))
    print('stacking MAE: [%.8f]' % stacking_mae)


def test_classifier():
    from cie.models.ensemble import GradientBoostingClassifier
    from cie.data import make_classification
    X, y = make_classification(n_samples=20)
    X = CieDataFrame.to_cie_data(X)
    y = CieDataFrame.to_cie_data(y)
    labels, y = np.unique(y, return_inverse=True)
    gb = GradientBoostingClassifier()
    gb.fit(X, y)
    test_deviance = np.zeros(gb.n_estimators, dtype=np.float64)
    for i, y_pred in enumerate(gb.staged_decision_function(X)):
        test_deviance[i] = gb.loss_(y, y_pred)
    # plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],'-')
    # plt.show()


def test_regressor():
    from cie.models.ensemble import XGBRegressor
    from cie.data import make_regression
    X, y = make_regression(n_samples=20)
    X = CieDataFrame.to_cie_data(X)
    y = CieDataFrame.to_cie_data(y)
    labels, y = np.unique(y, return_inverse=True)
    xgb = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
    xgb.fit(X, y)
    y_pred = xgb.predict(X)
    print(r2_score(y, y_pred))


if __name__ == "__main__":
    print("program begins")
    test_stacking()
    test_classifier()
    test_regressor()
    print("program ends")
