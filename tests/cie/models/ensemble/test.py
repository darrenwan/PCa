# -*- coding: utf-8 -*-

from cie.common import logger
from cie.evaluate.evaluation import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

logger = logger.get_logger(name=logger.get_name(__file__))


def test_stacking():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from xgboost import XGBRegressor
    from cie.models.ensemble import StackingTransformer

    # Load demo data
    boston = load_boston()
    X, y = boston.data, boston.target

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


if __name__ == "__main__":
    print("program begins")
    test_stacking()
    print("program ends")
