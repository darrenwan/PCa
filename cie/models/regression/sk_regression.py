from sklearn import svm
import sklearn.linear_model as lm


class SVR(svm.SVR):
    pass


class Lasso(lm.Lasso):
    pass


class LassoLars(lm.LassoLars):
    pass


class LinearRegression(lm.LinearRegression):
    pass


class RidgeRegression(lm.Ridge):
    pass


class RidgeClassifier(lm.RidgeClassifier):
    pass


class SGDRegressor(lm.SGDRegressor):
    pass
