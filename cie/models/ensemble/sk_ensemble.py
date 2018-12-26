import sklearn.ensemble as em
import xgboost as xgb


class RandomForestClassifier(em.RandomForestClassifier):
    pass


class RandomForestRegressor(em.RandomForestRegressor):
    pass


class ExtraTreesClassifier(em.ExtraTreesClassifier):
    pass


class ExtraTreesRegressor(em.ExtraTreesRegressor):
    pass


class BaggingClassifier(em.BaggingClassifier):
    pass


class BaggingRegressor(em.BaggingRegressor):
    pass


class GradientBoostingClassifier(em.GradientBoostingClassifier):
    pass


class GradientBoostingRegressor(em.GradientBoostingRegressor):
    pass


class AdaBoostClassifier(em.AdaBoostClassifier):
    pass


class AdaBoostRegressor(em.AdaBoostRegressor):
    pass


class VotingClassifier(em.VotingClassifier):
    pass


class XGBClassifier(xgb.XGBClassifier):
    pass
