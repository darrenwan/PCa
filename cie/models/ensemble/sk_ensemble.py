import sklearn.ensemble as em
import xgboost as xgb
import lightgbm as lgb


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


class XGBBase(object):
    """
    对应xgboost.plotting的plot_importance, plot_tree, to_graphviz三个方法
    """

    def plot_importance(self):
        return xgb.plot_importance(self)

    def plot_tree(self):
        return xgb.plot_tree(self)

    def to_graphviz(self):
        return xgb.to_graphviz(self)


class XGBClassifier(xgb.XGBClassifier, XGBBase):
    __doc__ = """Implementation of the scikit-learn API for XGBoost classification.

        """ + '\n'.join(xgb.XGBModel.__doc__.split('\n')[2:])


class XGBRegressor(xgb.XGBRegressor, XGBBase):
    __doc__ = """Implementation of the scikit-learn API for XGBoost regression.
        """ + '\n'.join(xgb.XGBModel.__doc__.split('\n')[2:])


class LgbBase(object):
    """
    对应xgboost.plotting的plot_importance, plot_tree, plot_metric, create_tree_digraph方法
    """

    def plot_importance(self):
        return lgb.plot_importance(self)

    def plot_tree(self):
        return lgb.plot_tree(self)

    def plot_metric(self):
        return lgb.plot_metric(self)

    def create_tree_digraph(self):
        return lgb.create_tree_digraph(self)


class LGBMClassifier(lgb.LGBMClassifier, LgbBase):
    """
    LightGBM classifier
    """
    pass


class LGBMRegressor(lgb.LGBMRegressor, LgbBase):
    """
    LightGBM regressor
    """
    pass
