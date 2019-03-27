from cie.datasource import *
from cie.model_selection import train_test_split
from cie.models.ensemble import GradientBoostingClassifier, XGBClassifier
from cie.eda.visualization import *
from cie.evaluate.tuning import *
from cie.evaluate.metrics import *
from cie.feature_selection import SequentialFeatureSelector, Sfs
from cie.output import output_metrics_to_excel
from cie.models.ensemble import StackingTransformer
from cie.models.classification import LogisticRegression
from cie.data import CieDataFrame
import pickle
from cie.common import logger

logger = logger.get_logger(name=logger.get_name(__file__))


class Estimator(object):
    def __init__(self, model_file, feature_selection_file, stack_file):
        self.model_file = model_file
        self.feature_selection_file = feature_selection_file
        self.stack_file = stack_file
        self.config = {"has_imputation": False,
                       "has_feature_selection": False,
                       "has_split_val": False,
                       "has_tuning_param": True,
                       "has_model_combination": True}

    def _read_excel(self, file):
        params = {
            "sep": '\t',
            "encoding": 'utf-8',
            # "nrows": 20,
        }
        # 训练集、验证集
        channel = ExcelChannel(file)
        channel.open()
        Xy, Xy_columns = channel.read(**params)
        channel.close()
        return Xy, Xy_columns

    def feature_selection(self, X=None, y=None, train=True):
        if train:
            estimator = GradientBoostingClassifier()
            # selector = SequentialFeatureSelector(estimator,
            #                                      k_features=16,
            #                                      forward=True,
            #                                      floating=False,
            #                                      verbose=0,
            #                                      scoring='accuracy',
            #                                      cv=5).fit(X, y)
            # print(selector.k_feature_idx_)
            selector = Sfs(estimator,
                           verbose=0, scoring=None,
                           cv=5, n_jobs=2,
                           persist_features=None,
                           pre_dispatch='2*n_jobs',
                           clone_estimator=True).fit(X, y)

            with open(self.feature_selection_file, 'wb') as f:
                pickle.dump(selector, f)
            X = selector.transform(X)
        else:
            with open(self.feature_selection_file, 'rb') as f:
                selector = pickle.load(f)
            X = selector.transform(X)
        return X

    def tuning_param(self, train_x, train_y, estimater=None, gs_param=None,
                     scores={"neg_log_loss": ("neg_log_loss", None)}):
        """
        尝试不同的参数，查看指标。
        :return:
        """
        # 参数tuning
        best_estimator = grid_search_tuning(
            estimater, train_x, train_y, None, None, param_grid=gs_param, cv=5,
            scores=scores,
            regressor=False)

        return best_estimator

    # 模型融合
    def model_conbine(self, estimator_l1=None, estimator_l2=None, X=None, y=None, X_test=None, y_test=None, train=True):
        # Init 1st level estimators
        if train:
            # X_train_orig, X_test_orig = X, X_test
            # Stacking
            estimator_l1 = [(k, v) for k, v in estimator_l1.items()]

            stack = StackingTransformer(estimators=estimator_l1,
                                        regression=False,
                                        shuffle=True,
                                        random_state=0,
                                        needs_proba=True,
                                        verbose=0)
            stack = stack.fit(X, y)
            with open(self.stack_file, 'wb') as f:
                pickle.dump(stack, f)
        else:
            with open(self.stack_file, 'rb') as f:
                stack = pickle.load(f)
        X = stack.transform(X)
        if train:
            estimator_l2 = estimator_l2.fit(X, y)
        else:
            estimator_l2 = None

        # if X_test is not None:
        #     X_test = stack.transform(X_test)
        #     y_pred = estimator_l2.predict_proba(X_test)
        #     stacking_score = -log_loss(y_test, y_pred, labels=[0, 1, 2])
        #
        #     for k, v in estimator_l1.items():
        #         single_model = v
        #         single_model = single_model.fit(X_train_orig, y)
        #         y_pred = single_model.predict_proba(X_test_orig)
        #         score = -log_loss(y_test, y_pred, labels=[0, 1, 2])
        #         print('单模型%s log_loss: [%.8f]' % (k, score))
        #     print('stacking log_loss: [%.8f]' % stacking_score)
        return estimator_l2, X, X_test

    def train(self, source):
        Xy, Xy_columns = self._read_excel(source)
        if self.config.get("has_imputation", False):
            # eda分析
            stat_missing_values(Xy)
        X_train = Xy.drop(['PID', '分组'], axis=1)
        y_train = Xy["分组"].to_frame()
        X_train = CieDataFrame(X_train)
        y_train = CieDataFrame(y_train)
        if self.config.get("has_feature_selection", False):
            # 特征选择
            X_train = self.feature_selection(X_train, y_train)

        model_candidates = {
            'LogisticRegression':
                {
                    "estimator": LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'),
                    "param": {'solver': ['lbfgs']}
                },
            'GradientBoostingClassifier':
                {
                    "estimator": GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=8,
                                                            min_samples_split=50),
                    "param": {'n_estimators': list(range(500, 600, 50)), 'max_depth': list(range(3, 10, 2)),
                              'min_samples_split': list(range(50, 60, 10)), 'learning_rate': [0.1, 0.6, 2]}
                },
            'XGBClassifier':
                {
                    "estimator": XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                                               n_estimators=500, max_depth=8),
                    "param": {'n_estimators': range(500, 600, 50), 'max_depth': range(5, 10, 5),
                              'learning_rate': [0.5, 0.1, 1]}
                }
        }
        model_initialized = False
        if self.config.get("has_tuning_param", False):
            # 参数tuning
            best_model_candidates = dict()
            for key, value in model_candidates.items():
                best_est = self.tuning_param(X_train, y_train, estimater=value["estimator"], gs_param=value["param"])
                best_model_candidates[key] = best_est

            if self.config.get("has_model_combination", False):
                estimator_l2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=8,
                                                          min_samples_split=50)
                model, X_train, _ = self.model_conbine(estimator_l1=best_model_candidates,
                                                       estimator_l2=estimator_l2, X=X_train, y=y_train)
                model_initialized = True

        if not model_initialized:
            model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=8,
                                               min_samples_split=50).fit(X_train, y_train)
        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)
        return model

    def predict(self, source):
        Xy, Xy_columns = self._read_excel(source)
        if self.config.get("has_imputation", False):
            # eda分析
            stat_missing_values(Xy)
        X = Xy.drop(['PID', '分组'], axis=1)
        X = CieDataFrame(X)
        # y = Xy["分组"].to_frame()
        # y = CieDataFrame(y)
        if self.config.get("has_feature_selection", False):
            # 特征选择
            X = self.feature_selection(X, train=False)

        if self.config.get("has_model_combination", False):
            _, X, _ = self.model_conbine(X=X, train=False)
        with open(self.model_file, 'rb') as f:
            model = pickle.load(f)
        return model.predict_proba(X)

    def stat(self, train_source, test_source, output_file=None, sheet_name="sheet1"):
        Xy, Xy_columns = self._read_excel(train_source)
        X_train = Xy.drop(['PID', '分组'], axis=1)
        y_train = Xy["分组"].to_frame()
        X_train = CieDataFrame(X_train)
        y_train = CieDataFrame(y_train)

        Xy, Xy_columns = self._read_excel(test_source)
        X_test = Xy.drop(['PID', '分组'], axis=1)
        X_test = CieDataFrame(X_test)
        y_test = Xy["分组"].to_frame()
        y_test = CieDataFrame(y_test)
        if self.config.get("has_feature_selection", False):
            # 特征选择
            X_train = self.feature_selection(X_train, train=False)
            X_test = self.feature_selection(X_test, train=False)

        if self.config.get("has_model_combination", False):
            _, X_train, _ = self.model_conbine(X=X_train, train=False)
            _, X_test, _ = self.model_conbine(X=X_test, train=False)

        with open(self.model_file, 'rb') as f:
            model = pickle.load(f)
        data = {"train": (X_train, y_train), "test": (X_test, y_test)}
        output_metrics_to_excel(model,
                                output_file=output_file,
                                sheet_name=sheet_name, data=data)


if "__main__" == __name__:
    logger.info(f"program begin")
    train_file = "./data/train.xlsx"
    test_file = "./data/test.xlsx"
    output_file = "./output/metrics.xlsx"
    model_file = "./output/model.pkl"
    stack_file = "./output/stack.pkl"
    feature_selection_file = "./output/selector.pkl"
    est = Estimator(model_file, feature_selection_file, stack_file)
    # est.train(source=train_file)
    est.predict(source=test_file)
    est.stat(train_source=train_file, test_source=test_file, output_file=output_file)
