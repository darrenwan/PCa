from cie.datasource import *
from cie.model_selection import train_test_split
from cie.models.ensemble import GradientBoostingClassifier, XGBClassifier
from cie.eda.visualization import *
from cie.evaluate.tuning import *
from cie.evaluate.metrics import *
from cie.feature_selection import SequentialFeatureSelector, Sfs
from cie.output import output_metrics_to_excel, output_metrics_to_excel_v2
from cie.models.ensemble import StackingTransformer
from cie.models.classification import LogisticRegression, SVC, KNeighborsClassifier, SGDClassifier
from cie.data import CieDataFrame
from cie.utils import pickle
from cie.utils import class_weight
from cie.common import logger
from cie.feature_selection import *
from cie.preprocessing import *

logger = logger.get_logger(name=logger.get_name(__file__))


class Estimator(object):
    def __init__(self, model_file, feature_selection_file, stack_file, gbdt_encoder_file, gbdt_file, gbdt_lm_file):
        self.gbdt_encoder_file = gbdt_encoder_file
        self.gbdt_file = gbdt_file
        self.gbdt_lm_file = gbdt_lm_file
        self.model_file = model_file
        self.feature_selection_file = feature_selection_file
        self.stack_file = stack_file
        # 定义flag开关：提供特征选择，参数tuning，stacking模型融合，gbdt+lr
        self.config = {"has_imputation": False,
                       "has_feature_selection": True,
                       "has_split_val": False,
                       "has_tuning_param": False,
                       "has_model_combination": True,
                       "use_gbdt_lr": False}

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
        train = False
        columns = X.columns.values
        logger.info(f"feature_selection begin")
        if train:
            estimator = GradientBoostingClassifier()
            selector = SequentialFeatureSelector(estimator,
                                                 k_features=16,
                                                 forward=True,
                                                 floating=False,
                                                 verbose=0,
                                                 scoring='recall',
                                                 cv=10).fit(X, y)
            logger.info(f"feature_selection selected: {selector.k_feature_idx_}")
            # selector = Sfs(estimator,
            #                verbose=0, scoring=None,
            #                cv=5, n_jobs=2,
            #                persist_features=None,
            #                pre_dispatch='2*n_jobs',
            #                clone_estimator=True).fit(X, y)
            selector = SelectFromModel(GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=8,
                                                                  min_samples_split=50), threshold=None).fit(X, y)
            logger.info(f"feature_selection selected: {columns[selector._get_support_mask()]}")
            with open(self.feature_selection_file, 'wb') as f:
                pickle.dump(selector, f)
            X = selector.transform(X)
        else:
            with open(self.feature_selection_file, 'rb') as f:
                selector = pickle.load(f)
            X = selector.transform(X)
        logger.info("feature_selection end")
        return X

    def tuning_param(self, train_x, train_y, estimater=None, param_grid=None, fit_params=None):
        """
        尝试不同的参数，查看指标。
        :return:
        """
        logger.info(f"tuning_param begin")
        scores = {"recall": ("recall", None)}
        # 参数tuning
        best_estimator = grid_search_tuning(
            estimater, train_x, train_y, None, None, param_grid=param_grid, cv=10, fit_params=fit_params,
            scores=scores,
            regressor=False)
        logger.info(f"tuning_param end")
        return best_estimator

    # 模型融合
    def model_combination(self, estimator_l1=None, estimator_l2=None, X=None, y=None, X_test=None, y_test=None,
                          train=True):
        logger.info(f"model_combination begin")
        # Init 1st level estimators
        if train:
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
        logger.info(f"model_combination begin")
        return estimator_l2, X, X_test

    def gbdt_lr(self, X=None, y=None, train=True):
        logger.info(f"begin {train}")
        if train:
            # 训练集分两部分：一部分训练GBDT模型，另一部分训练LR模型。GBDT的输出(叶子节点）进行onehot变换后作为LR的输入。
            X_train, X_train_lr, y_train, y_train_lr = train_test_split(X, y, test_size=0.5)

            # tuning参数
            # param = {'min_samples_split': range(800, 1900, 200), 'min_samples_leaf': range(60, 101, 10)}
            # gs = GridSearchCV(
            #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7,
            #                                          max_features='sqrt', subsample=0.8, random_state=10),
            #     param_grid=param, scoring='roc_auc', iid=False, cv=5)
            # gs.fit(X, y)
            # logger.info(f"{gs.best_params_}, {gsearch3.best_score_}")
            grd = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, max_depth=8,
                                             min_samples_split=60, max_features='sqrt', subsample=0.7)
            grd_enc = OneHotEncoder()
            grd_lm = LogisticRegression(solver='liblinear', max_iter=2000, class_weight='balanced', C=0.15)
            sample_weights = class_weight.compute_sample_weight('balanced', y_train["分组0"])
            grd.fit(X_train, y_train, sample_weight=sample_weights)
            grd_enc.fit(grd.apply(X_train)[:, :, 0])

            sample_weights = class_weight.compute_sample_weight('balanced', y_train_lr["分组0"])
            grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr, sample_weight=sample_weights)

            with open(self.gbdt_encoder_file, 'wb') as f:
                pickle.dump(grd_enc, f)
            with open(self.gbdt_file, 'wb') as f:
                pickle.dump(grd, f)
            with open(self.gbdt_lm_file, 'wb') as f:
                pickle.dump(grd_lm, f)
            return grd_lm
        else:
            with open(self.gbdt_encoder_file, 'rb') as f:
                grd_enc = pickle.load(f)
            with open(self.gbdt_file, 'rb') as f:
                grd = pickle.load(f)
            with open(self.gbdt_lm_file, 'rb') as f:
                grd_lm = pickle.load(f)
            y_score = grd_lm.predict_proba(grd_enc.transform(grd.apply(X)[:, :, 0]))
            y = grd_lm.predict(grd_enc.transform(grd.apply(X)[:, :, 0]))

            return grd_lm, y, y_score

    def train(self, source):
        # 训练
        logger.info(f"train begin")
        # 读取数据
        Xy, Xy_columns = self._read_excel(source)
        if self.config.get("has_imputation", False):
            # eda分析
            stat_missing_values(Xy)
        X_train = Xy.drop(['PID', '分组', '分组0'], axis=1)
        if features is not None:
            X_train = X_train[features]
        y_train = Xy["分组0"].to_frame()
        X_train = CieDataFrame(X_train)
        y_train = CieDataFrame(y_train)

        # 特征选择
        if self.config.get("has_feature_selection", False):
            X_train = self.feature_selection(X_train, y_train)

        # TODO, trick, 待解决CieDataFrame的warning和set问题
        sample_weights = class_weight.compute_sample_weight("balanced", y_train["分组0"])
        model_candidates = {
            'SVC':
                {
                    "estimator": SVC(gamma=0.001, C=0.5, class_weight='balanced', probability=True),
                    "param": {'C': [0.5]},
                    "fit_params": {"sample_weight": sample_weights}
                },
            'SGDClassifier':
                {
                    'estimator': SGDClassifier(loss='modified_huber', penalty='l1', class_weight='balanced'),
                    "param": {'loss': ['modified_huber', 'log'], 'penalty': ['l2', 'l1', 'elasticnet']},
                    "fit_params": None
                },

            'KNN':
                {
                    'estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=3, weights='distance'),
                    "param": {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance'],
                              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                    "fit_params": None
                },
            'GradientBoostingClassifier':
                {
                    "estimator": GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, max_depth=5,
                                                            min_weight_fraction_leaf=0.1, max_features='sqrt',
                                                            subsample=0.8, min_samples_split=30,
                                                            random_state=10),
                    "param": {'n_estimators': [500, 800], 'min_weight_fraction_leaf': [0, 0.01, 0.1],
                              "max_depth": [4, 5, 6],
                              'min_samples_split': [30, 50, 60]},
                    "fit_params": {"sample_weight": sample_weights}
                },
            'XGBClassifier':
                {
                    "estimator": XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.01,
                                               min_child_weight=2, subsample=0.8,
                                               n_estimators=800, max_depth=4),
                    # "param": {'min_child_weight': [1, 2, 3, 4], 'max_depth': [4,6,8,10]},
                    "param": {'min_child_weight': [2], 'max_depth': [4]},
                    "fit_params": {"sample_weight": sample_weights}
                }
        }

        model_initialized = False

        # 参数tuning及模型融合
        best_model_candidates = dict()
        if self.config.get("has_tuning_param", False):
            for key, value in model_candidates.items():
                best_est = self.tuning_param(X_train, y_train, estimater=value["estimator"],
                                             param_grid=value["param"], fit_params=value["fit_params"])
                best_model_candidates[key] = best_est
        else:
            for key, value in model_candidates.items():
                best_model_candidates[key] = value["estimator"]

        if self.config.get("has_model_combination", False):
            estimator_l2 = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
            model, X_train, _ = self.model_combination(estimator_l1=best_model_candidates,
                                                       estimator_l2=estimator_l2, X=X_train, y=y_train)
            model_initialized = True

        # gbdt+lr
        if not model_initialized:
            model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=8,
                                               min_samples_split=50).fit(X_train, y_train)
        if self.config.get("use_gbdt_lr", False):
            model = self.gbdt_lr(X_train, y_train)

        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"train end")
        return model

    def predict(self, source):
        # 预测
        logger.info(f"predict begin")
        # 读取数据
        Xy, Xy_columns = self._read_excel(source)
        if self.config.get("has_imputation", False):
            # eda分析
            stat_missing_values(Xy)
        X = Xy.drop(['PID', '分组', '分组0'], axis=1)
        if features is not None:
            X = X[features]
        X = CieDataFrame(X)

        if self.config.get("has_feature_selection", False):
            # 特征选择
            X = self.feature_selection(X, train=False)

        if self.config.get("has_model_combination", False):
            # stacking
            model, X, _ = self.model_combination(X=X, train=False)
        elif self.config.get("use_gbdt_lr", False):
            # gbdt + lr
            model, X, X_score = self.gbdt_lr(X, train=False)
            return X_score
        else:
            with open(self.model_file, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"predict end")
            return model.predict_proba(X)

    def stat(self, train_source, test_source, output_file=None, sheet_name="sheet"):
        # 统计模型效果
        logger.info(f"stat begin")
        # 读取数据
        Xy, Xy_columns = self._read_excel(train_source)
        X_train = Xy.drop(['PID', '分组0', '分组'], axis=1)
        if features is not None:
            X_train = X_train[features]
        y_train = Xy["分组0"].to_frame()
        X_train = CieDataFrame(X_train)
        y_train = CieDataFrame(y_train)

        Xy, Xy_columns = self._read_excel(test_source)
        X_test = Xy.drop(['PID', '分组0', '分组'], axis=1)
        if features is not None:
            X_test = X_test[features]
        X_test = CieDataFrame(X_test)
        y_test = Xy["分组0"].to_frame()
        y_test = CieDataFrame(y_test)

        # 特征选择
        if self.config.get("has_feature_selection", False):
            X_train = self.feature_selection(X_train, train=False)
            X_test = self.feature_selection(X_test, train=False)

        # 统计结果输出
        if self.config.get("has_model_combination", False):
            model, X_train, _ = self.model_combination(X=X_train, train=False)
            _, X_test, _ = self.model_combination(X=X_test, train=False)
            with open(self.model_file, 'rb') as f:
                model = pickle.load(f)

            print(model.classes_)
            data = {"train": (X_train, y_train), "test": (X_test, y_test)}
            output_metrics_to_excel(model,
                                    output_file=output_file,
                                    sheet_name=sheet_name + "_gbdt_lr" if self.config.get("use_gbdt_lr", False)
                                    else sheet_name + "_stacking",
                                    data=data)
        elif self.config.get("use_gbdt_lr", False):
            _, y_train_pred, y_train_score = self.gbdt_lr(X_train, train=False)
            _, y_test_pred, y_test_score = self.gbdt_lr(X_test, train=False)
            data = {"train": (y_train, y_train_pred, y_train_score), "test": (y_test, y_test_pred, y_test_score)}
            output_metrics_to_excel_v2(output_file=output_file,
                                       sheet_name=sheet_name + "_gbdt_lr" if self.config.get("use_gbdt_lr", False)
                                       else sheet_name + "_stacking",
                                       data=data)
        logger.info(f"stat end")


if "__main__" == __name__:
    from os import mkdir, path

    model_folder = './model'
    output_folder = './output'
    for folder in [model_folder, output_folder]:
        if not path.exists(folder):
            mkdir(folder)
    features = None
    # features = """碱性磷酸酶_combine0
    # 血小板压积（PCT）_combine0
    # 血浆D-二聚体（D-Dimer）_combine0
    # 亚硝酸盐_combine0
    # 淋巴细胞计数_combine0
    # 白蛋白/球蛋白_combine0
    # 肌酐_combine0
    # 胆红素_combine0
    # 淋巴细胞百分比_combine0
    # 门冬氨酸氨基转移酶_combine0
    # 颜色_combine0
    # 乙型肝炎e抗体（HBeAb）_num_combine0
    # 乙型肝炎核心抗体（HBcAb）_num_combine0
    # 乙型肝炎表面抗体（HBsAb）_num_combine0
    # 乙型肝炎表面抗原（HBsAg）_num_combine0
    # 凝血酶原时间（PT）_num_combine0
    # 总胆红素
    # 甲胎蛋白
    # 粪白细胞
    # 葡萄糖
    # 血小板计数（PLT）
    # 乙型肝炎e抗原（HBeAg）_num
    # 直接胆红素_combine0
    # 抗HIV(1+2)抗体【【】】
    # 中性粒细胞百分比（GRAN%）_combine0
    # 前白蛋白_combine0
    # 国际标准化比值_combine0
    # 嗜碱性粒细胞百分比（BASO%）_combine0
    # 年龄
    # 活化部分凝血活酶时间（APTT）_combine0
    # 纤维蛋白原（FIB）_combine0
    # 丙氨酸氨基转氨酶_combine0
    # CA199-定性_combine0
    # 中性粒细胞计数_combine0
    # 尿素_combine0
    # 丙型肝炎抗体IgG_combine0
    # γ-谷氨酰转肽酶_combine0
    # 间接胆红素【【计算】】
    # CA125_combine0
    # CA199-定量_combine0""".split()
    # print(len(features))
    logger.info(f"program begin")
    train_file = "./data/train.xlsx"
    test_file = "./data/test.xlsx"
    output_file = path.join(output_folder, "metrics.xlsx")
    model_file = path.join(model_folder, "model.pkl")
    stack_file = path.join(model_folder, "stack.pkl")
    feature_selection_file = path.join(model_folder, "selector.pkl")
    gbdt_encoder_file = path.join(model_folder, "gbdt_onehot.pkl")
    gbdt_file = path.join(model_folder, "gbdt.pkl")
    gbdt_lm_file = path.join(model_folder, "gbdt_lm.pkl")
    est = Estimator(model_file, feature_selection_file, stack_file, gbdt_encoder_file, gbdt_file, gbdt_lm_file)
    est.train(source=train_file)
    est.predict(source=test_file)
    est.stat(train_source=train_file, test_source=test_file, output_file=output_file)
