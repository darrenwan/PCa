from cie.datasource import *
from cie.model_selection import train_test_split
from cie.models.ensemble import GradientBoostingClassifier, XGBClassifier
from cie.eda.visualization import *
from cie.evaluate.tuning import *
from cie.evaluate.metrics import *
from cie.feature_selection import SequentialFeatureSelector, Sfs
from cie.output import output_metrics_to_excel, output_metrics_to_excel_v2
from cie.models.ensemble import StackingTransformer
from cie.models.classification import LogisticRegression, SVC
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
        self.config = {"has_imputation": False,
                       "has_feature_selection": False,
                       "has_split_val": False,
                       "has_tuning_param": True,
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
        # # 处理不均衡问题
        # print(np.unique(train_y))
        # class_weights = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
        # print("====", class_weights)
        # neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
        #                                   needs_proba=True, class_weight=class_weights)
        # scores = {"neg_log_loss": (neg_log_loss_scorer, neg_log_loss_scorer)}

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
        logger.info(f"model_combination begin")
        return estimator_l2, X, X_test

    def gbdt_lr(self, X=None, y=None, train=True):
        if train:
            # 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，
            # 然后作为LR的特征。这样分成两部分是为了防止过拟合。
            X_train, X_train_lr, y_train, y_train_lr = train_test_split(X, y, test_size=0.5)

            # param = {'min_samples_split': range(800, 1900, 200), 'min_samples_leaf': range(60, 101, 10)}
            # gs = GridSearchCV(
            #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=7,
            #                                          max_features='sqrt', subsample=0.8, random_state=10),
            #     param_grid=param, scoring='roc_auc', iid=False, cv=5)
            # gs.fit(X, y)
            # logger.info(f"{gs.best_params_}, {gsearch3.best_score_}")
            grd = GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, max_depth=8,
                                             min_samples_split=60, max_features='sqrt', subsample=0.7, random_state=10)

            grd_enc = OneHotEncoder()
            grd_lm = LogisticRegression(solver='liblinear', max_iter=2000, class_weight='balanced', C=0.15)
            sample_weights = class_weight.compute_sample_weight('balanced', y_train["分组0"])
            grd.fit(X_train, y_train, sample_weight=sample_weights)
            grd_enc.fit(grd.apply(X_train)[:, :, 0])

            # 使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
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
            print(y)

            return grd_lm, y, y_score

    def train(self, source):
        logger.info(f"train begin")
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
        if self.config.get("has_feature_selection", False):
            # 特征选择
            X_train = self.feature_selection(X_train, y_train)

        # TODO, trick, 待解决CieDataFrame的warning和set问题
        class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train["分组0"])
        # class_weights = dict(zip(np.unique(y_train), class_weights))
        sample_weights = class_weight.compute_sample_weight("balanced", y_train["分组0"])

        model_candidates = {
            'SVC':
                {
                    "estimator": SVC(gamma=0.001, C=0.25, class_weight='balanced', probability=True),
                    "param": {'C': [0.5]},
                    "fit_params": {"sample_weight": sample_weights}
                },
            'GradientBoostingClassifier':
                {
                    "estimator": GradientBoostingClassifier(learning_rate=0.01, n_estimators=800, max_depth=8,
                                                            min_samples_split=60, max_features='sqrt', subsample=0.8,
                                                            random_state=10),
                    "param": {'n_estimators': [800],
                              'min_samples_split': [60]},
                    "fit_params": {"sample_weight": sample_weights}
                },
            'XGBClassifier':
                {
                    "estimator": XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.01,
                                               min_child_weight=1, subsample=0.8,
                                               n_estimators=800, max_depth=8),
                    # "param": {'min_child_weight': [1, 2, 3, 4], 'max_depth': [4,6,8,10]},
                    "param": {'min_child_weight': [2], 'max_depth': [4]},
                    "fit_params": {"sample_weight": sample_weights}
                }
        }
        # model_candidates = {
        #             'LogisticRegression':
        #                 {
        #                     "estimator": LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights,
        #                                                     multi_class='multinomial'),
        #                     "param": {'solver': ['lbfgs']},
        #                     "fit_params": {"sample_weight": sample_weights}
        #                 },
        #             'GradientBoostingClassifier':
        #                 {
        #                     "estimator": GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=8,
        #                                                             min_samples_split=50),
        #                     "param": {'n_estimators': list(range(500, 600, 50)), 'max_depth': list(range(3, 10, 2)),
        #                               'min_samples_split': list(range(50, 60, 10)), 'learning_rate': [0.1, 0.6, 2]},
        #                     "fit_params": {"sample_weight": sample_weights}
        #                 },
        #             'XGBClassifier':
        #                 {
        #                     "estimator": XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
        #                                                n_estimators=500, max_depth=8),
        #                     "param": {'n_estimators': range(500, 600, 50), 'max_depth': range(5, 10, 5),
        #                               'learning_rate': [0.5, 0.1, 1]},
        #                     "fit_params": {"sample_weight": sample_weights}
        #                 }
        #         }
        model_initialized = False
        if self.config.get("has_tuning_param", False):
            # 参数tuning
            best_model_candidates = dict()
            for key, value in model_candidates.items():
                best_est = self.tuning_param(X_train, y_train, estimater=value["estimator"],
                                             param_grid=value["param"], fit_params=value["fit_params"])
                best_model_candidates[key] = best_est

            if self.config.get("has_model_combination", False):
                # estimator_l2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=50, max_depth=8,
                #                                           min_samples_split=20)
                estimator_l2 = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
                model, X_train, _ = self.model_combination(estimator_l1=best_model_candidates,
                                                           estimator_l2=estimator_l2, X=X_train, y=y_train)
                model_initialized = True

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
        logger.info(f"predict begin")
        Xy, Xy_columns = self._read_excel(source)
        if self.config.get("has_imputation", False):
            # eda分析
            stat_missing_values(Xy)
        X = Xy.drop(['PID', '分组', '分组0'], axis=1)
        if features is not None:
            X = X[features]
        X = CieDataFrame(X)
        # y = Xy["分组"].to_frame()
        # y = CieDataFrame(y)
        if self.config.get("has_feature_selection", False):
            # 特征选择
            X = self.feature_selection(X, train=False)

        if self.config.get("has_model_combination", False):
            model, X, _ = self.model_combination(X=X, train=False)

        if self.config.get("use_gbdt_lr", False):
            model, X, X_score = self.gbdt_lr(X, train=False)
            return X_score
        else:
            # with open(self.model_file, 'rb') as f:
            #     model = pickle.load(f)
            logger.info(f"predict end")
            return model.predict_proba(X)

    def stat(self, train_source, test_source, output_file=None, sheet_name="sheet"):
        logger.info(f"stat begin")
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
        if self.config.get("has_feature_selection", False):
            # 特征选择
            X_train = self.feature_selection(X_train, train=False)
            X_test = self.feature_selection(X_test, train=False)

        if self.config.get("has_model_combination", False):
            model, X_train, _ = self.model_combination(X=X_train, train=False)
            _, X_test, _ = self.model_combination(X=X_test, train=False)

        if self.config.get("use_gbdt_lr", False):
            _, y_train_pred, y_train_score = self.gbdt_lr(X_train, train=False)
            _, y_test_pred, y_test_score = self.gbdt_lr(X_test, train=False)
            data = {"train": (y_train, y_train_pred, y_train_score), "test": (y_test, y_test_pred, y_test_score)}
            # print(data)
            output_metrics_to_excel_v2(output_file=output_file,
                                       sheet_name=sheet_name + "_gbdt_lr" if self.config.get("use_gbdt_lr", False)
                                       else sheet_name + "_stacking",
                                       data=data)
        else:
            # with open(self.model_file, 'rb') as f:
            #     model = pickle.load(f)

            print(model.classes_)
            data = {"train": (X_train, y_train), "test": (X_test, y_test)}
            # print(data)
            output_metrics_to_excel(model,
                                    output_file=output_file,
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
    features = """碱性磷酸酶_combine0
    血小板压积（PCT）_combine0
    血浆D-二聚体（D-Dimer）_combine0
    亚硝酸盐_combine0
    淋巴细胞计数_combine0
    白蛋白/球蛋白_combine0
    肌酐_combine0
    胆红素_combine0
    淋巴细胞百分比_combine0
    门冬氨酸氨基转移酶_combine0
    颜色_combine0
    乙型肝炎e抗体（HBeAb）_num_combine0
    乙型肝炎核心抗体（HBcAb）_num_combine0
    乙型肝炎表面抗体（HBsAb）_num_combine0
    乙型肝炎表面抗原（HBsAg）_num_combine0
    凝血酶原时间（PT）_num_combine0
    总胆红素
    甲胎蛋白
    粪白细胞
    葡萄糖
    血小板计数（PLT）
    乙型肝炎e抗原（HBeAg）_num
    直接胆红素_combine0
    抗HIV(1+2)抗体【【】】
    中性粒细胞百分比（GRAN%）_combine0
    前白蛋白_combine0
    国际标准化比值_combine0
    嗜碱性粒细胞百分比（BASO%）_combine0
    年龄
    活化部分凝血活酶时间（APTT）_combine0
    纤维蛋白原（FIB）_combine0
    丙氨酸氨基转氨酶_combine0
    CA199-定性_combine0
    中性粒细胞计数_combine0
    尿素_combine0
    丙型肝炎抗体IgG_combine0
    γ-谷氨酰转肽酶_combine0
    间接胆红素【【计算】】
    CA125_combine0
    CA199-定量_combine0""".split()
    print(len(features))
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
    # est.train(source=train_file)
    est.predict(source=test_file)
    est.stat(train_source=train_file, test_source=test_file, output_file=output_file)
