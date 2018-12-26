from cie.datasource import *
from cie.model_selection import train_test_split
from cie.models.ensemble import GradientBoostingClassifier, XGBClassifier
from cie.eda.visualization import *
from cie.evaluate.tuning import *
from cie.evaluate.metrics import *
from cie.output import output_metrics_to_excel
from cie.models.ensemble import StackingTransformer
from cie.models.classification import LogisticRegression
from cie.data import CieDataFrame
import pickle


# 模型融合
def model_conbine(X_train, y_train, X_val, y_val, X_test, y_test):
    # Init 1st level estimators
    estimator_l1 = [('LogisticRegression',
                     LogisticRegression(solver='lbfgs', max_iter=200, multi_class='multinomial')),
                    ('GradientBoostingClassifier',
                     GradientBoostingClassifier(learning_rate=0.05, n_estimators=230, max_depth=5,
                                                min_samples_split=50)),
                    ('XGBClassifier', XGBClassifier(random_state=0,
                                                    n_jobs=-1,
                                                    learning_rate=0.1,
                                                    n_estimators=230,
                                                    max_depth=5))]

    X_train_orig, X_test_orig = X_train, X_test
    # Stacking
    stack = StackingTransformer(estimators=estimator_l1,
                                regression=False,
                                shuffle=True,
                                random_state=0,
                                needs_proba=True,
                                verbose=0)
    stack = stack.fit(X_train, y_train)
    X_train = stack.transform(X_train)
    X_test = stack.transform(X_test)
    X_val = stack.transform(X_val)

    estimator_l2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=230, max_depth=5,
                                              min_samples_split=50)
    estimator_l2 = estimator_l2.fit(X_train, y_train)
    y_pred = estimator_l2.predict_proba(X_test)
    stacking_mae = -log_loss(y_test, y_pred, labels=[0, 1, 2])

    for item in estimator_l1:
        single_model = item[1]
        single_model = single_model.fit(X_train_orig, y_train)
        y_pred = single_model.predict_proba(X_test_orig)
        print()
        print(np.unique(y_pred), np.unique(y_test))
        print()
        pre_mae = -log_loss(y_test, y_pred, labels=[0, 1, 2])
        print('单模型%s log_loss: [%.8f]' % (item[0], pre_mae))
    print('stacking log_loss: [%.8f]' % stacking_mae)
    return estimator_l2, X_train, X_val, X_test


def tuning_param(train_x, train_y, test_x, test_y):
    """
    尝试不同的参数，查看指标。
    :return:
    """
    # gs_param = {'n_estimators': range(100, 250, 10)}
    # max_depth = {'n_estimators': range(5, 20, 5)}
    # gs_param = {'min_samples_split': range(10, 60, 10)}
    gs_param = {'learning_rate': [0.05, 0.1, 0.15]}
    # train_x, train_y, test_x, test_y = train_data[train_data.columns.difference(['label'])], train_data['label'], \
    #                                    test_data[train_data.columns.difference(['label'])], test_data['label']

    # 参数tuning
    best_estimator = grid_search_tuning(
        GradientBoostingClassifier(learning_rate=0.1, n_estimators=230, max_depth=5, min_samples_split=50),
        train_x, train_y, test_x, test_y, param_grid=gs_param, cv=5,
        scores={"neg_log_loss": ("neg_log_loss", None)},
        regressor=False)

    return best_estimator


# ===========结果===========
# 数据集	分组	support	敏感性	特异性	阳性预测值	阴性预测值	f1_score
# train	0	2002	0.9310689310689311	0.9806857313969157	0.9352734570998494	0.9793660287081339	0.9331664580725906
# train	1	4102	0.947586543149683	0.8879668049792531	0.8834090909090909	0.949778089231488	0.9143730886850152
# train	2	2577	0.7854093907644548	0.9567496723460026	0.8846153846153846	0.9134991396840294	0.832065775950668
# val	0	861	0.8629500580720093	0.972027972027972	0.9027946537059538	0.9592822636300897	0.8824228028503563
# val	1	1815	0.9322314049586777	0.8588667366211962	0.8628250892401835	0.9301136363636363	0.8961864406779662
# val	2	1045	0.7215311004784689	0.9316143497757847	0.8046958377801494	0.8954741379310345	0.7608476286579214
# test	0	233	0.9012875536480687	0.9349693251533743	0.7984790874524715	0.9707006369426752	0.8467741935483872
# test	1	472	0.8877118644067796	0.8472222222222222	0.8264299802761341	0.9020332717190388	0.8559754851889684
# test	2	343	0.6909620991253644	0.9418439716312057	0.8525179856115108	0.8623376623376623	0.7632850241545893

# ======stacking 融合结果，变差========
# 数据集	分组	support	敏感性	特异性	阳性预测值	阴性预测值	f1_score
# train	0	2863	0.9182675515193852	0.9774609497850928	0.9244022503516175	0.975517890772128	0.9213246889784475
# train	1	5917	0.9496366401892851	0.8892829606784888	0.8866971753195518	0.9508656224237428	0.9170882976987106
# train	2	3622	0.7832689122032026	0.9562642369020501	0.8807823657249302	0.9144973314453764	0.8291684933508696
# test	0	233	0.9012875536480687	0.9288343558282208	0.7835820895522388	0.9705128205128205	0.8383233532934131
# test	1	472	0.8771186440677966	0.8576388888888888	0.8346774193548387	0.894927536231884	0.8553719008264463
# test	2	343	0.6997084548104956	0.9375886524822695	0.8450704225352113	0.8651832460732984	0.7655502392344498


def feature_selection(X, y):
    print("begin to select features")
    from cie.feature_selection import Sfs
    # from mlxtend.feature_selection import SequentialFeatureSelector
    # X = X.iloc[:, :4]
    estimator = GradientBoostingClassifier()
    # sfs = SequentialFeatureSelector(estimator,
    #                                 k_features=2,
    #                                 forward=True,
    #                                 floating=False,
    #                                 verbose=0,
    #                                 scoring='accuracy',
    #                                 cv=5).fit(X, y)
    sfs = Sfs(estimator,
              verbose=0, scoring=None,
              cv=5, n_jobs=2,
              persist_features=None,
              pre_dispatch='2*n_jobs',
              clone_estimator=True).fit(X, y)
    print("selected features:", sfs.selected)
    print("end to select features")
    return sfs


if "__main__" == __name__:
    has_imputation = True
    has_feature_selection = True
    has_split_val = True
    has_tuning_param = False
    has_model_combination = True

    # 因为可能的参数比较多，可以通过定义参数字典来读取文件。
    # 如果一个source里面label和feature，可以通过label_index来指定label，将cols里面的其他列作为feature.
    params = {
        "sep": '\t',
        "encoding": 'utf-8',
        # "nrows": 20,
    }
    # 训练集、验证集
    channel = CsvChannel("/Users/wenhuaizhao/works/ml/CIE/tests/cie/data/胰腺癌原始数据465特征2018前_67特征_pid_CA125fill_18.txt")
    channel.open()
    Xy, Xy_columns = channel.read(**params)
    channel.close()

    # 测试集
    channel = CsvChannel("/Users/wenhuaizhao/works/ml/CIE/tests/cie/data/胰腺癌原始数据465特征2018后_67特征_pid_CA125fill_18.txt")
    channel.open()
    Xy_test, _ = channel.read(**params)
    channel.close()

    if has_imputation:
        # eda分析
        stat_missing_values(Xy)

    X_train = Xy.drop(['PID', '分组'], axis=1)
    y_train = Xy["分组"].to_frame()

    X_train = CieDataFrame.to_cie_data(X_train)
    y_train = CieDataFrame.to_cie_data(y_train)

    if has_split_val:
        seed = 301
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=seed)

    X_test = Xy_test.drop(['PID', '分组'], axis=1)
    y_test = Xy_test["分组"]
    X_test = CieDataFrame.to_cie_data(X_test)
    y_test = CieDataFrame.to_cie_data(y_test)

    if has_feature_selection:
        # 特征选择
        selector = feature_selection(X_train, y_train)

        selector_file = './feature_selection_sfs.pkl'
        with open(selector_file, 'wb') as file:
            pickle.dump(selector, file)
        with open(selector_file, 'rb') as file:
            selector = pickle.load(file)

        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)

    if has_tuning_param:
        # 参数tuning
        tuning_param(X_train, y_train, X_test, y_test)

    # 调优后的参数
    model = GradientBoostingClassifier(learning_rate=0.05, n_estimators=230, max_depth=5, min_samples_split=50).fit(
        X_train, y_train)

    model_file = './model.pkl'
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    if has_model_combination:
        # 模型融合
        model, X_train, X_val, X_test = model_conbine(X_train, y_train, X_val, y_val, X_test, y_test)

    data = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    output_metrics_to_excel(model,
                            output_file="/Users/wenhuaizhao/works/ml/CIE/tests/cie/data/"
                                        "table2和3的GBDT与CA199评价指标_20181217_2.xlsx",
                            sheet_name='table3的CA199阴性的GBDT评价指标', data=data)
