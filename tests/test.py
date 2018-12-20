from cie.datasource import *
from cie.model_selection import *
from sklearn.ensemble import GradientBoostingClassifier
from cie.evaluate.evaluate import *
from cie.evaluate.metrics import *
from functools import reduce
import pandas as pd

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

# 拆分训练集和验证集
seed = 301
X = Xy.drop(['PID', '分组'], axis=1)
y = Xy["分组"]
X_test = Xy_test.drop(['PID', '分组'], axis=1)
y_test = Xy_test["分组"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)

df_train = pd.concat([X_train, y_train], axis=1)
df_val = pd.concat([X_val, y_val], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# 模型训练
# model = GradientBoostingClassifier(random_state=10)
# model.fit(X_train, y_train)
# gs_param = {'n_estimators': range(100, 250, 10)}
# max_depth = {'n_estimators': range(5, 20, 5)}
# gs_param = {'min_samples_split': range(10, 60, 10)}
# gs_param = {'learning_rate': [0.05, 0.1, 0.15]}
best_estimator = GradientBoostingClassifier(learning_rate=0.05, n_estimators=230, max_depth=5, min_samples_split=50)
best_estimator.fit(X_train, y_train)
# best_estimator = grid_search_tuning(
#     GradientBoostingClassifier(learning_rate=0.1, n_estimators=230, max_depth=5, min_samples_split=50),
#     X_train, y_train, X_test, y_test, param_grid=gs_param, cv=5,
#     scores={"neg_log_loss": ("neg_log_loss", None)}, features_names=X.columns,
#     regressor=False)
model = best_estimator


def stat_metrics(data_set, X_val, y_val):
    # 测试
    y_val_pred = model.predict(X_val)
    y_val_pred_score = model.predict_proba(X_val)

    # 指标输出
    labels, auc, tpr, spc, ppv, npv, f1, acc, support = score_metrics(y_val, y_val_pred,
                                                                      score_classes=(y_val_pred_score, model.classes_))
    data_type = [data_set] * len(labels)
    res = np.array(
        reduce(lambda x1, x2: np.vstack((x1, x2)), [data_type, labels, auc, tpr, spc, ppv, npv, f1, acc, support])).T
    res = pd.DataFrame(res)
    return res


all_metrics = map(lambda arg: stat_metrics(*arg),
                  [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)])
data = reduce(lambda x1, x2: pd.concat([x1, x2]), all_metrics)
data.columns = ['数据集', '分组', 'auc', '敏感性', '特异性', '阳性预测值', '阴性预测值', 'f1_score', 'acc', 'support']
channel = ExcelChannel("/Users/wenhuaizhao/works/ml/CIE/tests/cie/data/table2和3的GBDT与CA199评价指标_20181217_2.xlsx")
channel.open()
params = {'sheet_name': 'table3的CA199阴性的GBDT评价指标', 'index': False}
channel.write(data=data[['数据集', '分组', 'support', '敏感性', '特异性', '阳性预测值', '阴性预测值', 'f1_score']], **params)
channel.close()


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
