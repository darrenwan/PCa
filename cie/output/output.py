from functools import reduce
from cie.evaluate.metrics import score_metrics
from cie.datasource import *
import numpy as np
import pandas as pd


def _stat_metrics(model, *args):
    data_set, (X_val, y_val) = args
    y_val_pred = model.predict(X_val)
    y_val_pred_score = model.predict_proba(X_val)

    labels, auc, tpr, spc, ppv, npv, f1, acc, support = score_metrics(y_val, y_val_pred,
                                                                      score_classes=(y_val_pred_score, model.classes_))
    data_type = [data_set] * len(labels)
    res = np.array(
        reduce(lambda x1, x2: np.vstack((x1, x2)), [data_type, labels, auc, tpr, spc, ppv, npv, f1, acc, support])).T
    res = pd.DataFrame(res)
    return res


def output_metrics_to_excel(model, output_file=None, sheet_name='sheet1', data=None):
    """
    输出'数据集', '分组', 'support', '敏感性', '特异性', '阳性预测值', '阴性预测值', 'f1_score'到excel
    :param model: 模型
    :param output_file: 输出excel文件名
    :param sheet_name: sheet名
    :param data: dict, 训练测试数据。key为字符串,比如"train"；value为tuple(x, y)
    :return:
    """
    all_metrics = map(lambda arg: _stat_metrics(model, *arg), data.items())
    data = reduce(lambda x1, x2: pd.concat([x1, x2]), all_metrics)
    data.columns = ['数据集', '分组', 'auc', '敏感性', '特异性', '阳性预测值', '阴性预测值', 'f1_score', 'acc', 'support']
    channel = ExcelChannel(output_file)
    channel.open()
    params = {'sheet_name': sheet_name, 'index': False}
    channel.write(data=data[['数据集', '分组', 'support', '敏感性', '特异性', '阳性预测值', '阴性预测值', 'f1_score']], **params)
    channel.close()