from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
import numpy as np


def score_metrics(y_true, y_pred, score_classes=(None, None), average="macro"):
    """
    计算统计指标，返回(labels, auc, tpr, spc, ppv, npv, f1, acc, support)，
    即(标签，AUC, 敏感性, 特异性, 阳性预测值, 阴性预测值, F1, 准确率, 样本量)
    Parameters
    ----------
    y_true : 真实标签
    y_pred : 预测标签
    score_classes : 预测对应的概率分数和对应的标签类
    average : 求auc时候的平均方式
        ``'micro'``:
            全局计算考虑来计算每个label的指标。
        ``'macro'``:
            每个label分别计算指标，再平均。
        ``'weighted'``:
            每个label分别计算指标，以每个label的样本数为权重进行求加权平均。
    Returns
    -------
    (labels, auc, tpr, spc, ppv, npv, f1, acc, support)，
        即(标签，AUC, 敏感性, 特异性, 阳性预测值, 阴性预测值, F1, 准确率, 样本量)
    """
    # labels = unique_labels(y_true, y_pred)
    y_score, labels = score_classes
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    #               Predicted     condition
    # True          tn  tn  fp  tn  tn
    #               tn  tn  fp  tn  tn
    # condition *** fn  fn  tp  fn  fn
    #               tn  tn  fp  tn  tn
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    support = confusion.sum(axis=1)
    fn = support - tp
    tn = confusion.sum() - (fp + fn + tp)

    p = tp + fn
    n = tn + fp
    # recall, sensitivity, tpr, 敏感性（真阳率）
    tpr = tp / p
    # specificity, 特异性
    spc = tn / n
    # precision, 精确度
    ppv = tp / (tp + fp)
    # negative predictive value
    npv = tn / (tn + fn)
    # fall-out / false positive rate
    fpr = fp / n
    # false negative rate
    fnr = fn / p
    # false discovery rate
    fdr = fp / (tp + fp)
    # f1
    f1 = 2 * ppv * tpr / (ppv + tpr)
    # accuracy
    acc = (tp + tn) / (p + n)
    # auc
    if y_score is not None:
        le = LabelEncoder()
        le.fit(labels)
        n_labels = len(labels)
        y_true = le.transform(y_true)
        y_true = label_binarize(y_true, np.arange(n_labels))
        auc = [roc_auc_score(y_true[:, idx], y_score[:, idx], average=average) for idx in range(n_labels)]
    else:
        auc = None
    tpr[np.isnan(tpr)] = 0.0
    ppv[np.isnan(ppv)] = 0.0
    npv[np.isnan(npv)] = 0.0
    f1[np.isnan(f1)] = 0.0
    return labels, auc, tpr, spc, ppv, npv, f1, acc, support


def scores_avg_metrics(y_true, y_pred, score_classes=(None, None)):
    """
    返回一系列average指标
    average : 求auc时候的平均方式
        ``'micro'``:
            全局计算考虑来计算每个label的指标。
        ``'macro'``:
            每个label分别计算指标，再平均。
        ``'weighted'``:
            每个label分别计算指标，以每个label的样本数为权重进行求加权平均。

    Parameters
    ----------
    y_true : 真实标签
    y_pred : 预测标签
    score_classes : 预测对应的概率分数和对应的标签类

    Returns
    -------
        labels: 分类标签
        dct: 字典，以average方法为key，包括（'no_avg'， 'macro', 'micro', 'weighted'），
            value为：ndarray:（precision, recall, f_score, true_sum，auc)，即（精确率，召回率，f-measure, 正样本数，auc),
            其中（precision, recall, f_score, true_sum）的true_sum为None.
        no_avg: 返回一个ndarray, [5, num_labels], 行分别表示precision, recall, f_score, true_sum，auc， 列表示分类数。
    """
    y_score, labels = score_classes
    dct = {}
    le = LabelEncoder()
    le.fit(labels)
    n_labels = len(labels)
    y_indicator_true = le.transform(y_true)
    y_indicator_true = label_binarize(y_indicator_true, np.arange(n_labels))
    for key in ['macro', 'micro', 'weighted']:
        metrics = list(precision_recall_fscore_support(y_true, y_pred, average=key))
        metrics.append(roc_auc_score(y_indicator_true, y_score, average=key))
        dct[key] = np.array(metrics)
    no_avg = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
    auc = [roc_auc_score(y_indicator_true[:, idx], y_score[:, idx], average=None) for idx in range(n_labels)]
    dct['no_avg'] = np.concatenate((np.array(no_avg), np.array(auc).reshape(1, 3)), axis=0)
    return labels, dct
