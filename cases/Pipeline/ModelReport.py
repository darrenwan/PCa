# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:55:50 2019

@author: atlan

sensitivity (true positive rate)
specificity (true negative rate)
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from itertools import product
    
#设置浮点精度,默认小数点后4位
np.set_printoptions(precision=4)
pd.set_option('display.float_format', lambda x: '%.4f' %(x))


def model_report(y_true, y_pred, y_proba):
    '''
    模型报告模块
    计算基于混淆矩阵的基础指标：准确率、SEN、SPE、F1、PPV、NPV，以及样本量Support
    
    Params:
    y_true : array, shape = [n_samples]
    Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
    Estimated targets as returned by a classifier.
    
    y_proba : array, shape = [n_samples, n_classes]
    Target probability, probability estimates of target classes as returned
    by "predict_proba" on classsifier, confidence values, or non-thresholded 
    measure of decisions(as returned by "decision_function" on some classifiers).
    
    Returns:
        混淆矩阵
        总体准确率
        包含多指标的预测报告
    '''
    #通过混淆矩阵计算如下6类指标
    confusion = confusion_matrix(y_true, y_pred)
    #print(confusion)
    classes = np.unique(y_true) #类名
    num_classes = len(classes) #类个数
    num_samples = confusion.sum() #support
    
    assert num_classes == y_proba.shape[1]
    
    # 整体的准确率
    acc = np.sum([confusion[i, i] for i in range(num_classes)]) / confusion.sum()
       
    report = []
    for i, class_ in enumerate(classes):       
        pred_pos_true = confusion[i, i] #预测正确阳性
        true_pos = confusion[i, :].sum() #真实阳性
        pred_pos = confusion[:, i].sum() #预测为阳性
        pred_neg = num_samples - pred_pos #预测为阴性
        true_neg = num_samples - confusion[i, :].sum() #真实阴性 = 总数-真实阳性
        neg_index = [j for j in range(num_classes) if j != i] #阴性索引
        neg_coordinate = product(neg_index, repeat=2) #confusio matrix上的阴性坐标        
        pred_neg_true = np.sum([confusion[x] for x in neg_coordinate]) #预测正确阴性 = 阴性坐标上的值
        
        #敏感性
        sensitivity =  pred_pos_true / true_pos
        #特异性
        specificity = pred_neg_true / true_neg
        #阳性预测值        
        positiveValue = pred_pos_true / pred_pos
        #阴性预测值       
        negativeValue = pred_neg_true / pred_neg
        #F1 score
        f1_score = 2 * positiveValue * sensitivity / (positiveValue + sensitivity)
        #AUC
        class_true = np.zeros_like(y_true)
        class_true[y_true == i] = 1
        auc = roc_auc_score(class_true, y_proba[:, i])
        #support
        support = true_pos
        
        report.append([sensitivity, specificity, positiveValue, negativeValue, f1_score, auc, support])
        
    columns = ['敏感性','特异性','阳性预测值','阴性预测值','F1-score','AUC','support']
    report = pd.DataFrame(report, index=classes, columns=columns)
    report.to_excel('modelReport.xlsx')
    return confusion, acc, report
