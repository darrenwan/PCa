# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:06:08 2018

@author: atlan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix, \
    roc_curve,brier_score_loss,precision_score,recall_score,f1_score, \
    mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

    
    
def plot_roc(model, X, y):
    """
    Plot ROC curve of 'model' on X and y.

    Parameters
    ----------
    model: fitted model
    X : array-like, shape [n_samples, n_feature]
    y: the label,must be binary(i.e.

    Returns
    -------
    ROC curve
    """
    #plot setting
    plt.style.use("ggplot")
    f = plt.subplot()
    f.set_title('ROC Curve (%s)' % model.__class__.__name__ , {'fontsize':20})
    f.set_xlabel('fpr', {'fontsize': 20})
    f.set_ylabel('tpr', {'fontsize': 20})
    
    proba = model.predict_proba(X)[:,1]
    fpr, tpr, thr = roc_curve(y, proba)
    line, = f.plot(fpr, tpr)
    
    f.legend(line, model.__class__.__name__, loc='lower right', fontsize='large')

def _round(x):
    if isinstance(x, float):
        return np.round(x, 3)
    else:
        return x    
    

def classific_report(model, X, y):
    """
    classification performence of 'model' on X and y.

    Parameters
    ----------
    model: fitted model
    X : array-like, shape [n_samples, n_feature]
    y: the label,must be binary(i.e.

    Returns
    -------
    the values of main evaluation metrics with respect to 'model'
    """
    print(model.__class__.__name__)
    
    pred = model.predict(X)
    print('%d个特征' % X.shape[1])
    acc = model.score(X, y)
    print('准确率', acc)
    
    confusionMatrix = confusion_matrix(y, pred)        
    print(confusionMatrix)
    
    if len(np.unique(y)) == 2:        
        proba = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, proba)
        print('AUC', auc)    

    
    sensitivity = confusionMatrix[0, 0] / confusionMatrix.sum(axis=0)[0]
    specificity = confusionMatrix[1, 1] / confusionMatrix.sum(axis=0)[1]
    print('敏感度: %.3f, 特异度: %.3f' % (sensitivity, specificity))
    
    performence = [model.__class__.__name__, acc, sensitivity, specificity]
    performence = list(map(_round, performence))
    return performence


def regression_report(model, X, y):
    """
    regression performence of 'model' on X and y.

    Parameters
    ----------
    model: fitted model
    X : array-like, shape [n_samples, n_feature]
    y: the label,must be binary(i.e.

    Returns
    -------
    the values of main evaluation metrics with respect to 'model'
    """
    print(model.__class__.__name__)
    performence = []
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    evs = explained_variance_score(y, pred)
    r2 = r2_score(y, pred)
    performence = [model.__class__.__name__, mae, mse, evs, r2]
    performence = list(map(_round, performence))
    return performence

    
    