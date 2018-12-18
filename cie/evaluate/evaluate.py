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
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV


def grid_search_tuning(estimator, x_train, y_train, x_test, y_test, param_grid=None, cv=5,
                       scores={"acc": ("accuracy", None)}, regressor=False):
    """

    :param estimator:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param param_grid:
    :param cv:
    :param scores:
    :param regressor:
    :return:
    """
    results = dict()
    for score in scores:
        print("# 超参数tuning: %s" % score)
        print()
        score_new = scores[score][0]
        score_func = scores[score][1]
        if scores[score] in ['f1', 'precision', 'recall']:
            score_new = '%s_macro' % score_new
        if regressor:
            gs = GridSearchCV(estimator, param_grid, cv=cv, scoring=score_new)
        else:
            gs = GridSearchCV(estimator, param_grid, cv=cv,
                              scoring={score: score_new}, refit=score, return_train_score=True)
        gs.fit(x_train, y_train)

        if regressor:
            best_index_ = gs.cv_results_["rank_test_score"].argmax()
            best_params_ = gs.cv_results_["params"][best_index_]
            print("交叉验证集上最好的参数：", best_params_)
        else:
            print("交叉验证集上最好的参数：", gs.best_params_)
        print("交叉验证集上的各个参数表现：")
        if regressor:
            means = gs.cv_results_['mean_test_score']
            stds = gs.cv_results_['std_test_score']
        else:
            means = gs.cv_results_['mean_test_' + score]
            stds = gs.cv_results_['std_test_' + score]
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("mean: %0.3f, std: (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        y_true, y_pred = y_test, gs.predict(x_test)
        if regressor:
            print("回归评估 %s: %.3f" % (score, score_func(y_true, y_pred)))
        else:
            print("测试集上的分类效果报告:")
            print()
            print(classification_report(y_true, y_pred))
            print()
        results.update(gs.cv_results_)
    X_axis = results['params']
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("Parameter")
    plt.ylabel("Score")

    ax = plt.gca()
    if not regressor:
        ax.set_ylim(0.01, 1.01)

    # Get the regular numpy array from the MaskedArray

    colormap = plt.cm.get_cmap(lut=len(scores) + 1)
    colors = [colormap(i) for i in range(len(scores))]
    for scorer, color in zip(sorted(scores), colors):
        for sample, style in (('train', '--'), ('test', '-')):
            if regressor:
                sample_score_mean = results['mean_%s_score' % sample]
                sample_score_std = results['std_%s_score' % sample]
            else:
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))
        if regressor:
            best_index = np.nonzero(results['rank_test_score'] == len(results['rank_test_score']))[0][0]
            best_score = results['mean_test_score'][best_index]
        else:
            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    return plt


def plot_learning_curve(estimator, title, X, y, ylim=None, scoring=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

    
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
