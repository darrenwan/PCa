from cie.feature_selection.sfs_alg import SequentialForwardSelector
# from cie.feature_selection.lr12 import FeatureSelectionLr12
import sklearn.feature_selection as fs
import mlxtend.feature_selection as xfs
from sklearn.feature_selection import chi2, f_classif, f_oneway, f_regression
import pandas as pd
import numpy as np
# from abc import ABCMeta, abstractmethod
# import six

__all__ = [
    'RFE',
    'SelectKBest',
    'SelectFromModel',
    'VarianceThreshold',
    'chi2',
    'f_classif',
    'f_oneway',
    'f_regression',
    'SequentialFeatureSelector', 'Sfs']


class SequentialFeatureSelector(xfs.SequentialFeatureSelector):
    """Sequential Feature Selection for Classification and Regression.

    Parameters
    ----------
    estimator : scikit-learn classifier or regressor
    k_features : int or tuple or str (default: 1)
        Number of features to select,
        where k_features < the full feature set.
        New in 0.4.2: A tuple containing a min and max value can be provided,
            and the SFS will consider return any feature ensemble between
            min and max that scored highest in cross-validtion. For example,
            the tuple (1, 4) will return any ensemble from
            1 up to 4 features instead of a fixed number of features k.
        New in 0.8.0: A string argument "best" or "parsimonious".
            If "best" is provided, the feature selector will return the
            feature subset with the best cross-validation performance.
            If "parsimonious" is provided as an argument, the smallest
            feature subset that is within one standard error of the
            cross-validation performance will be selected.
    forward : bool (default: True)
        Forward selection if True,
        backward selection otherwise
    floating : bool (default: False)
        Adds a conditional exclusion/inclusion if True.
    verbose : int (default: 0), level of verbosity to use in logging.
        If 0, no output,
        if 1 number of features in current set, if 2 detailed logging i
        ncluding timestamp and cv scores at step.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    cv : int (default: 5)
        Integer or iterable yielding train, test splits. If cv is an integer
        and `estimator` is a classifier (or y consists of integer class
        labels) stratified k-fold. Otherwise regular k-fold cross-validation
        is performed. No cross-validation if cv is None, False, or 0.
    n_jobs : int (default: 1)
        The number of CPUs to use for evaluating different feature subsets
        in parallel. -1 means 'all CPUs'.
    pre_dispatch : int, or string (default: '2*n_jobs')
        Controls the number of jobs that get dispatched
        during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
        Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
        None, in which case all the jobs are immediately created and spawned.
            Use this for lightweight and fast-running jobs,
            to avoid delays due to on-demand spawning of the jobs
        An int, giving the exact number of total jobs that are spawned
        A string, giving an expression as a function
            of n_jobs, as in `2*n_jobs`
    clone_estimator : bool (default: True)
        Clones estimator if True; works with the original estimator instance
        if False. Set to False if the estimator doesn't
        implement scikit-learn's set_params and get_params methods.
        In addition, it is required to set cv=0, and n_jobs=1.

    Attributes
    ----------
    k_feature_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    k_feature_names_ : array-like, shape = [n_predictions]
        Feature names of the selected feature subsets. If pandas
        DataFrames are used in the `fit` method, the feature
        names correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. New in v 0.13.0.
    k_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        sequential selection, where the dictionary keys are
        the lengths k of these feature subsets. The dictionary
        values are dictionaries themselves with the following
        keys: 'feature_idx' (tuple of indices of the feature subset)
              'feature_names' (tuple of feature names of the feat. subset)
              'cv_scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)
        Note that if pandas
        DataFrames are used in the `fit` method, the 'feature_names'
        correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. The 'feature_names' is new in v 0.13.0.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    """
    def fit(self, X, y=None):
        self.columns = X.columns
        if y is not None:
            y = y[y.columns[0]].values
            X = X.values
        return super(SequentialFeatureSelector, self).fit(X, y)

    def transform(self, X):
        arr_X = super(SequentialFeatureSelector, self).transform(X.values)
        print("self.k_feature_idx_", self.k_feature_idx_)
        columns = np.array(self.columns)[list(self.k_feature_idx_)]
        return pd.DataFrame(arr_X, columns=columns)


class RFE(fs.RFE):

    def fit(self, X, y=None):
        self.columns = X.columns
        if y is not None:
            y = y[y.columns[0]].values
            X = X.values
        return super(RFE, self).fit(X, y)

    def transform(self, X):
        arr_X = super(RFE, self).transform(X.values)
        columns = np.array(self.columns)[self.get_support()]
        return pd.DataFrame(arr_X, columns=columns)


class Sfs(SequentialForwardSelector):

    def fit(self, X, y=None):
        self.columns = X.columns
        if y is not None:
            y = y[y.columns[0]].values
            X = X.values
        return super(Sfs, self).fit(X, y)

    def transform(self, X):
        arr_X = super(Sfs, self).transform(X.values)
        columns = np.array(self.columns)[self.selected]
        return pd.DataFrame(arr_X, columns=columns)


class SelectFromModel(fs.SelectFromModel):

    def fit(self, X, y=None):
        self.columns = X.columns
        if y is not None:
            y = y[y.columns[0]].values
            X = X.values
        return super(SelectFromModel, self).fit(X, y)

    def transform(self, X):
        arr_X = super(SelectFromModel, self).transform(X.values)
        columns = np.array(self.columns)[self.get_support()]
        return pd.DataFrame(arr_X, columns=columns)


class VarianceThreshold(fs.VarianceThreshold):
    def fit(self, X, y=None):
        self.columns = X.columns
        if y is not None:
            y = y[y.columns[0]].values
            X = X.values
        return super(VarianceThreshold, self).fit(X, y)

    def transform(self, X):
        arr_X = super(VarianceThreshold, self).transform(X.values)
        columns = np.array(self.columns)[self.get_support()]
        return pd.DataFrame(arr_X, columns=columns)


class SelectKBest(fs.SelectKBest):

    def fit(self, X, y=None):
        self.columns = X.columns
        if y is not None:
            y = y[y.columns[0]].values
            X = X.values
        return super(SelectKBest, self).fit(X, y)

    def transform(self, X):
        arr_X = super(SelectKBest, self).transform(X.values)
        columns = np.array(self.columns)[self.get_support()]
        return pd.DataFrame(arr_X, columns=columns)

