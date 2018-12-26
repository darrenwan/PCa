from cie.feature_selection.sfs_alg import SequentialForwardSelector
from cie.feature_selection.lr12 import FeatureSelectionLr12
import sklearn.feature_selection as fs
import mlxtend.feature_selection as xfs
import pandas as pd
from abc import ABCMeta, abstractmethod
import six


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
    def fit(self, X, y):
        self.columns = X.columns
        return super(SequentialFeatureSelector, self).fit(X, y)

    def transform(self, X):
        arr_X = super(SequentialFeatureSelector, self).transform(X)
        return pd.DataFrame(arr_X, columns=self.columns)


class RFE(fs.RFE):
    """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    ``coef_`` attribute or through a ``feature_importances_`` attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, (default=0)
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from cie.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    def fit(self, X, y):
        self.columns = X.columns
        return super(RFE, self).fit(X, y)

    def transform(self, X):
        arr_X = super(RFE, self).transform(X)
        return pd.DataFrame(arr_X, columns=self.columns)


class Sfs(SequentialForwardSelector):

    def fit(self, X, y):
        self.columns = X.columns
        return super(Sfs, self).fit(X, y)

    def transform(self, X):
        arr_X = super(Sfs, self).transform(X)
        return pd.DataFrame(arr_X, columns=self.columns)


class SelectFromModel(fs.SelectFromModel):

    def fit(self, X, y):
        self.columns = X.columns
        return super(SelectFromModel, self).fit(X, y)

    def transform(self, X):
        arr_X = super(SelectFromModel, self).transform(X)
        return pd.DataFrame(arr_X, columns=self.columns)

