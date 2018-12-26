import types
import numpy as np
import scipy as sp
import scipy.stats
from copy import deepcopy
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.metrics import get_scorer


class SequentialForwardSelector(BaseEstimator, MetaEstimatorMixin):
    """
    特征序列前向选择，以score无提升作为stopping criterion.
    """
    def __init__(self, estimator,
                 verbose=0, scoring=None,
                 cv=5, n_jobs=1,
                 persist_features=None,
                 pre_dispatch='2*n_jobs',
                 clone_estimator=True):

        self.estimator = estimator
        self.pre_dispatch = pre_dispatch
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.clone_estimator = clone_estimator

        if self.clone_estimator:
            self.est_ = clone(self.estimator)
        else:
            self.est_ = self.estimator
        self.scoring = scoring

        if scoring is None:
            if self.est_._estimator_type == 'classifier':
                scoring = 'accuracy'
            elif self.est_._estimator_type == 'regressor':
                scoring = 'r2'
            else:
                raise AttributeError('Estimator must '
                                     'be a Classifier or Regressor.')
        if isinstance(scoring, str):
            self.scorer = get_scorer(scoring)
        else:
            self.scorer = scoring

        self.fitted = False
        self.subsets_ = {}
        self.interrupted_ = False

        self.selected = None

        # don't mess with this unless testing
        self._TESTING_INTERRUPT_MODE = False
        self.persist_features = persist_features

    def fit(self, X, y):
        """Perform feature selection and learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        if hasattr(X, 'loc'):
            X_ = X.values
        else:
            X_ = X

        from sklearn.model_selection import cross_val_score, StratifiedKFold
        max_k = X_.shape[1]
        if self.persist_features is not None:
            remaining = list(set(range(max_k)).difference(set(self.persist_features)))
            selected = set(self.persist_features.copy())
        else:
            remaining = list(range(max_k))
            selected = set()
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            all_subsets = []
            all_avg_scores = []
            all_cv_scores = []
            for candidate in remaining:
                features = selected | {candidate}
                cv_scores = cross_val_score(self.estimator,
                                        X_[:, tuple(features)], y,
                                        cv=self.cv,
                                        scoring=self.scoring,
                                        n_jobs=self.n_jobs,
                                        pre_dispatch="2*n_jobs")
                all_cv_scores.append(cv_scores)
                all_avg_scores.append(np.nanmean(cv_scores))
                all_subsets.append(candidate)
            # 找出最高的分数
            best = np.argmax(all_avg_scores)
            best_new_score = all_avg_scores[best]
            if current_score < best_new_score:
                remaining.remove(all_subsets[best])
                selected.add(all_subsets[best])
                current_score = best_new_score
                k = len(selected)
                self.subsets_[k] = {
                    'feature_idx': selected,
                    'cv_scores': all_cv_scores[best],
                    'avg_score': all_avg_scores[best]
                }
            else:
                # 没有提升，则stop criterion
                break
        self.selected = list(selected)
        self.fitted = True
        return self

    def transform(self, X):
        """Reduce X to its most important features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}

        """
        self._check_fitted()
        if hasattr(X, 'loc'):
            X_ = X.values
        else:
            X_ = X
        return X_[:, self.selected]

    def fit_transform(self, X, y, **fit_params):
        """Fit to training data then reduce X to its most important features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.

        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}

        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def get_metric_dict(self, confidence_interval=0.95):
        """Return metric dictionary

        Parameters
        ----------
        confidence_interval : float (default: 0.95)
            A positive float between 0.0 and 1.0 to compute the confidence
            interval bounds of the CV score averages.

        Returns
        ----------
        Dictionary with items where each dictionary value is a list
        with the number of iterations (number of feature subsets) as
        its length. The dictionary keys corresponding to these lists
        are as follows:
            'feature_idx': tuple of the indices of the feature subset
            'cv_scores': list with individual CV scores
            'avg_score': of CV average scores
            'std_dev': standard deviation of the CV score average
            'std_err': standard error of the CV score average
            'ci_bound': confidence interval bound of the CV score average

        """
        self._check_fitted()
        fdict = deepcopy(self.subsets_)
        for k in fdict:
            std_dev = np.std(self.subsets_[k]['cv_scores'])
            bound, std_err = self._calc_confidence(
                self.subsets_[k]['cv_scores'],
                confidence=confidence_interval)
            fdict[k]['ci_bound'] = bound
            fdict[k]['std_dev'] = std_dev
            fdict[k]['std_err'] = std_err
        return fdict

    @staticmethod
    def _calc_confidence(ary, confidence=0.95):
        std_err = scipy.stats.sem(ary)
        bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(ary))
        return bound, std_err

    def _check_fitted(self):
        if not self.fitted:
            raise AttributeError('SequentialForwardSelector has not been fitted, yet.')
