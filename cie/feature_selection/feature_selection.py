from sklearn.linear_model import LogisticRegression
import numpy as np


class FeatureSelectionLr12(LogisticRegression):
    """
    对于l1 penalty中系数为零的特征，并不代表他们不重要，可能只是几个同等相关性特征中保留下来的一个。
    Lr12的目的，是找出这些同等相关性特征，并把他们的重要性设置相同。
    """

    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=0.01,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None):

        self.threshold = threshold
        common_params = {'dual': dual, 'tol': tol, 'C': C, 'fit_intercept': fit_intercept,
                         'intercept_scaling': intercept_scaling, 'class_weight': class_weight,
                         'random_state': random_state, 'solver': solver, 'max_iter': max_iter,
                         'multi_class': multi_class, 'verbose': verbose, 'warm_start': warm_start, 'n_jobs': n_jobs}
        super(FeatureSelectionLr12, self).__init__(penalty='l1', **common_params)
        self.lr_l2 = LogisticRegression(penalty='l2', **common_params)

    def fit(self, X, y, sample_weight=None):
        super(FeatureSelectionLr12, self).fit(X, y, sample_weight=sample_weight)
        self.lr_l2.fit(X, y, sample_weight=sample_weight)
        l1_coef_ = self.coef_
        l2_coef_ = self.lr_l2.coef_

        def is_zero_like(x):
            return abs(float(x)) <= np.finfo(np.float).eps

        rows, cols = self.coef_.shape
        for row in range(rows):
            for col in range(cols):
                coef_ = l1_coef_[row][col]
                if not is_zero_like(coef_):
                    all_indexes = [col]
                    # 如果某l1系数a和l2系数b相差不大的情况下，
                    # 把l2中所有和这个l1系数相差不大（<threshold)且对应的l1系数为零的位置全部找出来,总计n个，
                    # 然后用a/n来替换所有这些位置的l1系数
                    all_indexes.extend([k for k in range(cols) if abs(
                        l2_coef_[row][col] - l2_coef_[row][k]) < self.threshold and col != k and l1_coef_[row][k] == 0])
                    all_indexes = np.asarray(all_indexes)
                    l1_coef_[row][all_indexes] = coef_ / len(all_indexes)
        return self
