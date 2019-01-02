import pandas as pd
import sklearn.datasets as datasets
from sklearn.datasets import load_boston

__all__ = ['CieDataFrame', 'make_classification', 'make_regression', 'load_boston']


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    args = locals()
    X, y = datasets.make_classification(**args)
    return CieDataFrame.to_cie_data(X), CieDataFrame.to_cie_data(y)


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    args = locals()
    X, y = datasets.make_regression(**args)
    return CieDataFrame.to_cie_data(X), CieDataFrame.to_cie_data(y)


class CieDataFrame(object):

    @staticmethod
    def to_cie_data(X, X_columns=None):
        """
        一列转化为pandas.Series, 多列转化为pandas.DataFrame
        :param X: 待转化的数据
        :param X_columns: 列名
        :return: 转化化的数据
        """
        if hasattr(X, 'loc'):
            data_x = X
            # if isinstance(data_x, pd.DataFrame):
            #     if data_x.shape[1] == 1:
            #         data_x = data_x[data_x.columns[0]]
        else:
            data_x = pd.DataFrame(X, columns=X_columns)
        return data_x
