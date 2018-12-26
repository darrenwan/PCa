import pandas as pd


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
            if isinstance(data_x, pd.DataFrame):
                if data_x.shape[1] == 1:
                    data_x = data_x[data_x.columns[0]]
        else:
            data_x = pd.DataFrame(X, columns=X_columns)
        return data_x
