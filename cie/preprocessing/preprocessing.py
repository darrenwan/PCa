# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:57:56 2018

@author: atlan
"""

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PowerTransformer, MinMaxScaler, \
    PolynomialFeatures, LabelBinarizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array

import numpy as np
import pandas as pd
from scipy.stats import skew
import scipy.special as ss
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['SkewPowerTransformer', 'OneHotEncoder', 'LabelEncoder', 'StandardScaler', 'PowerTransformer',
           'MinMaxScaler', 'VarianceThreshold', 'PolynomialFeatures', 'LabelBinarizer']


class SkewPowerTransformer(BaseEstimator, TransformerMixin):
    """
    针对偏度大于skew_thresh的特征，进行power转换，转换方式通过method指定，可支持yeo-johnson和box-cox
    详情参考sk-learn的PowerTransformer类。
    """

    def __init__(self, method='yeo-johnson', skew_thresh=0.3):
        self.method = method
        self.skew_thresh = skew_thresh
        self.skew_cols = None
        self.num_cols = None
        self.transformer = PowerTransformer(method=method)

    def fit(self, X, y=None):
        # 找出skew太大的列
        self.columns = X.columns
        X = X.values
        if y is not None:
            y = y[y.columns[0]].values
        skew_cols = Preprocess.find_skew_cols(X, skew_thresh=self.skew_thresh)
        num_cols = X.shape[1]
        self.num_cols = num_cols
        skew_cols.sort()
        self.skew_cols = skew_cols
        if skew_cols is not None and len(skew_cols) > 0:
            self.transformer.fit(X[:, skew_cols], y)
        return self

    def transform(self, X):
        X = X.values
        num_cols = X.shape[1]
        if num_cols != self.num_cols:
            raise ValueError("transform data must be the same dimension of fit data")
        skew_cols = self.skew_cols
        print("skew_cols", skew_cols)
        if skew_cols is not None and len(skew_cols) > 0:
            # 针对skew太大的列使用power变换
            skew_data = self.transformer.transform(X[:, skew_cols])
            x_merge = np.concatenate((X, skew_data), axis=1)
            x_new_cols = []
            for i in range(num_cols):
                if i not in self.skew_cols:
                    x_new_cols.append(i)
                else:
                    x_new_cols.append(num_cols + i)
            x_new = x_merge[:, x_new_cols]
        else:
            x_new = X
            x_new_cols = self.columns
        return pd.DataFrame(x_new, columns=x_new_cols)


class Preprocess(object):
    """
    预处理类.
    本类实现的主要功能有
    (1) fillna: 填补缺失值
    (2) label_encode: 将文本编码转为数字编码
    (3) standard_scale: 标准化缩放, 减去均值, 再除以标准差
    (4) one-hot编码: 将分类整数特征转换为one-of-K形式
    (5) feature_select: 特征选择
    (6) miss_rate: 缺失率统计
    """

    def __init__(self):
        self.ohe = OneHotEncoder()
        self.lbe = LabelEncoder()
        self.sds = StandardScaler()

    def check_X(self, X):
        X_ = check_array(X)
        return X_

    def transform_to_df(self, X, columns=None):
        X_ = pd.DataFrame(X, columns)
        return X_

    def fillna(self, X, col, method):
        """
        Parameters
        ----------
        X: 待处理的DataFrame
        col: 需要填充缺失值的列名称
        method: 填补方法,可选值为
                'remove_row,remove_col,mean,median,linear,additional'
    
        Returns
        -------
        没有返回值, inplace方式填充
        """
        if method == 'remove_row':
            # 按行删除
            X[col] = X[col].dropna(axis=0, inplace=False)
        elif method == 'remove_col':
            # 按列删除
            X[col] = X[col].dropna(axis=1, inplace=False)
        elif method == 'mean':
            fill_value = X[col].mean(0)
            X[col] = X[col].fillna(fill_value, inplace=False)
        elif method == 'median':
            fill_value = X[col].median(0)
            X[col] = X[col].fillna(fill_value, inplace=False)
        elif method == 'linear':
            X[col] = X[col].interpolate(method='linear', inplace=False)
        elif method == 'additional':
            X[col] = X[col].fillna('UN', inplace=False)  # 'UN' means unknown
        else:
            raise Exception("Available methods are ' \
                        remove_row,remove_col,mean,median,linear,additional', but got %s" % method)

    def label_encode(self, X, fit=False):
        """
        Encode labels with value between 0 and n_classes-1.
        
        Parameters
        ----------
        X: array-like of shape [n_samples]
            Target values.
        fit: True时调用的是fit_transform,False时调用的是transform;
            适用场景: 转换训练集时设为True, 随后转换测试集时设为False.
    
        Returns
        -------
        encoded data, array-like of shape [n_samples]
        """
        if X.ndim == 1:
            if fit == True:
                X = self.lbe.fit_transform(X)
            else:
                X = self.lbe.transform(X)
            return X
        elif X.ndim == 2:
            if not isinstance(X, pd.DataFrame):
                X = self.transform_to_df(X)
            if fit == True:
                X = X.apply(self.lbe.fit_transform)
            else:
                X = X.apply(self.lbe.transform)
            return X
        else:
            raise Exception('The input type is wrong.')

    def standard_scale(self, X, fit=False):
        """
        Standardize features by removing the mean and scaling to unit variance
        
        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        fit: True时调用的是fit_transform,False时调用的是transform;
            适用场景: 转换训练集时设为True, 随后转换测试集时设为False.
    
        Returns
        -------
        encoded data, array-like, shape [n_samples, n_features]
        """

        if X.ndim == 1:
            X = X[:, None]
        if fit == True:
            out = self.sds.fit_transform(X)
        else:
            out = self.sds.transform(X)
        return out

    def onehot_encode(self, X, fit=False):
        """
        Fit OneHotEncoder on X.
        Encode categorical integer features using a one-hot aka one-of-K scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            Input array of type int.
        fit: True时调用的是fit_transform,False时调用的是transform;
            适用场景: 转换训练集时设为True, 随后转换测试集时设为False.
            
        Returns
        -------
        encoded data, array-like, shape [n_samples, n_feature]
        """
        if X.ndim == 1:
            X = X[:, None]
        if fit == True:
            out = self.ohe.fit_transform(X).toarray()
        else:
            out = self.ohe.transform(X)
        return out

    def feature_select(self, X, threshold):
        """
        Feature selector that removes all low-variance features.
        
        Parameters
        ----------
        X: numpy array of shape [n_samples, n_features]
            Training set.
        
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
        Transformed array.
        """
        VarSel = VarianceThreshold(threshold)
        X = VarSel.fit_transform(X)
        return X

    def miss_rate(self, X):
        """
        统计数据集中所有特征的数据缺失率
        
        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
        
        Returns
        -------
        特征的缺失率
        """

        if not isinstance(X, pd.DataFrame):
            X = self.transform_to_df(X)
        missRate = X.apply(
            lambda col: 1 - pd.notna(col).sum() / len(col),
            axis=0)
        missRate.sort_values(inplace=True)
        return missRate

    @staticmethod
    def drop_na_col(data, thresh=0.9, data_type='dataframe'):
        """
        删除缺失值大于thresh的列
        :param data:
        :param data_type: dataframe or ndarray
        :param thresh:
        :return: tuple(保留的数据, 保留的列名）
        """
        Preprocess._check_type(data=data, data_type=data_type)
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        df_na = df.isnull().sum() / len(df)
        drop_columns = df_na[df_na > thresh].index.values
        remain = [i for i in df.columns.tolist() if i not in drop_columns]
        if isinstance(data, pd.DataFrame):
            return df[remain], remain
        else:
            return data[:, remain], remain

    @staticmethod
    def exp_method(func=None):
        """
        可通过string来获取值变换函数；或者直接定义函数，供FunctionTransformer使用。
        注意x的值严格要求为非负数。
        可用PowerTransformer来代替boxcox1p，同时Yeo-Johnson支持负数
        :param func: string或者function
        :return:
        """
        dct = {'boxcox1p': (1, ss.boxcox1p),
               'boxcox': (1, ss.boxcox),
               'log1p': (0, np.log1p),
               'log': (0, np.log)}
        # scipy.special的很多函数参数不是以k-v形式传递，作特殊处理。
        special = 0
        if isinstance(func, str):
            if func in dct:
                special = dct[func][0]
                func = dct[func][1]
            else:
                raise ValueError("{func} is not supported".format(func=func))

        def inner_func(x, **kw_args):
            if special:
                lmbda = kw_args.pop("lmbda", None)
                return func(x, lmbda, **kw_args)
            else:
                return func(x, **kw_args)

        return inner_func

    @staticmethod
    def find_skew_cols(data, skew_thresh=0.75):
        """
        找出偏态大于skew_thresh的特征
        :param data: ndarray或者dataframe, 特征数据
        :param skew_thresh: float, skew阈值
        :return: 发生转换的columns名称
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        # series.skew()
        skew_val = df.apply(lambda x: skew(x.dropna(), axis=0, bias=False)).sort_values(ascending=False)
        skew_val = pd.DataFrame({'skew': skew_val})
        skew_val = skew_val[abs(skew_val) > skew_thresh].dropna()
        cols = skew_val.index
        if isinstance(data, pd.DataFrame):
            return np.asanyarray(cols)
        else:
            return np.asanyarray(cols)

    @staticmethod
    def log1p(data):
        """
        将label进行高斯平滑，log1p
        :param data: ndarray或者dataframe, 待平滑的数据.
        :return: tuple（平滑后的数据，列名)
        """
        Preprocess._check_type(data)
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        df = np.log1p(df)
        if isinstance(data, pd.DataFrame):
            return df, np.asanyarray(df.columns.tolist())
        else:
            return df.values, np.asanyarray(df.columns.tolist())

    @staticmethod
    def _check_type(data=None, data_type=None):
        if data_type not in ["dataframe", "ndarray"]:
            raise ValueError("data must be dataframe or ndarray")
        if data is None:
            raise ValueError("data must be set")
        if data_type == 'dataframe':
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be the same with data_type")
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError("data must be the same with data_type")
