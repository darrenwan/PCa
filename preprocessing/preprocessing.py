# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:57:56 2018

@author: atlan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,\
    GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_array


        
class Preprocess():
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
            #按行删除
            X[col] = X[col].dropna(axis=0, inplace=False)
        elif method == 'remove_col':
            #按列删除
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
            X[col] = X[col].fillna('UN', inplace=False) #'UN' means unknown
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
    
        
        

        
        