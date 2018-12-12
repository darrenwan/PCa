# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:08:08 2018

@author: atlan
"""

import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from getData import Get_data
from preprocessing import Preprocess

#df = Get_Data('20180806胰腺癌.csv').read_file(encoding='gbk')


def seperate_X_y(data, y_name):
    """
    分离数据集中的X和y
    
    Parameters
    ----------
    data: DataFrame
    y_name: the label name of target variable which needed to predict
    
    Returns
    -------
    List containing train-test split of inputs.
    """
    y = data.pop(y_name).values
    X = data
    return X, y


def split_dataset(X, y, test_size=0.3):
    """
    把数据集随机分割为训练集和测试集
    
    Parameters
    ----------
    X: array-like
    y: with same length / shape[0] as X
    test_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size.
    
    Returns
    -------
    List containing train-test split of inputs.
    """
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    return train_X, test_X, train_y, test_y



def normalize_data(X, y, numeric_col, categorical_col, test_size=0.3):
    """
    使数据集标准化，分类变量做one-hot，数值变量做标准缩放，
    然后对数据做训练集和测试集分割.
    
    Parameters
    ----------
    X: array-like
    y: with same length / shape[0] as X
    numeric_col: column names of numeric features
    categorical_col: column names of categorical features
    test_size: the proportion of the dataset serve as test split
    
    Returns
    -------   
    normalized dataset List containing train-test split of inputs.
    
    """
    Preprocessor = Preprocess()
    y = Preprocessor.label_encode(y, True)
    categorical_data = Preprocessor.onehot_encode(Preprocessor.label_encode(
                            X[categorical_col], True), True)
    numeric_data = Preprocessor.standard_scale(X[numeric_col], True)
    X = np.hstack([categorical_data, numeric_data])
    trainX,testX,trainY,testY = split_dataset(X, y, test_size)
        
    return trainX, testX, trainY, testY

def drop_cols(X, cols):
    """
    删掉数据集中的某些列
    
    Parameters
    ----------
    X: a DataFrame 
    cols: array-like, the columns you want to remove
    
    Returns
    -------
    X_new: DatFrame whose cols have been removed
    """
    if not isinstance(X, pd.DataFrame):
        raise Exception("Expected data format is {}, but got {}".format('pd.DataFrame', type(X)))
    if not isinstance(cols, collections.Iterable):
        raise Exception('The cols must be iterable, such as list,array,tuple...') 
    for col in cols:
        X.pop(col)
    return X


def get_model_fields(file, col_name):
    """
    获取模型需要化验项名
    return: list
    """
#    file = "data/模型需要的表型.csv"
    table = pd.read_csv(file, encoding="gbk", engine='python')
    assert col_name in table.columns, "Column %s not found!" % col_name
    fields = table[col_name].unique().tolist()
    return fields

   
def get_fields_default(file):
    """
    获取模型有默认值的化验项名及默认值
    return: dict
    """
    file = "data/模型需要的表型.csv"
    table = pd.read_csv(file, encoding="gbk")
    fields = table[table["默认值"].notnull()][['标准项目名称', '默认值']]
    fileds_default = {row[0]:row[1] for _,row in fields.iterrows()}
    return fileds_default


