# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:59:17 2018

@author: atlan
"""

import numpy as np
import pandas as pd



#读取数据
from getData import Get_data

datafile = '201800904胰腺癌数据集参数选择两两入组+特征.csv'


getData = Get_data(datafile)

data = getData.read_file(encoding='gbk', engine='python')
print(data.shape)

#数据预处理
from preprocessing import Preprocess

Preprocessor = Preprocess()

miss = Preprocessor.miss_rate(data)

from dataManipulate import get_model_fields
categorical_col = get_model_fields('缺失率统计.csv', '特征')
numeric_col = ['年龄']

from dataManipulate import seperate_X_y, split_dataset
X, y = seperate_X_y(data, '分组')
Preprocessor.fillna(X, categorical_col, 'additional')


from dataManipulate import normalize_data

train_X, test_X, train_y, test_y = normalize_data(X, y, numeric_col, categorical_col, 0.3)


#建模
from models import Modeling
Model = Modeling('classification')

Model.model_select(train_X, test_X, train_y, test_y, cv=5)




