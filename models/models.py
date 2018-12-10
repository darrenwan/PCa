# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:35:04 2018

@author: atlan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,\
    GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import roc_curve
from evaluate import classific_report, plot_roc
from sklearn.model_selection import cross_validate



class Modeling():
    """
    该类实例化时接受一个目标参数target指定任务类型，包含“分类,回归,聚类”
    """
    def __init__(self, target):
        if target == 'classification':
            self.models = [LogisticRegression(), 
                           SVC(), 
                           MLPClassifier(),
                           RandomForestClassifier(),
                           GradientBoostingClassifier(),
                           ]
        elif target == 'regression':
            self.models = [LinearRegression(), 
                           SVR(),
                           MLPRegressor(),
                           RandomForestRegressor(),
                           GradientBoostingRegressor(),
                           ]
        elif target == 'cluster':
            self.models = [KMeans]
        else:
            raise Exception("Available values of 'target' are 'classfication','regression',\
                            'cluster'; but got %s" % target)
            
    def set_model(model,**kwargs):
        """
        设置模型参数
        """
        
        model.set_params(**kwargs)
#        return model
            
    def round_numeric(self, x, decimals=3):
        if isinstance(x, float):
            return np.round(x, 3)
        else:
            return x
        
    
    def fit(self, model, X, y):
        model.fit(X, y)
        self.model = model
       
    def model_select(self, train_X, test_X, train_y, test_y, cv=5):
        """
        
        
        """
        print('%d个特征' % train_X.shape[1])
        
        performences = []
        for model in self.models:            
            #全特征
            scores = cross_validate(model, train_X, train_y, cv=cv)            
            train_score = scores['train_score'].mean()
            test_score = scores['test_score'].mean()
            self.fit(model, train_X, train_y)
            model_report = classific_report(model, test_X, test_y)
            performences.append(model_report + [train_score, test_score])
        print(performences)
           


    def cross_validation(self, model, X, y, cv=5):
        scores = cross_validate(model, X, y, cv=cv)
        return scores
        
    def predict(self, X):
        
        pred = self.model.predict(X)
        return pred
    
    
    
        