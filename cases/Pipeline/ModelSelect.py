# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:57:21 2019

@author: atlan
"""
import os
import numpy as np
import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.externals import joblib
from ModelReport import model_report
from DataProcess import DataSet


#读取配置文件
config_file='Configure.ini'        
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file, encoding='utf-8')
#设置浮点精度
float_precision = config.getint('DEFAULT', 'FloatPrecision')
np.set_printoptions(precision=float_precision)
pd.set_option('display.float_format', lambda x: '%.{}f'.format(float_precision) %(x))


class ModelSelect():
    def __init__(self):
        #设置随机种子
        self.seed = config.get('DEFAULT', 'Random_seed')
        if self.seed == '':
            self.seed = None  
        self.n_jobs = 6
        self.trainData = config.get('TASK_ATTR', 'Train_file')
        self.testData = config.get('TASK_ATTR', 'Test_file')
        self.num_class = config.getint('TASK_ATTR', 'Num_Class')
        self.Data = DataSet()
        
    def set_randomState(self, seed):
        '''
        设置全局随机种子
        '''
        self.seed = seed

    def _instantiate_models(self):        
        lr = LogisticRegression(multi_class='auto', solver='liblinear')
        svc = SVC(gamma='scale', probability=True)
        mlp = MLPClassifier()
        gauss = GaussianNB()
        knn = KNeighborsClassifier()
        ada = AdaBoostClassifier()
        rf = RandomForestClassifier()
        gb = GradientBoostingClassifier()
        xg = xgb.XGBClassifier(objective='multi:softmax',
                               num_class=self.num_class)
        lg = lgb.LGBMClassifier(objective='multiclass')

        self.models = [lr, svc, mlp, gauss, knn, ada, rf, gb, xg, lg]
              

#
#    def feature_select(self, X):        
#        ##特征选取
#        #featTable = pd.read_excel('特征重要性.xlsx')
#        #feats = featTable[featTable[1] >= 0][0].values.tolist()
#        #特征选择
#        var = X.var(axis=0)
#        select = var[var >= 0.05 * (1-0.05)].index.tolist()
#        X = X[select]
#        return X
        

    def model_select(self, scorer=None):
        """
        scorer: 
            参考 https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
        """
        self._instantiate_models()
        train_X, train_y, test_X, test_y = self.Data.read_data()
                
#        k_fold = KFold(n_splits=5,random_state=self.seed)      
        result = []
        for model in self.models:
            name = type(model).__name__
            print(name)
            param_grid = eval(config.get('MODEL_PARAMS', name + '__params'))
            best_params, best_param_model, best_result = self.grid_search(
                    train_X, train_y, model, param_grid, scoring=scorer)            
            mean_train_score = best_result['mean_train_score']
            mean_test_score = best_result['mean_test_score']
            print('Best Results:')
            print('Params:', best_params)
            print('Train score:', mean_train_score)
            print('Test score:', mean_test_score)
            result.append((name, mean_test_score))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        best_model, best_score = result[0][0], result[0][1]
        print('Best Model: %s, Best Score: %f' % (best_model, best_score))
        self.save_model(best_model)
        return best_model
                   
    def grid_search(self, X, y, model, param_grid, **kwargs):
        searcher = GridSearchCV(model, param_grid, n_jobs=self.n_jobs, **kwargs)        
        searcher.fit(X, y)
        best_params = searcher.best_params_
        best_param_model = searcher.best_estimator_
        best_index = searcher.best_index_
        cv_results = pd.DataFrame(searcher.cv_results_)
        best_result = cv_results.iloc[best_index]
        return best_params, best_param_model, best_result
       
           
    def save_model(self, model):
        joblib.dump(model, 'Model')

                
    def feature_importance(self):
        #特征重要性
        assert hasattr(self._best_model, 'feature_importances_'), \
            '%s has no feature_importances_' % self._best_model 
        train_X = self.read_data()[0]
        feats = train_X.columns
        
        feaImp = self._best_model.feature_importances_
        feaImp = [(i, j) for i, j in zip(feats, feaImp)]
        feaImp = sorted(feaImp, key=lambda i: i[1], reverse=True)
        feaImp = pd.DataFrame(feaImp)
        feaImp = feaImp[feaImp[1] > 0]
        feaImp.to_excel('特征重要性.xlsx', index=False)
        print(feaImp)

       
    def main(self):
        self.model_select()

if __name__ == '__main__':
    m = ModelSelect()
    m.main()