# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:31:33 2019

@author: atlan
"""

import numpy as np
import pandas as pd
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import re
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from time import time
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from configparser import ConfigParser, ExtendedInterpolation
import pymysql.cursors


import warnings
warnings.filterwarnings("ignore")

#读取配置文件
config_file='Configure.ini'     
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file, encoding='utf-8')

#设置浮点精度
precision = config.get('DEFAULT', 'FloatPrecision')
np.set_printoptions(precision=precision)
pd.set_option('display.float_format', lambda x: '%.{}f'.format(precision) %(x))


class DataSet():
    def __init__(self):
        '''
        初始化参数:
            特征类型统计表
            训练数据集特征列表
            记录预处理流程的Logger文件
            训练好的Model文件
        '''
        self.Label_file = config.get('TRAIN', 'Label_file')
        self.Label_name = config.get('TRAIN', 'Label_name')
        self.Label_encode = eval(config.get('TRAIN', 'Label_encode'))
        feat_file = config.get('TASK_ATTR', 'Feature_file')
        self.Numeric = pd.read_excel(feat_file, index_col='feat') 
        self.Features = self.Numeric.index.tolist() #原始特征
        self.Features.remove('PID')
        self.Examine_name = config.get('TRAIN', 'Examine_name')
        
        train_file = config.get('TASK_ATTR', 'Train_file')
        self.TrainFeats = pd.read_excel(train_file).columns.tolist()
        log_file = config.get('TASK_ATTR', 'Log_file')
        self.Log = joblib.load(log_file) #特征操作日志
        model_file = config.get('TASK_ATTR', 'Model_file')
        self.model = joblib.load(model_file)
        self.mysql = eval(config.get('DEPLOY', 'MySQL'))

       
    def trim(self, x):
        if isinstance(x, str):
            x = x.strip()
            x = x.replace('<', '')
            x = x.replace('>', '')
            x = x.replace('=', '')
        return x


    def check_numeric(self, data):
        """
        Input：DataFrame
        Output：1) 伪分类列转换后的data                
        """
#        data = pd.read_excel(df)
        print('trim')
        data = data.applymap(self.trim)
        print(data.shape)       
        for k, c in enumerate(data.columns):
            check = self.Numeric.loc[c]['check']
            print(c, check)
            if check:
                column = data[c]
                values = column.unique()               
                strs = []
                for i in values:
                    if not isinstance(i, float): #非浮点
                        try:
                            float(i)
                        except ValueError:                            
                            strs.append(i)
                if len(strs) > 0:
                    data[c] = data[c].replace(strs, np.nan) #删掉字符串
                    data[c] = data[c].astype(float)
                    print(data[c].head())
        print(data.shape)
        return data

    def label(self):
        """
        读取label数据,里面包含"年龄", "性别"
        """   
        label = pd.read_csv(self.Label_file, sep='\t', encoding='utf-8', engine='python')
        label['性别'] = label['性别'].apply(lambda x: 0 if x=='男' else 1)
        Ys = self.Label_encode
        label[self.Label_name] = label[self.Label_name].apply(lambda x: Ys[x] if x in Ys else '-1')
        label = label[[isinstance(i, int) for i in label[self.Label_name]]]
        label_file_header = eval(config.get('TRAIN', 'Label_file_header'))
        label = label[label_file_header]
        return label

    def read_mysql(self):
        # Connect to the database
        connection = pymysql.connect(**self.mysql)        
        try:
            with connection.cursor() as cursor:
                # Read a single record
                sql = "select * from `lab_result_for_ai`"
                cursor.execute(sql)
                result = cursor.fetchall()
                print(result[:1])
        finally:
            connection.close()
      
        #化验标准名转化
        examineName = pd.read_csv(self.Examine_name, sep='\t', encoding='utf-8', engine='python')
        examineMap = {i[0]: i[1] for i in examineName.values}
           
        data = defaultdict(dict) #带仪器信息
        for row in tqdm(result):
            pid = row['PATIENT_ID']
            device = row['INSTRUMENT_ID']
            examine = examineMap.get(row['REPORT_ITEM_NAME'], row['REPORT_ITEM_NAME'])
            examine_extended = examine + '【【' + device + '】】'
            result = str(row['RESULT'])
            #特征过滤，选择与训练集一致的特征
            if examine_extended in self.Features: 
                data[pid][examine_extended] = result
            
        data = pd.DataFrame(data).transpose()
        data = self.check_numeric(data)
        print('Data: ',data)
        label = self.label()
        data = pd.merge(data, label, how='inner', left_index=True, right_on='PID')        
        print('Data: ', data)
        #与标准数据集对齐
        stdData = pd.DataFrame(columns=self.Features)
        data = pd.concat([data, stdData], axis=0, join='outer')
               
        data.to_excel('./预测/rawdata.xlsx', index=False)
       
    def read_data(self):
        """
        Params:
            file: csv文件
        Columns:
            ['PID', '仪器编号', '化验单位', '化验名称', '化验时间', '化验结果（定性）新',
            '化验结果（定量）']
        Return:
        self.data: 分型特征原始数据
        """
        assert self.file.endswith('.txt')
        dataset = pd.read_csv(self.file, sep='\t', encoding='utf-8', engine='python', nrows=None)
        print(dataset.shape)
#        dataset['性别'] = dataset['性别'].apply(lambda x: 0 if x=='男' else 1)
        dataset.fillna('', inplace=True)
        #化验标准名转化
        examineName = pd.read_csv(self.Examine_name, sep='\t', encoding='utf-8', engine='python')
        examineMap = {i[0]: i[1] for i in examineName.values}
       
        data = defaultdict(dict) #带仪器信息
        for _, row in tqdm(dataset.iterrows()):
            pid = row['PID']
            device = row['仪器编号']
            examine = examineMap.get(row['化验名称'], row['化验名称'])
            examine_extended = examine + '【【' + device + '】】'
            if row['化验结果（定量）'] != '':
                result = str(row['化验结果（定量）'])
            else:
                result = str(row['化验结果（定性）'])
            #特征过滤，选择与训练集一致的特征
            if examine_extended in self.Features: 
                data[pid][examine_extended] = result
            
        data = pd.DataFrame(data).transpose()
        data = self.check_numeric(data)
       
        label = self.label()
        data = pd.merge(data, label, how='inner', left_index=True, right_on='PID')
        print('Data: ',data)
               
        #与标准数据集对齐
        stdData = pd.DataFrame(columns=self.Features)
        data = pd.concat([data, stdData], axis=0, join='outer')
               
        data.to_excel('./预测/rawdata.xlsx', index=False)
    
    def process(self, transformer, x):
        if isinstance(transformer, tuple): #带mode参数
            processor, mode = transformer[0], transformer[1]['mode']
            x = [i if i in processor.classes_ else mode for i in x]
        else:
            processor = transformer
        if processor.__class__.__name__ == 'SimpleImputer':
            x = x.astype('object')
        x = processor.transform(np.expand_dims(x, axis=1)) #np.expand_dims(x, axis=1)
        return x

    def standardize(self, data):
        '''
        [('合并', numericFeats, combine)]
        [('合并', (feats, FillValue[numeric]), combine)]
        [('转换', '年龄', Std)]
        [('转换', baseFeat, [Encoder, Imp])]
        [('转换', baseFeat, [Imp, (Encoder, {'mode': mode})])]
        [('转换', feat, Encoder)]
        [('合并', (subFeat, 0), baseFeat), ('转换', baseFeat, Imp)]
        [('合并', (subFeat, ''), baseFeat), ('转换', baseFeat, [Imp, (Encoder, {'mode': mode})])]
        '''     
        for items in self.Log:
            for item in items:                
                name, feat, op = item[0], item[1], item[2]
                if name == '合并':
                    subFeat, fillValue = feat[0], feat[1]
                    data[subFeat] = data[subFeat].fillna(fillValue)
                    data[op] = data[subFeat].sum(axis=1)
                elif name == '转换':
                    if isinstance(op, list): #多个
                        for transformer in op:                         
                            data[feat] = self.process(transformer, data[feat])                        
                    else: #单个                      
                        data[feat] = self.process(op, data[feat])
        
        #与训练数据集特征对齐
        NormalData = data[self.TrainFeats]
        NormalData.to_excel('./预测/标准数据.xlsx')


    def predict(self, data):
        data.pop('PID')
        y_true = data.pop('分组').tolist()
        print(data.shape)
        pred = self.model.predict(data)
        print('Pred:', pred)
        print('Score:', accuracy_score(y_true, pred))
        return pred

                
    def main(self):
        self.read_mysql()
        data = pd.read_excel('./预测/rawdata.xlsx')
        self.standardize(data)
        data1 = pd.read_excel('./预测/标准数据.xlsx', )
        pred = self.predict(data1)
        return pred


if __name__ == '__main__':
    processor = DataSet()
    pred = processor.main()
    print(pred)

