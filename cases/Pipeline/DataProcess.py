# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:16:53 2019

@author: atlan
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import re
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from time import time
import pickle
from sklearn.externals import joblib
from copy import deepcopy
from configparser import ConfigParser, ExtendedInterpolation

import warnings
warnings.filterwarnings("ignore")

#读取配置文件
config_file='Configure.ini'     
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file, encoding='utf-8')
#设置浮点精度
float_precision = config.getint('DEFAULT', 'FloatPrecision')
np.set_printoptions(precision=float_precision)
pd.set_option('display.float_format', lambda x: '%.{}f'.format(float_precision) %(x))


class DataSet():
    def __init__(self):             
        self.rawDir = config.get('TRAIN', 'Data_dir')
        self.labelData = config.get('TRAIN', 'Label_file')
        self.labelName = config.get('TRAIN', 'Label_name')
        self.label_file_header = config.get('TRAIN', 'Label_file_header')
        self.Label_encode = eval(config.get('TRAIN', 'Label_encode'))
        self.seed = config.get('DEFAULT', 'Random_seed')
        if self.seed == '':
            self.seed = None      
        self.feature_file = config.get('TRAIN', 'Specified_feature')
        self.Log = []
        
        #最终标准数据
        self.output_dir = config.get('TASK_ATTR', 'Output_dir')
        self.trainData = config.get('TASK_ATTR', 'Train_file')
        self.testData = config.get('TASK_ATTR', 'Test_file')

        
    def set_randomState(self, seed):
        '''
        设置全局随机种子
        '''
        self.seed = seed
        
        
    def trim(self, x):        
        if isinstance(x, str):            
            x = x.strip()
            x = x.replace('<', '')
            x = x.replace('>', '')
            x = x.replace('=', '')
#            if '-' in x:
#                x = np.array(x.split('-')).astype(int).mean()
        return x
    
    
    def check_numeric(self, data):
        """
        Input：DataFrame formatted data
        Output：1) 伪分类列转换后的data
                2) Numeric
        """
        MaxStrRatio = config.getfloat('TRAIN', 'Max_str_ratio')
        print('trim')
        data = data.applymap(self.trim)
        Numeric = []
        for k, c in enumerate(data.columns):
#            column = df[c].fillna('')
            column = data[c]
            values = column.unique()
            strnum = 0 #字符串数量
            strs = []
            for i in values:
                if not isinstance(i, float): #非浮点
                    try:
                        float(i)
                    except ValueError:
                        strnum += 1
                        strs.append(i)
            strratio = strnum / len(values)
            if strratio <= 0.3:
                check = 1
            else:
                check = 0
            Numeric.append([c, check, strratio, strs])
        Numeric = pd.DataFrame(Numeric, columns=['feat', 'check', 'strratio', 'strs'])
        Numeric = Numeric.set_index('feat')
        Numeric.to_excel(self.output_dir + '特征类型统计.xlsx', index=True)

        categCols = Numeric[np.logical_and(Numeric['strratio'] <= MaxStrRatio,
                                           Numeric['strs'].apply(len) > 0)]
        for feat, row in categCols.iterrows():
            if feat == 'PID':
                continue
            Str = row['strs']
            data[feat] = data[feat].apply(self.trim)
            data[feat] = data[feat].replace(Str, np.nan) #删掉字符串
        return data, Numeric

    def label(self):
        """
        读取label数据
        """        
        label = pd.read_csv(self.labelData, sep='\t', encoding='utf-8', engine='python')
        label['性别'] = label['性别'].apply(lambda x: 0 if x=='男' else 1)
        Ys = self.Label_encode
        label[self.labelName] = label[self.labelName].apply(lambda x: Ys[x] if x in Ys else '-1')
        label = label[[isinstance(i, int) for i in label[self.labelName]]]
        label_file_header = eval(config.get('TRAIN', 'Label_file_header'))
        label = label[label_file_header]
        return label

    def read_rawData(self):
        """
        Columns:
            ['PID', '仪器编号', '化验单位', '化验名称', '化验时间', '化验结果（定性）新',
            '化验结果（定量）']
        Return:
        self.basedata: 无分型原始数据
        self.data: 分型特征原始数据
        """
        collect = []
        root = self.rawDir
        for file in os.listdir(root):
            if not file.startswith('mongo'):
                continue
            filepath = root + file
            data = pd.read_csv(filepath, sep='\t', encoding='utf-8', engine='python')
#            print(data.shape)
            collect.append(data)
        dataset = pd.concat(collect, axis=0, ignore_index=True)
        dataset.fillna('', inplace=True)

        #化验标准名转化
        examine_file = config.get('TRAIN', 'Examine_name')
        examineName = pd.read_csv(examine_file, sep='\t', encoding='utf-8', engine='python')
        examineMap = {i[0]: i[1] for i in examineName.values}

        basedata = defaultdict(dict)
        data = defaultdict(dict) #带仪器信息
        self.Units = defaultdict(dict) #检验项不同仪器的化验单位
        for _, row in tqdm(dataset.iterrows()):
            pid = row['PID']
            device = row['仪器编号']
            unit = row['化验单位'].lower()
            examine = examineMap.get(row['标准项目名称'], row['标准项目名称'])
            if row['化验结果（定量）'] != '':
                result = str(row['化验结果（定量）'])
            else:
                result = str(row['化验结果（定性）'])

            basedata[pid][examine] = result
            data[pid][examine + '【【' + device + '】】'] = result
            self.Units[examine + '【【' + device + '】】'] = unit
        basedata = pd.DataFrame(basedata).transpose()
        basedata.to_excel(self.output_dir + 'basedata.xlsx', index=True)

        data = pd.DataFrame(data).transpose()
        data.to_excel(self.output_dir + 'rawdata.xlsx', index=True)

        joblib.dump(self.Units, self.output_dir + 'Units')


    def missingRate(self):
        """
        Return: self.Features, 符合缺失率要求的特征
        """
        #计算缺失率，选择特征
        print('缺失率')
        #合并
        basedata = pd.read_excel(self.output_dir + 'basedata.xlsx')
        label = self.label()

        basedata = pd.merge(basedata, label, how='inner', left_index=True, right_on='PID')
        basedata.drop(['PID'], axis=1, inplace=True) #移除非特征列
        MissingRate = basedata.groupby([self.labelName]).apply(
                lambda col: 1 - pd.notna(col).sum() / len(col)).transpose()
        print(MissingRate.head())
        MissingRate.sort_values(by=[0,1,2], inplace=True, ascending=True)
        MissingRate.to_excel(self.output_dir + '缺失率统计.xlsx', index=True)
        #满足缺失要求的特征
        Features = MissingRate[MissingRate.max(axis=1) <= 0.3].index.tolist()
        #合并医生指定特征
        specify = eval(config.get('TRAIN', 'Specify'))
        Features = list(set(Features).union(set(specify)))
        return Features

    def featMap(self, data):
        '''
        合并X、y数据
        根据缺失率要求创建入选特征映射
        '''
        print('特征映射')       
#        Features = self.missingRate()
        Features = pd.read_csv('18个特征.txt', squeeze=True, header=None, engine='python').values

        #所有X特征
        Feats = data.columns.values.astype(str)
        FeatMap = defaultdict(list) #原始特征到特征分型的映射，key为base特征，value为特征分型
        for i in Feats:
            base = re.match('(.*)【【', i).group(1) #原始base特征
            if base in Features:
                FeatMap[base].append(i)
        return FeatMap
    
    
    def merge(self):
        print('\n合并、拆分数据')
        data = pd.read_excel(self.output_dir + 'rawdata.xlsx')    
        self.FeatMap = self.featMap(data)
        #选中分型列       
        columns = self.FeatMap.values()
        columns = [i for j in columns for i in j]
        data = data[columns]
        #合并Label
        label = self.label()
        data = pd.merge(data, label, how='inner', left_index=True, right_on='PID')      
        print('data shape:', data.shape)

        #合并后重新check_numeric、process伪分类列
        data, self.Numeric = self.check_numeric(data)        
        print('合并后data shape:', data.shape)
        print('Shape: %s, %d个数值列' % (data.shape, self.Numeric['check'].sum()))
        data.to_excel(self.output_dir + 'dataset.xlsx', index=True)
        
             
    def standardize(self):
        #合并
        #1、数值型: 先转换，不管有无分型；
        #2、分类型 且 无分型：填充缺失-转换
        #3、有分型：1）数值型，合并，2）分类型，合并，然后转换        
        data = pd.read_excel(self.output_dir + 'dataset.xlsx')
        Y = data.pop(self.labelName)
        
        self.Units = joblib.load(self.output_dir + 'Units')
        #（一）分型不一致
        #按照分型的数据类型self.labelName，从Map中分离出来
        for baseFeat in tqdm(self.FeatMap.copy()):
            subFeat = self.FeatMap[baseFeat] #分型
            print(baseFeat)
            if len(subFeat) > 1: #有分型
                subNumeric = defaultdict(list)
                for feat in subFeat:
                    numeric = self.Numeric.loc[feat]['check']
                    subNumeric[numeric].append(feat) #获取特征类型
                #如果存在不同数据类型，分离数值型
                if len(subNumeric.keys()) == 2:
                    numericFeats = subNumeric[1]
                    #更新FeatMap
                    subFeat = list(set(subFeat).difference(set(numericFeats))) #分类型
                    self.FeatMap[baseFeat] = subFeat
                    #添加数值组合特征
                    combine = baseFeat + '_num' #组合特征名
                    self.FeatMap[combine] = numericFeats #更新映射
                    #更新Numeric
                    self.Numeric.loc[combine] = {'check': 1, 'strratio': 0, 'strs': []}
                    

        #（二）合并单位相同的特征
        #合并以后，分型从Map中删掉，同时对应的列也从data中删掉
        #合并列同时添加到Map中和data中
        FillValue = {1: 0, 0: ''} #根据类型填充值
        for baseFeat in tqdm(self.FeatMap.copy()):
            subFeat = self.FeatMap[baseFeat] #分型
            print(baseFeat)
            if len(subFeat) > 1: #有分型
                subUnits = defaultdict(list) #单位相同的subfeat
                for feat in subFeat:
                    print(feat)
                    unit = self.Units[feat]
                    subUnits[unit].append(feat)
                for k, item in enumerate(subUnits.items()):
                    unit, feats = item[0], item[1]
                    if len(feats) > 1: #存在多个特征，满足合并条件
                        numeric = self.Numeric.loc[feats[0]]['check']
                        #多个feat单位相同删掉feats，添加combine特征
                        subFeat = list(set(subFeat).difference(set(feats)))
                        combine = baseFeat + '_combine' + str(k) #组合特征名
                        subFeat.append(combine)
                        self.FeatMap[baseFeat] = subFeat #更新映射

                        data[feats] = data[feats].fillna(FillValue[numeric])
                        data[combine] = data[feats].sum(axis=1) #合并
                        data[combine] = data[combine].replace([0, ''], np.nan)
                        data.drop(feats, axis=1, inplace=True)
                        #更新Numeric
                        self.Numeric.loc[combine] = {'check': numeric, 'strratio': 0, 'strs': []}
                        
                        #记录日志
                        self.Log.append([('合并', (feats, FillValue[numeric]), combine)])
                        

        #划分数据集
        train_X,test_X,train_y,test_y = train_test_split(data, Y, test_size=0.3, random_state=self.seed)
        print('Train shape:', train_X.shape, 'Test shape:', test_X.shape)


        #年龄
        Std = StandardScaler()
        train_X['年龄'] = Std.fit_transform(train_X[['年龄']])
        test_X['年龄'] = Std.transform(test_X[['年龄']])
        self.Log.append([('转换', '年龄', deepcopy(Std))])

        #转换，合并
        EncoderMap = {1: RobustScaler(), 0: LabelEncoder()}
        ImpMap = {1: SimpleImputer(missing_values=np.nan, strategy='mean'),
                  0: SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='UN')}
        for baseFeat in tqdm(self.FeatMap):
            subFeat = self.FeatMap[baseFeat] #分型
            if len(subFeat) == 1: #无分型
                baseFeat = subFeat[0]
                print(baseFeat)
                numeric = self.Numeric.loc[baseFeat]['check'] #获取特征类型
                if numeric: #数值型
                    Encoder = EncoderMap[numeric]
                    train_X.loc[:, baseFeat] = Encoder.fit_transform(train_X[[baseFeat]].values).ravel()
                    test_X.loc[:, baseFeat] = Encoder.transform(test_X[[baseFeat]].values).ravel()
                    Imp = ImpMap[numeric]
                    train_X[baseFeat] = Imp.fit_transform(train_X[[baseFeat]])
                    test_X[baseFeat] =Imp.transform(test_X[[baseFeat]])
                   
                    #记录日志
                    self.Log.append([('转换', baseFeat, [deepcopy(Encoder), deepcopy(Imp)])])
                else: #分类型
                    Imp = ImpMap[numeric]
                    train_X[baseFeat] = Imp.fit_transform(train_X[[baseFeat]].astype('object'))
                    test_X[baseFeat] = Imp.transform(test_X[[baseFeat]].astype('object'))
                    Encoder = EncoderMap[numeric]
                    mode = train_X[baseFeat].mode().values[0]                   
                    train_X[baseFeat] = Encoder.fit_transform(train_X[[baseFeat]].astype(str).values).ravel()
                    test_X[baseFeat] = [i if i in Encoder.classes_ else mode for i in test_X[baseFeat]]
                    test_X[baseFeat] = Encoder.transform(test_X[[baseFeat]].values).ravel()
                    
                    #记录日志
                    self.Log.append([('转换', baseFeat, [deepcopy(Imp), (deepcopy(Encoder), {'mode': mode})])])
            else: #有分型
                #3-(1)特征有分型, 合并分型形成base特征，然后删掉分型
                numeric = [self.Numeric.loc[feat]['check'] for feat in subFeat] #获取特征类型
                #验证分型一致
                if len(set(numeric))  > 1:
                    print(baseFeat, '分型类型不一致')
                    continue
#                print(baseFeat, subFeat, numeric)
                numeric = numeric[0]
                if numeric: #数值型
                    #转换
                    for feat in subFeat:
                        Encoder = EncoderMap[numeric]
                        train_X[feat] = Encoder.fit_transform(train_X[[feat]].values).ravel()
                        test_X[feat] = Encoder.transform(test_X[[feat]].values).ravel()
                        
                        #记录日志
                        self.Log.append([('转换', feat, deepcopy(Encoder))])
                    #合并
                    train_X[subFeat] = train_X[subFeat].fillna(0)
                    test_X[subFeat] = test_X[subFeat].fillna(0)
                    train_X[baseFeat] = train_X[subFeat].sum(axis=1)
                    train_X.drop(subFeat, axis=1, inplace=True)
                    test_X[baseFeat] = test_X[subFeat].sum(axis=1)
                    test_X.drop(subFeat, axis=1, inplace=True)
                    
                    #2019年3月5日，补充
                    train_X[baseFeat] = train_X[baseFeat].replace(0, np.nan)
                    test_X[baseFeat] = test_X[baseFeat].replace(0, np.nan)
                    
                    #填充
                    Imp = ImpMap[numeric]
                    train_X[baseFeat] = Imp.fit_transform(train_X[[baseFeat]].astype('object'))
                    test_X[baseFeat] =Imp.transform(test_X[[baseFeat]].astype('object'))
                    
                    #记录日志
                    self.Log.append([('合并', (subFeat, 0), deepcopy(baseFeat)), 
                                     ('转换', baseFeat, deepcopy(Imp))
                                     ])
                else: #3-(2)
                    #合并
                    train_X[subFeat] = train_X[subFeat].fillna('')
                    test_X[subFeat] = test_X[subFeat].fillna('')
                    train_X[baseFeat] = train_X[subFeat].sum(axis=1)
                    test_X[baseFeat] = test_X[subFeat].sum(axis=1)
                    train_X.drop(subFeat, axis=1, inplace=True)
                    test_X.drop(subFeat, axis=1, inplace=True)
                    
                    #2019年3月5日，补充
                    train_X[baseFeat] = train_X[baseFeat].replace('', np.nan)
                    test_X[baseFeat] = test_X[baseFeat].replace('', np.nan)
                    
                    #填充
                    Imp = ImpMap[numeric]
                    train_X[baseFeat] = Imp.fit_transform(train_X[[baseFeat]].astype('object'))
                    test_X[baseFeat] =Imp.transform(test_X[[baseFeat]].astype('object'))
                    #转换
                    Encoder = EncoderMap[numeric]
                    global x
                    x = train_X[baseFeat]
                    mode = train_X[baseFeat].mode().values[0]
                    train_X[baseFeat] = Encoder.fit_transform(train_X[[baseFeat]].values).ravel()
                    test_X[baseFeat] = [i if i in Encoder.classes_ else mode for i in test_X[baseFeat]]
                    test_X[baseFeat] = Encoder.transform(test_X[[baseFeat]].values).ravel()
                    
                    #记录日志
                    self.Log.append([('合并', (subFeat, ''), baseFeat), 
                                     ('转换', baseFeat, [deepcopy(Imp), (deepcopy(Encoder), {'mode': mode})])
                                     ])
                                    

        print('合并完成, Train shape:', train_X.shape, 'Test shape:', test_X.shape)
        pd.concat([train_X, train_y], axis=1).to_excel(self.trainData, index=False)
        pd.concat([test_X, test_y], axis=1).to_excel(self.testData, index=False)
        
        #保存日志
        joblib.dump(self.FeatMap, self.output_dir + 'featMap')
        joblib.dump(self.Log, self.output_dir + 'Logger')
        
    
    def read_data(self):
        train = pd.read_excel(self.trainData)
        test = pd.read_excel(self.testData)
        train.pop('PID')
        test.pop('PID')              
        train_y = train.pop(self.labelName)
        train_X = train
        test_y = test.pop(self.labelName)
        test_X = test
        print('Train Shape:', train_X.shape)
        print('Test Shape:', test_X.shape)
        return train_X, train_y, test_X, test_y
        
        
    def main(self):
#        self.read_rawData()
        self.merge()
        self.standardize()

if __name__ == '__main__':
    processor = DataSet()
    processor.main()

