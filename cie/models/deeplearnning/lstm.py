#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhy
# Copyright 2018 @


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import logging
from gensim.models import word2vec,keyedvectors
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import sklearn.metrics as metrics

import multiprocessing

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Embedding,LSTM,Dropout,Dense,Bidirectional,Flatten,Reshape
from tensorflow.keras.layers import BatchNormalization,Conv1D,MaxPool1D ,MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.initializers import Constant



# numpy.set_printoptions(threshold=numpy.inf)

work_dir = "/Users/hitales/Documents/nlpwork/sle"
os.chdir(work_dir)

vac_dir = os.path.join(work_dir, 'data')
source_file = os.path.join(vac_dir,"result_SLE14439_2018-10-27.csv")
sentence_file = os.path.join(vac_dir,"sentence_file.txt")
model_file = os.path.join(vac_dir,"vec.model")
vec_file = os.path.join(vac_dir,"vord.vector.txt")

# train_file = os.path.join(vac_dir,"result_LN0-1_2018-10-27 13_05_27.csv")
train_file = os.path.join(vac_dir,"result_LN0-1_2018-10-27 13_05_27.csv")


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class rnnConfig(object):
    """循环神经网络常用的配置参数"""
    MAX_SEQUENCE_LENGTH = 300  # 每个文本或者句子的截断长度，只保留1000个单词}
    MAX_NUM_WORDS = 24000  # 用于构建词向量的词汇表数量
    EMBEDDING_DIM = 200  # 词向量维度
    VALIDATION_SPLIT = 0.2

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class simpleLstm(object):
    """构建简单的lstm模型，并训练.

    Examples
    --------


    """
    def __init__(self,config):
        self.config = config
        self.x_train = config.x_train
        self.y_test = config.y_test
        self.x_test = config.x_test
        self.y_train = config.y_train
        self.model = self.build_model(self.x_train,self.y_train)

    # -------------------------------------------------------------------------
    #  构建模型
    # -------------------------------------------------------------------------

    def build_model(self,x_train,y_train):

        embedding_layer = Embedding(input_dim=self.config.num_words,
                                    output_dim=self.config.EMBEDDING_DIM,
                                    weights=[self.get_embedding_matrix()],
                                    input_length=self.config.MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        sequence_input = Input(shape=(x_train.shape[1],), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(128, return_sequences=True)(embedded_sequences)
        x = Dense(128, activation='relu')(x)
        preds = Dense(y_train.shape[1], activation='sigmoid')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return self


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def fit(self,x_train, y_train):
        """训练一个lstm模型

        Parameters
        ----------
        X :


        Returns
        -------
        self : object
            实例本身.
        """
        print('Train...')
        self.model.fit(x_train, y_train,
                  epochs=3,validation_split=0.1)#,class_weight = {0:0.6158206,1:2.65851064})
        # validation_data=(x_test, y_test))

        score, acc = self.model.evaluate(self.x_test, self.y_test)
        self.model.summary()
        print('Test score:', score)
        print('Test accuracy:', acc)


        # Return fitted word2vec instance
        return score, acc

        # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def predict_point_by_point(self,model, data):
        predicted = model.predict(data)
        print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
        predicted = np.reshape(predicted, (len(predicted),))
        return predicted


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def model_evaluation(self,model, x_train, y_train, x_test, y_test):
        # 对训练集预测
        y_train1 = y_train[:, 1]
        y_test1 = y_test[:, 1]
        train_result = pd.value_counts(y_train1)
        train_total = len(y_train1)
        train_posi_c = train_result[1]
        train_nega_c = train_result[0]

        train_posi_r = train_posi_c / train_total
        train_nega_r = train_nega_c / train_total

        test_result = pd.value_counts(y_test1)
        test_total = len(y_test1)
        test_posi_c = test_result[1]
        test_nega_c = test_result[0]

        test_posi_r = test_posi_c / test_total
        test_nega_r = test_nega_c / test_total

        dtrain_predictions = model.predict_classes(x_train)
        dtrain_predprob = model.predict_classes(x_train)[:, 1]
        dtest_predictions = model.predict_classes(x_test)
        dtest_predprob = model.predict_classes(x_test)[:, 1]

        train_f1_score = f1_score(y_train1, dtrain_predictions)
        test_f1_score = f1_score(y_test1, dtest_predictions)

        # 输出模型的一些结果

        print('-------训练集--------->')
        print("总数:%d \n正样本:%d \n负样本:%d \n正样本占比:%.2f%% \n负样本占比:%.2f%%" % (
            train_total, train_posi_c, train_nega_c, train_posi_r * 100, train_nega_r * 100))
        print("准确率 : %.4g" % accuracy_score(y_train1, dtrain_predictions))
        print("AUC 得分 (训练集): %f" % roc_auc_score(y_train1, dtrain_predprob))
        print("f1_score (训练集): %f" % train_f1_score)
        print(confusion_matrix(y_train1, dtrain_predictions))

        # 输出模型的一些结果
        print('-------测试集--------->')
        print("总数:%d \n正样本:%d \n负样本:%d \n正样本占比:%.2f%% \n负样本占比:%.2f%%" % (
            test_total, test_posi_c, test_nega_c, test_posi_r * 100, test_nega_r * 100))
        print("准确率 : %.4g" % accuracy_score(y_test1, dtest_predictions))
        print("AUC 得分 (测试集): %f" % roc_auc_score(y_test1, dtest_predprob))
        print("f1_score (测试集): %f" % test_f1_score)
        print(confusion_matrix(y_test1, dtest_predictions))

        print ("max ks 训练集：%.5f,测试集：%.5f" % self.__ks_func(y_train1,dtrain_predictions,y_test1,dtest_predictions))

    def __ks_func(self,y_train,dtrain_predictions,y_test,dtest_predictions):
        fpr_train, tpr_train, threshold = metrics.roc_curve(y_train, dtrain_predictions, pos_label=1)
        ks_train = max(abs(fpr_train - tpr_train))
        fpr_test, tpr_test, _ = metrics.roc_curve(y_test, dtest_predictions, pos_label=1)
        ks_test = max(abs(fpr_test - tpr_test))

        return ks_train,ks_test


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def save_model(self,model_file):
        """保存模型.
        """
        self.model.save(model_file)

    def load_model(self,model_file):
            """加载模型.
            """
            return self.model.load(model_file)
