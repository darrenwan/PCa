#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhy
# Copyright 2018 @


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import logging
import multiprocessing

import pandas as pd
import numpy as np
from gensim.models import word2vec,keyedvectors



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class Word2VecConfig(object):
    """word2vec常用的配置参数"""

    size=200      # 词向量的维度
    window=5      # 词向量训练时的上下文扫描窗口大小
    min_count= 5  #
    iter=50,
    workers=multiprocessing.cpu_count() #训练的进程数（并行），默认是当前运行机器的处理器核数


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ClassWord2Vec(object):
    """常用的词向量训练方法参考与封装.

    Examples
    --------
    wconfig = Word2VecConfig()
    model = ClassWord2Vec(wconfig).fit(sentences_file)
    a = "[症状]多关节痛"
    b = "[诊断]关节痛"
    model.similarity(a, b)
    model.most_similar(a)

    # 保存模型
    print("save model")
    model.save(model_file)
    print("save model success")
    model = word2vec.Word2Vec.load(model_file)

    print("save w2v")
    model.wv.save_word2vec_format(vec_file, binary=False)
    print("save w2v success")

    vec = word2vec.KeyedVectors.load_word2vec_format(vec_file, binary=False)

    """
    def __init__(self,config):
        self.config = config
        self.model = None


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def fit(self, sentence_file):
        """训练一个词向量模型

        Parameters
        ----------
        X : 分好的句子


        Returns
        -------
        self : object
            实例本身.
        """
        sentences = word2vec.LineSentence(sentence_file)
        self.model = word2vec.Word2Vec(sentences,
                                       size=self.config.size,
                                       window=self.config.size,
                                       min_count=self.config.size,
                                       iter=self.config.size,
                                       workers=multiprocessing.cpu_count())


        # Return fitted word2vec instance
        return self

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def transform(self, X):
        """Transform (predict) 词向量.

        Parameters
        ----------
        X : 需要计算的词

        Returns
        -------
        numpy.ndarray : 返回词向量.
        """
        return self.model[X]


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def split_sentence_from_csv(source_file,sentence_file):  # file_name = train_file
        """把CSV写成词向量需要的句子形式

        Parameters
        ----------
        X : 文件

        Returns
        -------
        file: 返回句子文件.
        """

        # if not os.path.exists(source_file):
        #     destDF = pd.read_csv(source_file, encoding='utf-8')  # ,encoding = 'utf-8'
        # else:
        #     print('文件不存在')

        destDF = pd.read_csv(source_file, encoding='utf-8')  # ,encoding = 'utf-8'
        target = open(sentence_file, 'w', encoding='utf8')

        i = 0
        for index, item in destDF.iterrows():
            line_str = eval(item['Item'])

            if len(line_str) > 0:  # 如果句子非空
                for x in line_str:
                    target.writelines(str(x))
                    target.writelines(" ")
            target.writelines("\n")
            i += 1
        target.close()
        print("end!")
        return target


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def transform(self, X):
        """Transform (predict) 词向量.

        Parameters
        ----------
        X : 需要计算的词

        Returns
        -------
        numpy.ndarray : 返回词向量.
        """
        return self.model[X]

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def save_model(self, model,model_file):
        """保存模型.
        """
        model.save(model_file)

    def load_model(self,model_file):
            """加载模型.
            """
            return word2vec.Word2Vec.load(model_file)


class BertWord2Vec(object):
    """
    bert的词向量训练方法,参考ClassWord2vec.

    Examples
    --------
    """

    def __init__(self,config):
        self.config = config
        self.model = None


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def fit(self, sentence_file):
        """训练一个词向量模型

        Parameters
        ----------
        X : 分好的句子


        Returns
        -------
        self : object
            实例本身.
        """
        pass
        return self

