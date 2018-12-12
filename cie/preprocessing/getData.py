# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:04:16 2018

@author: atlan
"""

import numpy as np
import pandas as pd

class Get_data():
    """
    从本地目录读入数据文件.
    支持的文件格式：csv,xls,xlsx

    Parameters
    ----------
    __init__: 实例化本类时传入文件路径
    read_file: 可以接受pd.read_csv或pd.read_excel指定的参数

    Returns
    -------
    获取的数据以DataFrame形式返回.
    """
    def __init__(self, file):
        self.file = file
        
    def _check_file(self):
        if self.file.endswith('.csv'):
            self._read_func = pd.read_csv
        elif self.file.endswith('.xls') or self.file.endswith('.xlsx'):
            self._read_func = pd.read_excel
        else:
            raise Exception('The file format is not accepted!')
        
    def read_file(self, **args):
        self._check_file()
        data = self._read_func(self.file, **args)
        return data