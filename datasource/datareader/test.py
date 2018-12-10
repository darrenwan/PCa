# -*- coding: utf-8 -*-
import pytest
from cie.datareader import *


def test_excel_reader():
    channel = ExcelChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.xlsx")
    channel.open()
    params = {
        "header": [0],
        "sheet_name": 0,
        'encoding': 'gbk',
        "usecols": list(range(2)),

    }
    y, x = channel.read_xy(**params)
    channel.close()


def test_csv_reader():
    channel = CsvChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.csv")
    channel.open()
    params = {
        "header": 0,
        "sep": ',',
        'encoding': 'gbk',
        "usecols": list(range(2)),

    }
    y, x = channel.read_xy(**params)
    channel.close()


if __name__ == '__main__':
    pytest.main([__file__])
