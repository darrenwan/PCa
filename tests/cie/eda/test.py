# -*- coding: utf-8 -*-

from cie.common import logger
from cie.datasource import *
from cie.eda import *

from sklearn.datasets import make_classification
x, y = make_classification(n_features=4, random_state=0)


logger = logger.get_logger()


def test_data_reader():
    channel = CsvChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.csv")
    channel.open()
    params = {
        "label_index": 1,
        "header": 0,
        "sep": ',',
        "encoding": 'gbk',
        "nrows": 20000,
        "usecols": [4, 5, 6, 7],

    }
    data, columns = channel.read(**params)
    print(data, columns)
    stat_freq(data, force_cat_cols=[columns[2]])
    # stat_freq(x, names=x_name, force_cat_cols=[x_name[0]])
    # stat_freq(y, names=y_name)
    stat_dist(data, num_bins=10)
    channel.close()


if __name__ == "__main__":
    print("program begins")
    # f()
    test_data_reader()
    print("program ends")
