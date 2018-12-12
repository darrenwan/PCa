# -*- coding: utf-8 -*-
import cie
from cie.models import Model
from cie.models import Sequential
from cie.models import Input

from cie.preprocessing import *
from cie.classification import *
from cie.common import logger
from cie.datareader import *
from cie.visualization import *


from sklearn.datasets import make_classification
x, y = make_classification(n_features=4, random_state=0)


logger = logger.get_logger()


def f():
    sequential = Sequential()
    sequential.add(StandardScaler())
    sequential.add(SVC(kernel='linear'))
    sequential.compile()
    sequential.fit(x, y)
    res = sequential.predict(x)
    print(res)


def test_data_reader():
    channel = CsvChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.csv")
    channel.open()
    params = {
        "header": 0,
        "sep": ',',
        "encoding": 'gbk',
        "nrows": 20,
        "usecols": [1, 4],

    }
    y_name, y, x_name, x = channel.read_xy(label=1, **params)
    print(y_name, y, x_name, x)
    # stat_labels(y, names =)
    stat_freq(x, names=x_name, categorical=False)
    stat_freq(y, names=y_name, categorical=True)
    channel.close()


if __name__ == "__main__":
    print("program begins")
    # f()
    test_data_reader()
    print("program ends")
