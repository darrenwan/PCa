from cie.data import *
import pytest


def test_make_classification():
    make_classification(n_samples=10)


def test_make_regression():
    make_regression()


def test_data_type():
    X = [[1, 2, 4], [4, 5, 6]]
    cols = ['a', 'b', 'c']
    df = CieDataFrame(X, cols)
    # print(dir(df.get_inner_df()))
    print(isinstance(df, CieDataFrame))
    print(df)
    # print(df.to_csv('./1.txt'))


if __name__ == '__main__':
    # test_make_classification()
    # test_data_type()
    pytest.main([__file__])
