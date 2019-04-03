import pandas as pd
import numpy as np
import sklearn.datasets as datasets

__all__ = ['CieDataFrame', 'CieDataFrame2', 'make_classification', 'make_regression', 'load_boston', 'load_iris']


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    args = locals()
    X, y = datasets.make_classification(**args)
    return CieDataFrame(X), CieDataFrame(y)


def make_regression(n_samples=100, n_features=100, n_informative=10,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                    random_state=None):
    args = locals()
    X, y = datasets.make_regression(**args)
    return CieDataFrame(X), CieDataFrame(y)


def load_boston(return_X_y=False):
    data = datasets.load_boston(return_X_y=return_X_y)
    return CieDataFrame(data.data), CieDataFrame(data.target)


def load_iris(return_X_y=False):
    data = datasets.load_iris(return_X_y=return_X_y)
    return CieDataFrame(data.data), CieDataFrame(data.target)


class CieDataFrame2(object):

    @staticmethod
    def to_cie_data(X, X_columns=None):
        if hasattr(X, 'loc'):
            data_x = X
        else:
            data_x = pd.DataFrame(X, columns=X_columns)
        return data_x


CieDataFrame = pd.DataFrame


class CieDataFrame1(object):
    """
    待完善
    """

    def __init__(self, X, X_columns=None):
        if hasattr(X, 'loc'):
            if not isinstance(X, pd.DataFrame):
                data_x = pd.DataFrame(X)
            else:
                data_x = X
        else:
            data_x = pd.DataFrame(X, columns=X_columns)
        self.inner_df = data_x

    def get_inner_df(self):
        return self.inner_df

    @property
    def _constructor(self):
        return CieDataFrame

    @property
    def T(self):
        return self.inner_df.T

    @property
    def at(self):
        return self.inner_df.at

    @property
    def axes(self):
        """
        Return a list with the row axis labels and column axis labels as the
        only members. They are returned in that order.
        """
        return self.inner_df.axes

    @property
    def blocks(self):
        return self.inner_df.blocks

    @property
    def columns(self):
        return self.inner_df.columns

    @property
    def dtypes(self):
        return self.inner_df.dtypes

    @property
    def empty(self):
        return self.inner_df.empty

    @property
    def ftypes(self):
        return self.inner_df.ftypes

    @property
    def iloc(self):
        return self.inner_df.iloc

    @property
    def index(self):
        print("====")
        return self.inner_df.index

    @property
    def ix(self):
        return self.inner_df.ix

    @property
    def loc(self):
        return self.inner_df.loc

    @property
    def ndim(self):
        return self.inner_df.ndim

    @property
    def shape(self):
        return self.inner_df.shape

    @property
    def size(self):
        return self.inner_df.size

    @property
    def style(self):
        return self.inner_df.style

    @property
    def values(self):
        return self.inner_df.values

    def __getitem__(self, key):
        return self.inner_df.__getitem__(key)

    def __setitem__(self, key, value):
        return self.inner_df.__setitem__(key, value)

    def __delitem__(self, key):
        self.inner_df.__delitem__(key)

    def __str__(self):
        return self.inner_df.__str__()

    def iteritems(self):
        self.inner_df.iteritems()

    def iterrows(self):
        self.inner_df.iterrows()

    def itertuples(self, index=True, name="Pandas"):
        return self.inner_df.itertuples(index=index, name=name)

    def __len__(self):
        return self.inner_df.__len__()

    def dot(self, other):
        return self.inner_df.dot(other.inner_df)

    @classmethod
    def from_dict(cls, data, orient='columns', dtype=None):
        return pd.DataFrame.from_dict(data=data, orient=orient, dtype=dtype)

    def to_dict(self, orient='dict', into=dict):
        print("====")
        return self.inner_df.to_dict(orient=orient, into=into)

    @classmethod
    def from_records(cls, data, index=None, exclude=None, columns=None,
                     coerce_float=False, nrows=None):
        args = locals()
        args.pop("cls")
        return pd.DataFrame.from_records(**args)

    def to_records(self, index=True, convert_datetime64=True):
        print("====")
        return self.inner_df.to_records(index=index, convert_datetime64=convert_datetime64)

    @classmethod
    def from_items(cls, items, columns=None, orient='columns'):
        return pd.DataFrame.from_items(items, columns=columns, orient=orient)

    @classmethod
    def from_csv(cls, path, header=0, sep=',', index_col=0, parse_dates=True,
                 encoding=None, tupleize_cols=None,
                 infer_datetime_format=False):
        args = locals()
        args.pop("cls")
        return pd.DataFrame.from_csv(**args)

    def to_csv(self, path_or_buf=None, sep=",", na_rep='', float_format=None,
               columns=None, header=True, index=True, index_label=None,
               mode='w', encoding=None, compression=None, quoting=None,
               quotechar='"', line_terminator='\n', chunksize=None,
               tupleize_cols=None, date_format=None, doublequote=True,
               escapechar=None, decimal='.'):
        args = locals()
        args.pop("self")
        return self.inner_df.to_csv(**args)

    def to_excel(self, excel_writer, sheet_name='Sheet1', na_rep='',
                 float_format=None, columns=None, header=True, index=True,
                 index_label=None, startrow=0, startcol=0, engine=None,
                 merge_cells=True, encoding=None, inf_rep='inf', verbose=True,
                 freeze_panes=None):
        args = locals()
        args.pop("self")
        return self.inner_df.to_excel(**args)

    def transpose(self, *args, **kwargs):
        args = locals()
        args.pop("self")
        return self.inner_df.transpose(**args)

    def select_dtypes(self, include=None, exclude=None):
        args = locals()
        args.pop("self")
        return self.inner_df.select_dtypes(**args)

    def to_xarray(self):
        print("====")
        return self.inner_df.to_xarray()

    def to_sparse(self, fill_value=None, kind='block'):
        args = locals()
        args.pop("self")
        return self.inner_df.to_sparse(**args)

    def get_value(self, index, col, takeable=False):
        args = locals()
        args.pop("self")
        return self.inner_df.get_value(**args)

    def get_values(self):
        print("====")
        return self.inner_df.get_values()

    def get(self, key, default=None):
        print("====")
        args = locals()
        args.pop("self")
        return self.inner_df.get(**args)

    def astype(self, dtype, copy=True, errors='raise', **kwargs):
        print("====")
        args = locals()
        args.pop("self")
        return self.inner_df.astype(**args)

    def where(self, cond, other=np.nan, inplace=False, axis=None, level=None,
              errors='raise', try_cast=False, raise_on_error=None):
        args = locals()
        args.pop("self")
        return self.inner_df.where(**args)

    def to_dense(self):
        return self.inner_df.to_dense()

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None, float_format=None,
                  sparsify=None, index_names=True, justify=None,
                  line_width=None, max_rows=None, max_cols=None,
                  show_dimensions=False):
        args = locals()
        args.pop("self")
        return self.inner_df.to_string(**args)

    def to_parquet(self, fname, engine='auto', compression='snappy',
                   **kwargs):
        args = locals()
        args.pop("self")
        return self.inner_df.to_parquet(**args)

    def __unicode__(self):
        return self.inner_df.__unicode__()

    def as_matrix(self, columns=None):
        return self.inner_df.as_matrix(columns=columns)

    def as_blocks(self, copy=True):
        return self.inner_df.as_blocks(copy=copy)

    def xs(self, key, axis=0, level=None, drop_level=True):
        args = locals()
        args.pop("self")
        return self.inner_df.xs(**args)

    def __bytes__(self):
        return self.inner_df.__bytes__()

    def __repr__(self):
        return self.inner_df.__repr__()

    def __array__(self, dtype=None):
        return self.inner_df.__array__(dtype=dtype)

    def __array_wrap__(self, result, context=None):
        args = locals()
        args.pop("self")
        return self.inner_df.__array_wrap__(**args)
