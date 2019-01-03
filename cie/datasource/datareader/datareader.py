import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from cie.common import inherit_doc


@inherit_doc
class BaseChannel(object):
    __metaclass__ = ABCMeta

    def __init__(self, source=None):
        """
        construct a channel.
        :param source: source name. e.g. file path for excel/csv, database name for mongodb
        """
        self.source = source

    @classmethod
    def to_array(cls, d):
        return d.values

    @abstractmethod
    def open(self):
        raise NotImplementedError()

    @abstractmethod
    def read(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


class ExcelChannel(BaseChannel):
    """
    excel channel: read/write
    >>>channel = ExcelChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.xlsx")
    >>>channel.open()
    >>>params = {
    >>>    "label_index": 1,
    >>>    "header": [0],
    >>>    "sheet_name": 0,
    >>>    "encoding": 'gbk',
    >>>    "usecols": list(range(5)),
    >>>}
    >>>data = channel.read(**params)
    >>>print(data[:5])
    >>>channel.close()
    """

    def open(self):
        pass

    def close(self):
        pass

    def read(self, data_type='dataframe', **kwargs):
        """
        从excel读写数据
        :param data_type: dataframe or ndarray
        :param kwargs: `label_index` 必须为int，表示excel的第label_index列; `usecols` 必须为int的list.
        :return: 如果是ndarray, 返回四元组: （label名字, label值, feature列名, feature值）.
                 如果是dataframe，返回二元组: (dataframe， 列名）
        """
        if kwargs is None:
            raise ValueError("kwargs must be set")
        label = kwargs.pop("label_index", None)
        if label is not None:
            if not isinstance(label, int):
                raise ValueError("label_index must be type of int")
        if 'usecols' in kwargs:
            if label is not None:
                if label in kwargs['usecols']:
                    kwargs['usecols'].sort()
                    label = kwargs['usecols'].index(label)
                else:
                    kwargs['usecols'].append(label)
                    kwargs['usecols'].sort()
        data = pd.read_excel(self.source, **kwargs)
        columns = data.columns.values
        if data_type == 'dataframe':
            # if label is not None:
            #     data.rename(columns={data.columns[label]: "label"}, inplace=True)
            res = (data, columns)
        else:
            if label is not None:
                labels = data.iloc[:, label]
                features = data.iloc[:, [j for j, c in enumerate(data.columns) if j != label]]
                res = (np.array([columns[label]]), self.to_array(labels),
                       np.delete(columns, label), self.to_array(features))
            else:
                res = (None, None, columns, self.to_array(data))
        return res

    def write(self, data, **kwargs):
        # df = pd.DataFrame(data)
        writer = pd.ExcelWriter(self.source, engine='xlsxwriter')
        data.to_excel(writer, **kwargs)


class CsvChannel(BaseChannel):
    """
    csv channel: read/write
    >>>channel = CsvChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.csv")
    >>>channel.open()
    >>>params = {
    >>>    "label_index":1
    >>>    "header": 0,
    >>>    "sep": ',',
    >>>    "encoding": 'gbk',
    >>>    "usecols": list(range(2)),
    >>>}
    >>>y, x = channel.read(**params)
    >>>channel.close()
    """

    def open(self):
        pass

    def close(self):
        pass

    def read(self, data_type='dataframe', **kwargs):
        """
        从csv读写数据
        :param data_type: dataframe or ndarray
        :param kwargs: `label_index` 必须为int，表示csv的第label_index列; `usecols` 必须为int的list.
        :return: 如果是ndarray, 返回四元组: （label名字, label值, feature列名, feature值）.
                 如果是dataframe，返回二元组: (dataframe， 列名）
        """
        if kwargs is None:
            raise ValueError("kwargs must be set")
        label = kwargs.pop("label_index", None)
        if label is not None:
            if not isinstance(label, int):
                raise ValueError("label_index must be type of int")
        if 'usecols' in kwargs:
            if label is not None:
                if label not in kwargs['usecols']:
                    kwargs['usecols'].append(label)
                kwargs['usecols'].sort()
                label = kwargs['usecols'].index(label)
        data = pd.read_csv(self.source, **kwargs)
        columns = data.columns.values
        if data_type == 'dataframe':
            # if label is not None:
            #     data.rename(columns={data.columns[label]: "label"}, inplace=True)
            res = (data, data.columns.values)
        else:
            if label is not None:
                labels = data.iloc[:, label]
                features = data.iloc[:, [j for j, c in enumerate(data.columns) if j != label]]
                res = (np.array([columns[label]]), self.to_array(labels),
                       np.delete(columns, label), self.to_array(features))
            else:
                res = (None, None, columns, self.to_array(data))
        return res

    def write(self, data, **kwargs):
        # df = pd.DataFrame(data)
        data.to_csv(self.source, **kwargs)


class MongoChannel(BaseChannel):
    """
    mongodb channel: read/write
    >>>param_dct = {
    >>>    "local": False,
    >>>    "ssh_addr": "47.99.191.80",
    >>>    "ssh_port": 22,
    >>>    "ssh_user": "",
    >>>    "ssh_host_key": None,
    >>>    "ssh_pwd": None,
    >>>    "ssh_pkey": None,
    >>>    "mongo_host": "",
    >>>    "mongo_port": 3717,
    >>>    "db_user": "",
    >>>    "db_pwd": "",
    >>>    "db_name": "admin",
    >>>    "col_name": "",
    >>>    "auth_method": "SCRAM-SHA-1",
    >>>}
    """

    def open(self, **kwargs):
        """
        open mongodb channel
        :param kwargs: parameter mapping.
                       `local` directly access mongodb if True, otherwise using ssh to access the mongodb.
        :return:
        """
        from pymongo import MongoClient
        param_dct = {
            "local": False,
            "ssh_addr": "47.99.191.80",
            "ssh_port": 22,
            "ssh_user": "",
            "ssh_host_key": None,
            "ssh_pwd": None,
            "ssh_pkey": None,
            "mongo_host": "",
            "mongo_port": 3717,
            "db_user": "",
            "db_pwd": "",
            "db_name": "admin",
            "col_name": "",
            "auth_method": "SCRAM-SHA-1",
        }
        local = kwargs.get("local")
        param_dct.update(kwargs)
        if not local:
            from sshtunnel import SSHTunnelForwarder
            server = SSHTunnelForwarder(
                ssh_address_or_host=(param_dct["ssh_addr"], param_dct["ssh_port"]),
                ssh_username=param_dct["ssh_user"],
                ssh_pkey=param_dct["ssh_pkey"],
                ssh_host_key=param_dct["ssh_host_key"],
                ssh_password=param_dct["ssh_pwd"],
                remote_bind_address=(param_dct["mongo_host"], param_dct["mongo_port"])
            )
            server.start()

            client = MongoClient("127.0.0.1",
                                 server.local_bind_port)
            database = client[param_dct["db_name"]]
            database.authenticate(param_dct["db_user"], param_dct["db_pwd"], mechanism=param_dct["auth_method"])
            db_col = database[param_dct["col_name"]]
        else:
            client = MongoClient(param_dct["mongo_host"], param_dct["mongo_port"])
            database = client[param_dct["db_name"]]
            database.authenticate(param_dct["db_user"], param_dct["db_pwd"], mechanism=param_dct["auth_method"])
            db_col = database[param_dct["col_name"]]
        setattr(self, "db_col", db_col)
        setattr(self, "client", client)
        if server:
            setattr(self, "server", server)

    def read(self, query=None, projection=None):
        """
        read data from the database.
        :param query: optional, a SON object specifying elements which must be present for a document
                                to be included in the result set
        :param projection: optional, a list of field names that should be returned in the result
                             set or a dict specifying the fields to include or exclude. If projection
                             is a list “_id” will always be returned. Use a dict to exclude fields
                             from the result (e.g. projection={‘_id’: False}).
        :return: A dataframe to the documents that match the query criteria.
        """
        db_col = getattr(self, "db_col")
        data = db_col.find(query, projection)
        return pd.DataFrame(list(data))

    def write(self, data):
        """
        Insert an iterable of documents.
        :param data: A dataframe of documents to insert.
        :return: An instance of InsertManyResult.
        """
        data = data.to_dict('records')
        db_col = getattr(self, "db_col")
        from collections import Iterable
        if isinstance(data, Iterable):
            db_col.insert_many(data)
        else:
            raise ValueError("data must be an iterable of documents")

    def close(self):
        """
        close the mongodb.
        :return:
        """
        server = getattr(self, "server")
        client = getattr(self, "client")
        if server:
            server.close()
        if client:
            client.close()
