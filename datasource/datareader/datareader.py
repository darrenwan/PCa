import pandas as pd
import numpy as np


class BaseChannel(object):
    def __init__(self, source):
        """
        construct a channel.
        >>> channel = CsvChannel("file.csv")
        >>> channel.open()
        >>> channel.read(param_dct)
        >>> channel.close()
        :param source: source name. e.g. file path for excel/csv, database name for mongodb
        """
        self.source = source

    @classmethod
    def to_array(cls, d):
        return d.values

    def open(self):
        pass

    def read(self):
        pass

    def close(self):
        pass


class ExcelChannel(BaseChannel):
    """
    excel channel: read/write
    >>>channel = ExcelChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.xlsx")
    >>>channel.open()
    >>>params = {
    >>>    "header": [0],
    >>>    "sheet_name": 0,
    >>>    "encoding": 'gbk',
    >>>    "usecols": list(range(5)),
    >>>}
    >>>data = channel.read(**params)
    >>>print(data[:5])
    >>>channel.close()
    """

    def read_xy(self, label=None, **kwargs):
        """
        read data from excel
        :param label: int index of the label column
        :param kwargs:
        :return:
        """
        if label is None:
            raise ValueError("label column must be set")
        if 'usecols' in kwargs:
            if label in kwargs['usecols']:
                kwargs['usecols'].sort()
                label = kwargs['usecols'].index(label)
            else:
                kwargs['usecols'].append(label)
                kwargs['usecols'].sort()
        data = pd.read_excel(self.source, **kwargs)
        columns = data.columns.values
        labels = data.iloc[:, label]
        features = data.iloc[:, [j for j, c in enumerate(data.columns) if j != label]]
        return np.array([columns[label]]), self.to_array(labels), np.delete(columns, label), self.to_array(features)

    @classmethod
    def write(cls, data, file_name, sheet_name='Sheet1'):
        df = pd.DataFrame(data)
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
        df.to_excel(writer, sheet_name=sheet_name)


class CsvChannel(BaseChannel):
    """
    csv channel: read/write
    >>>channel = CsvChannel("/Users/wenhuaizhao/works/ml/test/20180806胰腺癌.csv")
    >>>channel.open()
    >>>params = {
    >>>    "header": 0,
    >>>    "sep": ',',
    >>>    "encoding": 'gbk',
    >>>    "usecols": list(range(2)),
    >>>}
    >>>y, x = channel.read(**params)
    >>>channel.close()
    """

    def read_xy(self, label=None, **kwargs):
        """
        read data from csv
        :param label: int index of the label column
        :param kwargs: parameters
        :return:
        """
        if label is None:
            raise ValueError("label column must be set")
        if 'usecols' in kwargs:
            if label in kwargs['usecols']:
                kwargs['usecols'].sort()
                label = kwargs['usecols'].index(label)
            else:
                kwargs['usecols'].append(label)
                kwargs['usecols'].sort()
        data = pd.read_csv(self.source, **kwargs)
        columns = data.columns.values
        labels = data.iloc[:, label]
        features = data.iloc[:, [j for j, c in enumerate(data.columns) if j != label]]
        return np.array([columns[label]]), self.to_array(labels), np.delete(columns, label), self.to_array(features)

    @classmethod
    def write(cls, data, file_name, **kwargs):
        df = pd.DataFrame(data)
        df.to_csv(file_name, kwargs)


class MongoChannel(BaseChannel):
    """
    mongodb channel: read/write
    >>>param_dct = {
    >>>    "ssh_addr": "47.99.191.80",
    >>>    "ssh_port": 22,
    >>>    "mongo_user": "",
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

    def open(self, local=False, **kwargs):
        """
        open mongodb channel
        :param local: directly access mongodb if True, otherwise using ssh to access the mongodb
        :param kwargs: parameter mapping
        :return:
        """
        from pymongo import MongoClient
        param_dct = {
            "ssh_addr": "47.99.191.80",
            "ssh_port": 22,
            "mongo_user": "",
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
        param_dct.update(kwargs)
        if not local:
            from sshtunnel import SSHTunnelForwarder
            server = SSHTunnelForwarder(
                ssh_address_or_host=(param_dct["ssh_addr"], param_dct["ssh_port"]),
                ssh_username=param_dct["mongo_user"],
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
        :param query: optional, a SON object specifying elements which must be present for a document to be
                      included in the result set
        :param projection: optional, a list of field names that should be returned in the result set or a dict
                           specifying the fields to include or exclude. If projection is a list “_id” will always be
                           returned. Use a dict to exclude fields from the result (e.g. projection={‘_id’: False}).
        :return: A cursor to the documents that match the query criteria.
        """
        db_col = getattr(self, "db_col")
        data = db_col.find(query=query, projection=projection)
        return data

    def write(self, data):
        """
        Insert an iterable of documents.
        :param data: A iterable of documents to insert.
        :return: An instance of InsertManyResult.
        """
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
