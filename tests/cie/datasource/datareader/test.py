# -*- coding: utf-8 -*-
import pymongo
import sys
from sshtunnel import SSHTunnelForwarder
from cie.datasource import *
import pandas as pd
import json

# os.chdir('/Users/hitales/Documents')
#步骤1 连接数据库函数

class ConnectMongoDB():

    def __init__(self):
        self.db = self.get_connection()

    def get_connection(self):
        db_connection_configuration = {
            "ssh_tunnel_address": "121.199.24.144",  # 跳板机地址
            "ssh_tunnel_port": 22,  # 跳板机端口
            "ssh_tunnel_user_name": "nancy",  # 跳板机用户名
            "ssh_tunnel_password": "ypbacYgog5jk1yE{",  # 跳板机密码
            "mongodb_address": "dds-bp1b01a6213fd5441.mongodb.rds.aliyuncs.com",  # MongoDB数据库地址
            "mongodb_port": 3717,  # MongoDB数据库端口
            "mongodb_user_name": "sle",  # MongoDB数据库用户名
            "mongodb_password": "Yiy1health_2017",  # MongoDB数据库密码
            "mongodb_name": "SLE-LN-Prediction-Patient-Data",  # MongoDB数据库名称
        }

        server = SSHTunnelForwarder(
            (
                db_connection_configuration["ssh_tunnel_address"],
                db_connection_configuration["ssh_tunnel_port"]
            ),
            ssh_username=db_connection_configuration["ssh_tunnel_user_name"],
            ssh_password=db_connection_configuration["ssh_tunnel_password"],
            remote_bind_address=(
                db_connection_configuration["mongodb_address"],
                db_connection_configuration["mongodb_port"]
            ))
        server.start()
        client = pymongo.MongoClient('localhost', server.local_bind_port)
        client[db_connection_configuration["mongodb_name"]].authenticate(
            db_connection_configuration["mongodb_user_name"],
            db_connection_configuration["mongodb_password"],
            mechanism='SCRAM-SHA-1')
        db = client[db_connection_configuration["mongodb_name"]]
        db = client['SLE-LN-Prediction-Patient-Data']
        return db


    # 从json 文件导入mogodb
    def import_from_json(self, filename, tablename):
        f = open(filename)
        s = []
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
                s.append(d)
            except json.decoder.JSONDecodeError:
                print('Error on line', i + 1, ':\n', repr(line))
        self.db[tablename].insert(s)
        print("complete!!")
        return 0


def test_csv_reader():
    channel = CsvChannel("/Users/wenhuaizhao/works/ml/CIE/tests/cie/data/胰腺癌原始数据465特征2018后_67特征_CA125fill.txt")
    channel.open()
    params = {
        "sep": '\t',
        "encoding": 'utf-8',
        "nrows": 20,
    }
    X, columns = channel.read(**params)
    print(X, columns)


def test_mongodb():
    fields = ['patientid', 'recordid', '化验组', '化验', '化验描绘词', '化验变化态', '数值', '化验数值高峰值', '数值单位', \
              '化验定性结果', '化验定性结果高峰值', '异常', '否定词', '化验条件', '时间', '段落标题', '化验组样本', '化验名称样本']

    db = ConnectMongoDB().get_connection()
    # db = MongoClient()

    proj = {}
    for it in fields:
        proj[it] = '$' + it
    project = {'$project': proj}
    # limit ={'$limit':13}

    pipe = [project]

    cursor = db["ALA"].aggregate(pipe, allowDiskUse=True)
    df = pd.DataFrame(list(cursor))
    cursor.close()

    # cursor = db..ALA.find_one()


def test_mongo():
    param_dct = {
        "local": False,
        "ssh_addr": "47.99.46.195",  # 跳板机地址
        "ssh_port": 22,  # 跳板机端口
        "ssh_user": "hitales",  # 跳板机用户名
        "ssh_pkey": './private_key',  # 跳板机private key
        "mongo_host": 'dds-bp1d8d8bebde53b41.mongodb.rds.aliyuncs.com',  # mongodb地址
        "mongo_port": 3717,  # mongodb端口
        "db_user": "root",  # mongodb 用户名
        "db_pwd": "Yiy1health_2017",  # mongodb 密码
        "db_name": "admin",  # mongodb database
        "col_name": "cnki-data",  # mongodb collection名
        "auth_method": "SCRAM-SHA-1",  # mongodb 加密机制
    }
    channel = MongoChannel()
    channel.open(**param_dct)
    query = {"fst_cat": "medicine", "item_name": {"$regex": "可乐定"}}
    df = channel.read(query=query)
    print(df)
    channel.close()


if __name__ == '__main__':
    # test_csv_reader()
    test_mongo()




