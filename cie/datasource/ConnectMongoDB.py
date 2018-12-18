#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:02:59 2018

@author: hitales
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:13:49 2018

@author: hitales
"""

#库
import pymongo
import sys
from sshtunnel import SSHTunnelForwarder
# pandas处理表格数据
import pandas as pd
import json
import  os

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

# MongoDB table

class MongoDB_Table():

    def __init__(self,tablename):
        self.tablename = tablename
        self.pid_file = "/Users/hitales/Documents/nlpwork/sle/alin_api/SLE14439.csv"
        self.ADI_Fields = ['patientid', 'recordid', '诊断', '不否定词', '诊断修饰', '部位', '部位描绘', '诊断程度', \
                           '诊断程度状态判定词', '诊断修饰状态判定词','段落标题', '时间']
        self.ALA_Fields = ['patientid','recordid','化验组','化验','化验描绘词','化验变化态','数值','化验数值高峰值','数值单位',\
                            '化验定性结果','化验定性结果高峰值','异常','否定词','化验条件','时间','段落标题','化验组样本','化验名称样本']
        self.ADR_Fields = ['patientid','recordid','段落标题','用药','用药方式','是否使用','频次','剂量数值','剂量单位',\
                           '用药持续时间','时间','部位','部位描绘','取药量','带药规格','病历示总剂量数值','病历示总剂量数值单位']
        self.ASY_Fields = ['patientid','recordid','症状','症状变化','部位','部位描绘','组织实体','段落标题','时间','症状描绘','否定词']
        self.FCTDES_Fields = ['patientid','recordid','段落标题','部位','部位描绘','组织实体','否定词','功能','功能描述','体征定量','观察值单位','时间']
        self.fields = self.get_self_fields(tablename)


    def get_self_fields(self,tablename):
        fields = ''
        if tablename == 'ADI':
            fields = self.ADI_Fields
        elif tablename == 'ALA':
            fields = self.ALA_Fields
        elif tablename == 'ADR':
            fields = self.ADR_Fields
        elif tablename == 'ASY':
            fields = self.ASY_Fields
        elif tablename == 'FCTDES':
            fields = self.FCTDES_Fields
        else:
            print("tablename lack !")

        return fields

    def read_dataframe_from_mongodb(self,db):

        # fields = ['patientid', 'recordid', '诊断', '不否定词', '诊断修饰', '部位', '部位描绘', '诊断程度', \
        #                    '诊断程度状态判定词', '诊断修饰状态判定词','段落标题', '时间']

        fields = self.fields
        tablename = self.tablename
        if self.tablename == "FCTDES":
            tablename ="ASY"

        proj = {}
        for it in fields:
            proj[it] = '$' + it
        project ={'$project':proj}

        pipe = [project]

        cursor = db[tablename].aggregate(pipe, allowDiskUse=True)
        df = pd.DataFrame.from_records(list(cursor))
        cursor.close()

        # 过滤 14439
        filter = pd.read_csv(self.pid_file)
        df = pd.merge(df,filter,how='inner',left_on="patientid",right_on='PID')
        del df["PID"]

        #处理数据库没有的字段
        mcol = set(fields) - set(df.columns)
        if len(mcol)>0:
            for col in mcol:
                df[col]=''

        df = df[fields]
        print(self.tablename+" dataframe success !")
        return df

    def save_dataframe_from_mongodb(self,db):

        # fields = ['patientid', 'recordid', '诊断', '不否定词', '诊断修饰', '部位', '部位描绘', '诊断程度', \
        #                    '诊断程度状态判定词', '诊断修饰状态判定词','段落标题', '时间']

        fields = self.fields
        tablename = self.tablename
        if self.tablename == "FCTDES":
            tablename ="ASY"

        proj = {}
        for it in fields:
            proj[it] = '$' + it
        project ={'$project':proj}

        pipe = [project]

        cursor = db[tablename].aggregate(pipe, allowDiskUse=True)
        df = pd.DataFrame.from_records(list(cursor))
        cursor.close()

        # 过滤 14439
        filter = pd.read_csv(self.pid_file)
        df = pd.merge(df,filter,how='inner',left_on="patientid",right_on='PID')
        del df["PID"]

        #处理数据库没有的字段
        mcol = set(fields) - set(df.columns)
        if len(mcol)>0:
            for col in mcol:
                df[col]=''

        df = df[fields]
        file_name = self.tablename+'_mongo.csv'
        df.to_csv(file_name)
        print(self.tablename+" dataframe success !")
        return df

    def read_dataframe_from_table(self,file_name='',filter_successed=False):
        # queryArgs = {}
        # # projectionFields = ['patientid', 'recordid', '诊断', '不否定词', '诊断修饰', '部位', '部位描绘', '诊断程度', '诊断程度状态判定词',
        # #                     '诊断修饰状态判定词', '段落标题', '时间']  # 用列表指定，结果中一定会返回_id这个字段
        # # tablename ='ADI' fields = ADI_Fields
        # cursor = db[tablename].find(queryArgs,no_cursor_timeout=True)
        # time.sleep(1)
        # df = pd.DataFrame.from_records(list(cursor))
        # cursor.close()
        #
        # #处理数据库没有的字段
        # mcol = set(fields) - set(df.columns)
        # if len(mcol)>0:
        #     for col in mcol:
        #         df[col]=''
        # target_dir = '/Users/hitales/Documents/nlpwork/sle/alin_api/'
        # file_name = target_dir+'ADR_mongo.csv'
        # file_name = self.tablename+'_mongo.csv'
        # file_name = 'data_9'
        df =None
        try:
            df = pd.read_csv(file_name,index_col=0,na_values=['nan'],keep_default_na=False)
        except FileNotFoundError:
            print("error: cache 文件不存在，尚未导出："+file_name)


        #过滤掉已经处理的pid  succfile=target_dir + 'success_ADR.csv'
        if filter_successed:
            succfile = 'success_{}.csv'.format(self.tablename)
            if os.path.exists(succfile):
                try:
                    filter = pd.read_csv(succfile)
                    filter.columns = ["PID"]
                    filter.drop_duplicates(inplace=True)
                    pidcol = pd.merge(df, filter, how='left', left_on="patientid", right_on='PID')
                    df = pidcol[pidcol['PID'].isnull()]
                except Exception:
                    print("warn: successpid 文件不存在！" )
                    pass


        return df

    # 诊断信息转换
    def trans_ADI_fields_post(self,db):

        fields = self.ADI_Fields
        adi_tb = self.get_dataframe_from_table(db,'ADI',fields)
        itels = []

        for _, item in adi_tb.iterrows():
            dc = {}
            for pv in fields:
                # print(item[pv])
                if pv == 'patientid':
                    pvo = 'patientId'
                    dc[pvo] = item[pv]
                elif pv == 'recordid':
                    pvo = 'recordId'
                    dc[pvo] = item[pv]
                else:
                    dc[pv] = item[pv]
            itels.append(dc)
        return itels


    # 化验信息转换
    def trans_fields_post(self,file_name ):

        adi_tb = self.read_dataframe_from_table(file_name)
        itels = []

        for _, item in adi_tb.iterrows():
            dc = {}
            for pv in self.fields:
                # print(item[pv])
                if pv == 'patientid':
                    pvo = 'patientId'
                    dc[pvo] = item[pv]
                elif pv == 'recordid':
                    pvo = 'recordId'
                    dc[pvo] = item[pv]
                else:
                    try:
                        dc[pv] = item[pv]
                    except KeyError:
                        dc[pv] =None
                        print("KeyError:"+str(pv))

            itels.append(dc)
        return itels,adi_tb


    def write_to_mongodb(self,db,out_df,out_table_name): # tb_name = "ADI_ALIN"

        db[out_table_name].drop()
        db.create_collection(out_table_name)
        db[out_table_name].insert(json.loads(out_df.T.to_json()).values())
        print("write success!")
        # sledb["ASY_SLE_MATCH"].drop()




###
import uuid
from pymongo import MongoClient
#两地址
CONN_ADDR1 = 'dds-bp1baff8ad4002a41567.mongodb.rds.aliyuncs.com:3717'
CONN_ADDR2 = 'dds-bp1baff8ad4002a42323.mongodb.rds.aliyuncs.com:3717'
REPLICAT_SET = 'mgset-1441984463'
username = 'huiyong_zhang'
password = 'ORzmD9U$26z%'
#获取mongoclient
client = MongoClient([CONN_ADDR1, CONN_ADDR2], replicaSet=REPLICAT_SET)
#授权. 这里的user基于admin数据库授权
client.admin.authenticate(username, password)
#使用test数据库的collection:HDP-live 做例子, 插入doc, 然后根据DEMO名查找
demo_name = 'python-' + str(uuid.uuid1())
print('demo_name:', demo_name)
doc = dict(DEMO=demo_name, MESG="Hello ApsaraDB For MongoDB")
doc_id = client.test.testColl.insert(doc)
print('doc_id:', doc_id)
for d in client.test.testColl.find(dict(DEMO=demo_name)):
    print('find documents:', d)

