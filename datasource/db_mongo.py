import pymongo
from sshtunnel import SSHTunnelForwarder
import numpy as np
import pandas as pd

#cur.execute("SELECT Host,User FROM user")

class MongoClient(object):
    def __init__(self, 
                 host='dds-bp1f7b56b50093041.mongodb.rds.aliyuncs.com',
                 port = 3717,
                 user = 'health',
                 password = 'feoGPh7<fhwczti7',

                 local = True,
                 ssh_host = '121.199.24.144',
                 ssh_port = 22,
                 ssh_user = 'yh',
                 ssh_password = 'gy3gxUw1x[qqPxyb'):
        
        self.user = user
        self.password = password
        
        if local:
            server = SSHTunnelForwarder((ssh_host, ssh_port,), 
                                        ssh_username=ssh_user,
                                        ssh_password=ssh_password,
                                        remote_bind_address=(host, port))
            server.start()
            self.client = pymongo.MongoClient('localhost', server.local_bind_port)
        else:
            self.client = pymongo.MongoClient(host, port)


    def close(self):
        self._cursor.close()
        self._conn.close()

    def set_database(self, database):
        self.database = database
        
    def merge_documents(self, documents):
        from collections import defaultdict
        records = defaultdict()
        
        
            
    def execute(self, sql, database='dpp-test', collection='candidate'):
        print('将执行的sql语句是: db["%s"]["%s"].%s' % (database, collection, sql))
        
        self._conn = self.client[database]
        self._conn.authenticate(self.user, self.password, mechanism="SCRAM-SHA-1")
        
        try:
            documents = eval('self._conn[collection].' + sql)            
            return documents
        except Exception as e:
            print('Error:', e)


