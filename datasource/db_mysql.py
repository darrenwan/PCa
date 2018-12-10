# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:02:22 2018

@author: atlan
"""

import pymysql
import numpy as np
import pandas as pd

#cur.execute("SELECT Host,User FROM user")

class MySQLClient(object):
    def __init__(self, 
                 host='rm-bp1a049g618bz7l2q.mysql.rds.aliyuncs.com',
                 port = 3306,
                 user = 'healthread',
                 password = 'rqqgK#a1nIfc3yvq',
                 database = 'yixianxiangguan'):
        
        self._conn = pymysql.connect(
                host = host,
                port = port,
                user = user,
                password = password,
                database = database,
                )
        self._cursor = self._conn.cursor()              


    def close(self):
        self._conn.close()


            
    def execute(self, sql, columns=None):
        print('将执行的sql语句是: %s' % sql)
        
        try:
            self._cursor.execute(sql)
            results = self._cursor.fetchall()
            df = pd.DataFrame(np.array(results), columns=columns)
            return df
        except Exception as e:
            print('Error:', e)
#                    'Error: unable to fetch data')
        
            
            
            