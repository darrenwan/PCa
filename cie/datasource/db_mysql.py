# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:02:22 2018

@author:
"""

import pymysql
import numpy as np
import pandas as pd
from sshtunnel import SSHTunnelForwarder
from cie.common.settings import MYSQL_CONFIG

#cur.execute("SELECT Host,User FROM user")

class MySQLClient(object):
    def __init__(self,config):


        server = SSHTunnelForwarder((config["ssh_host"], config["ssh_port"]),
                                    ssh_username=config["ssh_user"],
                                    # ssh_password=config["ssh_password"],
                                    ssh_private_key=config["ssh_private_key"],
                                    remote_bind_address=(config["host"], config["port"])
                                    )
        server.start()
        print ("SERVER ALIVE: {0}".format(server.is_alive))
        self._conn = pymysql.connect(
            host = 'localhost',
            port = server.local_bind_port,
            user = config["user"],
            password = config["password"],
            database = config["dbname"]
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


if __name__ == '__main__':
    msq = MySQLClient(MYSQL_CONFIG)
    tb = msq.execute(sql="SELECT * FROM iiy.`20181126批次` t ;")

