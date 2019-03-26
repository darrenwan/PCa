# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:41:21 2019

@author: atlan
"""

import pymysql.cursors
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='hitales',
                             db='medical',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select * from `lab_result_for_ai`"
        cursor.execute(sql)
        result = cursor.fetchall()
        print(result)
finally:
    connection.close()
    
    

#化验标准名转化
examineName = pd.read_csv('raw/化验标准名转化.txt', sep='\t', encoding='utf-8', engine='python')
examineMap = {i[0]: i[1] for i in examineName.values}
   
data = defaultdict(dict) #带仪器信息
for row in tqdm(result):
    pid = row['PATIENT_ID']
    device = row['INSTRUMENT_ID']
    examine = examineMap.get(row['REPORT_ITEM_NAME'], row['REPORT_ITEM_NAME'])
    examine_extended = examine + '【【' + device + '】】'
    result = str(row['RESULT'])
    #特征过滤，选择与训练集一致的特征
 
    data[pid][examine_extended] = result
    
data = pd.DataFrame(data).transpose()
#data = self.check_numeric(data)
   


    