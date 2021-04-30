# MODULE IMPORT & ETC
#======================================================================================================================#
import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from datetime import date, timedelta
from imblearn.over_sampling import RandomOverSampler
from gensim.models.fasttext import FastText as FT_gensim
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, GlobalMaxPool1D, SimpleRNN
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from DNS_Encoding.data_nodns_del import data_nodns_del
from DNS_Encoding.make_fasttext_model_vector2 import make_fasttext_model_vector2
from DNS_Encoding.making_training_bydns import making_training_bydns
import pymysql
from IP_Encoding.ipencoding_65536_delH import parallelize_dataframe
from IP_Encoding.ipencoding_65536_delH import ipencoding_65536_delH
from Etc_features.age_onehot import age_onehot
from Etc_features.gender_onehot import gender_onehot
from Etc_features.port_onehot import port_onehot
from Etc_features.protocol_onehot import protocol_onehot
from Etc_features.train_lastdnstimetolive import train_lastdnstimetolive
from Etc_features.train_delta_dn_link_data_size import train_delta_dn_link_data_size
from Etc_features.train_delta_up_link_data_size import train_delta_up_link_data_size
from Etc_features.train_synackmsssize import train_synackmsssize
from Etc_features.train_delta_dn_link_payload_size import train_delta_dn_link_payload_size
from Etc_features.train_delta_up_link_data_size import train_delta_up_link_data_size
from DNS_Encoding.data_nodns_unknown_del import data_nodns_unknown_del
from Etc_features.train_delta_up_link_payload_size import train_delta_up_link_payload_size
from Etc_features.train_delta_dn_link_payload_size import train_delta_dn_link_payload_size



warnings.filterwarnings('ignore')

# ======================================================================================================================#
# Input, Save data name
# ======================================================================================================================#
ExecutionTime = datetime.now()
CREATE_TIME = pd.to_datetime(ExecutionTime)

### 시스템용. 
# Input data
#current = str(datetime.now())[:13] + ":00"
#before1hour = str(datetime.now() - timedelta(hours=1))
#before1hour = before1hour[:13] + ":00"
#datatable = before1hour.replace("-", "")[:8]
#before1hour_date = 'T_AD_XDR_HTTP_KNOWN_' + datatable


###  데모용
# 테스트용
#current = '2020-01-21 01:00'
#before1hour = '2020-01-21 00:00'

datatable = '20200121'
before1hour_date = 'T_AD_XDR_HTTP_KNOWN_' + datatable


# 저장할 모델 이름



#======================================================================================================================#
# Data Load
#======================================================================================================================#

#http = pd.read_csv('http_classifier_train_20191212.csv')

conn = pymysql.connect(host ='192.168.6.89' , port = 3306,
                       user = 'atom', password = 'atom', db = '5G_Probe_App_Discovery', charset = 'utf8',
                      cursorclass = pymysql.cursors.DictCursor)

# 커서 가져오기 - 시간 정확히 다시 확인 필요
curs = conn.cursor()


# 전체 데이터 가져오기. 
#sql = "SELECT * FROM " + before1hour_date

## 현 시간으로부터 1시간전 가져오기
sql = "SELECT * FROM " + before1hour_date + " WHERE OngoingFlag = 1 AND SummaryCreateTime >= unix_timestamp(date_add(now(), interval -60 minute)) AND SummaryCreateTime < unix_timestamp(now()) ;"




curs.execute(sql)

# 데이터 가져오기
http = pd.DataFrame(curs.fetchall())
http = http[['SummaryCreateTime', 'IMSI', 'Gender', 'Age', 'App_ID', 'App_Group_Code', 'App_Port','Protocol', 'IP_Version', 'App_IPv4', 'App_IPv6',  'DNS.QueryName', 'HTTP.Request.Host', 'DNS.LastDnsTimeToLive', 'Traffic.Delta.Dnlink.DataSize', 'Traffic.Delta.Uplink.DataSize', 'TCP.Delta.MSS.SynMssSize', 'TCP.Delta.MSS.SynAckMssSize', 'Traffic.Delta.Uplink.RetransPayloadSize', 'Traffic.Delta.Dnlink.RetransPayloadSize', 'HTTP.Delta.Quility.TotResponseTime']].copy()

# ======================================================================================================================#
# Preprocessing
# ======================================================================================================================#
# ----------------------- 1. 동일Rule인데 app_id가 다른 경우 삭제
a = http.reset_index()
group1 = a.groupby(['Protocol', 'App_Port','DNS.QueryName', 'App_IPv4', 'App_IPv6']).agg(
    {"App_ID": lambda x: x.nunique()}).reset_index()
group2 = a.groupby(['Protocol', 'App_Port', 'DNS.QueryName', 'App_IPv4', 'App_IPv6'])['index'].unique().reset_index()

group1 = group1[group1['App_ID'] >= 2].index.tolist()
group2 = group2[group2.index.isin(group1)].reset_index(drop=True)

index_du = []
for i in range(len(group2)):
    index_du += list(group2['index'][i])

index_du.sort()
http = http[~http.index.isin(index_du)].reset_index(drop=True)




appid = list(http.groupby(['App_ID']).count().index)

# -*- coding: utf-8 -*-

#======================================================================================================================#

# ----------------------- 0. 정확도 판단을 위해 appid rule별 데이터 만들기

# ----------------------- 2. IP encoding

http_ip65536_delH = parallelize_dataframe(http, ipencoding_65536_delH)

# ----------------------- 3. DNS encoding

http_1 = data_nodns_unknown_del(http)
http_vector, http_modelinit, max_length_http = make_fasttext_model_vector2(http_1)
httpdns_training = making_training_bydns(http, http_vector)

# ----------------------- 4.  ETC Features encoding

http_age = age_onehot(http)
http_gender = gender_onehot(http)
http_port = port_onehot(http)
http_protocol = protocol_onehot(http)
http_lastdnstimetolive = train_lastdnstimetolive(http)
http_delta_dn_link_data_size = train_delta_dn_link_data_size(http)
http_delta_up_link_data_size = train_delta_up_link_data_size(http)
http_synackmsssize = train_synackmsssize(http)
http_delta_dn_link_payload_size = train_delta_dn_link_payload_size(http)
http_delta_up_link_payload_size = train_delta_up_link_payload_size(http)

http_training_ip65536_delH = pd.concat(
    [http_ip65536_delH, httpdns_training, http_port, http_protocol,
                                             http_lastdnstimetolive,http_delta_dn_link_data_size,http_delta_up_link_data_size,http_synackmsssize,http_delta_dn_link_payload_size,
                                             http_delta_up_link_payload_size], axis=1)



http_training_ip65536_delH = pd.concat([http_training_ip65536_delH, http[['App_ID']]], axis = 1)

a = pd.DataFrame(columns = http_training_ip65536_delH.columns)

for i in range(len(appid)):
    if len(http_training_ip65536_delH[http_training_ip65536_delH.App_ID == appid[i]])<6:
        xx = http_training_ip65536_delH.loc[list(http_training_ip65536_delH[http_training_ip65536_delH.App_ID == appid[i]].index)]
        xx['anomaly_score'] = [2 for i in range(len(xx))]
        a = pd.concat([a,xx])

            
    else:
        clf = IsolationForest(n_estimators=50, contamination=0.2, random_state=35, verbose=1)
        clf.fit(http_training_ip65536_delH[http_training_ip65536_delH.App_ID == appid[i]].drop(['App_ID'], axis = 1))
        y_pred = clf.score_samples(http_training_ip65536_delH[http_training_ip65536_delH.App_ID == appid[i]].drop(['App_ID'], axis = 1)) #1,-1로 나온다.
        xx = http_training_ip65536_delH.loc[list(http_training_ip65536_delH[http_training_ip65536_delH.App_ID == appid[i]].index)]
        xx['anomaly_score'] = list(y_pred)
        a = pd.concat([a,xx])
        
        
###### -1이 이상치, 1이 정상.

http = pd.concat([http, a[['anomaly_score']]], axis = 1)
b = pd.DataFrame(columns = http.columns)

for i in range(len(appid)):
    
    for j in range(len(http[http.App_ID == appid[i]].groupby(['Protocol', 'App_Port','DNS.QueryName', 'App_IPv4', 'App_IPv6']).count().index)):
  
        ss = http[http.App_ID == appid[i]].groupby(['Protocol', 'App_Port','DNS.QueryName', 'App_IPv4', 'App_IPv6']).count().index[j]
        xx = ((http[['Protocol', 'App_Port','DNS.QueryName', 'App_IPv4', 'App_IPv6']] ==list(ss)).sum(axis=1)==5)
        
        
        if sum(xx) >= 6:
        # 이게 각 rule 별 anomaly_score다.    
            if sum(http.loc[xx.index[xx], 'anomaly_score'] <-0.5)/len(http.loc[xx.index[xx]]) >= 0.6:
                b = pd.concat([b, http.loc[xx.index[xx]]])
       
        

        
b = b.drop(['anomaly_score'], axis=1)

test = b.reset_index(drop=True)



http = http.drop(['anomaly_score'], axis=1)

model = load_model('http_appid_model_20200120.h5')

with open('http_id_encoder_20200120.p', 'rb') as f:
    encoding = pickle.load(f)
    
    
from DNS_Encoding.data_nodns_del import data_nodns_del
from DNS_Encoding.make_fasttext_model_vector2 import make_fasttext_model_vector2
from DNS_Encoding.making_training_bydns import making_training_bydns
http_1 = data_nodns_del(http)
http_vector, http_modelinit, max_length_http = make_fasttext_model_vector2(http_1)

from modelling.make_test_modelinput_65536_delH import make_test_modelinput_http

modelinput = pd.DataFrame(make_test_modelinput_http(http, test, http_modelinit, max_length_http)[0])

y_prob = model.predict(modelinput) 
y_classes = y_prob.argmax(axis=-1)


result = pd.DataFrame(columns = list(encoding.classes_), data= y_prob).transpose()
result.columns = [str(i) for i in range(len(result.columns))]


a = pd.DataFrame(columns = ['app_id', 'Prob'])
for i in range(len(result.columns)):
    a = pd.concat([a, pd.DataFrame({'app_id':result.iloc[:, i].sort_values(ascending = False)[:10].index, 'Prob':result.iloc[:, i].sort_values(ascending = False)[:10].values})])
    

a = a.reset_index(drop = True)
test = test[['SummaryCreateTime', 'IMSI', 'App_ID']]

newdf = pd.DataFrame(np.repeat(test.values,10,axis=0))
newdf.columns = ['SUMMARY_CREATE_TIME', 'IMSI', 'OLD_APP_ID']

newdf['RANK'] = [1,2,3,4,5,6,7,8,9,10] * (int(len(newdf)/10))

output = pd.merge(newdf,a, left_index= True, right_index=True)

output = output[['SUMMARY_CREATE_TIME', 'IMSI', 'RANK', 'OLD_APP_ID', 'app_id', 'Prob']]

output.columns = ['SUMMARY_CREATE_TIME', 'IMSI', 'RANK', 'OLD_APP_ID', 'RECOMMEND_APP_ID', 'RECOMMEND_PROB']

import pandas as pd
from sqlalchemy import create_engine
import pymysql
# MySQL Connector using pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

engine = create_engine("mysql+mysqldb://atom:"+"atom"+"@192.168.6.89/5G_Probe_App_Discovery", encoding='utf-8')
conn = engine.connect()
output.to_sql(name='T_AD_RESULT_HTTP_RULE_ERROR', con=engine, if_exists='append', index=False)