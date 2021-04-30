import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline



def data_nodns_unknown_del(data):
    # 밑에서 사용할 함수 미리 정의
    data = data.reset_index()

    # http 데이터 처리
    # http 데이터 처리
    if 'HTTP.Request.Host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data.loc[(pd.isna(data['HTTP.Request.Host']) == True) & (
                    pd.isna(data['DNS.LastDnsTimeToLive']) == False), 'HTTP.Request.Host'] = data[
            'DNS.QueryName']

        # UNKNOWN DATA 제거
        data = data[data['App_ID'] != '0050']
        # request host(dns query name) 결측치 제거.
        dns_data = data[['HTTP.Request.Host']]
        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]
        dns_nona_data = dns_nona_data.rename(columns={'HTTP.Request.Host': 'request_host'})

        train_data = dns_nona_data.request_host.str.replace(".", " ")
        train_data = train_data.apply(lambda x: x.lstrip(' ')).tolist()

    # transport data 처리
    else:

        # unknown data 제거
        data = data[data['App_ID'] != '01BB']
        data = data[data['App_ID'] != 'C001']
        dns_data = data[['DNS.QueryName']]

        # dns query name 결측치 제거.

        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]
        dns_nona_data = dns_nona_data.rename(columns={'DNS.QueryName': 'dns_querey_name'})

        train_data = dns_nona_data.dns_querey_name.str.replace(".", " ")
        train_data = train_data.apply(lambda x: x.lstrip(' ')).tolist()

    # 전처리된 dns를 fasttext가 쓸 수 있게끔, 더블리스트 형태로.

    word2vec_train_data = []
    for i in range(len(train_data)):
        word2vec_train_data.append(train_data[i].split(' '))

    return word2vec_train_data