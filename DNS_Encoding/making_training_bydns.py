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



def making_training_bydns(data, empty):
    # http 데이터 처리
    if 'request_host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data = data[data['App_ID'] != '0050']
        data = data[['HTTP.Request.Host', 'DNS.QueryName']]
        data = data.reset_index()
        data = data.drop('index', axis=1)

        data.loc[(pd.isna(data['HTTP.Request.Host']) == True) & (
                    pd.isna(data['DNS.QueryName']) == False), 'HTTP.Request.Host'] = data['DNS.QueryName']

        data_index = data.index
        data_nona = data[np.logical_not(pd.isna(data['HTTP.Request.Host']))]
        data_yes_index = data_nona.index
        data_no_index = data[pd.isna(data['HTTP.Request.Host'])].index

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    # transport data 처리
    else:

        # unknown data 제거
        data = data[data['App_ID'] != '01BB']
        data = data[data['App_ID'] != 'C001']

        data = data[['DNS.QueryName']]
        data = data.reset_index()
        data = data.drop('index', axis=1)

        data_index = data.index
        data_data_nona = data[np.logical_not(pd.isna(data['DNS.QueryName']))]
        data_yes_index = data_data_nona.index
        data_no_index = data[pd.isna(data['DNS.QueryName'])].index

        # dns query name 결측치 제거.

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    return data_0

