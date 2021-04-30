import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from gensim.models.fasttext import FastText as FT_gensim
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, GlobalMaxPool1D, SimpleRNN
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score



def preprocessing(data):
    if 'request_host' in data.columns:

        data = data[data['app_id'] != '0050']

    else:

        data = data[data['app_id'] != '01BB']
        data = data[data['app_id'] != 'C001']

    return data

def ipencoding_minmax(data):
    data_ip = data[['app_ip_version', 'app_ip_v4', 'app_ip_v6']]
    data_ip['app_ip_v4'] = data_ip.loc[:, 'app_ip_v4'].apply(hex).str.replace('0x', '').str.zfill(8)
    data_ip['app_ip_v4'] = "0:0:0:0:0:ffff:" + data_ip.loc[:, 'app_ip_v4'].str.slice(0, 4) + ":" + data_ip.loc[:,
                                                                                                   'app_ip_v4'].str.slice(
        4, 8)

    data_ip.loc[data_ip['app_ip_version'] == 1, 'app_ip'] = data_ip['app_ip_v4']
    data_ip.loc[data_ip['app_ip_version'] == 2, 'app_ip'] = data_ip['app_ip_v6']

    split = data_ip['app_ip'].str.split(':').apply(lambda x: pd.Series(x))

    for i in range(0, len(split.columns)):
        split[split.columns[i]] = split[split.columns[i]].astype(str).apply(lambda x: int(x, 16))

    scaler = MinMaxScaler()
    split = pd.DataFrame(scaler.fit_transform(split))
    split.columns = ['Aclass', 'Bclass', 'Cclass', 'Dclass', 'Eclass', 'Fclass', 'Gclass', 'Hclass']

    return split



def execute_variablename(data):
    if data.columns[0] != 'transport.summary_create_time':

        header = data.columns.str.split('http.').str[1].tolist()
        data.columns = header

    else:

        header = data.columns.str.split('transport.').str[1].tolist()
        data.columns = header

    return data


def age_onehot(data):
    age = data[['age']]
    age.loc[(age['age'].str.isnumeric() == False) | (age['age'].isnull()), "age"] = "999"
    age.age = age.age.astype(int)

    age.loc[(age['age']) < 10, "age_range"] = '10살이하'
    age.loc[(age['age'] >= 10) & (age['age'] < 20), "age_range"] = '10대'
    age.loc[(age['age'] >= 20) & (age['age'] < 30), "age_range"] = '20대'
    age.loc[(age['age'] >= 30) & (age['age'] < 40), "age_range"] = '30대'
    age.loc[(age['age'] >= 40) & (age['age'] < 50), "age_range"] = '40대'
    age.loc[(age['age'] >= 50) & (age['age'] < 60), "age_range"] = '50대'
    age.loc[(age['age'] >= 60) & (age['age'] < 70), "age_range"] = '60대'
    age.loc[(age['age'] >= 70) & (age['age'] < 999), "age_range"] = '70대이상'
    age.loc[(age['age']) == 999, "age_range"] = 'null'
    age = age.drop(['age'], axis=1)
    # age = pd.DataFrame(age.age_range.astype('category').cat.codes)
    age = pd.get_dummies(age.age_range, prefix='age_')

    return age.reset_index(drop=True)

def age_onehot(data):
    age = data[['age']]
    age.loc[(age['age'].str.isnumeric() == False) | (age['age'].isnull()), "age"] = "999"
    age.age = age.age.astype(int)

    age.loc[(age['age']) < 10, "age_range"] = '10살이하'
    age.loc[(age['age'] >= 10) & (age['age'] < 20), "age_range"] = '10대'
    age.loc[(age['age'] >= 20) & (age['age'] < 30), "age_range"] = '20대'
    age.loc[(age['age'] >= 30) & (age['age'] < 40), "age_range"] = '30대'
    age.loc[(age['age'] >= 40) & (age['age'] < 50), "age_range"] = '40대'
    age.loc[(age['age'] >= 50) & (age['age'] < 60), "age_range"] = '50대'
    age.loc[(age['age'] >= 60) & (age['age'] < 70), "age_range"] = '60대'
    age.loc[(age['age'] >= 70) & (age['age'] < 999), "age_range"] = '70대이상'
    age.loc[(age['age']) == 999, "age_range"] = 'null'
    age = age.drop(['age'], axis=1)
    # age = pd.DataFrame(age.age_range.astype('category').cat.codes)
    age = pd.get_dummies(age.age_range, prefix='age_')

    return age.reset_index(drop=True)


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[columns]
    return d



def gender_onehot(data):
    gender = data[['gender']]
    gender.loc[(gender['gender'].str.isnumeric() == False) | (gender['gender'].isnull()), "gender"] = "999"
    # gender.gender = gender.gender.astype('category')
    gender = pd.get_dummies(gender.gender, prefix='gender_')

    return gender.reset_index(drop=True)


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[columns]
    return d

def port_onehot(data):
    port = data[['app_port']]
    #port = pd.DataFrame(port.app_port.astype('category').cat.codes)
    #port = pd.DataFrame(port.app_port.astype('category'))
    port = pd.get_dummies(port.app_port, prefix = 'port_')
    return port.reset_index(drop = True)


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[columns]
    return d



def protocol_onehot(data):
    protocol = data[['protocol']]
    #protocol = pd.DataFrame(protocol.protocol.astype('category').cat.codes)
    protocol = pd.get_dummies(protocol.protocol, prefix = 'proto_')
    return protocol.reset_index(drop = True)

def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[columns]
    return d

def time_hour_onehot(data):
    data['summary_create_time'] = pd.to_datetime(data['summary_create_time'], unit='s')
    hour = pd.DataFrame(data['summary_create_time'].dt.hour)
    hour = pd.get_dummies(hour.summary_create_time, prefix='hour_')
    return hour.reset_index(drop=True)

def protocol_onehot_test(data, testdata):
    columns = list(protocol_onehot(data).columns)

    return fix_columns(protocol_onehot(testdata), columns)


def protocol_onehot(data):
    protocol = data[['protocol']]
    #protocol = pd.DataFrame(protocol.protocol.astype('category').cat.codes)
    protocol = pd.get_dummies(protocol.protocol, prefix = 'proto_')
    return protocol.reset_index(drop = True)

def port_onehot_test(data, testdata):
    columns = list(port_onehot(data).columns)

    return fix_columns(port_onehot(testdata), columns)

def port_onehot(data):
    port = data[['app_port']]
    #port = pd.DataFrame(port.app_port.astype('category').cat.codes)
    #port = pd.DataFrame(port.app_port.astype('category'))
    port = pd.get_dummies(port.app_port, prefix = 'port_')
    return port.reset_index(drop = True)

def gender_onehot_test(data, testdata):
    columns = list(gender_onehot(data).columns)

    return fix_columns(gender_onehot(testdata), columns)


def gender_onehot(data):
    gender = data[['gender']]
    gender.loc[(gender['gender'].str.isnumeric() == False) | (gender['gender'].isnull()), "gender"] = "999"
    # gender.gender = gender.gender.astype('category')
    gender = pd.get_dummies(gender.gender, prefix='gender_')

    return gender.reset_index(drop=True)
def age_onehot_test(data, testdata):
    columns = list(age_onehot(data).columns)

    return fix_columns(age_onehot(testdata), columns)



def time_hour_onehot(data):
    data['summary_create_time'] = pd.to_datetime(data['summary_create_time'], unit='s')
    hour = pd.DataFrame(data['summary_create_time'].dt.hour)
    hour = pd.get_dummies(hour.summary_create_time, prefix='hour_')
    return hour.reset_index(drop=True)

def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[columns]
    return d


def time_hour_onehot_test(data, testdata):
    columns = list(time_hour_onehot(data).columns)

    return fix_columns(time_hour_onehot(testdata), columns)



def data_nodns_del(data):
    # 밑에서 사용할 함수 미리 정의

    def remove_values_from_list(the_list, val):
        return [value for value in the_list if value != val]

    data = data.reset_index()

    # http 데이터 처리
    if 'request_host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        for i in range(len(data)):
            if (pd.isna(data['request_host'][i]) == True) and (pd.isna(data['dns_querey_name'][i]) == False):
                data['request_host'][i] = data['dns_querey_name'][i]

                # request host(dns query name) 결측치 제거.
        dns_data = data[['request_host']]
        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]

        train_data = []
        for i in range(len(dns_nona_data)):
            train_data.append(' '.join(remove_values_from_list(dns_nona_data.iloc[i, 0].split('.'), '')))

    # transport data 처리
    else:

        # unknown data 제거

        dns_data = data[['dns_querey_name']]

        # dns query name 결측치 제거.

        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]
        train_data = []

        for i in range(len(dns_nona_data)):
            train_data.append(' '.join(remove_values_from_list(dns_nona_data.iloc[i, 0].split('.'), '')))

    # 전처리된 dns를 fasttext가 쓸 수 있게끔, 더블리스트 형태로.

    word2vec_train_data = []
    for i in range(len(train_data)):
        word2vec_train_data.append(train_data[i].split(' '))

    return word2vec_train_data




def data_nodns_unknown_del(data):
    # 밑에서 사용할 함수 미리 정의

    def remove_values_from_list(the_list, val):
        return [value for value in the_list if value != val]

    data = data.reset_index()

    # http 데이터 처리
    if 'request_host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        for i in range(len(data)):
            if (pd.isna(data['request_host'][i]) == True) and (pd.isna(data['dns_querey_name'][i]) == False):
                data['request_host'][i] = data['dns_querey_name'][i]

        # UNKNOWN DATA 제거
        data = data[data['app_id'] != '0050']
        # request host(dns query name) 결측치 제거.
        dns_data = data[['request_host']]
        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]

        train_data = []
        for i in range(len(dns_nona_data)):
            train_data.append(' '.join(remove_values_from_list(dns_nona_data.iloc[i, 0].split('.'), '')))

    # transport data 처리
    else:

        # unknown data 제거
        data = data[data['app_id'] != '01BB']
        data = data[data['app_id'] != 'C001']
        dns_data = data[['dns_querey_name']]

        # dns query name 결측치 제거.

        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]
        train_data = []

        for i in range(len(dns_nona_data)):
            train_data.append(' '.join(remove_values_from_list(dns_nona_data.iloc[i, 0].split('.'), '')))

    # 전처리된 dns를 fasttext가 쓸 수 있게끔, 더블리스트 형태로.

    word2vec_train_data = []
    for i in range(len(train_data)):
        word2vec_train_data.append(train_data[i].split(' '))

    return word2vec_train_data



def making_test_bydns(data, empty):
    # http 데이터 처리
    if 'request_host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data = data[['request_host', 'dns_querey_name']]
        data = data.reset_index(drop=True)


        for i in range(len(data)):
            if (pd.isna(data['request_host'][i]) == True) and (pd.isna(data['dns_querey_name'][i]) == False):
                data['request_host'][i] = data['dns_querey_name'][i]

        data_index = data.index
        data_nona = data[np.logical_not(pd.isna(data['request_host']))]
        data_yes_index = data_nona.index
        data_no_index = data[pd.isna(data['request_host'])].index

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    # transport data 처리
    else:

        # unknown data 제거
        data = data[['dns_querey_name']]
        data = data.reset_index(drop=True)

        data_index = data.index
        data_data_nona = data[np.logical_not(pd.isna(data['dns_querey_name']))]
        data_yes_index = data_data_nona.index
        data_no_index = data[pd.isna(data['dns_querey_name'])].index

        # dns query name 결측치 제거.

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    return data_0

def making_training_bydns(data, empty):
    # http 데이터 처리
    if 'request_host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data = data[data['app_id'] != '0050']
        data = data[['request_host', 'dns_querey_name']]
        data = data.reset_index()
        data = data.drop('index', axis=1)

        for i in range(len(data)):
            if (pd.isna(data['request_host'][i]) == True) and (pd.isna(data['dns_querey_name'][i]) == False):
                data['request_host'][i] = data['dns_querey_name'][i]

        data_index = data.index
        data_nona = data[np.logical_not(pd.isna(data['request_host']))]
        data_yes_index = data_nona.index
        data_no_index = data[pd.isna(data['request_host'])].index

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    # transport data 처리
    else:

        # unknown data 제거
        data = data[data['app_id'] != '01BB']
        data = data[data['app_id'] != 'C001']

        data = data[['dns_querey_name']]
        data = data.reset_index()
        data = data.drop('index', axis=1)

        data_index = data.index
        data_data_nona = data[np.logical_not(pd.isna(data['dns_querey_name']))]
        data_yes_index = data_data_nona.index
        data_no_index = data[pd.isna(data['dns_querey_name'])].index

        # dns query name 결측치 제거.

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    return data_0


def update_fasttextmodel_vector(data, model_gensim, max_length, vector_size=5):


    model_gensim.build_vocab(data, update=True)
    model_gensim.train(data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)

    length = []
    for i in range(len(data)):
        length.append(len(data[i]))

    dns = []
    for i in range(len(data)):
        dns.append(model_gensim[data[i]].reshape(-1))

    ## dns 벡터 길이 맞추기. 기존 training 과 비교, 결측치 대비 , 오류방지
    if length == []:
       return pd.DataFrame(columns = [i for i in range(vector_size * max_length)]), model_gensim, max_length

    elif max_length < max(length):
         max_length = max(length)


    empty = pd.DataFrame(columns=[i for i in range(vector_size * max_length)])
    ###########

    for i in range(len(dns)):
        if vector_size * max_length > len(dns[i]):

            # dns 길이가 안 맞을때는 0을 앞에다 채워넣는방식으로.
            empty.loc[i] = np.concatenate((dns[i], np.zeros(vector_size * max_length - len(dns[i])), ))

        else:
            empty.loc[i] = dns[i]

    return empty, model_gensim, max_length


def make_fasttext_model_vector(word2vec_train_data, vector_size=5):
    model_gensim = FT_gensim(size=vector_size)
    model_gensim.build_vocab(word2vec_train_data)
    model_gensim.train(word2vec_train_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)

    length = []
    for i in range(len(word2vec_train_data)):
        length.append(len(word2vec_train_data[i]))

    dns = []
    for i in range(len(word2vec_train_data)):
        dns.append(model_gensim[word2vec_train_data[i]].reshape(-1))

    ##########
    if length == []:

        return pd.DataFrame(columns = [i for i in range(max_length*ve)]), model_gensim

    else:

        empty = pd.DataFrame(columns=[i for i in range(vector_size * max(length))])
    ########

    for i in range(len(dns)):
        if vector_size * max(length) > len(dns[i]):

            # dns 길이가 안 맞을때는 0을 앞에다 채워넣는방식으로.
            empty.loc[i] = np.concatenate((dns[i], np.zeros(vector_size * max(length) - len(dns[i])), ))

        else:
            empty.loc[i] = dns[i]

    return empty, model_gensim, max(length)


def train_synackmsssize(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['synackmsssize']].astype('float64'))
    output = min_max_scaler.transform(data[['synackmsssize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['synackmsssize']

    return pd.DataFrame(output)

def train_delta_up_link_data_size(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['delta_up_link_data_size']].astype('float64'))
    output = min_max_scaler.transform(data[['delta_up_link_data_size']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_up_link_data_size']

    return pd.DataFrame(output)


def train_lastdnstimetolive(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['lastdnstimetolive']].astype('float64'))
    output = min_max_scaler.transform(data[['lastdnstimetolive']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['lastdnstimetolive']

    return pd.DataFrame(output)

def train_delta_dn_link_data_size(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['delta_dn_link_data_size']].astype('float64'))
    output = min_max_scaler.transform(data[['delta_dn_link_data_size']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_dn_link_data_size']

    return pd.DataFrame(output)




def make_test_modelinput_minmax(train, test, modelinit, max_length):
    empty, modelupdate, max_length = update_fasttextmodel_vector(data_nodns_del(execute_variablename(test)), modelinit,
                                                                 max_length)
    result_dns = making_test_bydns(test, empty)
    result_ip = ipencoding_minmax(test)
    result_age = age_onehot_test(train, test)
    result_gender = gender_onehot_test(train, test)
    result_port = port_onehot_test(train, test)
    result_protocol = protocol_onehot_test(train, test)
    result_time_hour = time_hour_onehot_test(train, test)
    result_train_delta_dn_link_data_size = train_delta_dn_link_data_size(test)
    result_train_delta_up_link_data_size = train_delta_up_link_data_size(test)
    result_train_lastdnstimetolive = train_lastdnstimetolive(test)
    result_train_synackmsssize = train_synackmsssize(test)

    test_ipminmax = pd.concat(
        [result_ip, result_dns, result_age, result_gender, result_port, result_protocol, result_time_hour, result_train_delta_dn_link_data_size, result_train_delta_up_link_data_size, result_train_lastdnstimetolive, result_train_synackmsssize], axis=1)

    return test_ipminmax, modelupdate, max_length
