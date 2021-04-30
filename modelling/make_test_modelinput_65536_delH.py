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
import multiprocessing
from multiprocessing import Pool


def ipencoding_65536_delH(data):
    data_ip = data[['IP_Version', 'App_IPv4', 'App_IPv6']].copy()
    data_ip['App_IPv4'] = data_ip.loc[:, 'App_IPv4'].astype(int).apply(hex).str.replace('0x', '').str.zfill(8)
    data_ip['App_IPv4'] = "0:0:0:0:0:ffff:" + data_ip.loc[:, 'App_IPv4'].str.slice(0, 4)
    data_ip.loc[data_ip['IP_Version'] == 1, 'app_ip'] = data_ip['App_IPv4']
    data_ip.loc[data_ip['IP_Version'] == 2, 'app_ip'] = data_ip['App_IPv6']

    split = data_ip['app_ip'].str.split(':').apply(lambda x: pd.Series(x))
    split = split.iloc[ :, 0:7]

    for i in range(0, len(split.columns)):
        split[split.columns[i]] = split[split.columns[i]].astype(str).apply(lambda x: int(x, 16))
        split[split.columns[i]] = split[split.columns[i]] / 65535

    split.columns = ['Aclass', 'Bclass', 'Cclass', 'Dclass', 'Eclass', 'Fclass', 'Gclass']

    return split

def parallelize_dataframe(df, func):
    num_cores = multiprocessing.cpu_count()

    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df



def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    #extra_cols = set(d.columns) - set(columns)
    #if extra_cols:
    #    print("extra columns:", extra_cols)

    d = d[columns]
    return d


def age_onehot(data):
    age = data[['Age']].copy()
    age.Age = age.Age.astype(str)
    age.loc[(age['Age'].str.isnumeric() == False) | (age['Age'].isnull()), "Age"] = "999"
    age.Age = age.Age.astype(int)

    age.loc[(age['Age']) < 10, "age_range"] = '10살이하'
    age.loc[(age['Age'] >= 10) & (age['Age'] < 20), "age_range"] = '10대'
    age.loc[(age['Age'] >= 20) & (age['Age'] < 30), "age_range"] = '20대'
    age.loc[(age['Age'] >= 30) & (age['Age'] < 40), "age_range"] = '30대'
    age.loc[(age['Age'] >= 40) & (age['Age'] < 50), "age_range"] = '40대'
    age.loc[(age['Age'] >= 50) & (age['Age'] < 60), "age_range"] = '50대'
    age.loc[(age['Age'] >= 60) & (age['Age'] < 70), "age_range"] = '60대'
    age.loc[(age['Age'] >= 70) & (age['Age'] < 999), "age_range"] = '70대이상'
    age.loc[(age['Age']) == 999, "age_range"] = 'x'
    age = age.drop(['Age'], axis=1)
    age = pd.get_dummies(age.age_range, prefix='age_')

    if age.shape[1] < 9:
        age_list = ['age__10살이하', 'age__10대', 'age__20대', 'age__30대', 'age__40대', 'age__50대', 'age__60대', 'age__70대이상', 'age__x']
        a = list(set(age_list) - set(age.columns))
        for i in range(len(a)):
            age[a[i]] = 0
    age = age.sort_index(axis=1)
    return age.reset_index(drop=True)

def age_onehot_test(data, testdata):
    columns = list(age_onehot(data).columns)

    return fix_columns(age_onehot(testdata), columns)


def gender_onehot(data):
    gender = data[['Gender']].copy()
    gender.Gender = gender.Gender.astype(str)
    gender.loc[(gender['Gender'].str.isnumeric() == False) | (gender['Gender'].isnull()) | (gender['Gender'] == "0"), "Gender"] = "999"
    gender.loc[(gender['Gender']) == "999", "Gender"] = 'x'
    gender = pd.get_dummies(gender.Gender, prefix='gender_')

    if gender.shape[1] < 3:
        gender_list = ['gender__1', 'gender__2', 'gender__x']
        a = list(set(gender_list) - set(gender.columns))
        for i in range(len(a)):
            gender[a[i]] = 0

    gender = gender.sort_index(axis=1)
    return gender.reset_index(drop=True)



def gender_onehot_test(data, testdata):
    columns = list(gender_onehot(data).columns)

    return fix_columns(gender_onehot(testdata), columns)



def port_onehot(data):
    port = data[['App_Port']].copy()
    port_list = port['App_Port'].value_counts(sort=True, ascending=False)

    if len(port_list) < 10:
        port = pd.get_dummies(port.App_Port, prefix='port_')
        port['port_x'] = 0

    else:
        port_list = port_list[:10]

        a = port[port['App_Port'].isin(port_list.index)]
        b = port[~port['App_Port'].isin(port_list.index)]
        b['App_Port'] = 'x'
        port = pd.concat([a, b]).sort_index()
        port = pd.get_dummies(port.App_Port, prefix='port_')

    port = port.sort_index(axis=1)
    return port.reset_index(drop=True)



def port_onehot_test(data, testdata):
    columns = list(port_onehot(data).columns)

    return fix_columns(port_onehot(testdata), columns)


def protocol_onehot(data):
    protocol = data[['Protocol']].copy()
    protocol_list = protocol['Protocol'].value_counts(sort=True, ascending=False)

    if len(protocol_list) < 10:
        protocol = pd.get_dummies(protocol.Protocol, prefix='protocol_')
        protocol['protocol_x'] = 0

    else:
        protocol_list = protocol_list[:10]
        a = protocol[protocol['Protocol'].isin(protocol_list.index)]
        b = protocol[~protocol['Protocol'].isin(protocol_list.index)]
        b['Protocol'] = 'x'
        protocol = pd.concat([a, b]).sort_index()
        protocol = pd.get_dummies(protocol.Protocol, prefix='protocol_')

    protocol = protocol.sort_index(axis=1)
    return protocol.reset_index(drop = True)



def protocol_onehot_test(data, testdata):
    columns = list(protocol_onehot(data).columns)

    return fix_columns(protocol_onehot(testdata), columns)


def time_hour_onehot(data):
    hour = data[['SummaryCreateTime']].copy()
    hour['SummaryCreateTime'] = hour['SummaryCreateTime'].apply(
        lambda x: datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H-%M'))
    hour = pd.DataFrame(hour['SummaryCreateTime'].apply(lambda x: x[11:13]))
    hour = pd.get_dummies(hour.SummaryCreateTime, prefix='hour_')

    if hour.shape[1] < 24:
        hour_list = ['hour__00', 'hour__01', 'hour__02', 'hour__03', 'hour__04', 'hour__05', 'hour__06', 'hour__07',
                     'hour__08', 'hour__09', 'hour__10', 'hour__11',
                     'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16', 'hour__17', 'hour__18', 'hour__19',
                     'hour__20', 'hour__21', 'hour__22', 'hour__23']
        a = list(set(hour_list) - set(hour.columns))
        for i in range(len(a)):
            hour[a[i]] = 0

    hour.sort_index(axis=1)
    return hour.reset_index(drop=True)


def time_hour_onehot_test(data, testdata):
    columns = list(time_hour_onehot(data).columns)

    return fix_columns(time_hour_onehot(testdata), columns)


def train_synackmsssize(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['TCP.Delta.MSS.SynAckMssSize']].astype('float64'))
    output = min_max_scaler.transform(data[['TCP.Delta.MSS.SynAckMssSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['synackmsssize']

    return pd.DataFrame(output)


def train_lastdnstimetolive(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['DNS.LastDnsTimeToLive']].astype('float64'))
    output = min_max_scaler.transform(data[['DNS.LastDnsTimeToLive']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['lastdnstimetolive']

    return pd.DataFrame(output)


def train_delta_up_link_data_size(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Uplink.DataSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Uplink.DataSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_up_link_data_size']

    return pd.DataFrame(output)


def train_delta_dn_link_data_size(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Dnlink.DataSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Dnlink.DataSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_dn_link_data_size']

    return pd.DataFrame(output)


def train_delta_up_link_payload_size(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Uplink.RetransPayloadSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Uplink.RetransPayloadSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_up_link_payload_size']

    return pd.DataFrame(output)


def train_delta_dn_link_payload_size(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Dnlink.RetransPayloadSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Dnlink.RetransPayloadSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_dn_link_payload_size']

    return pd.DataFrame(output)


def train_quality_total_response_time(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['HTTP.Delta.Quility.TotResponseTime']].astype('float64'))
    output = min_max_scaler.transform(data[['HTTP.Delta.Quility.TotResponseTime']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['quality_total_response_time']

    return pd.DataFrame(output)


def train_synmsssize(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['TCP.Delta.MSS.SynMssSize']].astype('float64'))
    output = min_max_scaler.transform(data[['TCP.Delta.MSS.SynMssSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['synmsssize']

    return pd.DataFrame(output)



def data_nodns_del(data):
    # 밑에서 사용할 함수 미리 정의
    data = data.reset_index()

    # http 데이터 처리
    if 'HTTP.Request.Host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data.loc[(pd.isna(data['HTTP.Request.Host']) == True) & (
                    pd.isna(data['DNS.QueryName']) == False), 'HTTP.Request.Host'] = data['DNS.QueryName']

                # request host(dns query name) 결측치 제거.
        dns_data = data[['HTTP.Request.Host']]
        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]
        dns_nona_data = dns_nona_data.rename(columns={'HTTP.Request.Host': 'request_host'})

        train_data = dns_nona_data.request_host.str.replace(".", " ")
        train_data = train_data.apply(lambda x:x.lstrip(' ')).tolist()

    # transport data 처리
    else:

        # unknown data 제거

        dns_data = data[['DNS.QueryName']]

        # dns query name 결측치 제거.

        dns_nona_data = dns_data[np.logical_not(pd.isna(dns_data).values)]
        dns_nona_data = dns_nona_data.rename(columns={'DNS.QueryName': 'dns_querey_name'})

        train_data = dns_nona_data.dns_querey_name.str.replace(".", " ")
        train_data = train_data.apply(lambda x:x.lstrip(' ')).tolist()

    # 전처리된 dns를 fasttext가 쓸 수 있게끔, 더블리스트 형태로.

    word2vec_train_data = []
    for i in range(len(train_data)):
        word2vec_train_data.append(train_data[i].split(' '))

    return word2vec_train_data


def data_nodns_unknown_del(data):
    # 밑에서 사용할 함수 미리 정의
    data = data.reset_index()

    # http 데이터 처리
    # http 데이터 처리
    if 'HTTP.Request.Host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data.loc[(pd.isna(data['HTTP.Request.Host']) == True) & (
                    pd.isna(data['DNS.LastDnsTimeToLive']) == False), 'HTTP.Request.Host'] = data[
            'DNS.LastDnsTimeToLive']

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
        dns_data = data[['DNS.LastDnsTimeToLive']]

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



def making_test_bydns(data, empty):
    # http 데이터 처리
    if 'HTTP.Request.Host' in data.columns:

        ## REQUEST HOST 와 dns querey name 통합
        data = data[['HTTP.Request.Host', 'DNS.QueryName']].copy()
        data = data.reset_index(drop=True)

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
        data = data[['DNS.QueryName']].copy()
        data = data.reset_index(drop=True)

        data_index = data.index
        data_data_nona = data[np.logical_not(pd.isna(data['DNS.QueryName']))]
        data_yes_index = data_data_nona.index
        data_no_index = data[pd.isna(data['DNS.QueryName'])].index

        # dns query name 결측치 제거.

        data_0 = pd.DataFrame(index=data_index, columns=empty.columns).fillna(0)
        empty.index = data_yes_index
        data_0.loc[data_yes_index] = empty

    return data_0



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


"""
def update_fasttextmodel_vector(data, model_gensim, max_length, vector_size=5):

    model_gensim.build_vocab(data, update=True)
    model_gensim.train(data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)

    length = []
    for i in range(len(data)):
        length.append(len(data[i]))

    df = pd.DataFrame({"col": data})
    df['dns'] = df.col.apply(lambda x: model_gensim[x].reshape(-1))
    dns = list(df['dns'])

    ## dns 벡터 길이 맞추기. 기존 training 과 비교, 결측치 대비 , 오류방지
    if length == []:
        pd.DataFrame(
            columns=[i for i in range(vector_size * max_length)]), model_gensim, max_length

    elif max_length < max(length):
        max_length = max(length)

    empty = pd.DataFrame(columns=[i for i in range(vector_size * max_length)])

    lens = np.array([len(i) for i in dns])

    # Mask of valid places in each row
    mask = np.arange(vector_size * max_length) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(dns)
    empty = pd.DataFrame(out)


    return empty, model_gensim, max_length
"""



def predict_fasttextmodel_vector(data, model_gensim, max_length, vector_size=5):

    length = []
    for i in range(len(data)):
        length.append(len(data[i]))

    df = pd.DataFrame({"col": data})
    df['dns'] = df.col.apply(lambda x: model_gensim[x].reshape(-1))
    dns = list(df['dns'])

    ## dns 벡터 길이 맞추기. 기존 training 과 비교, 결측치 대비 , 오류방지
    if length == []:
        pd.DataFrame(
            columns=[i for i in range(vector_size * max_length)]), model_gensim, max_length

    elif max_length < max(length):
        max_length = max(length)

    empty = pd.DataFrame(columns=[i for i in range(vector_size * max_length)])

    lens = np.array([len(i) for i in dns])

    # Mask of valid places in each row
    mask = np.arange(vector_size * max_length) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(dns)
    empty = pd.DataFrame(out)

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


def make_test_modelinput_http(train, test, modelinit, max_length):
    #empty, modelupdate, max_length = update_fasttextmodel_vector(data_nodns_del(execute_variablename(test)), modelinit, max_length)

    empty, modelupdate, max_length = predict_fasttextmodel_vector(data_nodns_del(test), modelinit, max_length)

    result_dns = making_test_bydns(test, empty)
    result_ip = parallelize_dataframe(test, ipencoding_65536_delH)
    result_age = age_onehot_test(train, test)
    result_gender = gender_onehot_test(train, test)
    result_port = port_onehot_test(train, test)
    result_protocol = protocol_onehot_test(train, test)
    result_time_hour = time_hour_onehot_test(train, test)

    result_train_lastdnstimetolive = train_lastdnstimetolive(test)
    result_train_delta_dn_link_data_size = train_delta_dn_link_data_size(test)
    result_train_delta_up_link_data_size = train_delta_up_link_data_size(test)
    result_train_synackmsssize = train_synackmsssize(test)
    result_train_delta_dn_link_payload_size = train_delta_dn_link_payload_size(test)
    result_train_delta_up_link_payload_size = train_delta_up_link_payload_size(test)
    result_train_quality_total_response_time = train_quality_total_response_time(test)
    result_train_synmsssize  = train_synmsssize(test)

    test_ip65536_delH = pd.concat(
        [result_ip, result_dns, result_port, result_train_delta_dn_link_data_size, result_train_delta_up_link_data_size, result_train_lastdnstimetolive, result_train_synackmsssize,
         result_train_delta_dn_link_payload_size, result_train_delta_up_link_payload_size, result_train_quality_total_response_time, result_train_synmsssize], axis=1)

    return test_ip65536_delH, modelupdate, max_length


def make_test_modelinput_http_appgroup(train, test, modelinit, max_length):
    #empty, modelupdate, max_length = update_fasttextmodel_vector(data_nodns_del(execute_variablename(test)), modelinit, max_length)

    empty, modelupdate, max_length = predict_fasttextmodel_vector(data_nodns_del(test), modelinit, max_length)

    result_dns = making_test_bydns(test, empty)
    result_ip = parallelize_dataframe(test, ipencoding_65536_delH)
    result_age = age_onehot_test(train, test)
    result_gender = gender_onehot_test(train, test)
    result_port = port_onehot_test(train, test)
    result_protocol = protocol_onehot_test(train, test)
    result_time_hour = time_hour_onehot_test(train, test)

    result_train_lastdnstimetolive = train_lastdnstimetolive(test)
    result_train_delta_dn_link_data_size = train_delta_dn_link_data_size(test)
    result_train_delta_up_link_data_size = train_delta_up_link_data_size(test)
    result_train_synackmsssize = train_synackmsssize(test)
    result_train_delta_dn_link_payload_size = train_delta_dn_link_payload_size(test)
    result_train_delta_up_link_payload_size = train_delta_up_link_payload_size(test)
    result_train_quality_total_response_time = train_quality_total_response_time(test)
    result_train_synmsssize  = train_synmsssize(test)

    test_ip65536_delH = pd.concat(
        [result_ip, result_dns, result_port, result_train_delta_dn_link_data_size, result_train_delta_up_link_data_size, result_train_lastdnstimetolive, result_train_synackmsssize,
         result_train_delta_dn_link_payload_size, result_train_delta_up_link_payload_size, result_train_quality_total_response_time], axis=1)

    return test_ip65536_delH, modelupdate, max_length














def make_test_modelinput_tr(train, test, modelinit, max_length):
    #empty, modelupdate, max_length = update_fasttextmodel_vector(data_nodns_del(execute_variablename(test)), modelinit, max_length)

    empty, modelupdate, max_length = predict_fasttextmodel_vector(data_nodns_del(test), modelinit, max_length)

    result_dns = making_test_bydns(test, empty)
    result_ip = parallelize_dataframe(test, ipencoding_65536_delH)
    result_age = age_onehot_test(train, test)
    result_gender = gender_onehot_test(train, test)
    result_port = port_onehot_test(train, test)
    result_protocol = protocol_onehot_test(train, test)
    result_time_hour = time_hour_onehot_test(train, test)

    result_train_lastdnstimetolive = train_lastdnstimetolive(test)
    result_train_delta_dn_link_data_size = train_delta_dn_link_data_size(test)
    result_train_delta_up_link_data_size = train_delta_up_link_data_size(test)
    result_train_synackmsssize = train_synackmsssize(test)
    result_train_delta_dn_link_payload_size = train_delta_dn_link_payload_size(test)
    result_train_delta_up_link_payload_size = train_delta_up_link_payload_size(test)


    test_ip65536_delH = pd.concat(
        [result_ip, result_dns, result_port, result_protocol,  result_train_delta_dn_link_data_size, result_train_delta_up_link_data_size, result_train_lastdnstimetolive, result_train_synackmsssize,
         result_train_delta_dn_link_payload_size, result_train_delta_up_link_payload_size], axis=1)


    return test_ip65536_delH, modelupdate, max_length