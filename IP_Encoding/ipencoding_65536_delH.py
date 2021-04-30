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

