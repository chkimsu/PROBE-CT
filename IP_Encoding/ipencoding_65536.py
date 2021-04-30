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


## transport, http에서는 ip 없는것 없다. 모두 ip 있다. 변수명 떼주고, 모두 65536으로 나눠주는 함수.
def ipencoding_65536(data):
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
        split[split.columns[i]] = split[split.columns[i]] / 65535

    split.columns = ['Aclass', 'Bclass', 'Cclass', 'Dclass', 'Eclass', 'Fclass', 'Gclass', 'Hclass']

    return split.reset_index(drop=True)