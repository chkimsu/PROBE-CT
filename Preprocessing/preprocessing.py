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

## UNKNONW DATA 없애는 함수

def preprocessing(data):
    if 'request_host' in data.columns:
        data = data[data['app_id'] != '0050']

    else:

        data = data[data['app_id'] != '01BB']
        data = data[data['app_id'] != 'C001']

    return data
 

    """
    if 'request_host' in data.columns:
        known = data[data['app_id'] != '0050']
        unknown = data[~data.index.isin(known.index)]

    else:
        known = data[(data['app_id'] != '01BB') & (data['app_id'] != 'C001')]
        unknown = data[~data.index.isin(known.index)]

    return known,  unknown

    """
    




