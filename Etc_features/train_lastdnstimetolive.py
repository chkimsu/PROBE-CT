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
from sklearn.preprocessing import MinMaxScaler



def train_lastdnstimetolive(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['DNS.LastDnsTimeToLive']].astype('float64'))
    output = min_max_scaler.transform(data[['DNS.LastDnsTimeToLive']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['lastdnstimetolive']


    return pd.DataFrame(output)