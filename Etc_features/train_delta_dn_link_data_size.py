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


def train_delta_dn_link_data_size(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Dnlink.DataSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Dnlink.DataSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_dn_link_data_size']

    return pd.DataFrame(output)