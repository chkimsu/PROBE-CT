import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def train_delta_up_link_data_size(data):

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Uplink.DataSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Uplink.DataSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_up_link_data_size']


    return pd.DataFrame(output)