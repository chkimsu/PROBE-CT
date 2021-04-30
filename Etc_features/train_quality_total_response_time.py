import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def train_quality_total_response_time(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['HTTP.Delta.Quility.TotResponseTime']].astype('float64'))
    output = min_max_scaler.transform(data[['HTTP.Delta.Quility.TotResponseTime']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['quality_total_response_time']


    return pd.DataFrame(output)
