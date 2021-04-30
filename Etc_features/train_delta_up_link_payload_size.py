import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def train_delta_up_link_payload_size(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['Traffic.Delta.Uplink.RetransPayloadSize']].astype('float64'))
    output = min_max_scaler.transform(data[['Traffic.Delta.Uplink.RetransPayloadSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['delta_up_link_payload_size']


    return pd.DataFrame(output)