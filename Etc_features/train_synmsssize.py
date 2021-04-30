import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def train_synmsssize(data):
    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(data[['TCP.Delta.MSS.SynMssSize']].astype('float64'))
    output = min_max_scaler.transform(data[['TCP.Delta.MSS.SynMssSize']].astype('float64'))
    output = pd.DataFrame(output)
    output.columns = ['synmsssize']

    return pd.DataFrame(output)