import pandas as pd
import os
import gensim
import numpy as np
import time
import pickle
import warnings
from gensim.models.fasttext import FastText as FT_gensim
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


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