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


def make_fasttext_model_vector(word2vec_train_data, vector_size=5):
    model_gensim = FT_gensim(size=vector_size,  seed=1234)
    model_gensim.build_vocab(word2vec_train_data)
    model_gensim.train(word2vec_train_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)

    length = []
    for i in range(len(word2vec_train_data)):
        length.append(len(word2vec_train_data[i]))

    dns = []
    for i in range(len(word2vec_train_data)):
        dns.append(model_gensim.wv[word2vec_train_data[i]].reshape(-1))

    ##########
    if length == []:

        return pd.DataFrame(columns = [i for i in range(max_length*ve)]), model_gensim

    else:

        empty = pd.DataFrame(columns=[i for i in range(vector_size * max(length))])
    ########

    for i in range(len(dns)):
        if vector_size * max(length) > len(dns[i]):

            # dns 길이가 안 맞을때는 0을 뒤에다 채워넣는방식으로.
            empty.loc[i] = np.concatenate((dns[i], np.zeros(vector_size * max(length) - len(dns[i]))))

        else:
            empty.loc[i] = dns[i]

    return empty, model_gensim, max(length)