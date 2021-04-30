import os
import pandas as pd
import gensim
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats
from datetime import datetime
import multiprocessing
from multiprocessing import Pool

def parallelize_dataframe(df, func):
    num_cores = multiprocessing.cpu_count()

    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def prob_app_id(df):
    a = pd.DataFrame(columns=['RECOMMEND_PROB_APP_ID'])
    for i in range(len(df)):
        x = pd.DataFrame(df.iloc[i, :].sort_values(ascending=False)[:10])
        x.columns = ['RECOMMEND_PROB_APP_ID']
        a = a.append(x)
    return a

def prob_app_group(df):
    b = pd.DataFrame(columns=['RECOMMEND_PROB_APP_GROUP'])
    for i in range(len(df)):
        x = pd.DataFrame(df.iloc[i, :].sort_values(ascending=False)[:10])
        x.columns = ['RECOMMEND_PROB_APP_GROUP']
        b = b.append(x)
    return b


def result_appid_1211(appid_model,encoder_id, test_raw, test):
    # ------------------- app id
    # result_csv_1211 부분
    result = pd.DataFrame(appid_model.predict(test))
    result.columns = encoder_id.classes_
    # result_transpose = result.transpose()

    a = parallelize_dataframe(result, prob_app_id)
    a['APP_ID'] = a.index
    a = a.reset_index(drop=True)


    index_list = list(np.arange(0, len(a), 10))
    index_list2 = list(np.arange(1, len(a), 10))

    list_df = a[a.index.isin(index_list)].reset_index(drop=True)
    list2_df = a[a.index.isin(index_list2)].reset_index(drop=True)

    index_diff = pd.DataFrame(list_df['RECOMMEND_PROB_APP_ID'] - list2_df['RECOMMEND_PROB_APP_ID'])
    index_diff.index = index_list

    index_diff['NEW_APP_ID_FLAG'] = 0
    index_diff.loc[index_diff['RECOMMEND_PROB_APP_ID'] < 0.65, "NEW_APP_ID_FLAG"] = 1

    index_diff['NEW_APP_ID_PROB'] = scipy.stats.norm(0, 0.4).pdf(list_df['RECOMMEND_PROB_APP_ID'] - list2_df['RECOMMEND_PROB_APP_ID']) * 100

    a = pd.merge(a[['RECOMMEND_PROB_APP_ID', 'APP_ID']], index_diff[['NEW_APP_ID_FLAG', 'NEW_APP_ID_PROB']], how='left',
                          left_index=True, right_index=True)

    a[['NEW_APP_ID_FLAG', 'NEW_APP_ID_PROB']] = a[['NEW_APP_ID_FLAG', 'NEW_APP_ID_PROB']].fillna(method='pad')


    test_index = test_raw[['SummaryCreateTime','IMSI', 'App_Group_Code', 'App_ID']]
    test_index = pd.DataFrame(test_index.iloc[np.repeat(np.arange(len(test_index)), 10)]).reset_index(drop=True)
    s = pd.Series(np.arange(1, 11))
    s = pd.DataFrame(pd.concat([s] * len(test))).rename(columns={0: 'RANK'}).reset_index(drop=True)
    df_list = [test_index, s, a]
    a = pd.concat(df_list, axis=1)

    a = a[['SummaryCreateTime', 'IMSI', 'RANK', 'APP_ID','RECOMMEND_PROB_APP_ID', 'NEW_APP_ID_FLAG', 'NEW_APP_ID_PROB']]
    a = a.rename(columns = {'SummaryCreateTime':'SUMMARY_CREATE_TIME'})

    # 최종삭제
    #a = a[a['RANK'] == 1].reset_index(drop=True)


    return a

def result_appgroup_1211(appgroup_model, encoder_group, test_raw, test):

    result = pd.DataFrame(appgroup_model.predict(test))
    result.columns = encoder_group.classes_
    #result_transpose = result.transpose()

    b = parallelize_dataframe(result, prob_app_group)
    b['APP_GROUP'] = b.index
    b = b.reset_index(drop=True)

    return b