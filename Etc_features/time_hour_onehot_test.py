import pandas as pd
import os
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from datetime import date, timedelta



def time_hour_onehot(data):
    hour = data[['SummaryCreateTime']].copy()
    hour['SummaryCreateTime'] = hour['SummaryCreateTime'].apply(
        lambda x: datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H-%M'))
    hour = pd.DataFrame(hour['SummaryCreateTime'].apply(lambda x: x[11:13]))
    hour = pd.get_dummies(hour.SummaryCreateTime, prefix='hour_')

    if hour.shape[1] < 24:
        hour_list = ['hour__00', 'hour__01', 'hour__02', 'hour__03', 'hour__04', 'hour__05', 'hour__06', 'hour__07',
                     'hour__08', 'hour__09', 'hour__10', 'hour__11',
                     'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16', 'hour__17', 'hour__18', 'hour__19',
                     'hour__20', 'hour__21', 'hour__22', 'hour__23']
        a = list(set(hour_list) - set(hour.columns))
        for i in range(len(a)):
            hour[a[i]] = 0

    hour = hour.sort_index(axis=1)
    return hour.reset_index(drop=True)

def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d

def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())
    """
    extra_cols = set(d.columns) - set(columns)

    if extra_cols:
        print("extra columns:", extra_cols)
    """
    d = d[columns]
    return d


def time_hour_onehot_test(data, testdata):
    columns = list(time_hour_onehot(data).columns)

    return fix_columns(time_hour_onehot(testdata), columns)