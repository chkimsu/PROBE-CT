import pandas as pd
import os
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from datetime import date, timedelta



def gender_onehot(data):
    gender = data[['Gender']].copy()
    gender.Gender = gender.Gender.astype(str)
    gender.loc[(gender['Gender'].str.isnumeric() == False) | (gender['Gender'].isnull()) | (gender['Gender'] == "0"), "Gender"] = "999"
    gender.loc[(gender['Gender']) == "999", "Gender"] = 'x'
    gender = pd.get_dummies(gender.Gender, prefix='gender_')

    if gender.shape[1] < 3:
        gender_list = ['gender__1', 'gender__2', 'gender__x']
        a = list(set(gender_list) - set(gender.columns))
        for i in range(len(a)):
            gender[a[i]] = 0

    gender = gender.sort_index(axis=1)
    return gender.reset_index(drop=True)




def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    return d

def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())
    d = d[columns]
    return d

def gender_onehot_test(data, testdata):
    columns = list(gender_onehot(data).columns)
    return fix_columns(gender_onehot(testdata), columns)