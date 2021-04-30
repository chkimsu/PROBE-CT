import pandas as pd
import os
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from datetime import date, timedelta

def age_onehot(data):
    age = data[['Age']].copy()
    age.Age = age.Age.astype(str)
    age.loc[(age['Age'].str.isnumeric() == False) | (age['Age'].isnull()), "Age"] = "999"
    age.Age = age.Age.astype(int)

    age.loc[(age['Age']) < 10, "age_range"] = '10살이하'
    age.loc[(age['Age'] >= 10) & (age['Age'] < 20), "age_range"] = '10대'
    age.loc[(age['Age'] >= 20) & (age['Age'] < 30), "age_range"] = '20대'
    age.loc[(age['Age'] >= 30) & (age['Age'] < 40), "age_range"] = '30대'
    age.loc[(age['Age'] >= 40) & (age['Age'] < 50), "age_range"] = '40대'
    age.loc[(age['Age'] >= 50) & (age['Age'] < 60), "age_range"] = '50대'
    age.loc[(age['Age'] >= 60) & (age['Age'] < 70), "age_range"] = '60대'
    age.loc[(age['Age'] >= 70) & (age['Age'] < 999), "age_range"] = '70대이상'
    age.loc[(age['Age']) == 999, "age_range"] = 'x'
    age = age.drop(['Age'], axis=1)
    age = pd.get_dummies(age.age_range, prefix='age_')

    if age.shape[1] < 9:
        age_list = ['age__10살이하', 'age__10대', 'age__20대', 'age__30대', 'age__40대', 'age__50대', 'age__60대', 'age__70대이상',
                    'age__x']
        a = list(set(age_list) - set(age.columns))
        for i in range(len(a)):
            age[a[i]] = 0
    age = age.sort_index(axis=1)
    return age.reset_index(drop=True)