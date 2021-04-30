import pandas as pd
import os
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from datetime import date, timedelta


def port_onehot(data):
    port = data[['App_Port']].copy()
    port_list = port['App_Port'].value_counts(sort=True, ascending=False)

    if len(port_list) < 10:
        port = pd.get_dummies(port.App_Port, prefix='port_')
        port['port_x'] = 0

    else:
        port_list = port_list[:10]

        a = port[port['App_Port'].isin(port_list.index)]
        b = port[~port['App_Port'].isin(port_list.index)]
        b['App_Port'] = 'x'
        port = pd.concat([a, b]).sort_index()
        port = pd.get_dummies(port.App_Port, prefix='port_')

    port = port.sort_index(axis=1)
    return port.reset_index(drop=True)