import pandas as pd
import os
import numpy as np
import time
import pickle
import warnings
from datetime import datetime
from datetime import date, timedelta


def protocol_onehot(data):
    protocol = data[['Protocol']].copy()
    protocol_list = protocol['Protocol'].value_counts(sort=True, ascending=False)

    if len(protocol_list) < 10:
        protocol = pd.get_dummies(protocol.Protocol, prefix='protocol_')
        protocol['protocol_x'] = 0

    else:
        protocol_list = protocol_list[:10]
        a = protocol[protocol['Protocol'].isin(protocol_list.index)]
        b = protocol[~protocol['Protocol'].isin(protocol_list.index)]
        b['Protocol'] = 'x'
        protocol = pd.concat([a, b]).sort_index()
        protocol = pd.get_dummies(protocol.Protocol, prefix='protocol_')

    protocol = protocol.sort_index(axis=1)
    return protocol.reset_index(drop = True)
