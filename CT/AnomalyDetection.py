import os
import sys
import time, datetime
from datetime import timedelta
import pandas as pd
import os
import numpy as np
import sys
import time
import pickle
import warnings
from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Conv1D, GlobalMaxPool1D, SimpleRNN
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import rrcf

warnings.filterwarnings('ignore')

path = "./"
file_list = os.listdir(path)



if str(sys.argv[1])+'-'+str(sys.argv[2])+'.p' in file_list:
    with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'rb') as f:
        data = pickle.load(f)
    
    if len(data) < 287:
        data.append(float(sys.argv[4]))
        
        with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'wb') as f:
            pickle.dump(data, f)
        
        print('0x64'+str(800), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sep= ',')
        


    else:
        # 여기가 진짜 실행하는 코드
        
        data = list(map(float, data))
        xx = pd.DataFrame(data)
        xx.columns = ['data']
        
        train = np.concatenate([xx.data.values, np.array([float(sys.argv[4])])])
                    
        num_trees = 64
        shingle_size = 144
        tree_size = 256


        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)

        points = rrcf.shingle(train, size=shingle_size)


        avg_codisp = {}


        for index, point in enumerate(points):

            for tree in forest:
        
                if len(tree.leaves) > tree_size:
                    tree.forget_point(index - tree_size)  
                tree.insert_point(point, index=index)
                if not index in avg_codisp:
                    avg_codisp[index] = 0
                avg_codisp[index] += tree.codisp(index) / num_trees
        
        avg_codisp_list = pd.Series([v for v in avg_codisp.values()])

        avg_codisp_list = ((avg_codisp_list - avg_codisp_list.min())
                      / (avg_codisp_list.max() - avg_codisp_list.min()))
        
        
              
        a = datetime.strptime(sys.argv[3], "%Y-%m-%dT%H:%M:%S")
        
        
        data.append(float(sys.argv[4]))
        
        
        with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'wb') as f:
            pickle.dump(data[1:], f)

        print('status','slice-id','type', 'time', 'value', sep = ',')

        if 1 > avg_codisp_list.values[-1] > 0:
            print('0x64'+str(200), sys.argv[1], sys.argv[2],sys.argv[3], avg_codisp_list.values[-1], sep= ',')
        else:
            print('0x64'+str(500), sys.argv[1], sys.argv[2],sys.argv[3], avg_codisp_list.values[-1], sep= ',')

                
else:
    
    with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'wb') as f:
        pickle.dump(list(sys.argv[4]), f)
             
    print('0x64'+str(800), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sep= ',')

    
    
    

   