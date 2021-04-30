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

warnings.filterwarnings('ignore')

path = "./"
file_list = os.listdir(path)

# slice-id 와 type에 맞는 model. 
model = load_model(str(sys.argv[1])+'-'+str(sys.argv[2])+'.h5')



if str(sys.argv[1])+'-'+str(sys.argv[2])+'.p' in file_list:
    with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'rb') as f:
        data = pickle.load(f)
    
    if len(data) < 143:
        data.append(float(sys.argv[4]))
        
        with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'wb') as f:
            pickle.dump(data, f)
        
        print('0x64'+str(800), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sep= ',')
        


    else:
        # 여기가 진짜 실행하는 코드
        
        data = list(map(float, data))
        prediction = np.concatenate([pd.DataFrame(data).values, np.reshape(float(sys.argv[4]),(1,1))])
                    
        result = model.predict(np.reshape(prediction,(1,144,1))/1e7)*1e7
          
        a = datetime.strptime(sys.argv[3], "%Y-%m-%dT%H:%M:%S")
        
        
        data.append(float(sys.argv[4]))
        
        
        with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'wb') as f:
            pickle.dump(data[1:], f)

        print('status','slice-id','type', 'time', 'value', sep = ',')
        for i in range(12):
            if result[0][i] > 0:
                print('0x64'+str(200), sys.argv[1], sys.argv[2],str((a+timedelta(minutes=5*(i+1))).date()) + 'T'+ str((a+timedelta(minutes=5*(i+1))).time()), result[0][i], sep= ',')
            else:
                print('0x64'+str(500), sys.argv[1], sys.argv[2],str((a+timedelta(minutes=5*(i+1))).date()) + 'T'+ str((a+timedelta(minutes=5*(i+1))).time()), result[0][i], sep= ',')

                
else:
    
    with open(str(sys.argv[1])+'-'+str(sys.argv[2])+'.p', 'wb') as f:
        pickle.dump(list(sys.argv[4]), f)
             
    print('0x64'+str(800), sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sep= ',')

    
    
    

   