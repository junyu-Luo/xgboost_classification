'''
logloss of train :0.0412
AUC: 0.7042
ACC: 0.7600
Recall: 0.4250
Precesion: 0.9444
F1-score: 0.5862
test_y [1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1
 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 1 1 1 0 1 1 0 0 0 0 1 0 1 0 1 1 0 0
 0 0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 0 0]
y_pred [0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0]
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from get_all_data import get_data
from xgboost import plot_importance
from matplotlib import pyplot as plt

bst2 = xgb.Booster(model_file='./model/xgb.model')

train_X, test_X, train_y, test_y = get_data(ues_smote=True)

# print(test_X.shape)
# random = np.random.random((1,135))
# print(random)
dtest = xgb.DMatrix(test_X) #np.array([[0]*135])

y_pred = bst2.predict(dtest)

# print('test_y', test_y,len(test_y))
# print('dtest', dtest.num_col())
# print('y_pred',y_pred)

pred_list = []
for pred in y_pred:
    pred_list.append(pred[1])

pred = np.array(pred_list)
print(pred)