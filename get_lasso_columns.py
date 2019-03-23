from get_all_data import get_data
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_excel('./data/dataset.xlsx')
# train_y = np.array(df['target'])
# train_X = np.array(df.drop(["target"], axis=1))
columns = [column for column in df]
columns.remove('target')
train_X, test_X, train_y, test_y = get_train_test(ues_smote=True)

model_lasso = linear_model.LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(train_X, train_y)  # 此处 alpha 为通常值 #fit 把数据套进模型里跑

sorce = list(model_lasso.coef_)
feaute = list(columns)

# print(len(sorce),len(feaute))
lis = list(zip(sorce,feaute))



sorted_dict = sorted(lis, key=lambda x:x[0], reverse=False)
print(sorted_dict)
key,val = zip(*sorted_dict)
# print(len(val[:10]+val[-10:]),val[:10]+val[-10:])

factor = list(val[:12]+val[-13:])
print(factor)