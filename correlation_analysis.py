# -*- coding: utf-8 -*-
import sklearn
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
from pylab import mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from xgboost import plot_importance
import matplotlib.pyplot as plt
import uuid
from get_all_data import get_data

X_train, X_test, y_train, y_test = get_data(ues_smote=True)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
useTrainCV = True
cv_folds = None
early_stopping_rounds = 50

xgb1 = XGBClassifier(
                    alpha=1,  # L1正则化系数，默认为1
                    seed=4,  # 随机种子 复现
                    scale_pos_weight=1,  # 正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时scale_pos_weight=10。
                    num_class=2,
                    nthread=-1,  # nthread=-1时，使用全部CPU进行并行运算（默认）nthread=1时，使用1个CPU进行运算。
                    silent=1,  # silent=0时，不输出中间过程（默认）silent=1时，输出中间过程

                    subsample=0.8,  # 使用的数据占全部训练集的比例。防止overfitting。默认值为1，典型值为0.5-1。
                    colsample_bytree=0.8,  # 使用的特征占全部特征的比例。防止overfitting。默认值为1，典型值为0.5-1。
                    colsample_bylevel=0.7,

                    learning_rate=0.01,  # 学习率，控制每次迭代更新权重时的步长，值越小，训练越慢。默认0.3，典型值为0.01-0.2。
                    n_estimators=1000000,  # 总共迭代的次数，即决策树的个数，数值大没关系，cv会自动返回合适的n_estimators
                    max_depth=5,  # 树的深度，默认值为6，典型值3-10。
                    min_child_weight=2,  # 值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。默认值为1
                    gamma=0,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                    objective='multi:softprob',
                    )

if useTrainCV:
    xgb_param = xgb1.get_xgb_params()

    xgtrain = xgb.DMatrix(X_train, label=y_train)

    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], folds=cv_folds,
                      metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)

    n_estimators = cvresult.shape[0]
    xgb1.set_params(n_estimators=n_estimators)
    # print(cvresult)
# Fit the algorithm on the data
xgb1.fit(X_train, y_train, eval_metric='mlogloss')
# Predict training set:
train_predprob = xgb1.predict_proba(X_train)
logloss = metrics.log_loss(y_train, train_predprob)

# Predict training set:
print("logloss of train :%.4f" % logloss)
y_pred = np.array(xgb1.predict(X_test))
predictions = [round(value) for value in y_pred]
print('AUC: %.4f' % metrics.roc_auc_score(y_test, y_pred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, predictions))
print('Recall: %.4f' % metrics.recall_score(y_test, predictions))
print('Precesion: %.4f' % metrics.precision_score(y_test, predictions))
print('F1-score: %.4f' % metrics.f1_score(y_test, predictions))
print('test_y',y_test)
print('y_pred',y_pred)
def get_score(self, fmap='', importance_type='weight'):
    """Get feature importance of each feature.
    Importance type can be defined as:
        'weight' - the number of times a feature is used to split the data across all trees.
        'gain' - the average gain of the feature when it is used in trees
        'cover' - the average coverage of the feature when it is used in trees

    Parameters
    ----------
    fmap: str (optional)
       The name of feature map file
    """

    if importance_type not in ['weight', 'gain', 'cover']:
        msg = "importance_type mismatch, got '{}', expected 'weight', 'gain', or 'cover'"
        raise ValueError(msg.format(importance_type))

    # if it's weight, then omap stores the number of missing values
    if importance_type == 'weight':
        # do a simpler tree dump to save time
        trees = self.get_dump(fmap, with_stats=False)

        fmap = {}
        for tree in trees:
            for line in tree.split('\n'):
                # look for the opening square bracket
                arr = line.split('[')
                # if no opening bracket (leaf node), ignore this line
                if len(arr) == 1:
                    continue

                # extract feature name from string between []
                fid = arr[1].split(']')[0].split('<')[0]

                if fid not in fmap:
                    # if the feature hasn't been seen yet
                    fmap[fid] = 1
                else:
                    fmap[fid] += 1

        return fmap

    else:
        trees = self.get_dump(fmap, with_stats=True)

        importance_type += '='
        fmap = {}
        gmap = {}
        for tree in trees:
            for line in tree.split('\n'):
                # look for the opening square bracket
                arr = line.split('[')
                # if no opening bracket (leaf node), ignore this line
                if len(arr) == 1:
                    continue

                # look for the closing bracket, extract only info within that bracket
                fid = arr[1].split(']')

                # extract gain or cover from string after closing bracket
                g = float(fid[1].split(importance_type)[1].split(',')[0])

                # extract feature name from string before closing bracket
                fid = fid[0].split('<')[0]

                if fid not in fmap:
                    # if the feature hasn't been seen yet
                    fmap[fid] = 1
                    gmap[fid] = g
                else:
                    fmap[fid] += 1
                    gmap[fid] += g

        # calculate average value (gain/cover) for each feature
        for fid in gmap:
            gmap[fid] = gmap[fid] / fmap[fid]

        return gmap

plot_importance(xgb1)

dic = (xgb1.get_booster().get_score(importance_type='weight'))
print(len(dic),dic)

def get_dic():
    data = pd.read_excel('./data/all_0.xlsx')
    columns = [column for column in data]
    columns.remove('target')

    dic = {}
    for i in range(len(columns)):
        dic['f'+str(i)] = columns[i]
    return dic

conv = get_dic()


lis_x = []
list_y = []
for key,val in dic.items():
    lis_x.append(key)
    list_y.append(val)
list_x = []
for x in lis_x:
    for key,val in conv.items():
        if x == key:
            list_x.append(val)

# mpl.rcParams['font.sans-serif'] = ['FangSong']    # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False        # 解决保存图像是负号'-'显示为方块的问题　
# fig = plt.figure()
# plt.bar(lis_x, lis_y, 0.4, color="green")
# plt.xlabel("特征")
# plt.ylabel("分数")
# plt.title("xgboost特征重要程度")
# plt.show()

data = pd.read_excel("./data/nan.xlsx")
for i in range(len(lis_x)):
    data.loc[i, "特征"] = list_x[i]
    data.loc[i, "分数"] = list_y[i]
# writer = pd.ExcelWriter("../out/corr"+str(uuid.uuid1())+".xlsx")
data.to_excel('out.xlsx')
# writer.save()

