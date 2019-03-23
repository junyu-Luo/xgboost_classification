import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
# from get_lasso_data import get_data
from get_all_data import get_data


train_X, test_X, train_y, test_y = get_data(ues_smote=True)
# print('train_X:', train_X.shape, train_X.dtype)
# print('train_y:', train_y.shape, train_y.dtype)
# print('X_test:', test_X.shape)
# print('y_test:', test_y.shape)

def modelfit(alg, train_X, train_y, test_X, test_y, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(train_X, label=train_y)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                          metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)

        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators=n_estimators)
        print(cvresult)
    # Fit the algorithm on the data
    alg.fit(train_X, train_y, eval_metric='mlogloss')

    # Predict training set:
    train_predprob = alg.predict_proba(train_X)
    logloss = metrics.log_loss(train_y, train_predprob)

    # Print model report:
    print("logloss of train :%.4f" % logloss)
    y_pred = np.array(alg.predict(test_X))
    predictions = [round(value) for value in y_pred]
    print('AUC: %.4f' % metrics.roc_auc_score(test_y, y_pred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y, predictions))
    print('Recall: %.4f' % metrics.recall_score(test_y, predictions))
    print('Precesion: %.4f' % metrics.precision_score(test_y, predictions))
    print('F1-score: %.4f' % metrics.f1_score(test_y, predictions))
    print('test_y',test_y)
    print('y_pred',y_pred)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

xgb_params = XGBClassifier(
    alpha=1,  # L1正则化系数，默认为1
    seed=4, # 随机种子 复现
    scale_pos_weight=1,  # 正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时scale_pos_weight=10。
    num_class=2,
    nthread=-1,# nthread=-1时，使用全部CPU进行并行运算（默认）nthread=1时，使用1个CPU进行运算。
    silent=1,  # silent=0时，不输出中间过程（默认）silent=1时，输出中间过程

    subsample=0.8,  # 使用的数据占全部训练集的比例。防止overfitting。默认值为1，典型值为0.5-1。
    colsample_bytree=0.8, # 使用的特征占全部特征的比例。防止overfitting。默认值为1，典型值为0.5-1。
    colsample_bylevel=0.7,

    learning_rate=0.01, # 学习率，控制每次迭代更新权重时的步长，值越小，训练越慢。默认0.3，典型值为0.01-0.2。
    n_estimators=1000000,  # 总共迭代的次数，即决策树的个数，数值大没关系，cv会自动返回合适的n_estimators
    max_depth=6,  # 树的深度，默认值为6，典型值3-10。
    min_child_weight=1,  # 值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。默认值为1
    gamma=0, # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
    objective='multi:softprob',

)


modelfit(xgb_params,train_X,train_y,test_X,test_y,cv_folds=kfold)




