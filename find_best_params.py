import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
# from get_lasso_data import get_data
from get_all_data import get_data
from functools import reduce
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn
    import numpy
    import pandas
    import kmeans_smote

train_X, test_X, train_y, test_y = get_data(ues_smote=True)
# print('train_X:', train_X.shape, train_X.dtype)
# print('train_y:', train_y.shape, train_y.dtype)
# print('X_test:', test_X.shape)
# print('y_test:', test_y.shape)

def modelfit(alg, train_X, train_y, test_X, test_y, useTrainCV=True, cv_folds=None, early_stopping_rounds=35):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(train_X, label=train_y)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                          metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)

        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators=n_estimators)
        # print(cvresult)
    # Fit the algorithm on the data
    alg.fit(train_X, train_y, eval_metric='mlogloss')

    # Predict training set:
    # train_predprob = alg.predict_proba(train_X)
    # logloss = metrics.log_loss(train_y, train_predprob)

    # Print model report:
    # print("logloss of train :%.4f" % logloss)
    y_pred = np.array(alg.predict(test_X))
    predictions = [round(value) for value in y_pred]
    AUC = metrics.roc_auc_score(test_y, y_pred)
    ACC = metrics.accuracy_score(test_y, predictions)
    Recall = metrics.recall_score(test_y, predictions)
    Precesion = metrics.precision_score(test_y, predictions)
    F1_score = metrics.f1_score(test_y, predictions)
    print('AUC: %.4f' % AUC)
    print('ACC: %.4f' % ACC)
    print('Recall: %.4f' % Recall)
    print('Precesion: %.4f' % Precesion)
    print('F1_score: %.4f' % F1_score)
    # print('test_y',test_y)
    # print('y_pred',y_pred)
    return AUC,ACC,Recall,Precesion,F1_score


def permutation(lists, code='|'):
    '''输入多个列表组成的列表, 输出其中每个列表所有元素可能的所有排列组合
    code用于分隔每个元素'''

    def myfunc(list_name1, list_name2):
        return [str(i) + code + str(j) for i in list_name1 for j in list_name2]

    return reduce(myfunc, lists)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)


learning_rate_list =[0.01,0.05,0.005]
max_depth_list = [1,2,3,4,5,6,7,8,9,10]
min_child_weight_list = [1,2,3,4,5,6,7,8,9,10]
colsample_bytree_list = [0.7,0.8,0.9]
subsample_list = [0.7,0.8,0.9]


perm = permutation([learning_rate_list, max_depth_list, min_child_weight_list,colsample_bytree_list,subsample_list])
# columns = ['learning_rate', 'max_depth', 'min_child_weight','colsample_bytree','subsample','AUC','ACC','Recall','Precesion','F1_score']
df = pd.read_excel('./data/params.xlsx')
columns = [column for column in df]
for new_loc in columns:
    df[new_loc] = [np.nan] * len(perm)

for i in range(len(perm)):
    print('总共循环次数：',len(perm),'。现在已循环次数：',i)
    seq_list = perm[i].split('|')
    xgb_params = XGBClassifier(
        alpha=1,  # L1正则化系数，默认为1
        seed=0, # 随机种子 复现
        scale_pos_weight=1,  # 正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时scale_pos_weight=10。
        num_class=2,
        nthread=-1,# nthread=-1时，使用全部CPU进行并行运算（默认）nthread=1时，使用1个CPU进行运算。
        silent=1,  # silent=0时，不输出中间过程（默认）silent=1时，输出中间过程
        colsample_bylevel=0.7,
        n_estimators=1000000,  # 总共迭代的次数，即决策树的个数，数值大没关系，cv会自动返回合适的n_estimators
        gamma=0, # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
        objective='multi:softprob',

        learning_rate=seq_list[0],  # 学习率，控制每次迭代更新权重时的步长，值越小，训练越慢。默认0.3，典型值为0.01-0.2。
        max_depth=seq_list[1],  # 树的深度，默认值为6，典型值3-10。
        min_child_weight=seq_list[2],  # 值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。默认值为1
        colsample_bytree=seq_list[3],  # 使用的特征占全部特征的比例。防止overfitting。默认值为1，典型值为0.5-1。
        subsample=seq_list[4],  # 使用的数据占全部训练集的比例。防止overfitting。默认值为1，典型值为0.5-1。
    )

    early_stopping = int(seq_list[5])
    AUC,ACC,Recall,Precesion,F1_score = modelfit(xgb_params,train_X,train_y,test_X,test_y,cv_folds=kfold)
    df['learning_rate'][i] = str(seq_list[0])
    df['max_depth'][i] = str(seq_list[1])
    df['min_child_weight'][i] = str(seq_list[2])
    df['colsample_bytree'][i] = str(seq_list[3])
    df['subsample'][i] = str(seq_list[4])
    df['AUC'][i] = AUC
    df['ACC'][i] = ACC
    df['Recall'][i] = Recall
    df['Precesion'][i] = Precesion
    df['F1_score'][i] = F1_score

df.to_excel('find_param.xlsx')





