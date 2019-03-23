import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from kmeans_smote import KMeansSMOTE

def array_joint(array_1, array_2):
    return np.concatenate((array_1, array_2), axis=0)


def shuffled(X, y,random_seed):
    assert len(X) == len(y)
    np.random.seed(random_seed)
    p = np.random.permutation(len(X))
    return X[p], y[p]




# smote=True 使用上采样填充数据
def split_dataset(X_1, y_1, X_0, y_0, test_size=0.2, random_seed=0,smote=True):
    if smote:
        dataX_1, testX_1, datay_1, testy_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=random_seed)
        dataX_0, testX_0, datay_0, testy_0 = train_test_split(X_0, y_0, test_size=0.067, random_state=random_seed)
        test_X, test_y = shuffled(array_joint(testX_1, testX_0), array_joint(testy_1, testy_0),random_seed=random_seed)
        # print(len(testy_1), len(testy_0), len(testy_1) + len(testy_0))
        addX, addy = shuffled(array_joint(dataX_1, dataX_0), array_joint(datay_1, datay_0),random_seed=random_seed)
        kmeans_smote = KMeansSMOTE(
            kmeans_args={
                'n_clusters': 2
            },
            smote_args={
                'k_neighbors': 2
            },
            random_state=random_seed
        )
        X_resampled, y_resampled = kmeans_smote.fit_sample(addX, addy)
        # datay_1 = np.ones(len(smote_dataX_1), dtype=np.int16)
        train_X, train_y = shuffled(X_resampled, y_resampled,random_seed=random_seed)
    else:
        dataX_1, testX_1, datay_1, testy_1 = train_test_split(X_1, y_1, test_size=0.21, random_state=random_seed)
        dataX_0, testX_0, datay_0, testy_0 = train_test_split(X_0, y_0, test_size=0.069, random_state=random_seed)
        test_X, test_y = shuffled(array_joint(testX_1, testX_0), array_joint(testy_1, testy_0),random_seed=random_seed)
        train_X, train_y = shuffled(array_joint(dataX_1, dataX_0), array_joint(datay_1, datay_0),random_seed=random_seed)

    return train_X,test_X,train_y,test_y


def get_data(ues_smote):
    factor_1 = ['PCT_PRE_FEV1_V1', 'SIX_MWD', 'PCT_PRE_FEF2575_V1', 'IGE_RESULT', 'VCAM1', 'LPA', 'CXCL9', 'CXCL10',
                'NPPB_PH', 'CDH1', 'CCL13', 'TNFRSF1A', 'IL18', 'CCL4', 'FTL_FTH1', 'CCL24', 'CHGA', 'IL1RN', 'F7',
                'CCL22']

    factor_2 = ['PCT_PRE_FEV1_V1', 'SIX_MWD', 'VCAM1', 'SERPINA3', 'CCL3', 'IGE_RESULT', 'NPPB_PH', 'CXCL9', 'LPA',
                'CCL20', 'IL2RA', 'CDH1', 'TNFRSF10C', 'CCL4', 'CCL13', 'TNFRSF1A', 'CCL24', 'IL1RN', 'CHGA', 'IL18',
                'CCL22', 'FTL_FTH1', 'KITLG', 'CCL18', 'ICAM1']

    factor_3 = ['PCT_PRE_FEV1_V1', 'SIX_MWD', 'PCT_PRE_FEF2575_V1', 'IL16', 'IGE_RESULT', 'CCL3', 'CXCL10', 'IGE', 'LPA', 'CDH1', 'TNFRSF1A', 'CXCL9', 'CCL24', 'CHGA', 'FTL_FTH1', 'CCL11', 'CCL4', 'CCL22', 'F7']

    factor_all = list(set(factor_1 + factor_2))

    label_1 = pd.read_excel('./data/all_1.xlsx')
    label_0 = pd.read_excel('./data/all_0.xlsx')

    y_1 = np.array(label_1['target'])
    X_1 = np.array(label_1[factor_3])
    # X_1 = np.array(label_1.drop(["target"], axis=1))
    # print(X_1)

    y_0 = np.array(label_0['target'])
    X_0 = np.array(label_0[factor_3])
    # X_0 = np.array(label_0.drop(["target"], axis=1))

    train_X, test_X, train_y, test_y = split_dataset(X_1, y_1, X_0, y_0,smote=ues_smote)

    return train_X, test_X, train_y, test_y

if __name__ == '__main__':
    train_X, test_X, train_y, test_y = get_data(ues_smote=True)
    print('train_X:', train_X.shape, train_X.dtype)
    print('train_y:', train_y.shape, train_y.dtype)
    print('X_test:', test_X.shape)
    print('y_test:', test_y.shape)



