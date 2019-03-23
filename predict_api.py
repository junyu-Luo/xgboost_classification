'''
logloss of train :0.0425
AUC: 0.9719
ACC: 0.9727
Recall: 0.9831
Precesion: 0.9667
F1-score: 0.9748
'''

import numpy as np
import xgboost as xgb

def copd_predict(seq_list):
    '''
    输入：一个25维有序list
            ["AGE","DAY","过去1年急性加重次数","过去1年内门诊加重次数","过去一年内住院加重次数","高血压病","前列腺增生","SABA+SAMA","全身用糖皮质激素","β-内酰胺类抗菌药","调脂类药","解热镇痛抗炎药","嗜酸性粒细胞数","痰嗜酸性粒细胞比例","中性粒细胞数","血红蛋白","白细胞","二氧化碳分压（测定）","肺泡动脉氧分压差","血沉","氧分压（测定）","肌酐","嗜酸性粒细胞比率","D二聚体(ELISA法)","尿素氮"]
    输出：pred_probability 再入院的概率
          pred_vale 是否再入院
     '''
    gy_model = xgb.Booster(model_file='./model/gy_xgb.model')
    list2Matrix = xgb.DMatrix(np.array([seq_list]))
    y_pred = gy_model.predict(list2Matrix)
    pred_list = []
    for pred in y_pred:
        pred_list.append(pred[1])
    pred_probability = np.array(pred_list)
    pred_vale = np.array(pred_list) > 0.5
    return pred_probability,pred_vale


if __name__ == '__main__':
    input_yes = [71, 5, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0.03, 1.5, 6.1, 131, 5.1, 43.3, 26.6, 100, 87.8, 97, 0.4, 474, 5]
    input_no = [73, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.23, 1.5, 3.2, 112, 2.1, 46.5, 13.7, 32, 82.7, 107, 4.2, 897, 5.7]
    probability,vale = copd_predict(input_yes)
    print('probability',probability)
    print('vale',vale)


