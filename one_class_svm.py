import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from get_all_data import shuffled,array_joint
from sklearn.model_selection import train_test_split

label_1 = pd.read_excel('./data/all_1.xlsx')
label_0 = pd.read_excel('./data/all_0.xlsx')

y_1 = np.array(label_1['target'])
X_1 = np.array(label_1.drop(["target"], axis=1))

y_0 = np.array(label_0['target'])
X_0 = np.array(label_0.drop(["target"], axis=1))


dataX_1, testX_1, datay_1, testy_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=0)
dataX_0, testX_0, datay_0, testy_0 = train_test_split(X_0, y_0, test_size=0.067, random_state=0)
test_X, test_y = shuffled(array_joint(testX_1, testX_0), array_joint(testy_1, testy_0),random_seed=0)





# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
clf.fit(dataX_1)
# y_pred_test = clf.predict(test_X)
print(len(clf.predict(dataX_1)),clf.predict(dataX_1))
print()
print(len(clf.predict(dataX_1)),clf.predict(dataX_1))
print()
print(len(clf.predict(test_X)),clf.predict(test_X))
print()
print(len(test_y),test_y)

