import sklearn
import pandas
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv


data_frame = read_csv('breast-cancer.csv')

data = data_frame.drop(['diagnosis'], 1)
target = data_frame['diagnosis']

data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2)

classes = ['m' 'b']

clf = svm.SVC()
clf.fit(data_train, target_train)

target_pred = clf.predict(data_test)

acc = metrics.accuracy_score(target_test, target_pred)

print(acc)