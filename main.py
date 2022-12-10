import sklearn
import pandas
import numpy
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv 
from multiprocessing import Pool

LOOP_COUNT = 10

def load_data():
    data_frame = read_csv('breast-cancer.csv')

    data = data_frame.drop(labels = ['diagnosis'], axis = 1)
    target = data_frame['diagnosis']

    return data_frame, data, target

def run_svm(data, target, default_acc = numpy.array([0]), dropped_column = None, return_mean = False, multiplier = 1):
    acc_array = numpy.empty(LOOP_COUNT * multiplier)
    for count, value in enumerate(acc_array):
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2, shuffle=True)
        
        classes = ['m' 'b']

        clf = svm.SVC(kernel="linear", C=2)

        clf.fit(data_train, target_train)

        target_pred = clf.predict(data_test)

        acc_array[count] = metrics.accuracy_score(target_test, target_pred)

    if (return_mean):
        return acc_array.mean()
    else:
        if (acc_array.mean() >= default_acc.mean()):
            return None
        else:
            return dropped_column


data_frame, data, target = load_data()
data = data.drop(labels = "id", axis = 1)


feature_array = data.columns
acc_array = numpy.empty(LOOP_COUNT)
feature_arr = []
drop_columns = []
    
default_acc = run_svm(data, target, return_mean = True, multiplier = 10)

print(default_acc)

for counter, column in enumerate(data):
    feature_arr.append((data.drop(labels = [column], axis = 1), target, default_acc, column, False))

with Pool() as p:
    drop_columns.append(p.starmap(run_svm, feature_arr))
    drop_columns = drop_columns[0]

# for data_dropped in feature_arr:
#     drop_columns.append(run_svm(data_dropped[0], data_dropped[1], data_dropped[2], data_dropped[3], data_dropped[4]))

drop_columns = [x for x in drop_columns if x is not None]

print(drop_columns)

final_data = data.drop(labels = drop_columns, axis = 1)

print(run_svm(final_data, target, return_mean = True, multiplier = 10))

print("Done")