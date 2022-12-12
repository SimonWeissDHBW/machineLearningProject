import sklearn
import numpy
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from pandas import read_csv 
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns


LOOP_COUNT = 1

def load_data():
    data_frame = read_csv('breast-cancer.csv')

    data = data_frame.drop(labels = ['diagnosis'], axis = 1)
    target = data_frame['diagnosis']

    return data_frame, data, target

def find_best_params(data, target):
    grid_params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.1, 0.5 , 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
    # grid_params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.1,1,5,10,50,100,500,1000], 'gamma':[1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]}
    clf = GridSearchCV(svm.SVC(),grid_params, n_jobs=-1)

    data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2, shuffle=True)
    clf.fit(data_train, target_train)

    return clf.best_params_

def run_svm(data, target, default_acc = numpy.array([0]), dropped_column = None, return_mean = False, multiplier = 1, params = None):
    acc_array = numpy.empty(LOOP_COUNT * multiplier)
    for count, value in enumerate(acc_array):
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2, shuffle=True)

        clf = svm.SVC(C = params['C'], kernel = params['kernel'], cache_size = 8000)
        # clf = svm.SVC(kernel = params['kernel'], C = params['C'], gamma = params['gamma'])

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

if(__name__ == "__main__"):
    print("Starting")

    data_frame, data, target = load_data()
    data = data.drop(labels = "id", axis = 1)

    sns.pairplot(data_frame, hue="diagnosis", vars=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"])
    plt.show()

    feature_array = data.columns
    acc_array = numpy.empty(LOOP_COUNT)
    feature_arr = []
    drop_columns = []

    best_params = find_best_params(data, target)
    print(best_params)
        
    default_acc = run_svm(data, target, return_mean = True, multiplier = 10, params = best_params)

    print(default_acc)

    for counter, column in enumerate(data):
        feature_arr.append((data.drop(labels = [column], axis = 1), target, default_acc, column, False, 1, best_params))

    with Pool() as p:
        drop_columns.append(p.starmap(run_svm, feature_arr))
        drop_columns = drop_columns[0]

    drop_columns = [x for x in drop_columns if x is not None]

    print(drop_columns)

    final_data = data.drop(labels = drop_columns, axis = 1)

    print(run_svm(final_data, target, return_mean = True, multiplier = 10, params = best_params))

    print("Done")