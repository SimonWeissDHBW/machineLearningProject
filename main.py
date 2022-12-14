import sklearn
import numpy
import time
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from pandas import read_csv 
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns

LOOP_COUNT = 100
SHOW_PLOT = True
MULTIPLIER = 10
LINE = "---------------------------------------------------"
CACHE_SIZE = 12000

KERNEL_LIST = ['linear', 'poly', 'rbf', 'sigmoid']
C_LIST = [0.1, 0.5 , 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
GAMMA_LIST = ['scale']

def print_pretty(*args):
    for arg in args:
        print(arg)
    print(LINE)

def load_data():
    data_frame = read_csv('breast-cancer.csv')

    data = data_frame.drop(labels = ['diagnosis'], axis = 1)
    target = data_frame['diagnosis']

    return data_frame, data, target

def find_best_params(data, target):
    grid_params = {'kernel': KERNEL_LIST, 'C': C_LIST, 'gamma':['scale']}
    clf = GridSearchCV(svm.SVC(),grid_params, n_jobs=-1)

    data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2, shuffle=True)
    clf.fit(data_train, target_train)

    return clf.best_params_

def plot_data(data_frame):
    if (SHOW_PLOT):
        sns.pairplot(                   # Plotting data of first 5 columns
            data_frame, 
            hue="diagnosis", 
            vars=["radius_mean", 
                "texture_mean", 
                "perimeter_mean", 
                "area_mean", 
                "smoothness_mean"])        
        plt.show()                      # Showing plot 

def run_svm(data, target, default_acc = numpy.array([0]), dropped_column = None, return_mean = False, multiplier = 1, params = None):
    acc_array = numpy.empty(LOOP_COUNT * multiplier)
    for count, value in enumerate(acc_array):
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2, shuffle=True)

        clf = svm.SVC(C = params['C'], kernel = params['kernel'], cache_size = CACHE_SIZE)

        clf.fit(data_train, target_train)

        target_pred = clf.predict(data_test)

        acc_array[count] = metrics.accuracy_score(target_test, target_pred)

    if (return_mean):
        # print(acc_array)
        return acc_array.mean()
    else:
        if (acc_array.mean() >= default_acc.mean()):
            return None
        else:
            return dropped_column

def drop_columns(data, target, default_acc, best_params):

    # Create necessary arrays
    feature_arr = []                                # Array for multiprocessing
    dropped_columns = []                            # Array for dropped columns

    for counter, column in enumerate(data):
        feature_arr.append((data.drop(labels = [column], axis = 1), target, default_acc, column, False, 1, best_params))

    with Pool() as p:
        dropped_columns.append(p.starmap(run_svm, feature_arr))
        dropped_columns = dropped_columns[0]

    dropped_columns = [x for x in dropped_columns if x is not None]

    return dropped_columns

if(__name__ == "__main__"):

    begin = time.time()

    print_pretty("\n\n\nStarting")

    # Getting data
    data_frame, data, target = load_data()          # Getting data from csv file
    data = data.drop(labels = "id", axis = 1)       # Dropping id column, because it is not needed

    # Plot example data from first 5 columns
    plot_data(data_frame)

    best_params = find_best_params(data, target)    # Finding best params for SVM with GridSearchCV (hyperparameter tuning)
    print_pretty(best_params)
        
    default_acc = run_svm(                          # Getting accuracy with all features
        data,
        target,
        return_mean = True, 
        multiplier = MULTIPLIER, 
        params = best_params)

    print_pretty("Accuracy with all features: ", default_acc)

    dropped_columns = drop_columns(
        data, 
        target, 
        default_acc, 
        best_params)

    print_pretty("Dropped Columns: ", dropped_columns)

    final_data = data.drop(labels = dropped_columns, axis = 1)

    final_acc = run_svm(
        final_data, 
        target, 
        return_mean = True, 
        multiplier = MULTIPLIER, 
        params = best_params)

    print_pretty("Accuracy with dropped features: ", final_acc)

    end = time.time()
    print_pretty("Done. Time Taken: ", end - begin)
