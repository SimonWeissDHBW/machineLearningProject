import sklearn
import numpy
from sklearn import svm, metrics
                                         
CACHE_SIZE = 12000

def run_svm(data, target, default_acc = numpy.array([0]), dropped_column = None, return_mean = False, repeats = 1, params = None):

    acc_array = numpy.empty(repeats)
    for count, value in enumerate(acc_array):
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.25, shuffle=True)

        clf = svm.SVC(C = params['C'], kernel = params['kernel'], cache_size = CACHE_SIZE)
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