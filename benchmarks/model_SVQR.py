
from liquidSVM import *
import numpy as np

def model_SVQR(experiment):

    X = experiment['X']
    X_train = experiment['X_train']  # data, numpy array of shape (number of features, number of examples)
    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    X_test = experiment['X_test']
    tau = experiment['tau']
    scale = experiment['scale']
    apply_log = experiment['apply_log']

    # quantile regression using SVMs
    model = qtSVM(X_train,y_train, weights=tau.tolist(),kernel='GAUSS_RBF')

    # compute predictions
    q_train, err = model.test(X_train)
    q_test, err = model.test(X_test)
    q_all, err = model.test(X)

    # inverse scaled data
    if apply_log:
        q_train = np.exp(q_train)
        q_test = np.exp(q_test)
        q_all = np.exp(q_all)
    q_train = q_train * scale
    q_test = q_test * scale
    q_all = q_all * scale

    experiment['model'] = model
    experiment['q_train'] = q_train
    experiment['q_test'] = q_test
    experiment['q_all'] = q_all
    experiment['costs'] = err

    return experiment