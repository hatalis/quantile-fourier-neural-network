"""
@author: Kostas Hatalis
"""
import numpy as np

def split_data(experiment):
    """
    Create features, split, and scale data.

    Arguments:
        experiment(dict): raw_data, percent_training, max_value

    Returns:
        experiment(dict): training/testing X and y, their size, scale, split point
    """

    raw_data = experiment['raw_data']
    percent_training = experiment['percent_training']
    max_value = experiment['max_value']
    apply_log = experiment['apply_log']

    # split data
    split_point = int(len(raw_data) * percent_training)
    y_train, y_test = raw_data[0:split_point].values, raw_data[split_point:].values

    # scale y to [0,max_value] if needed
    maxy, scale = np.max(y_train), 1
    if maxy > max_value:
        scale = (1/max_value)*maxy
        y_train = y_train / scale

    # apply log if needed
    if apply_log:
        y_train = np.log(y_train)

    # create X time features
    N_test = len(y_test)
    N_train = len(y_train)
    X_train = np.array(range(0, N_train))/N_train
    X_test = np.array(range(N_train,N_train+N_test))/N_test
    X_train = X_train.reshape((N_train, 1))
    X_test = X_test.reshape((N_test, 1))

    # whole X for plotting train and test predictions
    N = len(raw_data)
    X = np.array(range(0, N))/N
    X = X.reshape((N, 1))

    experiment['split_point'] = split_point
    experiment['scale'] = scale
    experiment['N_train'] = N_train
    experiment['X_train'] = X_train
    experiment['y_train'] = y_train
    experiment['N_test'] = N_test
    experiment['X_test'] = X_test
    experiment['y_test'] = y_test
    experiment['X'] = X
    experiment['N'] = N

    return experiment