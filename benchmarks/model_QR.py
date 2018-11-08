
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras import regularizers
from keras.models import Sequential
from keras.optimizers import SGD

def model_QR(experiment,method=0,poly=1):
    """
    Keras code for creating the following benchmarks:
        -linear quantile regression (QR)
        -polynomial quantile regression (PQR)
        -quantile regression neural network (QRNN)
    All models use the smooth approximation to pinball cost.

    Arguments:
        experiment (dict) - various experiment parameters
        method (int) - 0 = test QR, 1 = QRNN
        poly (int) - polynomial degree for features
    Returns:
        experiment (dict) - trained model and predicted quantiles
    """

    X = experiment['X']
    y_train = experiment['y_train']
    N_tau = experiment['N_tau']
    tau = experiment['tau']
    alpha = experiment['alpha']
    Lambda = experiment['Lambda']
    eta = experiment['eta']
    epochs = experiment['epochs']
    N_train = experiment['N_train']

    # Create polynomial features
    # X = np.arange(len(X)).reshape((len(X),1))
    X_poly = X
    if poly > 1:
        for p in range(2,poly+1):
            temp = X ** p
            X_poly = np.concatenate((X_poly, temp), axis=1)
    X = X_poly
    X_train = X[:N_train]
    X_test = X[N_train:]

    # -------------------------------------- build the model
    model = Sequential()
    if method == 0: # QR or PQR
        model.add(Dense(N_tau, input_dim=poly, kernel_initializer='uniform'))
    elif method == 1: # QRNN
        hidden_dims = int(N_train * 1)
        model.add(Dense(hidden_dims, input_dim=poly, kernel_regularizer=regularizers.l2(0.001),
                        kernel_initializer='uniform', activation='relu'))
        model.add(Dense(N_tau, kernel_initializer='uniform'))

    # -------------------------------------- compile and fit model
    sgd = SGD(lr=0.1, decay=0, momentum=0.00, nesterov=False)
    model.compile(loss=lambda Y, Q: pinball_loss(tau,Y,Q,alpha), optimizer=sgd)
    history = model.fit(X_train, y_train, epochs=10_000, verbose=0, batch_size=N_train)

    # -------------------------------------- estimate quantiles of testing data

    # compute predictions
    scale = experiment['scale']
    apply_log = experiment['apply_log']

    # model.load_weights('nd_weights.hdf5')
    q_train = model.predict(X_train)
    q_test = model.predict(X_test)
    q_all = model.predict(X)

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
    experiment['costs'] = history.history['loss']

    return experiment


def pinball_loss(tau, y, q,alpha):
    """
    Smooth Pinball loss function.

    Arguments:
        tau (ndarray) - quantile levels
        y (ndarray) - time series observations
        q (ndarray) - quantile predictions
        alpha (float) - smoothing rate
    Returns:
        quantile_loss (tensor) - loss
    """

    error = (y - q)
    quantile_loss = K.mean(tau * error + alpha * K.softplus(-error / alpha))

    return quantile_loss