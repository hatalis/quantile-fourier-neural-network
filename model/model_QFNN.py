'''
By Kostas Hatalis
'''

import numpy as np
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate
from keras import regularizers
from keras.engine.training import Model
from tensorflow import cos, sin, exp
from keras.backend import set_value
import os # remove certain tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K.set_floatx('float32')


def model_QFNN(experiment):
    """
    Initialize weights for the compiled model.

    Arguments:
        experiment (dict) - various QFNN parameters
    Returns:
        experiment (dict) - trained model and predicted quantiles
    """

    X_train = experiment['X_train']  # data, numpy array of shape (number of features, number of examples)
    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    X_test = experiment['X_test']
    N_tau = experiment['N_tau']
    tau = experiment['tau']

    smooth_loss = experiment['smooth_loss']
    alpha = experiment['alpha']
    Lambda = experiment['Lambda']
    eta = experiment['eta']
    epochs = experiment['epochs']
    batch_size = experiment['batch_size']
    N_train = experiment['N_train']
    g_dims = experiment['g_dims']

    # set batch size and hidden dim to size of train data
    hidden_dims = int(N_train*1)
    if batch_size == 0:
        batch_size = N_train

    # -------------------------------------- build the model

    # one dimensional input data
    input_data = Input(shape=(1,))
    cosine = Dense(hidden_dims, activation=cos)(input_data) # cosine nodes
    linear = Dense(g_dims, activation='linear')(input_data) # g(t) nodes

    # concatenate layers into one
    input_layer = Concatenate()([cosine, linear])

    # output layer with added L1 regularizer
    output_layer = Dense(N_tau, kernel_regularizer=regularizers.l1(Lambda))(input_layer)

    # compile keras model
    sgd = SGD(lr=eta, decay=0, momentum=0.00, nesterov=False)
    model = Model(inputs=[input_data], outputs=[output_layer])
    model.compile(loss=lambda Y, Q: pinball_loss(tau, Y, Q, alpha, smooth_loss), optimizer=sgd)

    # initialize weights
    model = initialize_weights(model, hidden_dims, g_dims, N_tau)

    # train model
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)

    # --------------------------------------

    # compute predictions
    X = experiment['X']
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


def initialize_weights(model, h_dims, g_dims, N_tau):
    """
    Initialize weights for the compiled model.

    Arguments:
        model (keras) - keras model of QFNN
        h_dims (int) - number of cos nodes
        g_dims (int) - number of g(t) nodes
        N_tau (int) - number of taus
    Returns:
        model (keras) - keras model of QFNN with initialized weights
    """

    variance = 0.00

    # set weights and bias's for sin kernel
    set_value(model.weights[0], (2*np.pi * np.arange(h_dims)).reshape((1, h_dims)).astype('float32'))  # cos/kernel
    set_value(model.weights[1], (np.ones((h_dims)) + np.random.normal(size=(h_dims)) * variance).astype('float32'))  # sin/bias

    # set weights and bias's for g(t) kernel
    set_value(model.weights[2], (np.ones((1, g_dims)) + np.random.normal(size=(1, g_dims)) * variance).astype('float32'))  # linear/kernel
    set_value(model.weights[3], (np.random.normal(size=(g_dims)) * variance).astype('float32'))  # linear/bias

    # initialize output layer
    set_value(model.weights[4], (np.random.normal(size=(h_dims + g_dims, N_tau)) * variance).astype('float32'))  # output/kernel
    set_value(model.weights[5], (np.random.normal(size=(N_tau)) * variance).astype('float32')) # output/bias


    return model


def pinball_loss(tau, y, q, alpha = 0.01, smooth_loss = 1):
    """
    Pinball loss function.

    Arguments:
        tau (ndarray) - quantile levels
        y (ndarray) - time series observations
        q (ndarray) - quantile predictions
        alpha (float) - smoothing rate
        smooth_loss (int) - use nonsmooth or smooth pinball function
    Returns:
        quantile_loss (tensor) - loss
    """

    error = (y - q)
    quantile_loss = 0
    if smooth_loss == 0: # pinball function
        quantile_loss = K.mean(K.maximum(tau * error, (tau - 1) * error))
    elif smooth_loss == 1: # smooth pinball function
        quantile_loss = K.mean(tau * error + alpha * K.softplus(-error / alpha))

    return quantile_loss

