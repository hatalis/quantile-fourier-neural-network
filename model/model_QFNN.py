'''
By Kostas Hatalis
'''

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam, SGD


from keras.layers import Dense, Input
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import regularizers
from keras.engine.training import Model
from keras.callbacks import ModelCheckpoint

from tensorflow import cos, sin, exp
from keras.backend import set_value

import matplotlib.pylab as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.set_floatx('float64')

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def model_QFNN(experiment):

    method = experiment['method']

    X_train = experiment['X_train']  # data, numpy array of shape (number of features, number of examples)
    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    X_test = experiment['X_test']
    N_tau = experiment['N_tau']
    tau = experiment['tau']

    smooth_loss = experiment['smooth_loss']
    kappa = experiment['kappa']
    alpha = experiment['alpha']
    margin = experiment['margin']
    Lambda = experiment['Lambda']

    # hidden_dims = experiment['hidden_dims']
    maxIter = experiment['maxIter']
    batch_size = experiment['batch_size']
    optimizer = experiment['optimizer']
    N_train = experiment['N_train']

    hidden_dims = int(N_train*1)
    if batch_size == 0:
        batch_size = N_train

    # -------------------------------------- build the model

    # one dimensional input data
    input_data = Input(shape=(1,), name='input_data')
    cosine = Dense(hidden_dims, activation=cos)(input_data) # periodical component of the series
    linear = Dense(1, activation='linear')(input_data) # g(t) function

    # concatenate layers into one
    input_layer = Concatenate()([cosine, linear])

    # output layer, add L1 regularizer as in the paper
    output_layer = Dense(N_tau, kernel_regularizer=regularizers.l1(Lambda))(input_layer)

    # compile keras model
    sgd = SGD(lr=0.5, decay=0, momentum=0.00, nesterov=False)
    model = Model(inputs=[input_data], outputs=[output_layer])
    model.compile(loss=lambda Y, Q: pinball_loss(tau, Y, Q, alpha, smooth_loss, kappa, margin), optimizer=sgd)

    # initialize weights
    model = initialize_weights(model, hidden_dims, 1, N_tau)

    # create model checkpoint
    # weights_path = 'nd_weights.hdf5'
    # model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # callbacks = [model_checkpoint]

    # train model
    history = model.fit(X_train, y_train, epochs=maxIter, verbose=0, batch_size=batch_size)#,callbacks=callbacks)

    # compute predictions
    X = experiment['X']
    raw_data = experiment['raw_data']
    scale = experiment['scale']
    apply_log = experiment['apply_log']
    split_point = experiment['split_point']
    N_PI = experiment['N_PI']
    N = experiment['N']

    # model.load_weights('nd_weights.hdf5')
    q_train = model.predict(X_train)
    q_test = model.predict(X_test)
    q_all = model.predict(X)


    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print(weights)

    if apply_log:
        q_train = np.exp(q_train)
        q_test = np.exp(q_test)
        q_all = np.exp(q_all)
    q_train = q_train * scale
    q_test = q_test*scale
    q_all = q_all * scale

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(np.array(raw_data), 'r',linewidth=1)
    for i in range(0, N_PI):
        ax.fill_between(range(N), q_all[:, i], q_all[:, -1 - i], alpha=2 / N_tau, facecolor='blue')
    # plt.plot(q_all)

    plt.ylim(0,1.5*np.max(np.array(raw_data)))
    plt.axvline(x=split_point, color='black')

    # plt.figure()
    # plt.plot(history.history['loss'])

    '''
    model = Sequential()
    if method == 0: # QRNN
        for i in range(0,len(layers_dims)-2):
            model.add(Dense(layers_dims[i+1], input_dim=layers_dims[i], kernel_regularizer=regularizers.l2(Lambda),
                            kernel_initializer='normal', activation=activation))
        model.add(Dense(layers_dims[-1], kernel_initializer='normal'))
    elif method == 1: # QAR
        model.add(Dense(layers_dims[-1], input_dim=layers_dims[0], kernel_regularizer=regularizers.l2(Lambda),
                        kernel_initializer='normal'))

    # -------------------------------------- compile and fit model
    model.compile(loss=lambda Y, Q: pinball_loss(tau,Y,Q,alpha,smooth_loss,kappa,margin), optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=maxIter, verbose=0, batch_size=batch_size)

    # -------------------------------------- estimate quantiles of testing data
    q_hat = model.predict(X_test)
    '''

    # experiment['q_hat'] = q_hat.T
    experiment['costs'] = history.history['loss']

    return experiment


def initialize_weights(model, h_dims, g_dims, N_tau):
    """
    Initialize weights for the compiled model.
        h_dims - number of cos nodes
        g_dims - number of g(t) nodes
    """
    noise = 0.001

    # set weights and bias's for sin kernel
    set_value(model.weights[0], (2*np.pi * np.arange(h_dims)).reshape((1, h_dims)).astype('float32'))  # cos/kernel
    set_value(model.weights[1], (np.ones((h_dims))).astype('float32'))  # sin/bias

    # set_value(model.weights[0], (2 * np.pi * np.floor(np.arange(h_dims) / 2))[np.newaxis, :].astype('float32'))
    # set_value(model.weights[1],(np.pi / 2 + np.arange(h_dims) % 2 * np.pi / 2).astype('float32'))


    # set weights and bias's for g(t) kernel
    set_value(model.weights[2], (np.ones((1, g_dims)) + np.random.normal(size=(1, g_dims)) * noise).astype('float32'))  # linear/kernel
    set_value(model.weights[3], (np.zeros((g_dims))).astype('float32'))  # linear/bias

    # set_value(model.weights[2], (np.ones(shape=(1, g_dims)) + np.random.normal(size=(1, g_dims)) * noise).astype('float32'))  # linear/kernel
    # set_value(model.weights[3], (np.random.normal(size=(g_dims)) * noise).astype('float32'))  # linear/bias

    # initialize output layer
    set_value(model.weights[4], (np.ones((h_dims + g_dims, N_tau))*0 + np.random.normal(size=(h_dims + g_dims, N_tau)) * noise).astype('float32'))  # output/kernel
    set_value(model.weights[5], (np.zeros((N_tau))).astype('float32')) # output/bias

    return model


# pinball loss function with penalty
def pinball_loss(tau, y, q, alpha = 0.01, smooth_loss = 1, kappa=0, margin=0):
    error = (y - q)
    diff = q[:, 1:] - q[:, :-1]

    quantile_loss = 0
    if smooth_loss == 0: # pinball function
        quantile_loss = K.mean(K.maximum(tau * error, (tau - 1) * error))
    elif smooth_loss == 1: # smooth pinball function
        quantile_loss = K.mean(tau * error + alpha * K.softplus(-error / alpha))

    penalty = kappa * K.mean(tf.square(K.maximum(tf.Variable(tf.zeros([1], dtype=tf.float64)), margin - diff)))

    return quantile_loss + penalty


