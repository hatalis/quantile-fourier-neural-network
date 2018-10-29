
from benchmarks.ETS import additive
import numpy as np
import scipy.stats as st

def model_ETS(experiment, season = 12):

    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    N_test = experiment['N_test']
    tau = experiment['tau']
    scale = experiment['scale']
    apply_log = experiment['apply_log']

    # x = list of training data (not numpy)
    # m = size of period
    # h = number of points to forecast
    x = y_train.ravel().tolist()
    Y, y_hat, alpha, beta, gamma, rmse = additive(x = x, m = season, fc = N_test)

    q_test = calculate_quantiles(y_hat, rmse, season, N_test, alpha, beta, gamma, tau)
    q_all = calculate_quantiles(Y, rmse, season, len(Y), alpha, beta, gamma, tau)

    # inverse scaled data
    if apply_log:
        # q_train = np.exp(q_train)
        q_test = np.exp(q_test)
        q_all = np.exp(q_all)
    # q_train = q_train * scale
    q_test = q_test * scale
    q_all = q_all * scale

    experiment['model'] = None
    experiment['q_train'] = 0
    experiment['q_test'] = q_test
    experiment['q_all'] = q_all
    experiment['costs'] = rmse

    return experiment


def calculate_quantiles(y_hat,rmse,m,N_test,alpha,beta,gamma,tau):
    # calculate quantiles
    mse = rmse ** 2
    mean = np.array(y_hat).reshape(N_test,1)
    variance = np.zeros((N_test,1))
    variance[0] = mse
    for i in range(1,N_test):
        sum_c = 0
        for j in range(1, i):
            d = 0
            if j%m == 0:
                d = 1
            sum_c = sum_c + (alpha + beta*j + gamma*d)**2
        variance[i] = mse * (1 + sum_c)


    Z = st.norm.ppf(tau)
    quantiles = mean + Z * np.sqrt(variance)

    return quantiles