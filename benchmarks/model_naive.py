import numpy as np
import statsmodels.api as sm
import matplotlib.pylab as plt
from scipy.stats import norm

def model_naive(experiment,method=1):

    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    X_train = experiment['X_train']
    X = experiment['X']
    tau = experiment['tau']
    scale = experiment['scale']
    apply_log = experiment['apply_log']

    # calculate linear trend
    model = sm.OLS(y_train, sm.add_constant(X_train))
    results = model.fit()

    # obtain trend parameters
    intercept = results.params[0]
    slope = results.params[1]

    # detrend training data
    y = slope * X_train + intercept
    y_diff = y_train - y

    # calculate uniform or normal quantiles
    if method == 0:
        a = np.min(y_diff)
        b = np.max(y_diff)
        quantiles = (1-tau)*a + tau*b
    else:
        quantiles = norm.ppf(tau, loc=np.mean(y_diff), scale=np.std(y_diff))

    y = slope*X_train + intercept
    q_train = y + quantiles

    y = slope*X + intercept
    q_all = y + quantiles

    y = slope*X_train + intercept
    q_test = y + quantiles

    # inverse scaled data
    if apply_log:
        q_train = np.exp(q_train)
        q_test = np.exp(q_test)
        q_all = np.exp(q_all)
    q_train = q_train * scale
    q_test = q_test * scale
    q_all = q_all * scale

    experiment['model'] = None
    experiment['q_train'] = q_train
    experiment['q_test'] = q_test
    experiment['q_all'] = q_all
    experiment['costs'] = 0

    return experiment