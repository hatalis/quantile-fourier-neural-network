
import numpy as np
import scipy.stats as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

def model_SARIMA(experiment, order=(1,1,1), seasonal_order=(1,0,1,12)):

    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    N_test = experiment['N_test']
    N_train = experiment['N_train']
    tau = experiment['tau']
    N_tau = experiment['N_tau']
    scale = experiment['scale']
    apply_log = experiment['apply_log']

    # Fit SARIMA model to training data
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(N_test)
    y_train_hat = model_fit.fittedvalues

    # Get confidence intervals of forecasts
    pred_ci = model_fit.get_forecast(steps=N_test).conf_int()

    # Get uppler and lower confidence bounds, calculate std
    lower,upper = np.array(pred_ci[:, 0]), np.array(pred_ci[:, 1])
    std = (upper-lower)/1.96/2

    # extract quantiles from normal distribution
    q_test = np.zeros((N_test,N_tau))
    for t in range(0,N_test):
        q_test[t,:] = st.norm.ppf(tau, loc=forecast[t], scale=std[t])

    # combine training fit with predicted quantiles to get q_all
    temp = np.ones((N_train,N_tau))*y_train_hat.reshape((N_train,1))
    q_all = np.concatenate((temp, q_test), axis=0)

    # inverse scaled data
    if apply_log:
        q_test = np.exp(q_test)
        q_all = np.exp(q_all)
    q_test = q_test * scale
    q_all = q_all * scale

    experiment['model'] = model
    experiment['q_train'] = 0
    experiment['q_test'] = q_test
    experiment['q_all'] = q_all
    experiment['costs'] = 0


    return experiment
