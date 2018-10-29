
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

def EDA(experiment):

    y_train = experiment['y_train']
    y_train = np.diff(y_train, n=1, axis=0)

    plot_acf(y_train,lags=48)
    plot_pacf(y_train,lags=48)

    return None