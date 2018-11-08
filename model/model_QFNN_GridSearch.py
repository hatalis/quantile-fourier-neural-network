

import numpy as np
np.random.seed(0)

from tensorflow import set_random_seed
set_random_seed(0)

from model.model_QFNN import model_QFNN

def model_QFNN_GridSearch(experiment):

    y_train = experiment['y_train']
    tau = experiment['tau']

    # Lambda = [  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
    #             0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
    #             0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
    #             0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0]

    step = 0.05
    dropout = np.arange(step, 0.6 + step, step)

    quantile_score = np.zeros((len(dropout),1))
    for idx, val in enumerate(dropout):
        try:
            experiment['dropout'] = val
            experiment = model_QFNN(experiment)
            q_train = experiment['q_train']
            QS_train = quantileScore(q_train, tau, y_train)
            quantile_score[idx] = QS_train
            print('{0:.2f} + {1:.4f}'.format(val,QS_train))
            # print(val,' + ',QS_train)
        except:
            print(val,' did not work.')

    id = int(np.argmin(quantile_score))
    print('===========================')
    print('Best = ',dropout[id],' + ',quantile_score[id])
    print('===========================')
    experiment['dropout'] = dropout[id]

    return model_QFNN(experiment)


def quantileScore(q_hat, tau, y):

    error = (y - q_hat)
    qscore = np.mean(np.maximum(tau * error, (tau - 1) * error))

    return qscore