

import numpy as np
from model.model_QFNN import model_QFNN
from evaluation.evaluate_PI_results import evaluate_PI_results


def model_QFNN_GridSearch(experiment):


    Lambda = [  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
                0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
                0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0]

    quantile_score = np.zeros((len(Lambda),1))
    for idx, val in enumerate(Lambda):
        experiment['Lambda'] = val
        experiment = model_QFNN(experiment)
        experiment = evaluate_PI_results(experiment)
        quantile_score[idx] = experiment['QS_train']
        print(val,' + ',experiment['QS_train'])

    id = int(np.argmin(quantile_score))
    print('===========================')
    print(quantile_score[id])
    print(Lambda[id])
    print('===========================')

    experiment['Lambda'] = Lambda[id]
    experiment = model_QFNN(experiment)

    return experiment