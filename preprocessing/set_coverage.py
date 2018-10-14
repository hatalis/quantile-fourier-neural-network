"""
@author: Kostas Hatalis
"""
import numpy as np

def set_coverage(experiment):
    """
    Formulates the tau level to create equally spaced quantiles (0,1)

    Arguments:
        experiment(dict): n_PI number of PIs to calculate

    Returns:
        experiment(dict): N_tau (num of taus), and taus
    """

    N_PI = experiment['N_PI']

    if N_PI == None: # test only median
        tau, N_tau = 0.5, 1
    else:
        step = 1 / (2 * N_PI + 1)
        tau = np.array(np.arange(step, 1.0, step))
        N_tau = len(tau)

    # can also also custom define taus here
    # tau = np.arange(0.01, 1.0, 0.01)
    # tau = np.array([0.025, 0.975])

    experiment['tau'] = tau
    experiment['N_tau'] = N_tau

    return experiment