"""
@author: Kostas Hatalis
"""
import matplotlib.pylab as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


from preprocessing.load_data import load_data
from preprocessing.split_data import split_data
from preprocessing.set_coverage import set_coverage

start_time = time.time()


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#--------------------------------------------------------------------------
""" 
Datasets:                   Methods:
    1 = Air Passengers          1 = FQNN
    2 = Sunspots                2 = 
    3 = Load Demand             3 = 
    4 = Internet Traffic        4 = 
    5 = Apple Stock
    6 = Solar Power
    7 = Wind Power
    8 = Wave Elevation
"""
experiment = {}
experiment['dataset'] = 1
experiment['method'] = 1
experiment['xlabel'] = 'Time (days)'
experiment['ylabel'] = 'Price ($)'
experiment['apply_log'] = 0 # log transform data
experiment['max_value'] = 9999 # max value to scale series down to
experiment['percent_training'] = 0.5
experiment['N_PI'] = None # num of PIs to estimate, if None calculate median only

experiment = load_data(experiment)
experiment = split_data(experiment)
experiment = set_coverage(experiment)


#--------------------------------------------------------------------------
# experiment['optimizer'] = 'Adam' # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
# experiment['activation'] = 'relu' # relu, sigmoid, tanh, softplus, elu, softsign, sigmoid, linear
# experiment['smooth_loss'] = 0 # 0 = pinball, 1 = smooth pinball loss
# experiment['maxIter'] = 2000
# experiment['batch_size'] = 200
# experiment['hidden_dims'] = [40] # number of nodes per hidden layer
# experiment['alpha'] = 0.01 # smoothing rate
# experiment['Lambda'] = 0.01 # regularization term
# experiment['n_tau'] = len(tau)
# experiment['tau'] = np.array(tau)
# experiment['kappa'] = 1000 # penalty term
# experiment['margin'] = 0 # penalty margin
# experiment['print_cost'] = 0 # 1 = plot quantile predictions
# experiment['plot_results'] = 1 # 1 = plot cost
#--------------------------------------------------------------------------

# experiment = detector(experiment,test_method=0) # 0 = QARNET, 1 = QAR
# experiment = evaluate_results(experiment)
# output_results(experiment)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()