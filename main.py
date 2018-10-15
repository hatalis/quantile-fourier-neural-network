"""
@author: Kostas Hatalis
"""
import matplotlib.pylab as plt
import time
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from preprocessing.load_data import load_data
from preprocessing.split_data import split_data
from preprocessing.set_coverage import set_coverage
from model.model_QFNN import model_QFNN

start_time = time.time()

#--------------------------------------------------------------------------
""" 
dataset:                    method:
    1 = Air Passengers          1 = QFNN
    2 = Sunspots                2 = 
    3 = Load Demand             3 = 
    4 = Internet Traffic        4 = 
    5 = Apple Stock
    6 = Solar Power
    7 = Wind Power
    8 = Wave Elevation
"""
experiment = {}
experiment['dataset'] = 6
experiment['method'] = 1
experiment['xlabel'] = 'Time (days)'
experiment['ylabel'] = 'Price ($)'
experiment['apply_log'] = 0 # log transform data
experiment['max_value'] = 10 # max value to scale series down to
experiment['percent_training'] = 0.5
experiment['N_PI'] = 50 # num of PIs to estimate, if 0 calculate median only
#--------------------------------------------------------------------------
experiment['optimizer'] = 'SGD' # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
experiment['smooth_loss'] = 1 # 0 = pinball, 1 = smooth pinball loss
experiment['maxIter'] = 10000
experiment['batch_size'] = 0 # if 0, make size = N_train
experiment['alpha'] = 0.01 # smoothing rate
experiment['Lambda'] = 0.0002 # regularization term
experiment['kappa'] = 0 # penalty term
experiment['margin'] = 0 # penalty margin
experiment['print_cost'] = 0 # 1 = plot quantile predictions
experiment['plot_results'] = 0 # 1 = plot cost
#--------------------------------------------------------------------------

experiment = load_data(experiment)
experiment = split_data(experiment)
experiment = set_coverage(experiment)
experiment = model_QFNN(experiment)

# experiment = evaluate_results(experiment)
# output_results(experiment)

print("--- %s seconds ---" % (time.time() - start_time))
plt.show()