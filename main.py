'''
By Kostas Hatalis
'''
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pylab as plt
import time

import numpy as np
np.random.seed(0)

from tensorflow import set_random_seed
set_random_seed(0)

from preprocessing.load_data import load_data
from preprocessing.split_data import split_data
from preprocessing.set_coverage import set_coverage
from evaluation.evaluate_PI_results import evaluate_PI_results
from evaluation.output_PI_results import output_PI_results
from model.model_QFNN import model_QFNN
from model.model_QFNN_GridSearch import model_QFNN_GridSearch
from benchmarks.model_QR import model_QR
from benchmarks.model_SVQR import model_SVQR
from benchmarks.model_ETS import model_ETS
from benchmarks.model_SARIMA import model_SARIMA
from EDA import EDA
from benchmarks.model_naive import model_naive

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
experiment['dataset'] = 3
experiment['xlabel'] = 'Time (seconds)'
experiment['ylabel'] = 'Wave Elevation (meters)'
experiment['ymax'] = 1.5
experiment['ymin'] = 0
experiment['labelH'] = 1.05

experiment['apply_log'] = 0 # log transform data
experiment['max_value'] = 10 # max value to scale series down to, if 0 then no scale
experiment['percent_training'] = 0.5
experiment['N_PI'] = 0 # num of PIs to estimate, if 0 calculate median only
experiment['print_cost'] = 0 # 1 = plot quantile predictions
experiment['plot_results'] = 1 # 1 = plot cost
#--------------------------------------------------------------------------
# QFNN parameters:
experiment['smooth_loss'] = 1 # 0 = pinball, 1 = smooth pinball loss
experiment['g_dims'] = 1 # number of g(t) nodes
experiment['epochs'] = 40_000 # number of training epochs
experiment['alpha'] = 0.01 # smoothing rate
experiment['eta'] = 0.5 # learning rate
experiment['Lambda'] = 0.000 # L1 reg. to output weights, 0.0003
experiment['dropout'] = 0.45 # droput rate
#--------------------------------------------------------------------------
# Data Preprocessing:
experiment = load_data(experiment)
experiment = split_data(experiment)
experiment = set_coverage(experiment)
# EDA(experiment)
# experiment['tau'] = np.array([0.005, 0.01,0.015,0.02,0.025,0.975,0.98,0.985,0.99,0.995])
# experiment['N_tau'] = 10
# experiment['N_PI'] = 5


#--------------------------------------------------------------------------
# Prediction Methods:
# experiment = model_QFNN(experiment)
# experiment = model_QFNN_GridSearch(experiment)
# experiment = model_QR(experiment,method=1,poly=1) # method = 0: QR, method = 1: QRNN
# experiment = model_SVQR(experiment)
# experiment = model_ETS(experiment, season = 24)
experiment = model_SARIMA(experiment, order=(1,1,1), seasonal_order=(1,1,1,24))
# experiment = model_naive(experiment,method=0) # 0 = UM, 1 = PM
#--------------------------------------------------------------------------

q_all = experiment['q_all']
# print(q_all.shape)
print('\n'.join(str(n) for n in q_all.ravel()))

# Evaluate and Output Results
experiment = evaluate_PI_results(experiment)
output_PI_results(experiment)

print("--- %s seconds ---" % (time.time() - start_time))
plt.show()