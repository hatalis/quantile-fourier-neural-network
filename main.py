'''
By Kostas Hatalis
'''
import matplotlib.pylab as plt
import time
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from preprocessing.load_data import load_data
from preprocessing.split_data import split_data
from preprocessing.set_coverage import set_coverage
from model.model_QFNN import model_QFNN
from evaluation.evaluate_PI_results import evaluate_PI_results
from evaluation.output_PI_results import output_PI_results
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
experiment['dataset'] = 1
experiment['xlabel'] = 'Time (days)'
experiment['ylabel'] = ''
experiment['apply_log'] = 1 # log transform data
experiment['max_value'] = 10 # max value to scale series down to
experiment['percent_training'] = 0.5
experiment['N_PI'] = 50 # num of PIs to estimate, if 0 calculate median only
experiment['print_cost'] = 0 # 1 = plot quantile predictions
experiment['plot_results'] = 1 # 1 = plot cost
#--------------------------------------------------------------------------
# QFNN parameters:
experiment['smooth_loss'] = 1 # 0 = pinball, 1 = smooth pinball loss
experiment['g_dims'] = 1 # number of g(t) nodes
experiment['batch_size'] = 0 # if 0, make size = N_train
experiment['epochs'] = 10_000 # number of training epochs
experiment['alpha'] = 0.01 # smoothing rate
experiment['Lambda'] = 0.0003 # L1 regularization term 0.0002
experiment['eta'] = 0.5 # learning rate
#--------------------------------------------------------------------------

experiment = load_data(experiment)
experiment = split_data(experiment)
experiment = set_coverage(experiment)
# EDA(experiment)

# experiment = model_QFNN(experiment)
# experiment = model_QR(experiment,method=0,poly=1) # method = 0: QR, method = 1: QRNN
# experiment = model_SVQR(experiment)
# experiment = model_ETS(experiment, season = 24)
# experiment = model_SARIMA(experiment, order=(1,1,1), seasonal_order=(1,0,1,12))
experiment = model_naive(experiment)

experiment = evaluate_PI_results(experiment)


# Lambda = [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
#           0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
# quantile_score = np.zeros((len(Lambda),1))
# for i in Lambda:
#     experiment['Lambda'] = i
#     experiment = model_QFNN(experiment)
#     experiment = evaluate_PI_results(experiment)
#     quantile_score[i] = experiment['QS']
#     print(Lambda,' + ',experiment['QS'])

output_PI_results(experiment)

print("--- %s seconds ---" % (time.time() - start_time))
plt.show()