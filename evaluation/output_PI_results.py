'''
By Kostas Hatalis
'''

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns; sns.set()

def output_PI_results(experiment):

    SHARP = experiment['SHARP']
    QS = experiment['QS']
    QS_train = experiment['QS_train']
    IS = experiment['IS']
    ACE = experiment['ACE']
    PICP = experiment['PICP']

    plot_results = experiment['plot_results']
    print_cost = experiment['print_cost']
    costs = experiment['costs']
    q_train = experiment['q_train']
    q_test = experiment['q_test']
    q_all = experiment['q_all']
    N_tau = experiment['N_tau']

    y_train = experiment['y_train']
    y_test = experiment['y_test']
    N_test = experiment['N_test']
    N_train = experiment['N_train']

    X = experiment['X']
    raw_data = experiment['raw_data']
    scale = experiment['scale']
    # model = experiment['model']
    split_point = experiment['split_point']
    N_PI = experiment['N_PI']
    N = experiment['N']

    y = experiment['raw_data']
    xlabel = experiment['xlabel']
    ylabel = experiment['ylabel']

    data_index = experiment['data_index']
    series = experiment['raw_data']

    ymin = experiment['ymin']
    ymax = experiment['ymax']
    labelH = experiment['labelH']
    # print('Results on Train Set:')
    # print('QS_train = ', QS_train)


    # print('\nResults on Test Set:')
    # print('PICP = ',PICP)
    # print('Sharpness = ',SHARP)
    # print('ACE = ', ACE)
    # print('IS = ',IS)
    # print('QS = ', QS)
    # print('')

    # print(PICP)
    # print(SHARP)
    # print(ACE)
    # print(IS)
    # print(QS)

    # print(' '.join(str(n) for n in PICP.ravel()))
    # print(' '.join(str(n) for n in SHARP.ravel()))
    # print(' '.join(str(n) for n in ACE.ravel()))
    # print(' '.join(str(n) for n in IS.ravel()))
    # print(' '.join(str(n) for n in QS.ravel()))

    # PIs = np.arange(0.1,1,0.1)
    # plt.figure()
    # plt.plot(PIs,PICP.T)
    # plt.xlim(0,1)

    # print(N_tau)
    # plt.figure()
    # ax = sns.heatmap(second_layer_biases)
    # ax.invert_yaxis()

    # extract weights from each layer:
    # first_layer_weights = model.layers[1].get_weights()[0]
    # first_layer_biases = model.layers[1].get_weights()[1]
    # first_layer_biases = np.reshape(first_layer_biases, (1,len(first_layer_biases)))
    # second_layer_weights = model.layers[4].get_weights()[0]
    # second_layer_biases = model.layers[4].get_weights()[1]
    # second_layer_biases = np.reshape(second_layer_biases, (1,len(second_layer_biases)))

    # plt.plot(PINC,SHARP)

    if print_cost:
        plt.figure()
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('epoch')

    if plot_results:
        if N_PI > 0:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_axisbelow(True)
            ax.axvspan(series.index[0], series.index[split_point], alpha=0.2, color='gray')
            x = range(N)
            plt.plot(series,'r',linewidth=1)
            for i in range(N_PI):
                y1 = q_all[:,i]
                y2 = q_all[:,-1-i]
                plt.fill_between(series.index, y1, y2, color='blue', alpha=2 / N_tau) # alpha=str(1/n_PIs)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            # plt.ylim(0,1.5*np.max(np.array(y)))
            plt.axvline(x=series.index[split_point], color='black')
            plt.grid(linewidth=0.4)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.grid(True)

            miny, maxy = ax.get_ylim()
            midpoint = int(len(y)*0.2)
            plt.text(series.index[midpoint],maxy* labelH, 'Training Samples')
            midpoint = int(len(y) * 0.7)
            plt.text(series.index[midpoint], maxy * labelH, 'Testing Samples')
            # plt.ylim(ymin, ymax)
            fig.savefig("pis.pdf", bbox_inches='tight')

        else: # plot median
            fig =  plt.figure()
            x = range(N)
            plt.plot(x,y,'r')
            plt.plot(x, q_all, 'b')
            # plt.ylim(0,1.5*np.max(np.array(y)))
            plt.axvline(x=split_point, color='black')
            # fig.savefig('median.svg', format='svg', dpi=1200)
            fig.savefig("median.pdf", bbox_inches='tight')
    return None