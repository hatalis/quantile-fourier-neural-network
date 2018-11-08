"""
@author: Kostas Hatalis
"""
import pandas as pd

def load_data(experiment):
    """
    Load in time series data.

    Arguments:
        experiment(dict): which dataset to use

    Returns:
        experiment(dict): raw series data
    """

    dataset = experiment['dataset']
    data = None

    if dataset == 1: # Air Passengers
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
        data = pd.read_csv('data\\air.csv', parse_dates=['Month'], index_col=0, date_parser=dateparse)

    elif dataset == 2: # Sunspots
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y')
        data = pd.read_csv('data\\sunspots.csv', parse_dates=['Year'], index_col=0, date_parser=dateparse)

    elif dataset == 3: # Real-Time Load Demand
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d %H:%M')
        data = pd.read_csv('data\\load.csv', parse_dates=[['Date', 'Hour']], index_col=0, date_parser=dateparse)

    elif dataset == 4: # Internet Traffic Data (in bits)
        dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %H:%M')
        data = pd.read_csv('data\\internet.csv', parse_dates=['Time'], index_col=0, date_parser=dateparse)
        # series = data['Internet']

    elif dataset == 5: # Apple Closing Stock Price
        dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
        data = pd.read_csv('data\\stock.csv', parse_dates=['Date'], index_col=0, date_parser=dateparse)

    elif dataset == 6: # Solar Power
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d %H:%M')
        data = pd.read_csv('data\\solar.csv', parse_dates=[['Date', 'Hour']], index_col=0, date_parser=dateparse)

    elif dataset == 7: # Wind Power
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d %H:%M')
        data = pd.read_csv('data\\wind.csv', parse_dates=[['Date', 'Hours']], index_col=0, date_parser=dateparse)

    elif dataset == 8: # Ocean Wave Elevation
        dateparse = lambda dates: pd.datetime.strptime(dates, '%M:%S.%f')
        data = pd.read_csv('data\\wave.csv', parse_dates=['Time'], index_col=0, date_parser=dateparse)

    data.columns = ['data']
    data.index.name = 'time'

    experiment['raw_data'] = data
    experiment['data_index'] = data.index

    return experiment