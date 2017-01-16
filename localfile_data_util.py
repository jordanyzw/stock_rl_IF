#coding=utf-8
from __future__ import print_function
import numpy as np
import talib
import pandas as pd
from matplotlib import pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def plot_variable_hist(X,label,range=None):
    if range == None:
        plt.hist(X, 50, normed=1, facecolor='green',histtype='barstacked')
    else:
        plt.hist(X,50,normed=1, facecolor='green',histtype='barstacked')
    plt.savefig(label + '.png')
    return

def read_local_file(filename):
    quotes = pd.read_csv(filename, header=0, encoding='cp936', names=['ticker','name','date','close',\
            'open', 'high', 'low', 'volume','tol_money','buy_volume','buy_money','sell_volume','sell_money'])
    return quotes

def get_local_data(ticker = "IF"):
    quotes = read_local_file("./If_index/" + ticker + "00_2010-01-01_2016-11-20.csv")
    dates = pd.to_datetime(quotes['date'])
    open_v = np.array(quotes['open'])
    close_v = np.array(quotes['close'])
    high_v = np.array(quotes['high'])
    low_v = np.array(quotes['low'])
    volume = np.array(quotes['volume'], dtype='float64')
    tol_money = np.array(quotes['tol_money'],dtype='float64')
    buy_money = np.array(quotes['buy_money'],dtype='float64')


    MFI = talib.MFI(high_v,low_v,close_v,volume) / 100
    #ADX = talib.ADX(high_v,low_v,close_v,timeperiod=7) / 100
    log_open = np.diff(np.log(open_v))
    log_close = np.diff(np.log(close_v))
    log_high = np.diff(np.log(high_v))
    log_low = np.diff(np.log(low_v))
    log_tol_money_diff = tol_money[1:] / tol_money[:-1]
    buy_proportion  = np.array(buy_money / tol_money)
    #log_volume = np.diff(np.log(volume))
    dates = dates[1:]
    close_diff =  np.diff(close_v)
    close_v = close_v[1:]
    #plot the data
    #if not os.path.exists(ticker + "_Hmm_price_predict"):
    #    os.mkdir(ticker + "_Hmm_price_predict")
    #os.chdir(ticker + "_Hmm_price_predict")
    plot_variable_hist(log_open,ticker + "log_open_diff")
    plot_variable_hist(log_close,ticker + "log_close_diff")
    plot_variable_hist(log_high, ticker + "log_high_diff")
    plot_variable_hist(log_low, ticker + "log_low_diff")
    plot_variable_hist(log_tol_money_diff,"log_tol_money_diff")
    plot_variable_hist(buy_proportion,"buy_proportion")
    plot_variable_hist(MFI[14:],"MFI",(0,1))
    #plot_variable_hist(ADX[14:],"ADX")
    #plot_variable_hist(log_volume, ticker + "log_volume_diff")


    # Pack diff and volume for training.
    print(len(log_open[13:]),len(buy_proportion[14:]),len(MFI[14:]))
    #X = np.column_stack([log_open[13:], log_close[13:], log_high[13:], \
    #                     log_low[13:], log_tol_money_diff[13:],buy_proportion[14:],MFI[14:]])
    X = np.column_stack([log_close[13:], log_tol_money_diff[13:],buy_proportion[14:],MFI[14:]])
    return X, close_v[13:], close_diff[13:], dates[14:]


