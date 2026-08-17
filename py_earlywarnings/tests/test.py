# -*- coding: utf-8 -*-

#import context
#assuming installed
import ews
from random import random
import pandas as pd
import numpy as np
import time


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import nbinom
import scipy as sc
from scipy.stats import kendalltau



sns.set_style("darkgrid",  {"axes.facecolor": ".9", "font.family": u'serif'})
#$matplotlib.rcParams['font.family'] = "sans-serif"

#Plot EWS and scaled EWS
def ews_plot(df, signals,
                 filename="./figs/decision_function.pdf", title=None):
    '''
    Plot EWS
    :param df: pandas.dataframe or dict with columns/entries named: time, timeseries and all EWS listed in signals
    :param signals: list of EWS to be plotted
    :param filename: name of output file
    :param title: title for figure. No title shown if None
    :return: returns matplotlib figure
    '''

    #plt.style.use('seaborn-darkgrid')
    # plt.rcParams.update({'figure.autolayout': True})

    fig, axes = plt.subplots(len(signals) + 1, 1, figsize=(6, 10), sharex=False, sharey=False)
    # add a big axes, hide frame
    fig.add_subplot(1, 1, 1, frameon=False)
    #  hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("time", labelpad=20)

    if (title != None):
        plt.title(title, y=1.02)
    fig.tight_layout(pad=2, w_pad=0.5, h_pad=0.5)

    t = df["time"]

    axes[0].plot(t, df["timeseries"])
    axes[0].locator_params(nbins=3, axis='y')
    axes[0].set_xlim(min(t),max(t))
    axes[0].set_xticklabels([])
    axes[0].set_title("time series", loc="left", fontsize=12)
    for j, signal in enumerate(signals):
        x = df[signal]
        axes[j + 1].plot(t, x)
        axes[j+1].set_xlim(min(t), max(t))
        axes[j + 1].locator_params(nbins=3, axis='y')
        axes[j + 1].set_title(signal, loc="left", fontsize=12)
        if (j != len(signals)-1):
            axes[j + 1].set_xticklabels([])


        else:
            for tick in axes[j + 1].get_xticklabels():
                tick.set_rotation(45)
    if(filename != None):
        fig.savefig(filename, format='pdf')
    return(fig)



data = pd.read_csv("./tests/sample_emerging.csv", sep = "\t", header=None)
x = data[2].astype(int).tolist()


## Gaussian data test
mu = 0
sigma = 0.1

x = np.random.normal(mu, sigma, 1000)
x = nbinom.rvs(10, 0.5, size=1000).astype(float)
x = np.array([nbinom.rvs(10, 0.9*(1.-0.9*i/1000.)) for i in range(1,1000)], dtype=float)
x = np.array([np.random.normal( i, np.sqrt(i)) for i in range(1,1000)], dtype=float)

#x[500] = np.nan
#x = np.zeros(1000)
ews_data = pd.DataFrame(ews.get_ews(x, 201, 1))
ews_data["time"] = ews_data.index.values


ews_plot(ews_data, ews_data.columns.values[:-1], "./tests/fig.pdf")

taus = {}
for signal in ews_data.columns:
    ts = ews_data[signal]
    ts = ts[~np.isnan(ts)]
    tau, p_value = kendalltau(np.arange(ts.shape[0]),ts)
    taus[signal] = tau

taus


s = np.random.negative_binomial(10, 0.3, size=100000)
start = time.time()
n = pd.DataFrame(ews.get_ews(s,100,1, method="new",mv_method="exp", kc=False, se=False))
end = time.time()
print("new", end - start)

start = time.time()
o = pd.DataFrame(ews.get_ews(s,100,1, method="old", kc=False, se=False))
end = time.time()
print("old", end - start)



start = time.time()
mv1 = ews.get_ews(x,windowsize=100,ac_lag=1,se=False,kc=False)
end = time.time()
print("1", end - start)

start = time.time()
mv1 = ews.get_ews(x,windowsize=100,ac_lag=1,se =True,kc=False)
end = time.time()
print("2", end - start)

np.nanmax(mv1-mv2)