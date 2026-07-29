import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import py.helper as h
import py.classifier_training as ct
import numpy as np


signals = h.signals()
sig_colors = h.ews_colours

f = "./data/backup-01-14" #+ "-lightweight" # Change from zenodo
ui = True  # use incidence
for ui in [0,1]:


    def get_dependence_on_ews(signals):

        ct.run_classifier_training(use_incidence=ui, use_cross_val=True,
                                   calculate_ews=False,
                                   folder=f, save_df=True,
                                   aggregation_period=4,
                                   training_signals=signals
                                   )

        # Calculate AUC for whole timeseries
        test = pd.read_csv(f+"/ews_data_test" + h.incidence_filepath(ui) +".csv")
        test = test[["is_test","Time","all"]]
        test["Time"] = test["Time"]/365 - 20
        test.index = test["Time"]
        auc = metrics.roc_auc_score(test.loc[test.index > -10, "is_test"],
                                    test.loc[test.index > -10, "all"])

        # Get AUC timeseries
        auc_test = pd.read_csv(f + "/auc_time_series" +
                                          h.incidence_filepath(ui) + ".csv")
        auc_test.index = auc_test["Time"]/365 - 20


        return auc_test["all"], auc

    auc_ts_loo = {}
    auc_loo = {}
    for s in signals:
        auc_ts_loo[s], auc_loo[s] = get_dependence_on_ews([s2 for s2 in
                                                         signals if s2 != s])
        auc_loo[s] = auc_ts_loo[s].loc[np.round(auc_ts_loo[s].index, 2) == -3].values[0]

        print(s)

    # sort by decreasing importance
    auc_loo = pd.Series(auc_loo)
    auc_loo = auc_loo.sort_values()

    auc_ts_ao = {}
    auc_ao = {}

    for i, s in enumerate(auc_loo.index):
        pfs = auc_loo.index.values[:(i+1)].tolist()
        auc_ts_ao[s], auc_ao[s] = get_dependence_on_ews(pfs)
        auc_ao[s] = auc_ts_ao[s].loc[np.round(auc_ts_ao[s].index, 2) == -3].values[0]

        print(s)

    auc_ao = pd.Series(auc_ao)




    def plot_leave_one_out(figsize=(6,6)):

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(5, 6)
        axes = np.array([plt.subplot(gs[0:2, 0:3]),
                         plt.subplot(gs[0:2, 3:6]), plt.subplot(gs[2:4, 0:3]),
                         plt.subplot(gs[2:4, 3:6])])

        fig.text(0.1,0.92,"Fit to simulated data \nwith one EWS left out")
        fig.text(0.54,0.92,"Fit to simulated data with \nincreasing number of EWS included")

        ax = axes[0]
        for i, s in enumerate(auc_loo.index):
            ax.plot(auc_ts_loo[s],
                     color=sig_colors[s], label=h.sig_labels()[s])

        ax.set_ylabel("AUC")
        ax.set_xlabel("Lead time")
        ax.set_title("a)", fontsize=10, loc="left")
        ax.set_ylim(0.2, 1)
        ax.set_xlim(-10, 0)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

        ax = axes[1]
        for i, s in enumerate(auc_loo.index):
            ax.plot(auc_ts_ao[s],
                    color=sig_colors[s], label=h.sig_labels()[s])

        # ax.set_ylabel("AUC")
        ax.set_yticklabels([])
        ax.set_xlabel("Lead time")
        ax.set_title("c)", fontsize=10, loc="left")
        ax.set_ylim(0.2, 1)
        ax.set_xlim(-10, 0)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)


        x_text = [h.sig_labels()[s] for s in auc_loo.index]

        ax = axes[2]
        for i, s in enumerate(auc_loo.index):
            #auc_plt = auc_ts_loo[s].loc[np.round(auc_ts_loo[s].index, 2) == -2]
            auc_plt = auc_loo[s]
            ax.scatter(i, auc_plt, color=sig_colors[s])
        ax.set_ylim(0.6, 0.9)
        #ax.set_xlim(-0.5,len(x_text))
        ax.set_ylabel("AUC (3 year lead time)")
        ax.set_xlabel("EWS left out")
        ax.set_title("b)", fontsize=10, loc="left")
        ax.set_xticks(np.arange(0, len(x_text)))
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)
        ax.set_xticklabels(x_text, rotation=45, ha="right")


        ax = axes[3]
        for i, s in enumerate(auc_loo.index):
            #auc_plt = auc_ts_ao[s].loc[np.round(auc_ts_ao[s].index, 2) == -2]
            auc_plt = auc_ao[s]
            ax.scatter(i, auc_plt, color=sig_colors[s])

        ax.set_ylim(0.6, 0.9)
        ax.set_yticklabels([])
        #print(ax.get_xlim())
        #ax.set_xlim(x_text[0],x_text[-1])
        ax.set_xlabel("EWS included (all to left)")
        ax.set_title("d)", fontsize=10, loc="left")
        ax.set_xticks(np.arange(0, len(x_text)))
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)
        ax.set_xticklabels(x_text, rotation=45, ha="right")

        #ax.set_xlim(x_text[0],x_text[-1])

        #fig.tight_layout(rect=(0, 0.05, 0.7, 1))
        fig.subplots_adjust(wspace=2, hspace=4)
        return fig, axes


    fig_lo, _ = plot_leave_one_out()

    fig_lo.savefig("./fig_leave_one_out" + h.incidence_filepath(ui) + ".png")

    # run training to restore files to results for using all EWS
    ct.run_classifier_training(use_incidence=ui, use_cross_val=True,
                               calculate_ews=True,
                               folder=f, save_df=True,
                               aggregation_period=4)



coefs_best = pd.read_csv(f+"/ews_weights_incidence.csv", header=None)
