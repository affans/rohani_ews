import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import py.helper as h
import py.classifier_training as ct
from sklearn.metrics import roc_auc_score



plt.style.use('seaborn-ticks')
signals = h.signals()


def plot_all_ews_sim(folder, use_incidence=True, figsize=(7, 4), auc_nc=None,
                     labels=["with covariates","no covariates"]):

    # no_covar is a legacy name
    plotted_signals = h.signals() + ["all"] + ["no_covar"]
    sig_labels = h.sig_labels()
    sig_labels["all"] = "Best fit to simulated\ndata ("+labels[0]+")"
    sig_labels["no_covar"] = "Best fit to simulated\ndata ("+labels[1]+")"

    auc_dict = pd.read_csv(folder + "/auc_time_series" +
                           h.incidence_filepath(use_incidence) + ".csv",
                           index_col=0)
    auc_dict["no_covar"] = auc_nc

    pal = h.ews_colours
    pal["all"] = pal["Best fit to pertussis"]
    pal["no_covar"] = pal["Best fit to simulated data"]



    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 6)
    axes = np.array([plt.subplot(gs[0:6, 0:6])])

    for i, s in enumerate(plotted_signals):
        ax = axes[0]

        ax.plot(auc_dict[s].index / 365 -20, auc_dict[s],
                label=sig_labels[s], color=pal[s])
        ax.set_ylabel("AUC")
        ax.set_xlabel("Lead time")
        ax.set_ylim(0.2, 1)
        ax.set_xlim(-10, 0)
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    fig.legend(bbox_to_anchor=(0.825, 0.5, 0, 0),
               loc="center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 0.7, 1))
    return fig, axes

def calculate_ews_using_other_fit(f_fit, f_data, use_incidence):
    params_df = pd.read_csv(f_data+"/simulation_parameters.csv")
    coefs_best = pd.read_csv(f_fit + "/ews_weights" +
                             h.incidence_filepath(use_incidence) + ".csv",
                             index_col=0, header=None)
    df = pd.concat(
        [pd.read_csv(f_data + "/data_" + str(i) + ".csv")
             .assign(model=i).assign(
            is_test=int(params_df.loc[i, "R0_f"] == 1.0))
         for i in params_df.index.values])
    # calculate EWS
    edf = h.get_ews(df, params_df, agg=4,
                    wtime=h.read_cross_val(f_fit, use_incidence)[0],
                    mv_method="exp", use_incidence=use_incidence)

    edf["Decision function"] = edf[signals].dot(coefs_best.loc[signals, 1]) +\
                               coefs_best.loc["intercept", 1]
    edf["Emergence risk"] = h.lr_emergence_risk(edf["Decision function"])
    edf.to_csv(f_data + "/ews_data_test_nocovars" +
               h.incidence_filepath(use_incidence) + ".csv")

f1 = "./data/backup-01-14" # + "-lightweight" # change from zenodo
f2 = "./data/backup-01-14" + "-covar" # change from zenodo


for ui in [True]:

    calculate_ews_using_other_fit(f1, f2, ui)

    edf = pd.read_csv(f2 + "/ews_data_test_nocovars" +
                      h.incidence_filepath(ui) + ".csv")
    edf = edf.dropna()

    auc_no_covars = edf.groupby("Time").apply(
        lambda x: roc_auc_score(x["is_test"], x["Emergence risk"]))

    fig, _ = plot_all_ews_sim(f2, use_incidence=ui, figsize=(7, 4),
                              auc_nc=auc_no_covars )
                             #labels=["convex only", "concave and convex"])

    fig.savefig("./fig_simulated_auc_performance_covar" +
                h.incidence_filepath(ui) + ".png")
