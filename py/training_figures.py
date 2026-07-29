import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gsp
import py.helper as h

plt.style.use('seaborn-ticks')

# sigs = ['autocorrelation','coefficient_of_variation', 'kurtosis',
# 'skewness',"ac2"]


def plot_all_ews_sim(folder, use_incidence=True, figsize=(4, 4)):

    plotted_signals = ["all"] + h.signals()
    sig_labels = h.sig_labels()
    sig_labels["all"] = "Best fit to simulated data"
    pal = h.ews_colours
    pal["all"] = pal["Best fit to simulated data"]
    auc_dict = pd.read_csv(folder + "/auc_time_series" +
                           h.incidence_filepath(use_incidence) + ".csv",
                           index_col=0)

    fig = plt.figure(figsize=figsize)
    gs = gsp.GridSpec(6, 6)
    axes = np.array([plt.subplot(gs[0:6, 0:6])])

    for i, s in enumerate(plotted_signals):
        ax = axes[0]

        ax.plot(auc_dict[s].index / 365 - 20, auc_dict[s],
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


# Cross-validation figure
def plot_cval(folder, use_incidence=True, figsize=(4, 4)):
    p_performance = pd.read_csv(folder+"/k-fold-cross-validation" +
                                h.incidence_filepath(use_incidence)
                                + ".csv", index_col=0)
    p_performance["test"] = np.sign(p_performance["auc"] + p_performance["std"]
                                    - p_performance["auc"].max()).astype(int)
    # get w min

    p_performance["is_max"] = 0
    p_performance.loc[p_performance["auc"].idxmax(), "is_max"] = 1
    w_min, c_min = h.read_cross_val(folder, use_incidence)

    print("Best:", w_min, c_min)
    p_performance["p"] = np.log10(1/p_performance["p"])
    auc_max = p_performance.iloc[p_performance["auc"].idxmax(axis=0)]
    p_max = auc_max["p"]
    w_max = auc_max["w"]
    print("Max.:", p_max, w_max)

    fig, ax = plt.subplots(figsize=figsize)
    t = p_performance[["p", "w", "auc"]]\
        .pivot_table(index="p", columns="w", values="auc")
    t1 = p_performance[["p", "w", "test"]]\
        .pivot_table(index="p", columns="w", values="test")
    # t2 = p_performance[["p", "w", "is_max"]]\
    #     .pivot_table(index="p", columns="w", values="is_max")

    sns.heatmap(t, vmin=0.5, vmax=0.7,
                cbar_kws={'label': 'AUC'}, cmap="Greens_r", ax=ax, square=True)

    ax.scatter(w_max//52-0.5, p_max+2.5, marker='x', s=50, color='black')
    ax.scatter(w_min//52-0.5, -np.log10(c_min)+2.5, marker='o', s=50,
               color='black')
    ax.text(w_max//52-0.25, p_max+2.25, "Max.", color='black')
    ax.text(w_min//52-0.25, -np.log10(c_min)+2.25, "Best", color='black')

    ax.invert_yaxis()

    ax.set_xlabel("Half-life")
    ax.set_ylabel("Penalty strength (log10)")
    ax.contour(t1.values, colors="black", levels=[0], origin="lower")
    # ax.contour(t2.values, colors="black", levels=[0], origin="upper")
    fig.tight_layout()
    return fig, ax


f = "./data/backup-01-14"  # + "-lightweight"
for ui in [False, True]:
    fig2, _ = plot_all_ews_sim(f, use_incidence=ui)
    # fig2.suptitle(f + " " + str(ui))

    fig2.savefig("./fig_simulated_auc_performance" + h.incidence_filepath(ui)
                 + ".svg")

    figcv, _ = plot_cval(f, use_incidence=ui)
    # figcv.suptitle(f+ " " + str(ui))

    figcv.savefig("./fig_cross_validation" + h.incidence_filepath(ui)
                  + ".svg")

coefs_best = pd.read_csv(f+"/ews_weights_incidence.csv", header=None)
print(coefs_best)
