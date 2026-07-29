import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
# from pyDOE import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import py.helper as h
import pandas as pd

# Select signals to be used in learning:
signals = h.signals()
# column names
sig_labels = h.sig_labels()


# plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-ticks')
pal = ["#0033ab","#d9006f"]
pal = h.pal
lw = "-lightweight/"
f = "./data/backup-01-14" #+lw # Change from zenodo


coefs_best = pd.read_csv(f + "/ews_weights" +
                         h.incidence_filepath(False) + ".csv",
                         index_col=0, header=None)

edf = pd.read_csv(f + "/ews_data_test.csv")
params_df = pd.read_csv(f + "/simulation_parameters.csv")
df_threshold = pd.read_csv(f+ "/optimum_thresholds.csv", header=None, index_col=0)
pr_threshold = h.lr_emergence_risk(df_threshold.loc["all",1])

ews_data = edf[edf.Time.between(10 * 365,20*365)].copy()
ews_data = ews_data[~ews_data.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
ews_data = ews_data[~ews_data[signals].isna().any(axis=1)]
ews_data["df"] = ews_data["all"] # Change from zenodo
ews_data["pr"] = h.lr_emergence_risk(ews_data["df"])

auc_best = ews_data.groupby("Time").apply(
    lambda x: roc_auc_score(x["is_test"], x["df"]))

times = ews_data["Time"].unique()
cbar = sns.cubehelix_palette(len(times))


np.random.seed(20)


def plot_early_warning_system():

    def convert_to_plot_format(df):
        df = df.melt(id_vars="Time", value_vars=signals)
        df["value_abs"] = np.abs(df["value"])
        df["value_norm"] = df["value_abs"]/df["value_abs"].max()
        df["value_sign"] = np.sign(df["value"]).astype(int) + 1

        df["number"] = pd.factorize(df["variable"])[0]
        df["colour"] = df["number"].apply(lambda j: h.long_pal[j])
        return df

    d_sample = ews_data[ews_data["model"] == 2].copy()
    d_sample["Time"] = np.round(d_sample["Time"]/365 - 10, 1)

    d_ews = d_sample.iloc[::13, :]
    d_ews["pred"] = d_ews["pr"] > \
                    h.lr_emergence_risk(df_threshold.loc["all", 1])
    d_raw = convert_to_plot_format(d_ews)

    d_ews2 = d_ews.copy()
    d_ews2[signals] = d_ews2[signals]*coefs_best.loc[signals, 1]
    d_weighted = convert_to_plot_format(d_ews2)

    axes_label_fontsize = 8
    dot_size = 70
    pc = h.plot_colours

    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    fig = plt.figure(figsize=(3.5, 6.5))
    gs = GridSpec(30, 1)

    axes = np.array([plt.subplot(gs[0:2]),
                     plt.subplot(gs[6:13]), plt.subplot(gs[17:24]),
                     plt.subplot(gs[29:30])])

    ax = axes[0]
    ax.plot(d_sample["Time"], d_sample["timeseries"], color=h.long_pal[7])
    ax.set_xlim(-0.4, 11)
    ax.set_xticks(np.arange(0, 11))
    ax.set_xticklabels(np.arange(-10, 1))
    ax.set_xlabel("Lead time", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False,
                bottom=False)

    ax = axes[1]
    for i, row in d_raw.iterrows():
        ax.scatter(row["Time"], row["number"], s=dot_size, #20*row["value"],
                    c=h.ews_colours[row["variable"]], alpha=row["value_norm"])
    ax.set_xlim(-0.6, 11)
    ax.set_xticks(np.arange(0, 11))
    ax.set_xticklabels(np.arange(-10, 1))
    ax.set_xlabel("Lead time", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_ylim(-0.5, len(signals))
    ax.set_yticks(np.arange(0,len(signals)))
    ax.set_yticklabels([h.sig_labels()[s] for s in signals],
                       color=pc["axis_text"])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False,
                bottom=False)

    ax = axes[2]
    for i, row in d_weighted.iterrows():
        ax.scatter(row["Time"], row["variable"], s=dot_size, #50*row["value"],
                   c=h.ews_colours[row["variable"]], alpha=row["value_norm"])
    ax.set_ylim(-0.5, len(signals))
    ax.set_xlim(-0.4, 11)
    ax.set_xticks(np.arange(0, 11))
    ax.set_xticklabels(np.arange(-10, 1))
    ax.set_xlabel("Lead time", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_yticks(np.arange(0,len(signals)))
    ax.set_yticklabels([h.sig_labels()[s] for s in signals],
                       color=pc["axis_text"])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False,
                bottom=False)

    ax = axes[3]
    for i, row in d_ews.iterrows():
        ax.scatter(row["Time"], row["is_test"], s=dot_size, #200*row["pr"],
                   c=h.long_pal[row["pred"]], alpha=row["pr"])
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([1])
    ax.set_yticklabels(["Emergence risk"], color=pc["axis_text"])
    ax.set_xlim(-0.4, 11)
    ax.set_xticks(np.arange(0, 11))
    ax.set_xticklabels(np.arange(-10, 1))
    ax.set_xlabel("Lead time", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    fig.tight_layout()

    return fig, axes



fig_ews, _ = plot_early_warning_system()
fig_ews.savefig("./fig_demo_lhs.svg", transparent=True)


## Learning algorithm figure

def plot_learning_algorithm():
    np.random.seed(21)

    axes_label_fontsize = 8
    dot_size=70
    pc = h.plot_colours

    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    fig = plt.figure(figsize=(3.,6.5))
    gs = GridSpec(30,1)

    axes = np.array([plt.subplot(gs[0:2]),
                     plt.subplot(gs[6:9]), plt.subplot(gs[18:21]),
                     plt.subplot(gs[26:30])])



    ax = axes[0]
    a = params_df[params_df["R0_f"] !=1].sample(n=10, axis=0)
    ax.scatter(np.log10(a["N_i"]), a["R0_i"], color=h.long_pal[0], label="Not emerging")
    b = params_df[params_df["R0_f"] ==1].sample(n=10, axis=0)
    ax.scatter(np.log10(b["N_i"]), b["R0_i"], color=h.long_pal[1], label="Emerging")
    ax.set_xlabel("Population size", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_ylabel("Initial $R_0$", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(np.log10(5e4), np.log10(5e6))
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([np.log10(5e4), np.log10(5e5), np.log10(5e6)])
    ax.set_xticklabels(["$5\\times 10^4$", "$5\\times 10^5$", "$5\\times 10^6$"])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    #ax.legend(loc="right",bbox_to_anchor=(1.95, 0.7), handletextpad=0.0)

    df = ews_data[ews_data["model"].isin(np.concatenate([b.index,a.index]))]
    df.index = df["Time"]/365 - 10
    dfg = df.groupby("model")


    ax = axes[1]
    for i, g in dfg:
        ax.plot(g["timeseries"], color=h.long_pal[int(g.iloc[0, :].loc["is_test"])])

    ax.set_xlim(-0.4, 11)
    ax.set_xticks(np.arange(0, 11))
    ax.set_xticklabels(np.arange(-10, 1))
    ax.set_xlabel("Lead time", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False, bottom=False)

    ax = axes[2]
    for i, g in dfg:
        gg = g[g.index > 0].sample(5)
        ax.scatter(gg["df"], gg["is_test"],
                   color=h.long_pal[int(g.iloc[0, :].loc["is_test"])])

    x_values = np.arange(-1.2, 2.2, 0.1)
    ax.plot(x_values, h.lr_emergence_risk(x_values), color=h.long_pal[3])
    ax.set_xlim(-1.5,2.5)
    ax.set_ylim(-0.2,1.2)
    ax.set_xlabel("Weighted sum of EWS", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_ylabel("Emergence risk", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False, bottom=False)

    ax = axes[3]

    for year in np.arange(1,10,2):
        ews_single_year = ews_data[ews_data["Time"] ==times[times>(20-year)*365].min()]
        fpr, tpr, thresholds = metrics.roc_curve(ews_single_year["is_test"], ews_single_year["df"])
        roc_auc =roc_auc_score(ews_single_year["is_test"], ews_single_year["df"])
        #df_c = thresholds[np.argmin(fpr - tpr)]
        ax.plot(fpr, tpr, label= str(year),
                  color=h.long_pal[3],
                  alpha= 1- year/12,
                linestyle="-", linewidth=1.5)


    ed = ews_data[ews_data["Time"] > 10 * 365]
    fpr, tpr, thresholds = metrics.roc_curve(ed["is_test"], ed["df"])
    roc_auc = roc_auc_score(ed["is_test"], ed["df"])
    # df_c = thresholds[np.argmin(fpr - tpr)]
    ax.plot(fpr, tpr, label=str("All"),
            color=h.long_pal[5],
            alpha=1,
            linestyle="-", linewidth=1.5)


    ax.set_xlabel("False positive rate", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_ylabel("True positive rate", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.legend(title="Lead time", frameon=False, fontsize=axes_label_fontsize,
              title_fontsize=axes_label_fontsize, ncol=2,
              loc='lower left', bbox_to_anchor=(0.6, -0.2))
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0,1)
    ax.set_yticks([0., 0.5, 1.0])
    ax.set_xticks([0., 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False, bottom=False)

    fig.tight_layout()
    return fig, axes

fig_learn, _ = plot_learning_algorithm()
fig_learn.savefig("./fig_demo_rhs.svg", transparent=True)


# fig.tight_layout()

# fig.savefig("./fig_demonstration.png")