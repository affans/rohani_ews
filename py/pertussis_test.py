import pandas as pd
import ews
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import py.helper as h
from sklearn import metrics
import py.ews_logistic_regression as elr
import pomegranate as pm




# EWS for pertussis
lw = "-lightweight"

folder = "./data/backup-01-14" #+ lw # change from zenodo
use_incidence = True
years=(1980, 2000)
significance_level=0.05


# Select signals to be used in learning:
signals = h.signals()

# Read in trained coefficients and thresholds:
coefs_best = pd.read_csv(folder + "/ews_weights" +
                         h.incidence_filepath(use_incidence) + ".csv",
                         index_col=0, header=None, squeeze=True)
# coefs_best["mean"] = 0.0

opt_thresh = pd.read_csv(folder + "/optimum_thresholds" +
                         h.incidence_filepath(use_incidence) + ".csv",
                         index_col=0, header=None, squeeze=True)
opt_thresh["Emergence risk"] = h.lr_emergence_risk(opt_thresh["all"])


def get_annual_data(use_incidence=True, multiplier=10**5):
    tdf, states_list, state_names, us_dem = h.read_pertussis_files()

    tdf = tdf[tdf.index > 1980]
    tdg = tdf.groupby("YEAR")[states_list].sum()

    if use_incidence:
        for s in states_list:
            n = h.get_pop_size(state_names.loc[s.replace(".", " "),
                                               "Abbreviation"], tdg.index,
                               us_dem)
            tdg[s] = (tdg[s] / n)  * multiplier

    return tdg

def do_gmm(tdg):
    tdg["YEAR"] = tdg.index
    tdm = tdg.melt(id_vars="YEAR")

    d1 = pm.ExponentialDistribution(0.51)
    d2 = pm.ExponentialDistribution(0.5)
    model = pm.GeneralMixtureModel([d1, d2], weights=[0.5, 0.5])
    model.fit(tdm["value"].values,
              stop_threshold=0.000001, inertia=0.)

    # todo: inefficient method of calculating threshold
    for i in np.arange(0, 10, 0.001):
        if model.predict(i)[0] == 0:
            threshold = i

    # todo: exact method of calculating. Unclear how to return weights from gmm
    l = np.array([model.distributions[0].parameters[0],
                  model.distributions[1].parameters[0]])
    w = (model.probability([0])[0] - l[1]) / (l[0] - l[1])
    w = np.array([w,1-w])
    threshold = 1/(l[1]-l[0])*np.log(w[1]*l[1]/(w[0]*l[0]))



    tdm_f = tdm[tdm["value"] > threshold]

    # first outbreak year
    f_outbreak_year = tdm_f.groupby("variable")["YEAR"].min()


    return f_outbreak_year, model, threshold


def plot_states_annual(ews_data, f_outbreak_year, states):
    opt_thresh = pd.read_csv(folder + "/optimum_thresholds" +
                             h.incidence_filepath(use_incidence) + ".csv",
                             index_col=0, header=None, squeeze=True)
    opt_thresh["Emergence risk"] = h.lr_emergence_risk(opt_thresh["all"])

    df_colour = "blue"
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 7.5))

    for i, s in enumerate(states):
        ax = axes[i // 4, i % 4]
        ax.plot(tdg["YEAR"], tdg[s], color="blue", alpha=0.5)
        ax.set_title(s, fontsize=10)
        ax.set_ylim(0, 20)
        df_ts = ews_data[ews_data["state"] == s]
        # ax.fill_between(df_ts.index, 0, df_ts["pred"]*20, alpha=0.2)

        try:
            ax.axvline(f_outbreak_year[s], c="red", alpha=0.7)
        except KeyError:
            pass

        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=2, trim=True, right=False, left=True)
        ax1.axhline(opt_thresh["Emergence risk"], color=df_colour)
        ax1.plot(df_ts["Emergence risk"], "-", color=df_colour)

        if i % 4 == 3 or i == len(states) - 1:
            ax1.set_ylabel("Emergence\n risk")

    fig.tight_layout()


def get_rolling_lr(use_incidence=True):
    def lr_wrapper(years=(1980, 2000)):
        lr = h.get_lr_pertussis(use_incidence=use_incidence, years=years,
                                significance_level=0.05)
        lr["year_s"] = years[0]
        lr["year_f"] = years[1]

        print(years)
        return lr

    test = [lr_wrapper((i, j)) for i in range(1980, 1981)
            for j in range(i + 1, 2001)]
    # tt = [item for sublist in test for item in sublist]

    test2 = pd.concat(test, axis=0)
    test2["state"] = test2.index
    test2 = test2.reset_index(drop=True)

    return test2

lr_rolling = get_rolling_lr(use_incidence=True)


tdg = get_annual_data(use_incidence=True, multiplier=10**5)
f_outbreak_year, model, threshold = do_gmm(tdg)

# get EWS data from other file
def plot_shifted_results(ews_data, lr_rolling, folder, use_incidence=True,
                         figsize=(4, 4.1)):
    def shift_year(x):
        try:
            x["shifted_year"] = x["Year"] - f_outbreak_year[x.name]
            print(x.name, "ok")
        except KeyError:
            x["shifted_year"] = np.nan
            print(x.name, "nan")
        return x

    opt_thresh = pd.read_csv(folder + "/optimum_thresholds" +
                             h.incidence_filepath(use_incidence) + ".csv",
                             index_col=0, header=None, squeeze=True)
    opt_thresh["Emergence risk"] = h.lr_emergence_risk(opt_thresh["all"])

    ews_data = ews_data.groupby("state").apply(shift_year)
    ews_data["pred1"] = ews_data["Emergence risk"] > 0.70
    e2g = ews_data.groupby(["shifted_year"])["pred"].mean()
    e2 = ews_data[ews_data["shifted_year"].between(-50, 10)]

    # todo: fix this
    lr_rolling["Year"] = lr_rolling["year_f"]
    t2 = lr_rolling.groupby("state").apply(shift_year)
    lr2g = t2.groupby(["shifted_year"])["emerging"].mean()

    axes_label_fontsize = 8
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    ax = axes[0]
    e2g = ews_data.groupby(["shifted_year"])["pred"].mean()
    opt_thresh_str = str(np.round(opt_thresh["Emergence risk"], 2))
    ax.plot(e2g, c="blue", label="c = " + opt_thresh_str + " (optimum)")

    for thresh in [0.6, 0.65, 0.7]:
        ews_data["pred1"] = ews_data["Emergence risk"] > thresh
        e2g = ews_data.groupby(["shifted_year"])["pred1"].mean()
        ax.plot(e2g, c="black", alpha=0.3 + 4 * (0.7 - thresh),
                label="c = " + thresh)

    ax.plot(lr2g, c="red", label="Linear regression")

    ax.set_xlim(-30, 0)
    ax.legend(loc="lower right", ncol=2, borderaxespad=0., frameon=False,
              fontsize=axes_label_fontsize)
    ax.set_xlabel("Lead time before first large outbreak (years)")
    ax.set_ylabel("Fraction above threshold")

    ax = axes[1]
    g2 = ews_data.groupby(["shifted_year"])["pred"].count()
    ax.plot(g2 / 12,
            c="blue")  # divide by number of months to get number of states
    ax.set_xlim(-30, 0)
    ax.set_xlabel("Lead time before first large outbreak (years)")
    ax.set_ylabel("Number of states in denominator")


    fig.tight_layout()


fig_shift = plot_shifted_results(ews_data=ews_data, lr_rolling=lr_rolling,
                                 folder=f, use_incidence=ui)

fig.savefig("./pertussis_frac_above_threshold.pdf")


# Make GMM plot

# Linear regression
ui = True
