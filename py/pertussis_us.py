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



# Options for script
# use_incidence = True
# use_cross_val = True
# calculate_ews = False
# group_by_time_from_bp = False
# cut_off = False
# folder = "./data/backup-09-08-1"
#folder = "./data/backup-16-07-1-pertussis"
#



def get_pertussis_ews(folder, lr_df, use_incidence=True, use_cross_val=True,
                      pertussis_fit_signals=None, start_year=1980):

    # Read cross-validation results
    if use_cross_val:
        w_min, c_min = h.read_cross_val(folder, use_incidence)
    else:
        w_min = 156.
        c_min = 0.0001

    agg = 4
    w = w_min//agg
    print(c_min)

    tdf, states_list, state_names, us_dem = h.read_pertussis_files()

    # Select signals to be used in learning:
    signals = h.signals()

    # Read in trained coefficients and thresholds:
    coefs_best = pd.read_csv(folder + "/ews_weights" +
                             h.incidence_filepath(use_incidence) + ".csv",
                             index_col=0, header=None, squeeze=True)
    #coefs_best["mean"] = 0.0
    df_c_dict = pd.read_csv(folder + "/optimum_thresholds" +
                            h.incidence_filepath(use_incidence) + ".csv",
                            index_col=0, header=None, squeeze=True)



    def get_state_ews(tdf, threshold=0):

        # can replace this with function defined in helper
        def get_df(row):
            return sum([row[signal] * coefs_best[signal]
                        for signal in signals]) + coefs_best.loc["intercept"]

        tdf = tdf[tdf.index > start_year]



        ews_states = []
        for state in states_list:

            x = tdf[state]

            if use_incidence:
                n = h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                   "Abbreviation"], x.index,
                                   us_dem)
                x *= 1e0/n

            ews_df = pd.DataFrame(
                        ews.get_ews(x, windowsize=w, ac_lag=1, se=False,
                                    kc=False, method="new",
                                    mv_method="exp"))
            ews_df["Time"] = x.index
            ews_df["ac2"] = x.ewm(w).corr(x.shift(2), "pearson")
            ews_df["sd_convexity"] = ews_df["standard_deviation"] - \
                                ews_df["standard_deviation"].shift(1)
            ews_df = ews_df.replace([np.inf, -np.inf], np.nan)
            ews_df = ews_df.dropna()
            # ews_df["Decision function"] = ews_df.apply(axis=1, func=get_df)
            ews_df["Decision function"] = ews_df[signals].dot(coefs_best[signals]) \
                        + coefs_best["intercept"]

            ews_df["Emergence risk"] = h.lr_emergence_risk(
                ews_df["Decision function"])
            ews_df["pred"] = ews_df.apply(axis=1, func=lambda j: int(
                j["Decision function"] > threshold))
            ews_df["state"] = state
            ews_states += [ews_df]

        return states_list, pd.concat(ews_states, axis=0)

    # Normalised confusion matrix and AUC:
    def get_stats(y_true, df, threshold):
        auc = metrics.roc_auc_score(y_true, df)
        cm = confusion_matrix(y_true, df > threshold)
        counts = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / counts
        return pd.Series({"auc": auc, "fpr": cm[0, 1], "tpr": cm[1, 1],
                          "negatives": counts[0, 0], "positives": counts[1, 0]})


    _, ews_state_data = get_state_ews(tdf, threshold=df_c_dict["all"])

    ews_state_data["is_test"] = ews_state_data.apply(lambda x:
                                                     int(lr_df.loc[x["state"],
                                                                   "emerging"]),
                                                     axis=1)

    if pertussis_fit_signals is None:
        test_signals = signals
    else:
        test_signals = pertussis_fit_signals

    # training filter
    esd_filtered = ews_state_data[ews_state_data["Time"].between(1980, 2000)]
    coefs_test, lr_clf_test = elr.ews_logistic_regression(esd_filtered,
                                                          standardise=False,
                                                          signals=test_signals,
                                                          do_pca=False,
                                                          # penalty="l1",
                                                          solver="liblinear")

    fpr, tpr, thresholds = metrics.roc_curve(esd_filtered["is_test"],
                                             lr_clf_test.decision_function(
                                                 esd_filtered[test_signals].values))

    auc_test = metrics.roc_auc_score(esd_filtered["is_test"],
                                     lr_clf_test.decision_function(
                                         esd_filtered[test_signals].values))

    df_c_test = thresholds[np.argmin(fpr - tpr)]
    fpr, tpr, thresholds = metrics.roc_curve(esd_filtered["is_test"],
                                             esd_filtered["Decision function"])
    df_c_best_pertussis = thresholds[np.argmin(fpr - tpr)]

    ews_state_data["Year"] = np.floor(ews_state_data["Time"])

    g = ews_state_data[ews_state_data["Time"] > 1970].groupby("Year")

    stats_signals = dict(zip(signals + ["timeseries"],
                             [g.apply(lambda x: get_stats(x["is_test"], x[signal],
                                                          df_c_dict[signal]))
                              for signal in signals + ["timeseries"]]))

    stats_signals["Best fit to simulated data"] = g.apply(
        lambda x: get_stats(x["is_test"], x["Decision function"], df_c_dict["all"]))
    stats_signals["Best fit to pertussis"] = g.apply(
        lambda x: get_stats(x["is_test"],
                            lr_clf_test.decision_function(x[test_signals].values),
                            df_c_test))
    stats_signals["Opt. threshold for pertussis"] = g.apply(
        lambda x: get_stats(x["is_test"], x["Decision function"],
                            df_c_best_pertussis))

    if pertussis_fit_signals is not None:
        ret_test = df_c_test
        ret_clf = lr_clf_test
    else:
        ret_test = None
        ret_clf = None

    test_coefs = pd.Series(dict(zip(test_signals ,lr_clf_test.coef_[0])))
    test_coefs["intercept"] = lr_clf_test.intercept_[0]
    #stats_signals.to_csv(folder + "/optimum_thresholds" +
    #                        h.incidence_filepath(use_incidence) + ".csv")

    return stats_signals, ews_state_data, states_list, auc_test, df_c_test, \
           test_coefs


# Plot AUC through time for pertussis
def plot_stats(df, plotted_signals, by_time_from_bp=False, figsize=(8, 4)):
    """
    Plot AUC and true/false postive rates for various signals
    :param df: Dict of data frames with columns "AUC", "tpr", and "fpr".
    :param plotted_signals: signals (dict indices) to be plotted
    :param by_time_from_bp: boolean, if true time series are grouped by time
    from breakpoint.
    :param figsize: Figure size
    :return:
    """

    fig, axes = plt.subplots(2, 2, sharex="all", figsize=figsize)
    for signal in plotted_signals:
        if signal == "Opt. threshold for pertussis":
            axes[0, 0].plot(df[signal]["auc"], linestyle=(0, (5, 10)))
        else:
            axes[0, 0].plot(df[signal]["auc"])
        axes[0, 0].set_ylabel("AUC")
        axes[0, 0].set_ylim(0.2, 1,)
        axes[1, 0].plot(df[signal]["tpr"], label=signal)
        axes[1, 0].set_ylabel("True positive rate")
        axes[1, 0].set_ylim(0., 1,)
        axes[1, 1].plot(df[signal]["fpr"])
        axes[1, 1].set_ylabel("False positive rate")
        axes[1, 1].set_ylim(0., 1,)
        axes[0, 1].plot(df[signal]["tpr"]-df[signal]["fpr"],
                        label=signal)
        axes[0, 1].set_ylabel("TPR- FPR")
        axes[0, 1].set_ylim(-0.2, 0.8)
    axes[1, 0].legend(loc=0)
    if by_time_from_bp:
        axes[1, 0].set_xlabel("Years since breakpoint")
        axes[1, 1].set_xlabel("Years since breakpoint")
    else:
        axes[1, 0].set_xlabel("Year")
        axes[1, 1].set_xlabel("Year")
    for ax in axes.flatten():
        ax.axvline(2000, linestyle="--", color="black", alpha=0.4)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    fig.tight_layout(rect=(0, 0, 1, 1))
    return fig, axes


def get_figure_multistate(data, lr_df,states_list, folder, use_incidence=True,
                          figsize=(8, 4), outbreak_year=None):

    threshold = pd.read_csv(folder + "/optimum_thresholds" +
                            h.incidence_filepath(use_incidence) + ".csv",
                            index_col=0, header=None, squeeze=True)["all"]

    # state names and abbreviations
    state_names = pd.read_csv("./data/states.csv", index_col=0)
    # population sizes:
    us_dem = pd.read_csv("./data/journal.pbio.1002172.s004.CSV").groupby(
        "state")

    tdf = pd.read_csv("./data/pertussis.51.12.csv", header=0, sep=",")
    tdf["time"] = tdf.YEAR + tdf.MONTH / 12.
    tdf = (tdf.fillna(method="ffill") + tdf.fillna(method="bfill"))/2
    tdf = tdf.set_index("time")


    ts = dict()
    for state in states_list:

            npop = h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem)
            if use_incidence:
                ts[state] = np.log10(1e0* (tdf[state].copy() + 1) / npop)
            else:
                ts[state] = np.log10(tdf[state].copy() + 1)

    x = np.arange(1980, 2001)
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    fig = plt.figure(figsize=figsize)
    a = 0 if len(states_list)%4 == 0 else 1
    gs = GridSpec((len(states_list)//4)+a, 4)
    axes = np.array([plt.subplot(gs[i//4, i%4])
                     for i in range(len(states_list))])

                    # if 4*i +j < len(states_list)
                    # for j in range(0, 4)
                    # for i in range(0, 3*(len(states_list)//4), 3)])

    for i, state in enumerate(states_list):
        if lr_df.loc[state, "emerging"]:
            df_colour = h.pal[2]
        else:
            df_colour = h.pal[0]
        pvalue = lr_df.loc[state, "p_value"]
        if pvalue < 1/1000:
            pvalue = "p < 0.001"
        elif pvalue > 1 - 1/1000:
            pvalue = "p > 0.999"
        else:
            pvalue = "p = " + str(np.round(pvalue, 3))

        df = data[data["state"] == state]["Emergence risk"]
        ax = axes[i]
        ax.plot(ts[state])
        ax.plot(x, lr_df.loc[state, "intercept"] + x*lr_df.loc[state, "coef"],
                color=h.pal[4])

        ax.set_xlim(1975, 2010)
        ax.set_title(letters[i]+") " + state_names.loc[state.replace(".", " "),
                        "Abbreviation"] + " " + pvalue,
                     loc="left", fontsize=10)

        if use_incidence:
            ax.set_ylim(-7, -3)
            yticks = np.arange(-7,-2)
            ax.set_yticks(yticks)
            ax.set_yticklabels(["$10^{" + str(i) + "}$" for i in yticks])
        else:
            ax.set_ylim(0, 4)
        is_emerging = np.array(
            [j > h.lr_emergence_risk(threshold) for j in df])
        ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[0] +
                        is_emerging * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        color=df_colour, alpha=0.3, edgecolor="black")

        ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
        ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)
        if i < len(states_list)-4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year")
        if i % 4 != 0:
            ax.set_yticklabels([])
        else:
            if use_incidence:
                ax.set_ylabel("Incidence")
            else:
                ax.set_ylabel("Cases")

        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=2, trim=True, right=False, left=True)
        ax1.axhline(h.lr_emergence_risk(threshold), color=df_colour)
        ax1.plot(df, "-", color=df_colour)
        if i % 4 == 3 or i == len(states_list) - 1:
            ax1.set_ylabel("Emergence\n risk")


        if outbreak_year is not None:
            if state in outbreak_year.index:
                ax.axvline(outbreak_year[state], color="black", alpha=0.8,
                           linestyle="--")

    fig.tight_layout()
    return fig, axes


def plot_new(df, lr_df, plotted_signals, figsize=(8, 6), use_incidence=True,
             axes_label_fontsize=10):
    pc = h.plot_colours

    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    # state names and abbreviations
    state_names = pd.read_csv("./data/states.csv", index_col=0)
    # population sizes:
    us_dem = pd.read_csv("./data/journal.pbio.1002172.s004.CSV").groupby(
        "state")

    tdf = pd.read_csv("./data/pertussis.51.12.csv", header=0, sep=",")
    states_list = tdf.columns[2:]
    tdf["time"] = tdf.YEAR + tdf.MONTH / 12.
    tdf = (tdf.fillna(method="ffill") + tdf.fillna(method="bfill"))/2
    tdf = tdf.set_index("time")


    us_pop = pd.DataFrame([h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem) for state in states_list]).sum(axis=0)

    ts = dict()
    states = ["California", "Massachusetts", "Louisiana"]
    for state in states:

            npop = h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem)
            if use_incidence:
                ts[state] = np.log10(1e0 * (tdf[state].copy() + 1) / npop)
            else:
                ts[state] = np.log10(tdf[state].copy() + 1)
    if use_incidence:
        us_ts = np.log10(1e0 * (tdf.sum(axis=1) + 1) / us_pop)
    else:
        us_ts = np.log10(tdf.sum(axis=1) + 1)


    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 6)
    axes = np.array([plt.subplot(gs[0:2, 0:6]),
                     plt.subplot(gs[2:4, 0:2]), plt.subplot(gs[2:4, 2:4]),
                     plt.subplot(gs[2:4, 4:6]),

                     plt.subplot(gs[4:6, 0:3]), plt.subplot(gs[4:6, 3:6])])


    x = np.arange(1980,2001)

    ax = axes[0]
    ax.plot(us_ts, color=h.long_pal[7])
    ax.set_title("a) Contiguous USA", loc="left",
                 color=pc["axis_text"], fontsize=axes_label_fontsize)
    ax.set_xlim(1950, 2010)
    if use_incidence:
        y_loc = 0.55-5
        yticks = np.array([-5,-4.75,-4.5,-4.25])
        ax.set_yticks(yticks)
        ax.set_yticklabels(["$10^{"+k+"}$" for
                            k in ["-5.00","-4.75","-4.50","-4.25"]])

        #ax.set_ylabel("Monthly incidence (log10)")
    else:
        y_loc = 3.9
        #ax.set_ylabel("Monthly cases (log10)")
    ax.text(1965, y_loc, "declining\nphase", color=pc["axis_text"], fontsize=10,
            horizontalalignment='center',
            verticalalignment='center',)
    ax.text(1990, y_loc, "emerging\nphase", fontsize=10, color=pc["axis_text"],
            horizontalalignment='center',
            verticalalignment='center',)
    ax.text(2005, y_loc, "endemic\nphase", fontsize=10, color=pc["axis_text"],
            horizontalalignment='center',
            verticalalignment='center',)

    letters = ["b","c","d"]
    for i, state in enumerate(states):
        ax = axes[1+i]
        ax.plot(ts[state], label="_nolegend_",color=h.long_pal[7])
        ax.plot(x, lr_df.loc[state, "intercept"] + x*lr_df.loc[state, "coef"],
                color=h.long_pal[3], label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_title(letters[i]+") " + state.replace(".", " "), loc="left",
                     color=pc["axis_text"], fontsize=axes_label_fontsize)
        if use_incidence:
            ax.set_ylim(-7, -4)
            yticks = np.arange(-7,-3)
            ax.set_yticks(yticks)
            ax.set_yticklabels(["$10^{" + str(i) + "}$" for i in yticks])
        else:
            ax.set_ylim(0, 3)


    for i, signal in enumerate(plotted_signals):
        ax = axes[4]
        ax.plot(df[signal]["tpr"], color=h.ews_colours[signal],
                label=signal if signal != "mean" else "Mean")
        ax.plot(df[signal]["fpr"], color=h.ews_colours[signal],
                        linestyle="--", label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_ylabel("Fraction positive",color=pc["axis_text"],
                     fontsize=axes_label_fontsize)
        ax.set_title("e)", loc="left", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)

        ax = axes[5]
        if signal == "Opt. threshold for pertussis":
            ax.plot(df[signal]["auc"], linestyle=(0, (5, 10)),
                            color=h.ews_colours[signal], label="_nolegend_")
        else:
            ax.plot(df[signal]["auc"], color=h.ews_colours[signal],
                    label="_nolegend_")
        ax.set_ylabel("AUC",color=pc["axis_text"],
                     fontsize=axes_label_fontsize)
        ax.set_ylim(0.2, 1)
        ax.set_xlim(1975, 2010)
        ax.set_title("f)", loc="left", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)

    for ax in axes:
        ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
        ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    fig.legend(bbox_to_anchor=(0.5, 0.0, 0., 0.), loc="lower center",
               ncol=2, borderaxespad=0., frameon=False,
               fontsize=axes_label_fontsize)

    if use_incidence:
        ylab = "Monthly incidence (log10)"
    else:
        ylab = "Monthly cases (log10)"

    fig.text(0.02, 0.66, ylab, va='center', rotation='vertical',
             fontsize=axes_label_fontsize, color=pc["axis_text"])
    fig.tight_layout(rect=(0.02, 0.05, 1, 1))
    return fig, axes


def plot_all_ews(df, figsize=(7, 4), lr_auc=None):
    sig_comb = ["Best fit to simulated data",
        # "Opt. threshold for pertussis",
         "Best fit to pertussis"]
    plotted_signals = sig_comb +  h.signals()


    sig_labels = h.sig_labels()
    for i in sig_comb:
        sig_labels[i] = i



    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 6)
    axes = np.array([plt.subplot(gs[0:6, 0:6])])


    # x = np.arange(1980,2001)

    for i, signal in enumerate(plotted_signals):
        ax = axes[0]
        if signal == "Opt. threshold for pertussis":
            ax.plot(df[signal]["auc"], linestyle=(0, (5, 10)),
                            color=h.ews_colours[signal], label=sig_labels[signal])
        else:
            ax.plot(df[signal]["auc"], color=h.ews_colours[signal],
                    label=sig_labels[signal])
        ax.set_ylabel("AUC")
        ax.set_xlabel("Year")
        ax.set_ylim(0.2, 1)
        ax.set_xlim(1975, 2010)

    if lr_auc is not None:
        auc_plt = lr_auc.loc[1980]
        ax.plot(auc_plt, c="black", label="Linear regression")

    for ax in axes:
        ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
        ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    fig.legend(bbox_to_anchor=(0.825, 0.5, 0, 0),
                          loc="center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 0.7, 1))
    return fig, axes


def plot_new2(df, lr_df, plotted_signals, figsize=(8, 6), use_incidence=True,
             axes_label_fontsize=10):
    pc = h.plot_colours

    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    # state names and abbreviations
    state_names = pd.read_csv("./data/states.csv", index_col=0)
    # population sizes:
    us_dem = pd.read_csv("./data/journal.pbio.1002172.s004.CSV").groupby(
        "state")

    tdf = pd.read_csv("./data/pertussis.51.12.csv", header=0, sep=",")
    states_list = tdf.columns[2:]
    tdf["time"] = tdf.YEAR + tdf.MONTH / 12.
    tdf = (tdf.fillna(method="ffill") + tdf.fillna(method="bfill"))/2
    tdf = tdf.set_index("time")


    us_pop = pd.DataFrame([h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem) for state in states_list]).sum(axis=0)

    ts = dict()
    states = ["Massachusetts", "Louisiana"]
    for state in states:

            npop = h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem)
            if use_incidence:
                ts[state] = np.log10(1e0 * (tdf[state].copy() + 1) / npop)
            else:
                ts[state] = np.log10(tdf[state].copy() + 1)
    if use_incidence:
        us_ts = np.log10(1e0 * (tdf.sum(axis=1) + 1) / us_pop)
    else:
        us_ts = np.log10(tdf.sum(axis=1) + 1)


    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 6)
    axes = np.array([plt.subplot(gs[0:2, 0:6]),
                     plt.subplot(gs[2:4, 0:3]),
                     plt.subplot(gs[2:4, 3:6]),

                     plt.subplot(gs[4:6, 0:3]), plt.subplot(gs[4:6, 3:6])])


    x = np.arange(1980,2001)

    ax = axes[0]
    ax.plot(us_ts, color=h.long_pal[7])
    ax.set_title("a) Contiguous USA", loc="left",
                 color=pc["axis_text"], fontsize=axes_label_fontsize)
    ax.set_xlim(1950, 2010)
    if use_incidence:
        y_loc = 0.55-5
        yticks = np.array([-5,-4.75,-4.5,-4.25])
        ax.set_yticks(yticks)
        ax.set_yticklabels(["$10^{"+k+"}$" for
                            k in ["-5.00","-4.75","-4.50","-4.25"]])

        #ax.set_ylabel("Monthly incidence (log10)")
    else:
        y_loc = 3.9
        #ax.set_ylabel("Monthly cases (log10)")
    ax.text(1965, y_loc, "declining\nphase", color=pc["axis_text"], fontsize=8,
            horizontalalignment='center',
            verticalalignment='center',)
    ax.text(1990, y_loc, "emerging\nphase", fontsize=8, color=pc["axis_text"],
            horizontalalignment='center',
            verticalalignment='center',)
    ax.text(2005, y_loc, "endemic\nphase", fontsize=8, color=pc["axis_text"],
            horizontalalignment='center',
            verticalalignment='center',)

    letters = ["b","c","d"]
    for i, state in enumerate(states):
        ax = axes[1+i]
        tsp = ts[state]
        ax.plot(tsp[tsp.index < 2010], label="_nolegend_",color=h.long_pal[7])
        ax.plot(x, lr_df.loc[state, "intercept"] + x*lr_df.loc[state, "coef"],
                color=h.long_pal[3], label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_title(letters[i]+") " + state.replace(".", " "), loc="left",
                     color=pc["axis_text"], fontsize=axes_label_fontsize)
        if use_incidence:
            ax.set_ylim(-7, -4)
            yticks = np.arange(-7,-3)
            ax.set_yticks(yticks)
            ax.set_yticklabels(["$10^{" + str(i) + "}$" for i in yticks])
        else:
            ax.set_ylim(0, 3)


    for i, signal in enumerate(plotted_signals):
        ax = axes[3]
        ax.plot(df[signal]["tpr"], color=h.ews_colours[signal],
                label=signal if signal != "mean" else "Mean")
        ax.plot(df[signal]["fpr"], color=h.ews_colours[signal],
                        linestyle="--", label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_ylabel("Fraction positive",color=pc["axis_text"],
                     fontsize=axes_label_fontsize)
        ax.set_title("d)", loc="left", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)

        ax = axes[4]
        if signal == "Opt. threshold for pertussis":
            ax.plot(df[signal]["auc"], linestyle=(0, (5, 10)),
                            color=h.ews_colours[signal], label="_nolegend_")
        else:
            ax.plot(df[signal]["auc"], color=h.ews_colours[signal],
                    label="_nolegend_")
        ax.set_ylabel("AUC", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)
        ax.set_ylim(0.2, 1)
        ax.set_xlim(1975, 2010)
        ax.set_title("e)", loc="left", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)

    for ax in axes:
        ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
        ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    fig.legend(bbox_to_anchor=(0.5, 0.0, 0., 0.), loc="lower center",
               ncol=3, borderaxespad=0., frameon=False,
               fontsize=axes_label_fontsize)

    if use_incidence:
        ylab = "Monthly incidence (log10)"
    else:
        ylab = "Monthly cases (log10)"

    fig.text(0.02, 0.66, ylab, va='center', rotation='vertical',
             fontsize=axes_label_fontsize, color=pc["axis_text"])
    #fig.tight_layout(rect=(0.02, 0.05, 1, 1))
    fig.subplots_adjust(left=0.19, bottom=None, right=0.85, top=None,
                            wspace=None, hspace=0.7)

    return fig, axes


def plot_new3(df, lr_df, plotted_signals, data, figsize=(8, 6),
              use_incidence=True, axes_label_fontsize=10, outbreak_year=None):
    pc = h.plot_colours

    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    # state names and abbreviations
    state_names = pd.read_csv("./data/states.csv", index_col=0)
    # population sizes:
    us_dem = pd.read_csv("./data/journal.pbio.1002172.s004.CSV").groupby(
        "state")

    tdf = pd.read_csv("./data/pertussis.51.12.csv", header=0, sep=",")
    states_list = tdf.columns[2:]
    tdf["time"] = tdf.YEAR + tdf.MONTH / 12.
    tdf = (tdf.fillna(method="ffill") + tdf.fillna(method="bfill"))/2
    tdf = tdf.set_index("time")


    us_pop = pd.DataFrame([h.get_pop_size(state_names
                                          .loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem)
                           for state in states_list]).sum(axis=0)

    ts = dict()
    states = ["Massachusetts", "Louisiana"]
    for state in states:

            npop = h.get_pop_size(state_names.loc[state.replace(".", " "),
                                                  "Abbreviation"],
                                  tdf.index.values, us_dem)
            if use_incidence:
                ts[state] = np.log10(1e0 * (tdf[state].copy() + 1) / npop)
            else:
                ts[state] = np.log10(tdf[state].copy() + 1)
    if use_incidence:
        us_ts = np.log10(1e0 * (tdf.sum(axis=1) + 1) / us_pop)
    else:
        us_ts = np.log10(tdf.sum(axis=1) + 1)


    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2)
    axes = np.array([plt.subplot(gs[0, 0:2]),
                     plt.subplot(gs[1, 0]),
                     plt.subplot(gs[1, 1]),
                     plt.subplot(gs[2, 0:2]),
                     plt.subplot(gs[3, 0:2])])


    x = np.arange(1980,2001)

    ax = axes[0]
    ax.plot(us_ts, color=h.long_pal[7])
    ax.set_title("a) Contiguous USA", loc="left",
                 color=pc["axis_text"], fontsize=axes_label_fontsize)
    ax.set_xlim(1950, 2010)
    if use_incidence:
        y_loc = 0.55-5
        yticks = np.array([-5.0,-4.6, -4.2])
        ax.set_yticks(yticks)
        ax.set_yticklabels(["$10^{"+str(k)+"}$" for
                            k in yticks])
        ax.set_ylabel("Monthly incidence", fontsize=axes_label_fontsize, color=pc["axis_text"])
    else:
        y_loc = 3.9
        #ax.set_ylabel("Monthly cases (log10)")
    ax.text(1965, y_loc, "declining\nphase", color=pc["axis_text"], fontsize=8,
            horizontalalignment='center',
            verticalalignment='center',)
    ax.text(1990, y_loc, "emerging\nphase", fontsize=8, color=pc["axis_text"],
            horizontalalignment='center',
            verticalalignment='center',)
    ax.text(2005, y_loc, "endemic\nphase", fontsize=8, color=pc["axis_text"],
            horizontalalignment='center',
            verticalalignment='center',)

    letters = ["b","c","d"]
    for i, state in enumerate(states):
        ax = axes[1+i]
        tsp = ts[state]
        ax.plot(tsp[tsp.index < 2010], label="_nolegend_",color=h.long_pal[7])
        ax.plot(x, lr_df.loc[state, "intercept"] + x*lr_df.loc[state, "coef"],
                color=h.long_pal[3], label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_title(letters[i]+") " + state.replace(".", " "), loc="left",
                     color=pc["axis_text"], fontsize=axes_label_fontsize)
        if use_incidence:
            ax.set_ylim(-7, -4)
            yticks = np.arange(-7,-3)
            ax.set_yticks(yticks)
            ax.set_yticklabels(["$10^{" + str(i) + "}$" for i in yticks])
        else:
            ax.set_ylim(0, 3)

        if outbreak_year is not None:
            if state in outbreak_year.index:
                ax.axvline(outbreak_year[state], color=pc["axis_text"], alpha=1,
                           linestyle="--")                # ax.set_xticks([1980,f_outbreak_year[state], 2000])


        if lr_df.loc[state, "emerging"]:
            df_colour = h.pal[2]
        else:
            df_colour = h.pal[0]

        fill_data = data[data["state"] == state]
        ax.fill_between(fill_data.index, ax.get_ylim()[0], ax.get_ylim()[0] +
                        fill_data["pred"] * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        color=df_colour, alpha=0.3, edgecolor="black")




    for i, signal in enumerate(plotted_signals):
        ax = axes[3]
        ax.plot(df[signal]["tpr"], color=h.ews_colours[signal],
                label=signal if signal != "mean" else "Mean")
        ax.plot(df[signal]["fpr"], color=h.ews_colours[signal],
                        linestyle="--", label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_ylabel("Fraction positive",color=pc["axis_text"],
                     fontsize=axes_label_fontsize)
        ax.set_title("d)", loc="left", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)

        ax = axes[4]
        if signal == "Opt. threshold for pertussis":
            ax.plot(df[signal]["auc"], linestyle=(0, (5, 10)),
                            color=h.ews_colours[signal], label="_nolegend_")
        else:
            ax.plot(df[signal]["auc"], color=h.ews_colours[signal],
                    label="_nolegend_")
        ax.set_ylabel("AUC", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)
        ax.set_ylim(0.2, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(1975, 2010)
        ax.set_title("e)", loc="left", color=pc["axis_text"],
                     fontsize=axes_label_fontsize)

    for ax in axes:
        ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
        ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    fig.legend(bbox_to_anchor=(0.5, 0.0, 0., 0.), loc="lower center",
               ncol=2, borderaxespad=0., frameon=False,
               fontsize=axes_label_fontsize)

    if use_incidence:
        ylab = "Monthly incidence"
    else:
        ylab = "Monthly cases "

    fig.text(0.02, 0.66, ylab, va='center', rotation='vertical',
             fontsize=axes_label_fontsize, color=pc["axis_text"])
    #fig.tight_layout(rect=(0.02, 0.05, 1, 1))
    fig.subplots_adjust(left=0.19, bottom=None, right=0.85, top=None,
                            wspace=None, hspace=0.7)

    return fig, axes



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

def do_gmm(df_annual):
    df_annual["YEAR"] = df_annual.index
    tdm = df_annual.melt(id_vars="YEAR")

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


def plot_states_annual(df_annual, ews_data, f_outbreak_year, states, folder,
                       use_incidence=True):
    opt_thresh = pd.read_csv(folder + "/optimum_thresholds" +
                             h.incidence_filepath(use_incidence) + ".csv",
                             index_col=0, header=None, squeeze=True)
    opt_thresh["Emergence risk"] = h.lr_emergence_risk(opt_thresh["all"])

    df_colour = "blue"
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 7.5))

    for i, s in enumerate(states):
        ax = axes[i // 4, i % 4]
        ax.plot(df_annual["YEAR"], df_annual[s], color="blue", alpha=0.5)
        ax.set_title(s, fontsize=10)
        ax.set_ylim(0, 20)
        df_ts = ews_data[ews_data["state"] == s]
        # ax.fill_between(df_ts.index, 0, df_ts["pred"]*20, alpha=0.2)

        try:
            ax.axvline(f_outbreak_year[s], color="black", alpha=0.8,
                       linestyle="--")
        except KeyError:
            pass

        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=2, trim=True, right=False, left=True)
        ax1.axhline(opt_thresh["Emergence risk"], color=df_colour)
        ax1.plot(df_ts["Emergence risk"], "-", color=df_colour)

        # ax1.axhline(10**5*opt_thresh["mean"], color="green")
        # ax1.plot(10**5*df_ts["mean"], "-", color="green")
        if i % 4 == 3 or i == len(states) - 1:
            ax1.set_ylabel("Emergence\n risk")

    fig.tight_layout()




def get_rolling_lr(linear_regression_df, use_incidence=True):
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
    auc_lr = test2.groupby(["year_s","year_f"]).apply(lambda x:
                    metrics.roc_auc_score(linear_regression_df["emerging"],
                                          1-x["p_value"]))

    return test2, auc_lr


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
    sim_colour = h.ews_colours["Best fit to simulated data"]
    ax.plot(e2g, c=sim_colour, label="c = " + opt_thresh_str + " (optimum)")

    ews_data["pred1"] = ews_data["mean"] > opt_thresh["mean"]/5
    e2g = ews_data.groupby(["shifted_year"])["pred1"].mean()
    ax.plot(e2g, c="green", label="Mean")

    ax.plot(lr2g, c="red", label="Linear regression")

    ax.set_xlim(-30, 0)
    ax.legend(ncol=2, borderaxespad=0., frameon=False,
              fontsize=axes_label_fontsize)
    ax.set_xlabel("Lead time before first large outbreak (years)")
    ax.set_ylabel("Fraction above threshold")
    sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    ax = axes[1]
    g2 = ews_data.groupby(["shifted_year"])["pred"].count()
    ax.plot(g2 / 12,
            c="black")  # divide by number of months to get number of states
    ax.set_xlim(-30, 0)
    ax.set_xlabel("Lead time before first large outbreak (years)")
    ax.set_ylabel("Number of states in denominator")
    sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    fig.tight_layout()
    return fig


# EWS for pertussis
lw = "-lightweight"

f = "./data/backup-01-14" #+ lw # change from zenodo

for ui in [True]:
    linear_regression_df = h.get_lr_pertussis(use_incidence=ui,
                                              years=(1980, 2000),
                                              significance_level=0.05)

    stats_signals, ews_data, states,_,dt_p,p_coef = get_pertussis_ews(f,
                                                        linear_regression_df,
                                                        use_incidence=ui,
                                                        use_cross_val=True,
                                                        start_year=1980,)



    _,_,_,_,dt_p,p_coef = get_pertussis_ews(f, linear_regression_df,
                                            use_incidence=ui,
                                            use_cross_val=True,
                                            start_year=1980,
                                            pertussis_fit_signals=h.signals()[3:6])

    p_coef["c"] = dt_p
    p_coef.to_csv(f+"/pertussis_coefficients"+ h.incidence_filepath(ui) +".csv")


    lr_rolling, auc_lr = get_rolling_lr(linear_regression_df,
                                        use_incidence=True)



    tdg = get_annual_data(use_incidence=True, multiplier=10 ** 5)
    f_outbreak_year, model, threshold = do_gmm(tdg)

    # Plot outbreak sizes and GMM fit
    def plot_outbreak_sizes(tdg, model, threshold):
        tdm = tdg.melt(id_vars="YEAR")

        os_density = tdm["value"].astype(int).value_counts() / tdm.shape[0]
        os_size = np.arange(0, os_density.index.max(),0.5)
        os_theory = pd.Series(model.probability(os_size), index=os_size)

        os_small = os_density[os_density.index +0.5 < threshold]
        os_large = os_density[os_density.index+0.5 >= threshold]


        fig = plt.figure(figsize=(6, 3))
        gs = GridSpec(1, 1)
        axes = np.array([plt.subplot(gs[0, 0])])
        ax = axes[0]
        ax.scatter(os_small.index +0.5, np.log10(os_small),
                   label="Small outbreaks", color=h.pal[0])
        ax.scatter(os_large.index+0.5, np.log10(os_large),
                   label="Large outbreaks", color=h.pal[2])
        ax.plot(os_theory.index, np.log10(os_theory),
                color=h.pal[1],
                label="GMM fit")
        ax.axvline(threshold, linestyle="--", color="grey")
        ax.set_ylim(-5, 0)
        ax.set_yticks(np.arange(-5,1))
        ax.set_xlim(-10, 120)
        ax.set_xlabel("Annual incidence per $10^5$")
        ax.set_ylabel("Log$_{10}$ probability")
        ax.legend()
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        fig.tight_layout(rect=[0.02, 0, 1, 1])

        print("lo_threshold:", threshold)

        return fig, axes


    fig_os,_ = plot_outbreak_sizes(tdg,model,threshold)



    plot_states_annual(tdg, ews_data, f_outbreak_year, states=states[:16],
                       folder=f,use_incidence=True)

    # All ews plot
    fa, _ = plot_all_ews(stats_signals, lr_auc=auc_lr)

    # Figure 4 plot
    stats = ["Best fit to simulated data", "Best fit to pertussis", "mean"]

    plt.style.use('seaborn-ticks')

    fig00, _ = plot_new3(stats_signals, linear_regression_df,
                         plotted_signals=stats, data=ews_data,
                         use_incidence=ui, figsize=(3.6, 4.5),
                         axes_label_fontsize=8,
                         outbreak_year=f_outbreak_year)

    fig00.subplots_adjust(left=0.19, bottom=0.14, right=0.95, top=0.96,
                          wspace=0.5, hspace=1)



    # Pertussis states plots
    fs1, _ = get_figure_multistate(ews_data, lr_df=linear_regression_df,
                                   states_list=states[:16],
                                   folder=f, figsize=(8, 6),
                                   outbreak_year=f_outbreak_year)

    fs2, _ = get_figure_multistate(ews_data, lr_df=linear_regression_df,
                                   states_list=states[16:32],
                                   folder=f, figsize=(8, 6),
                                   outbreak_year=f_outbreak_year)

    fs3, _ = get_figure_multistate(ews_data, lr_df=linear_regression_df,
                                   states_list=states[32:],
                                   folder=f, figsize=(8, 7.5),
                                   outbreak_year=f_outbreak_year)

    fig_shift = plot_shifted_results(ews_data=ews_data, lr_rolling=lr_rolling,
                                     folder=f, use_incidence=ui,figsize=(4,6))



    fig00.savefig("./fig_pertussis_usa2" + h.incidence_filepath(ui) + ".png")
    fig00.savefig("./fig_pertussis_usa2" + h.incidence_filepath(ui) + ".pdf")




    fs1.savefig("./fig_pertussis_states1"+ h.incidence_filepath(ui) + ".png")
    fs2.savefig("./fig_pertussis_states2"+ h.incidence_filepath(ui) + ".png")
    fs3.savefig("./fig_pertussis_states3"+ h.incidence_filepath(ui) + ".png")
    fa.savefig("./fig_pertussis_all_ews" + h.incidence_filepath(ui) + ".pdf")

    fig_shift.savefig("./fig_shifted_ews"+ h.incidence_filepath(ui) + ".pdf")

    fig_os.savefig("./fig_pertussis_os"+h.incidence_filepath(ui)+".png")


# Leave one out test (crude inefficient implementation)
lw = "-lightweight"

f = "./data/backup-01-14" #+ lw # change from zenodo
ui = True
linear_regression_df = h.get_lr_pertussis(use_incidence=ui, years=(1980, 2000),
                                          significance_level=0.05)
signals = h.signals()

leave_oo = {}
for s in signals:
    pfs = [ss for ss in signals if ss != s]
    leave_oo[s] = get_pertussis_ews(f, linear_regression_df,
                                    use_incidence=ui, use_cross_val=True,
                                    pertussis_fit_signals=pfs)
    print(s)

auc_loo = pd.Series(dict((zip(signals, [leave_oo[s][3] for s in signals]))))
auc_loo = pd.Series(dict((zip(signals,
                              [leave_oo[s][0]["Best fit to pertussis"].loc[1997, "auc"] for s in signals]))))

# leave_oo[s][0]['Best fit to pertussis']["auc"][1995]

# sort by decreasing importance
auc_loo = auc_loo.sort_values()

add_one = {}
for i, s in enumerate(auc_loo.index):
    pfs = auc_loo.index.values[:(i+1)]
    add_one[s] = get_pertussis_ews(f, linear_regression_df,
                                   use_incidence=ui, use_cross_val=True,
                                   pertussis_fit_signals=pfs)
    print(s)

auc_ao = pd.Series(dict((zip(signals, [add_one[s][3] for s in signals]))))
auc_ao = pd.Series(dict((zip(signals,
                              [add_one[s][0]["Best fit to pertussis"].loc[1997, "auc"] for s in signals]))))

# add_one[s][0]['Best fit to pertussis']["auc"][1995]

def plot_leave_one_out(figsize=(6,6)):

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 6)
    axes = np.array([plt.subplot(gs[0:2, 0:3]),
                     plt.subplot(gs[0:2, 3:6]), plt.subplot(gs[2:4, 0:3]),
                     plt.subplot(gs[2:4, 3:6])])

    fig.text(0.1,0.92,"Fit to pertussis with one EWS\nleft out")
    fig.text(0.54,0.92,"Fit to pertussis with increasing\nnumber of EWS included")

    ax = axes[0]
    for i, s in enumerate(auc_loo.index):
        ax.plot(leave_oo[s][0]['Best fit to pertussis']["auc"],
                color=h.ews_colours[s], label=h.sig_labels()[s])

    ax.set_ylabel("AUC")
    ax.set_xlabel("Year")
    ax.set_title("a)", fontsize=10, loc="left")
    ax.set_ylim(0.2, 1)
    ax.set_xlim(1975, 2010)
    ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
    ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
    sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    ax = axes[1]
    for i, s in enumerate(auc_loo.index):
        ax.plot(add_one[s][0]['Best fit to pertussis']["auc"],
                color=h.ews_colours[s], label=h.sig_labels()[s])

    # ax.set_ylabel("AUC")
    ax.set_yticklabels([])
    ax.set_xlabel("Year")
    ax.set_title("c)", fontsize=10, loc="left")
    ax.set_ylim(0.2, 1)
    ax.set_xlim(1975, 2010)
    ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
    ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
    sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    x_text = [h.sig_labels()[s] for s in auc_loo.index]
    x_colours = [h.ews_colours[s] for s in auc_loo.index]

    ax = axes[2]
    for i, s in enumerate(auc_loo.index):
        #auc_plt = leave_oo[s][0]["Best fit to pertussis"].loc[1995, "auc"]
        auc_plt = auc_loo[s]
        ax.scatter(i, auc_plt, color=h.ews_colours[s])
    ax.set_ylim(0.2, 1)
    # ax.set_xlim(-0.5,len(x_text))
    ax.set_ylabel("AUC (1997)")
    ax.set_xlabel("EWS left out")
    ax.set_title("b)", fontsize=10, loc="left")
    ax.set_xticks(np.arange(0, len(x_text)))
    sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)
    ax.set_xticklabels(x_text, rotation=45, ha="right")


    ax = axes[3]
    for i, s in enumerate(auc_loo.index):
        auc_plt = add_one[s][0]["Best fit to pertussis"].loc[1995, "auc"]
        auc_plt = auc_ao[s]
        ax.scatter(i, auc_plt, color=h.ews_colours[s])
    ax.set_ylim(0.2, 1)
    ax.set_yticklabels([])
    ax.set_xlabel("EWS included (all to left)")
    ax.set_title("d)", fontsize=10, loc="left")
    ax.set_xticks(np.arange(0, len(x_text)))
    sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)
    ax.set_xticklabels(x_text, rotation=45, ha="right")

    fig.subplots_adjust(wspace=2, hspace=4)
    return fig, axes


fig_lo, _ = plot_leave_one_out()

fig_lo.savefig("./fig_leave_one_out_pertussis"+ h.incidence_filepath(ui) + ".png")




# Convexity test


lw = "" #""-lightweight"

### TEMP FIX
fp_root = f = "./data/backup-01-" #+ lw



# Get convexity + concavity results
def gp_wrapper(f, ui):

    lr_df = h.get_lr_pertussis(use_incidence=ui, years=(1980, 2000),
                               significance_level=0.05)
    d = get_pertussis_ews(f, lr_df, use_incidence=ui)[0]

    for k, v in d.items():
        v["ews"] = k

    df = pd.concat([v for k, v in d.items()])
    df["folder"] = f
    df["use_incidence"] = ui
    return df

convexity_stats = pd.concat([gp_wrapper(f, ui) for ui in [1]
                             for f in [fp_root + "14"+lw,
                                       fp_root + "14-concave"+lw,
                                       fp_root + "14-convex"+lw]])
convexity_stats.to_csv("./data/pertussis_convexity_stats.csv")


# convexity stats plot

def convexity_stats_plot(df, use_incidence=True,
                         benchmark=fp_root,
                         figsize=(8, 3)):

    def get_label(s):
        if "concave" in s:
            y = "concave "
        elif "convex" in s:
            y = "convex "
        else:
            y = ""
        return "Best fit to " + y +"simulated data"


    df = df[df["use_incidence"] == use_incidence]

    bm = df[(df["ews"] == "Best fit to pertussis") &
            (df["folder"] == benchmark)].copy()
    bm["label"] = bm["ews"]

    df = df[df["ews"] == "Best fit to simulated data"]
    df["label"] = df["folder"].apply(get_label)

    df = pd.concat([df, bm])
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2)
    axes = np.array([plt.subplot(gs[0,0]),
                     plt.subplot(gs[0,1])])



    for i, f in enumerate(df["label"].unique()):
        ax = axes[1]
        x = df[df["label"] == f]

        ax.plot(x["auc"], color=h.pal[i], label="_nolegend_")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.2, 1)
        ax.set_xlim(1975, 2010)
        ax.set_title("b)", loc="left")

        ax = axes[0]
        ax.plot(x["tpr"], color=h.pal[i], label=f)
        ax.plot(x["fpr"], color=h.pal[i],
                        linestyle="--", label="_nolegend_")
        ax.set_xlim(1975, 2010)
        ax.set_ylabel("Fraction positive")
        ax.set_title("a)", loc="left")

    for ax in axes:
        ax.axvspan(2000, 2012, facecolor='0.2', alpha=0.2)
        ax.axvspan(1950, 1980, facecolor='0.2', alpha=0.2)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)

    fig.legend(bbox_to_anchor=(0.5, 0.0, 0., 0.), loc="lower center",
               ncol=2, borderaxespad=0., frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    return fig, axes


convexity_stats = pd.read_csv("./data/pertussis_convexity_stats.csv",
                              index_col="Year")


fig_conv, _ = convexity_stats_plot(convexity_stats, use_incidence=True)
fig_conv.savefig("./fig_pertussis_convexity" + h.incidence_filepath(True)
                 + ".png")





