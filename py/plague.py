import pandas as pd
import ews
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import py.helper as h
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats



pal = h.long_pal

signals = h.signals()
use_incidence = False
use_pertussis = False
lw = "-lightweight"
folder = "./data/backup-01-11-5" + lw

w_min, c_min = h.read_cross_val(folder, use_incidence)
agg = 4
w = w_min // agg
coefs_best = pd.read_csv(folder + "/ews_weights" +
                         h.incidence_filepath(use_incidence) + ".csv",
                         index_col=0, header=None, squeeze=True)
df_c_dict = pd.read_csv(folder + "/optimum_thresholds" +
                        h.incidence_filepath(use_incidence) + ".csv",
                        index_col=0, header=None, squeeze=True)
threshold = df_c_dict["all"]

# Using pertussis
if use_pertussis:
    signals = h.signals()[3:6]

    p_coef = pd.read_csv(folder+"/pertussis_coefficients_incidence.csv",
                         index_col=0, header=None, squeeze=True)
    dt_p = h.lr_emergence_risk(p_coef["c"])

    coefs_best = p_coef
    threshold=dt_p


def get_ews_local(x, w, signals):


    ews_df = pd.DataFrame(
        ews.get_ews(x, windowsize=w, ac_lag=1, se=False,
                    kc=False, method="new",
                    mv_method="exp"))

    ews_df["Time"] = x.index
    ews_df["ac2"] = x.ewm(w).corr(x.shift(2), "pearson")

    ews_df["Decision function"] = ews_df[signals].dot(coefs_best[signals]) \
        + coefs_best["intercept"]

    ews_df["Emergence risk"] = h.lr_emergence_risk(ews_df["Decision function"])

    ews_df["pred"] = ews_df.apply(axis=1, func=lambda j: int(
        j["Decision function"] > threshold))

    return ews_df


# Get plague data

df_p = pd.read_csv("data/Nguyen2017_plague_madagascar.csv")
df_p["all"] = df_p.iloc[:, 1:].sum(axis=1)
ews_p = get_ews_local(df_p["bubonic"], w, signals)


# Get Dengue data


# Cases
def get_lr_single_state(timeseries,sig_level=0.05):
    lr = LinearRegression(fit_intercept=True)
    # Cases

    x = np.log10(timeseries + 1)

    y = x.values

    x_train = x.index.values.reshape(-1, 1)
    y_train = y
    lr.fit(x_train, y_train)

    # t-test:
    # sum of square residuals:
    ssr = np.sum((y_train - lr.predict(X=x_train)) ** 2)
    xbar = np.mean(x_train)
    ybar = np.mean(y_train)
    var_x = np.sum((x_train - xbar) ** 2)
    # Standard error
    se = np.sqrt(ssr / (var_x * (len(x_train) - 2)))
    t_score = lr.coef_ / se
    # One tail test
    p_value = (1 - stats.t.cdf(t_score, (len(x_train) - 2)))

    return {"intercept": lr.intercept_,
            "coef": lr.coef_[0],
            "p_value": p_value[0],
            "ybar": ybar,
            "t_score": t_score,
            "emerging": p_value[0] < sig_level}




def plot_dengue_serotypes():
    dfd = pd.read_csv("./data/San_Juan_Training_Data.csv")


    fig, axes = plt.subplots(2,2, figsize=(8,4))
    pc = h.plot_colours
    axes_label_fontsize = 8
    pal = h.long_pal
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    pal = [pal[5], pal[2], pal[9]]
    abc = ["a", "b", "c", "d"]
    for i in range(0,4):
        st = str(i+1) #serotype
        st_col = "denv" + st+"_cases"
        ax = axes[i//2, i%2]
        ews_d = get_ews_local(dfd[st_col] / 1, w, signals)

        ax.plot(pd.to_datetime(dfd["week_start_date"]),
                 dfd[st_col], c=pal[1])
        ax.set_title(abc[i] +") DENV-"+st, loc="left",
                     fontsize=axes_label_fontsize, color=pc["axis_text"])
        ax.set_xlim(pd.to_datetime("1990"), pd.to_datetime("2010"))
        ax.set_xticks([pd.to_datetime("1990"), pd.to_datetime("1995"),
                       pd.to_datetime("2000"), pd.to_datetime("2005"),
                       pd.to_datetime("2010")])
        ax.set_ylabel("Weekly cases", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])
        ax.set_xlabel("Time", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)

    ax = axes[0,1]
    ax.annotate("1994 outbreak", xy=(pd.to_datetime("1995"), 15),
                xytext=(pd.to_datetime("1996"), 18),
                fontSize=axes_label_fontsize, color=pc["axis_text"],
                arrowprops=dict(facecolor=pc["axis"], shrink=0.1,
                                width=1, headwidth=2),
                horizontalalignment='left'
                )

    ax.annotate("2005 outbreak", xy=(pd.to_datetime("2005"), 23),
                xytext=(pd.to_datetime("2004"), 25),
                fontSize=axes_label_fontsize, color=pc["axis_text"],
                arrowprops=dict(facecolor=pc["axis"], shrink=0.1,
                                width=1, headwidth=2),
                horizontalalignment='right'
                )

    fig.tight_layout()
    return fig


dfd = pd.read_csv("./data/San_Juan_Training_Data.csv")

dfd["dt_week"] = pd.to_datetime(dfd["week_start_date"])
t_filter = dfd["dt_week"] >= pd.to_datetime("1995-01-01")
dfd = dfd[t_filter]


def lr_wrapper(t,sig_level):
    return pd.DataFrame(get_lr_single_state(dfd.loc[dfd["dt_week"] < t,
                                                    "denv2_cases"],
                                            sig_level=sig_level), index=[t])

lr_threshold=0.05
lr_test = pd.concat([lr_wrapper(t,lr_threshold)
                     for t in dfd["dt_week"].iloc[1:]])


def fig_dengue_all_ews():
    dfd = pd.read_csv("./data/San_Juan_Training_Data.csv")

    dfd["dt_week"] = pd.to_datetime(dfd["week_start_date"])
    t_filter = dfd["dt_week"] >= pd.to_datetime("1995-01-01")
    dfd = dfd[t_filter]

    pc = h.plot_colours
    axes_label_fontsize = 8
    pal = h.long_pal
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    pal = [pal[5], pal[2], pal[9]]

    fig, axes = plt.subplots(5,1, figsize=(3.6, 6))

    axes = axes.reshape(-1)

    ylims = [30,40,80,20]
    yticks = np.array([np.arange(0,31,10),
                       np.arange(0,41,10),
                       np.arange(0, 81, 20),
                       np.arange(0, 21, 5)])
    abc = ["b", "c", "d", "e"]

    ax = axes[0]
    ax.plot(df_p.index, df_p["bubonic"], color=pal[0],
            label="Bubonic")
    ax.set_title("a) Plague in Madagascar", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlim(0, 84) #71.5)
    ax.set_xticks([0, 14, 28, 42, 56, 70,84])
    ax.set_ylim(0, 20)
    yticks_plague = np.array([0,  4, 8, 12, 16, 20])
    ax.set_yticks(yticks_plague)
    ax.set_yticklabels(yticks_plague)

    ep_fill = ews_p 
    ax.fill_between(ep_fill.index, 0, ep_fill["pred"]*90, step="post",
                    color=pal[0], alpha=0.3, edgecolor="black")

    ax.set_ylabel("Daily cases", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_xlabel("Days since 2017-08-01", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])

    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)

    ax1 = ax.twinx()
    ax1.set_ylim(-1, 1)
    ax1.set_yticks([0, 1])
    ax1.tick_params(right=True, labelright=True)
    sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
    ax1.axhline(h.lr_emergence_risk(threshold), color=pal[0]) #not if pertussis
    ax1.plot(ep_fill["Emergence risk"], "-", color=pc["axis"])
    ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                   color=pc["axis_text"])
    ax1.tick_params(axis='both', color=pc["axis"],
                    labelcolor=pc["axis_text"],
                    which='both', labelsize=axes_label_fontsize)


    for i, ax in enumerate(axes[1:]):
        col = "denv"+str(i+1)+"_cases"

        ews_d = get_ews_local(dfd[col]/1, w, signals)

        ax.plot(pd.to_datetime(dfd["week_start_date"]), ews_d["timeseries"],
                color=pal[1])
        ep_fill = ews_d
        ep_fill.index = dfd["dt_week"]
        ax.fill_between(ep_fill.index, 0, ep_fill["pred"]*90, step="post",
                        color=pal[1], alpha=0.3, edgecolor="black")

        ax.set_xticks(pd.to_datetime(np.arange(1995,2010,2).astype(str)))
        ax.set_xlim(pd.to_datetime(["1995","2009"]))
        ax.set_ylim(0, ylims[i])
        ax.set_yticks(yticks[i])
        ax.set_ylabel("Weekly cases", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax.set_title(abc[i] + ") DENV-" +str(i+1) + " in San Juan, Puerto Rico",
                     fontsize=axes_label_fontsize, loc="left",
                     color=pc["axis_text"])
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)



        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        ax1.axhline(h.lr_emergence_risk(threshold), color=pal[1]) #not if pertussis
        ax1.plot(ews_d["Emergence risk"], "-", color=pc["axis"])
        ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                       color=pc["axis_text"])
        ax1.tick_params(axis='both', color=pc["axis"],
                        labelcolor=pc["axis_text"],
                        which='both', labelsize=axes_label_fontsize)
        sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)



        #ax.set_ylim(0,30)
        if i != len(axes) - 2:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Year", fontsize=axes_label_fontsize,
                          color=pc["axis_text"])


    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95,
                         wspace=None, hspace=1.5)

    return fig


fig_dengue_all = fig_dengue_all_ews()

if use_pertussis:
    up = "_using_pertussis"
else:
    up = ""

fig_dengue_all.savefig("./fig_plague_dengue" + up + ".png", dpi=600)
fig_dengue_all.savefig("./fig_plague_dengue" + up + ".pdf", dpi=600)



col = "denv"+str(2)+"_cases"

ews_d = get_ews_local(dfd[col]/1, w, signals)

#dfd.index = pd.to_datetime(dfd["week_start_date"])

def plague_etc_figure():

    pc = h.plot_colours
    axes_label_fontsize = 8
    pal = h.long_pal
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    pal = [pal[5], pal[2], pal[9]]

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 2.5))

    ax = axes[0]
    ax.plot(df_p.index, df_p["bubonic"], color=pal[0],
            label="Bubonic")
    ax.set_title("a) Plague in Madagascar", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlim(0, 84) #71.5)
    ax.set_xticks([0, 14, 28, 42, 56, 70,84])
    ax.set_ylim(0, 20)
    yticks = np.array([0,  4, 8, 12, 16, 20])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    ep_fill = ews_p #[ews_p.index < 65]
    ax.fill_between(ep_fill.index, 0, ep_fill["pred"]*90, step="post",
                    color=pal[0], alpha=0.3, edgecolor="black")

    ax.set_ylabel("Daily cases", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_xlabel("Days since 2017-08-01", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])

    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)

    ax1 = ax.twinx()
    ax1.set_ylim(-1, 1)
    ax1.set_yticks([0, 1])
    ax1.tick_params(right=True, labelright=True)
    sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
    ax1.axhline(h.lr_emergence_risk(threshold), color=pal[0]) #not if pertussis
    ax1.plot(ep_fill["Emergence risk"], "-", color=pc["axis"])
    ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                   color=pc["axis_text"])
    ax1.tick_params(axis='both', color=pc["axis"],
                    labelcolor=pc["axis_text"],
                    which='both', labelsize=axes_label_fontsize)


    ax = axes[1]
    ax.plot(dfd["dt_week"], dfd["denv2_cases"], color=pal[1])


    ax.set_title("b) DENV-2 in San Juan, Puerto Rico", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_ylim(0, 50)
    ax.set_yticks([0,10,20,30,40])

    ax.set_xlim(pd.to_datetime("1995"), pd.to_datetime("2009"))
    ep_fill = ews_d #[ews_d.index < 500]
    ax.fill_between(dfd["dt_week"].values, 0, ep_fill["pred"]*120, step="post",
                    color=pal[1], alpha=0.3, edgecolor="black")

    ax.set_ylabel("Weekly cases", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])

    ax.set_xlabel("Year", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])

    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)

    ax1 = ax.twinx()
    ax1.set_ylim(-1, 1)
    ax1.set_yticks([0, 1])
    ax1.tick_params(right=True, labelright=True)
    sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
    ax1.axhline(h.lr_emergence_risk(threshold), color=pal[1]) #not if pertussis
    ax1.plot(dfd["dt_week"], ep_fill["Emergence risk"], "-", color=pc["axis"])
    ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                   color=pc["axis_text"])
    ax1.tick_params(axis='both', color=pc["axis"],
                    labelcolor=pc["axis_text"],
                    which='both', labelsize=axes_label_fontsize)

    fig.subplots_adjust(left=0.14, bottom=0.18, right=0.85, top=0.92,
                        wspace=None, hspace=1.3)
    return fig


def dengue_lr_figure():

    pc = h.plot_colours
    axes_label_fontsize = 8
    pal = h.long_pal
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    pal = [pal[5], pal[2], pal[9], pal[6]]

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 2.5))

    for i, ax in enumerate(axes):

        ax.plot(dfd["dt_week"], dfd["denv2_cases"], color=pal[1])
        ax.set_ylim(0, 50)
        ax.set_xlim(pd.to_datetime("1995"), pd.to_datetime("2009"))

        ep_fill = ews_d

        ax.set_ylabel("Weekly cases", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])

        ax.set_xlabel("Year", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])


        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)

        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
        if i == 0:
            ax.set_title("a) DENV-2 using EWS method",
                         fontsize=axes_label_fontsize,
                         loc="left", color=pc["axis_text"])
            ax1.axhline(h.lr_emergence_risk(threshold), color=pal[1]) #not if pertussis
            ax1.plot(dfd["dt_week"], ep_fill["Emergence risk"], "-",
                     color=pc["axis"])
            ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                       color=pc["axis_text"])
            ax.fill_between(dfd["dt_week"].values, 0, ep_fill["pred"] * 120,
                            step="post",
                            color=pal[1], alpha=0.3, edgecolor="black")

        if i == 1:
            ax.set_title("b) DENV-2 using linear regression method",
                         fontsize=axes_label_fontsize,
                         loc="left", color=pc["axis_text"])
            ax1.axhline(1-lr_threshold, color=pal[3])
            ax1.plot(lr_test.index, 1-lr_test["p_value"], "-",
                     color=pc["axis"])
            ax1.set_ylabel("$1-$ p-value", fontsize=axes_label_fontsize,
                       color=pc["axis_text"])
            ax.fill_between(lr_test.index.values, 0, lr_test["emerging"] * 120,
                            step="post",
                            color=pal[3], alpha=0.3, edgecolor="black")

        ax1.tick_params(axis='both', color=pc["axis"],
                        labelcolor=pc["axis_text"],
                        which='both', labelsize=axes_label_fontsize)


    fig.subplots_adjust(left=0.14, bottom=0.18, right=0.85, top=0.92,
                        wspace=None, hspace=1.3)
    return fig




fig_plague = plague_etc_figure()

if use_pertussis:
    up = "_using_pertussis"
else:
    up = ""

fig_plague.savefig("./fig_plague_dengue" + up + ".png", dpi=600)
fig_plague.savefig("./fig_plague_dengue" + up + ".pdf", dpi=600)


fig_d = plot_dengue_serotypes()
fig_d.savefig("./fig_dengue_serotypes.png", dpi=600)
fig_d.savefig("./fig_dengue_serotypes.svg")


fig_dengue_lr = dengue_lr_figure()

fig_dengue_lr.savefig("./fig_dengue_lr.png", dpi=600)


fig, axes = plt.subplots(2)
ax = axes[0]
ax.plot(np.sqrt(df_p["bubonic"]))

ax.set_xlim(0, 115)
ax.set_ylabel("Daily incidence (sqrt)")
ax.set_title("a)", loc="left")
yticks = np.array([0,2,4,6,8])
ax.set_yticks(yticks)
ax.set_yticklabels(yticks**2)


ax = axes[1]
ax.plot(ews_p["Emergence risk"])
ax.axhline(h.lr_emergence_risk(threshold)) #not if pertussis
ax.fill_between(ews_p.index, 0, ews_p["pred"], alpha=0.5,step="post")
ax.set_xlim(0, 115)
ax.set_ylabel("Emergence risk")

ax.set_xlabel("Days since 2017-08-01")
ax.set_title("b", loc="left")
fig.tight_layout()

fig.savefig("./plague_magagascar.pdf")

