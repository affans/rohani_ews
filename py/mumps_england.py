from subprocess import call
import ews
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pomegranate as pm
import scipy.stats as stats
import seaborn as sns
from matplotlib.gridspec import GridSpec
import py.helper as h
from sklearn import metrics


# Options for script
use_incidence = False
use_cross_val = True
# calculate_ews = False
group_by_time_from_bp = False
cut_off = False
folder = "./data/backup-01-11-5" + "-lightweight"
# folder = "./data/backup-28-09-1/"
agg=4
# folder = "./data/backup-16-07-1-pertussis"

pal = sns.color_palette()

# Read cross-validation results
if use_cross_val:
    w_min, c_min = h.read_cross_val(folder, use_incidence)
else:
    w_min = 156.
    c_min = 0.0001
w = w_min//agg

# Select signals used in learning:
signals = h.signals()

# Read in trained coefficients and thresholds:
coefs_best = pd.read_csv(folder + "/ews_weights" +
                         h.incidence_filepath(use_incidence) + ".csv",
                         index_col=0, header=None, squeeze=True)
df_c_dict = pd.read_csv(folder + "/optimum_thresholds" +
                        h.incidence_filepath(use_incidence) + ".csv",
                        index_col=0, header=None, squeeze=True)


if use_incidence:
    coefs_best["standard_deviation"] = 0
    coefs_best["index_of_dispersion"] = 0
    coefs_best["mean"] = 0


def get_mumps_ews(x,coefs, return_signal="Emergence risk"):

    df = pd.DataFrame(
        ews.get_ews(x, windowsize=w, ac_lag=1, se=False,
                    kc=False, method="new", mv_method="exp"))

    df.index = x.index

    df["ac2"] = x.ewm(w).corr(x.shift(2), "pearson")

    df["Decision function"] = df[signals].dot(coefs[signals]) \
                     + coefs["intercept"]
    df["Emergence risk"] = h.lr_emergence_risk(df["Decision function"])

    return df[return_signal]

# Do analysis of england mumps data
def get_mumps_england(er_threshold,coefs, return_signal="Emergence risk"):

    def get_ews_wrapper(x):
        return get_mumps_ews(x,coefs, return_signal=return_signal)


    # Mumps England data
    def get_prediction(row):
        return [k > er_threshold for k in row]

    # National
    df_england = pd.read_csv("./data/mumps_england_4week.csv", index_col=0,
                             header=None)

    ews_mumps = get_ews_wrapper(df_england.loc[1990:, 1])

    # Regions
    df_regions = pd.read_csv("./data/mumps_england_4week_regions.csv",
                             index_col=0)
    ews_mumps_region = df_regions.loc[1990:2005].apply(get_ews_wrapper, axis=0)

    test = ews_mumps_region.apply(axis=0, func=get_prediction)

    # Local
    df_local = pd.read_csv("./data/mumps_england_4week_LA.csv", index_col=0,
                           header=[0,1])
    # Filter out local authorities which have no cases before 1994
    df_local = df_local.loc[:, (df_local[:1994].sum(axis=0) > 0)]
    ews_mumps_la = df_local.loc[1990:].apply(get_ews_wrapper, axis=0)
    test2 = ews_mumps_la.apply(axis=0, func=get_prediction)

    # Filter out local authorities without large outbreaks in after 1990,
    # this needs systematising


    outbreak_sizes = df_local.loc[2004:2007].sum(axis=0)


    d1 = pm.ExponentialDistribution(0.51)
    d2 = pm.ExponentialDistribution(0.5)
    model = pm.GeneralMixtureModel([d1, d2], weights=[0.5, 0.5])
    model.fit(outbreak_sizes[outbreak_sizes != 0].values,
              stop_threshold=0.000001, inertia=0.)
    # large outbreaks >= lo_threshold
    lo_threshold = np.sum([model.predict(0)[0] == model.predict(i)[0] for i in
            range(0, int(max(outbreak_sizes)))])

    large_outbreaks = df_local.loc[:, outbreak_sizes >=
                                      lo_threshold]

    d = {"country_cases": df_england,
         "country_er": ews_mumps,
         "country_pred": ews_mumps > er_threshold,
         "regions_cases": df_regions,
         "regions_er": ews_mumps_region,
         "regions_pred": test,
         "la_cases": df_local,
         "la_er": ews_mumps_la,
         "la_pred": test2,
         "large_outbreaks": large_outbreaks,
         "lo_threshold": lo_threshold,
         "GMM": model,
         "outbreak_sizes": outbreak_sizes
         }

    return d



# Example figure 3:
def plot_england_mumps(england_data, threshold, simulated_data=None):

    pc = h.plot_colours
    axes_label_fontsize = 8

    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})



    ed = england_data
    lo_columns = ed["large_outbreaks"].columns
    pal = ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]

    fig = plt.figure(figsize=(7.2, 5.))
    gs = GridSpec(6, 2)
    axes = np.array([plt.subplot(gs[0:4, 0]), plt.subplot(gs[0:2, 1]),
                     plt.subplot(gs[2:4, 1]),# plt.subplot(gs[2, 1]),
                     plt.subplot(gs[4:6, 0])
                     #, plt.subplot(gs[6:10, 1])
                     ])
    if simulated_data is not None:
        axes = np.concatenate((axes,np.array([plt.subplot(gs[4:6, 1])])))

    img=mpimg.imread('england_regions_districts.png')
    lum_img = img[:, :, 0]
    axes[0].imshow(img,  cmap="nipy_spectral")
    axes[0].axis('off')
    axes[0].set_title("a) Administrative units of England",
                      fontsize=axes_label_fontsize, loc="left",
                      color=pc["axis_text"])

    # Get time series for panels b to d
    la_max = ed["la_cases"][2004:2006].sum(axis=0).idxmax()
    region_max = ed["regions_cases"][2004:2006].sum(axis=0).idxmax()
    df_list = [ed["country_cases"][1], # ed["regions_cases"][region_max],
               ed["la_cases"][la_max]]
    ews_list = [ed["country_er"], # ed["regions_er"][region_max],
                ed["la_er"][la_max]]
    titles = ["b) England", # "c) "+region_max,
              "c) "+la_max[0]]
    ts_max = [200, # ??
              15]

    for i, df in enumerate(df_list):
        # df = df_list[i]
        ews_df = ews_list[i]
        # ts_max = df[1990:2005].max()/3
        ews_df = ews_df[ews_df.index < 2004]
        ax = axes[i+1]
        ax.plot(df.index, df, color=pal[i])
        ax.set_title(titles[i], fontsize=axes_label_fontsize,
                     loc="left", color=pc["axis_text"])
        ax.set_ylim(0, ts_max[i])
        ax.set_xlim(1990, 2005)
        ax.set_xticks([1990, 1995, 2000, 2005])
        is_emerging = np.array(
            [j > threshold for j in ews_df])
        ax.fill_between(ews_df.index, ax.get_ylim()[0],
                        is_emerging * ax.get_ylim()[1],
                        color=pal[i], alpha=0.3, edgecolor="black")
        ax.set_ylabel("Monthly\ncases", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])
        if i != 1:
            ax.tick_params(labelbottom=False)
        else:
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
        ax1.axhline(threshold, color=pal[i])
        ax1.plot(ews_df, "-", color=pc["axis"])
        ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                       color=pc["axis_text"])
        ax1.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)

    ax = axes[3]
    df_positive = pd.DataFrame({"country": ed["country_pred"],
                                "regions": ed["regions_pred"].mean(axis=1),
                                "la_lo": ed["la_pred"][lo_columns].mean(axis=1),
                                "la_no_lo": ed["la_pred"]
                                .drop(lo_columns, axis=1).mean(axis=1)})

    df_positive = df_positive[df_positive.index < 2004]

    # ax.plot(df_list[0]/df_list[0].max(), color="black",
    #        label="Normalised cases", alpha=0.2 )
    ax.plot(df_positive["country"], color=pal[0], label="All England")
    # ax.plot(df_positive["regions"], label="Regions ", color=pal[1])
    ax.plot(df_positive["la_lo"],
            label="LAs with large outbreaks",# (" +
                  #str(ed["la_pred"][lo_columns].shape[1])+" total)",
            color=pal[1])
    ax.plot(df_positive["la_no_lo"],
            label="Remaining LAs", # (" + str(ed["la_pred"].shape[1]) + " total)",
            color=pal[2])
    ax.set_ylabel("Fraction above threshold", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_xlabel("Year", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_title("d)", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlim(1990,2005)
    ax.set_xticks([1990, 1995, 2000, 2005])
    ax.set_xticklabels([1990, 1995, 2000, 2005])
    ax.set_ylim(0,1)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    ax.legend(fontsize=8, frameon=False, loc=(-0.02, 0.4))

    ax = axes[4]

    simulated_data = simulated_data[simulated_data.index < 2004]

    #ax.plot(simulated_data["regions"], label="Regions", color=h.pal[1])
    ax.plot(simulated_data["country"], label="All England", color=h.pal[0])
    ax.plot(simulated_data["la_lo"], label="LAs with large\noutbreaks",
            color=h.pal[1])
    ax.plot(simulated_data["la_no_lo"], label="Remaining LAs", color=h.pal[2])
    ax.set_xlim(1990, 2005)
    ax.set_ylim(0, 1)
    ax.set_xticks([1990, 1995, 2000, 2005])
    ax.set_xticklabels([1990, 1995, 2000, 2005])
    ax.set_title("e)", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlabel("Year",fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)


    fig.tight_layout()
    #fig_eng.subplots_adjust(left=0.08, right=0.93, top=0.96, bottom=0.08,
    #                        wspace=0.3, hspace=1E3)

    return fig, axes



# Breakdown of regions
def plot_regions(england_data, threshold):
    ed = england_data
    lo_columns = ed["large_outbreaks"].columns
    pal = ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(5, 2)
    axes = np.array([plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]),
                     plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                     plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]),
                     plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]),
                     plt.subplot(gs[4, 0])])

    regions = ed["regions_cases"].columns
    df_list = [ed["regions_cases"][r] for r in regions]
    ews_list = [ed["regions_er"][r] for r in regions]
    abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    titles = [abc[i] + ") " + regions[i] for i in range(len(regions))]

    for i in range(0, 9):
        df = df_list[i]
        ews_df = ews_list[i]
        ts_max = df[1990:2005].max()/3
        ax = axes[i]
        ax.plot(df.index, df, color=h.long_pal[i])
        ax.set_title(titles[i], loc="left", fontsize=10)
        ax.set_ylim(0, ts_max)
        ax.set_xlim(1990, 2005)
        ax.set_ylim(0, 25)
        ax.set_xticks([1990, 1995, 2000, 2005])
        is_emerging = np.array(
            [i > threshold for i in ews_df])
        ax.fill_between(ews_df.index, ax.get_ylim()[0],
                        is_emerging * ax.get_ylim()[1],
                        color=h.long_pal[i], alpha=0.3, edgecolor="black")
        if i < 7:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Year")

        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
        ax1.axhline(threshold, color=h.long_pal[i])
        ax1.plot(ews_df, "-", color="black")
        #if i % 2 == 0:
        #    ax.set_ylabel("Monthly\ncases")
        #else:
        #    ax1.set_ylabel("Emergence\n risk")
    fig.text(0.02, 0.5, 'Monthly cases',
             va='center', rotation='vertical')
    fig.text(0.97, 0.6, 'Emergence risk',
             va='center', rotation='vertical')

    fig.tight_layout(rect=[0.02, 0, 0.98, 1])
    return fig, axes


# Breakdown of regions: local authorities above threshold
def plot_regions_fractions(england_data, threshold):
    ed = england_data
    lo = ed["large_outbreaks"].columns
    pal = ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(5, 2)
    axes = np.array([plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]),
                     plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                     plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]),
                     plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]),
                     plt.subplot(gs[4, 0])])

    regions = ed["regions_cases"].columns
    df_list = [ed["la_cases"].loc[:,ed["la_cases"].columns.
                                  get_level_values('Anomymised_Region')
                                  == r] for r in regions]
    ews_list = [ed["regions_er"][r] for r in regions]
    pred_list = [ed["la_pred"].loc[:,ed["la_pred"].columns.
                                   get_level_values('Anomymised_Region')
                                   == r] for r in regions]
    abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    titles = [abc[i] + ") " + regions[i] for i in range(len(regions))]

    for i in range(0, 9):
        df = df_list[i].sum(axis=1)
        ews_df = ews_list[i]
        pr = pred_list[i]
        pr_lo = pr[lo[lo.get_level_values('Anomymised_Region') ==regions[i]]]
        ts_max = df[1990:2005].max()/3
        ax = axes[i]
        ax.plot(pr.index, pr.sum(axis=1), color=h.long_pal[i])
        ax.plot(pr.index, pr_lo.sum(axis=1), color="black",
                alpha=0.5, linestyle="--")
        ax.set_title(titles[i], loc="left", fontsize=10)
        ax.set_ylim(0, ts_max)
        ax.set_xlim(1990, 2005)
        ax.set_ylim(0, 10)
        ax.set_xticks([1990, 1995, 2000, 2005])
        is_emerging = np.array(
            [i > threshold for i in ews_df])
        ax.fill_between(ews_df.index, ax.get_ylim()[0],
                        is_emerging * ax.get_ylim()[1],
                        color=h.long_pal[i], alpha=0.3, edgecolor="black")
        if i < 7:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Year")

        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        # ax1 = ax.twinx()
        # ax1.set_ylim(-1, 1)
        # ax1.set_yticks([0, 1])
        # ax1.tick_params(right=True, labelright=True)
        # sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
        # ax1.axhline(1 / (1 + np.exp(-df_c_dict["all"])), color=h.long_pal[i])
        # ax1.plot(ews_df, "-", color="black")
        #if i % 2 == 0:
        #    ax.set_ylabel("")
        # else:
            # ax1.set_ylabel("Emergence\n risk")
    fig.text(0.02, 0.5, 'Number of local authorities above threshold',
             va='center', rotation='vertical')
    fig.tight_layout(rect=[0.02, 0, 1, 1])
    return fig, axes


# Breakdown of local authorities by region
def plot_local_authority_old(england_data, region, threshold):
    ed = england_data
    lo_columns = ed["large_outbreaks"].columns
    pal = ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]
    c_df = ed["la_cases"].loc[:, ed["la_cases"].columns.
                             get_level_values('Anomymised_Region')
                             == region]
    er_df = ed["la_er"].loc[:, ed["la_er"].columns.
                             get_level_values('Anomymised_Region')
                             == region]
    p_df = ed["la_pred"].loc[:, ed["la_pred"].columns.
                             get_level_values('Anomymised_Region')
                             == region]


    # number of local authorities
    n_las = c_df.shape[1]
    n_row = n_las//3 + (1 if n_las % 3 != 0 else 0)
    fig = plt.figure(figsize=(8, n_row))
    gs = GridSpec(n_row, 3)

    axes = []
    for i in range(n_las):
        axes += [plt.subplot(gs[i//3, i % 3])]



    # regions = ed["regions_cases"].columns
    # df_list = [ed["regions_cases"][r] for r in regions]
    # ews_list = [ed["regions_er"][r] for r in regions]
    abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i","j","k","l","m","n","o",
           "p","q","r","s","t","u","v","w","x","y","z","aa","bb"]

    for i, c in enumerate(c_df.columns):
        title = abc[i] + ") " + c[0]
        df = c_df[c]
        ews_df = er_df[c]
        ts_max = df[1990:2005].max()/3
        col = h.long_pal[i % len(h.long_pal)]
        ax = axes[i]
        ax.plot(df.index, df, color=col)
        ax.set_title(title, loc="left", fontsize=10,
                     color="red" if c in lo_columns else "black")
        ax.set_xlim(1990, 2005)
        ax.set_ylim(0, 5)
        ax.set_xticks([1990, 1995, 2000, 2005])
        is_emerging = np.array(
            [i > threshold for i in ews_df])
        ax.fill_between(ews_df.index, ax.get_ylim()[0],
                        is_emerging * ax.get_ylim()[1],
                        color=col, alpha=0.3, edgecolor="black")
        if i < n_las -3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Year")

        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
        ax1.axhline(threshold, color=col)
        ax1.plot(ews_df, "-", color="black")
        # if i % 2 == 0:
        #    ax.set_ylabel("Monthly\ncases")
        # else:
        #    ax1.set_ylabel("Emergence\n risk")
    fig.text(0.02, 0.5, 'Monthly cases',
             va='center', rotation='vertical')
    fig.text(0.97, 0.6, 'Emergence risk',
             va='center', rotation='vertical')

    fig.tight_layout(rect=[0.02, 0, 0.98, 1])
    return fig, axes


# Breakdown of local authorities by region
def plot_local_authority(england_data,fig_no, n_plots, threshold):
    ed = england_data
    lo_columns = ed["large_outbreaks"].columns.droplevel(1)
    pal = ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]


    c_df = ed["la_cases"].copy()
    c_df.columns = c_df.columns.droplevel(level=1)

    c_df.columns = [int(c.replace("Local Authority ", ""))
                           for c in c_df.columns]

    authorities = np.sort([c for c in c_df.columns])
    authorities = authorities[fig_no*n_plots:min((fig_no+1)*n_plots,
                                                 len(authorities))]

    #plot_authorities = ["Local Authority " + str(i) for i in authorities]

    plot_columns = authorities #c_df.columns.isin(plot_authorities)
    c_df = c_df.loc[:, plot_columns]

    er_df = ed["la_er"].copy()
    er_df.columns = er_df.columns.droplevel(level=1)
    er_df.columns = [int(c.replace("Local Authority ", ""))
                           for c in er_df.columns]
    er_df = er_df.loc[:, plot_columns]
    #
    # p_df = ed["la_pred"].loc[:, plot_columns].copy()
    # p_df.columns = p_df.columns.droplevel(level=1)
    # p_df.columns = [int(c.replace("Local Authority ", ""))
    #                        for c in p_df.columns]



    # number of local authorities
    n_las = c_df.shape[1]
    n_row = n_las//3 + (1 if n_las % 3 != 0 else 0)
    fig = plt.figure(figsize=(8, n_row))
    gs = GridSpec(n_row, 3)

    axes = []
    for i in range(n_las):
        axes += [plt.subplot(gs[i//3, i % 3])]


    abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i","j","k","l","m","n","o",
           "p","q","r","s","t","u","v","w","x","y","z","aa","bb"]


    for i, c in enumerate(c_df.columns):

        lo_text = "Local Authority " + str(c)
        title = abc[i] + ") " + lo_text
        df = c_df.loc[:,c]
        ews_df = er_df.loc[:,c]
        ts_max = df[1990:2005].max()/3
        # col = h.long_pal[i % len(h.long_pal)]
        col = h.pal[2] if lo_text in lo_columns else h.pal[0]

        ax = axes[i]
        ax.plot(df.index, df, color=col)
        ax.set_title(title, loc="left", fontsize=10,
                     color="black")
        ax.set_xlim(1990, 2005)
        ax.set_ylim(0, 5)
        ax.set_xticks([1990, 1995, 2000, 2005])
        is_emerging = np.array(
            [i > threshold for i in ews_df])
        ax.fill_between(ews_df.index, ax.get_ylim()[0],
                        is_emerging * ax.get_ylim()[1],
                        color=col, alpha=0.3, edgecolor="black")
        if i < n_las -3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Year")

        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
        ax1.axhline(threshold, color=col)
        ax1.plot(ews_df, "-", color="black")
        # if i % 2 == 0:
        #    ax.set_ylabel("Monthly\ncases")
        # else:
        #    ax1.set_ylabel("Emergence\n risk")
    fig.text(0.02, 0.5, 'Monthly cases',
             va='center', rotation='vertical')
    fig.text(0.97, 0.6, 'Emergence risk',
             va='center', rotation='vertical')

    fig.tight_layout(rect=[0.02, 0, 0.98, 1])
    return fig, axes



# Simulate "England" LAs, large outbreaks from GMM, overall cases roughly
# matching england
def get_simulated_england(england_data, threshold, coefs,
                          return_signal="Emergence risk", rng_seed=21):
    def get_ews_wrapper(x):
        return get_mumps_ews(x,coefs, return_signal=return_signal)



    def get_simulated_ews(model, runs):

        def england_sim_args(fsim):
            # load default simulator arguments
            sim_args = h.simulator_args(folder=fsim)

            sim_args["runs"] = runs
            sim_args["T"] = 30
            sim_args["R0_rs"] = 20
            sim_args["N_i"] = 2E5
            sim_args["R0_i"] = 0.2
            sim_args["N_f"] = sim_args["N_i"]
            sim_args["eta_i"] = 1 / 365
            sim_args["eta_f"] = sim_args["eta_i"]
            sim_args["rp_i"] = 0.1
            sim_args["rp_f"] = sim_args["rp_i"]
            sim_args["bb_a"] = 0.5
            sim_args["seed"] = rng_seed

            if model == 1:
                sim_args["R0_ramp"] = "bb"
                sim_args["R0_f"] = 1
                sim_args["eta_i"] = 1 / 60
                sim_args["eta_f"] = 1 / 60
                # sim_args["rp_f"] = 0.6
            else:
                sim_args["R0_ramp"] = "ou"
                sim_args["R0_f"] = sim_args["R0_i"]
                sim_args["eta_i"] = 1 / 180
                sim_args["eta_f"] = sim_args["eta_i"]

            return sim_args

        simulator = "./SEIR-Simulator-0.2.5/seir_simulator_gamma"

        e_sim_args = england_sim_args("./data/test/")
        pl = [k + '=' + str(e_sim_args[k]) for k in e_sim_args.keys()]
        comm = [simulator] + pl
        call(comm)

        df = pd.read_csv("./data/test/epi_data.csv")
        df["model"] = model
        df["is_test"] = model

        edf = h.get_ews(df, params_df=pd.DataFrame(e_sim_args, index=[model]),
                        agg=4,
                        wtime=w_min, mv_method="exp",
                        use_parallel=False, nc=1, use_incidence=False)

        edf["df"] = edf[signals].dot(coefs[signals]) \
                    + coefs["intercept"]
        edf["Emergence risk"] = h.lr_emergence_risk(edf["df"])
        edf["Prediction"] = edf["Emergence risk"] > threshold
        edf["Time"] = edf["Time"] / 365 - e_sim_args["R0_rs"]

        return edf

    df = pd.concat([get_simulated_ews(0, 157 - 16), get_simulated_ews(1, 16)])
    df["Time"] += 1994
    df = df.reset_index(drop=True)

    df_positive = df.groupby(["model", "Time"])["Prediction"].mean().unstack(
        "model")
    df_positive = df_positive.rename(columns={1: "la_lo", 0: "la_no_lo"})

    df_positive["combined"] = df.groupby(["Time"])["Prediction"].mean()

    df_country = df.groupby(["Time"])["timeseries"].sum()
    df_positive["country"] = get_ews_wrapper(df_country[1980:]) > threshold

    # Regions

    lo = england_data["large_outbreaks"].columns
    # number of LAs per region
    region_counts = pd.Series(
        [c[1] for c in england_data["la_er"].columns]).value_counts()
    region_counts = region_counts.sort_index(axis=0)
    # number of LAs with large outbreaks per region
    lo_counts = pd.Series([c[1] for c in lo]).value_counts()
    lo_counts = lo_counts.sort_index(axis=0)

    no_lo_region = pd.cut(df[df["model"] == 0]["run"], [0] +
                          (region_counts - lo_counts).cumsum().tolist(),
                          labels=region_counts.index)
    lo_region = pd.cut(df[df["model"] == 1]["run"],
                       [0] + lo_counts.cumsum().tolist(),
                       labels=lo_counts.index)
    df["region"] = pd.concat([no_lo_region, lo_region])

    df_regions = df.groupby(["Time", "region"]).sum()["timeseries"].unstack(
        "region")

    df_regions_ews = df_regions.apply(get_ews_wrapper)
    df_regions_pred = df_regions_ews > threshold
    df_positive["regions"] = df_regions_pred.mean(axis=1)

    return df_positive


# Plot outbreak sizes and GMM fit
def plot_outbreak_sizes(england_data):
    outbreak_sizes = england_data["outbreak_sizes"]
    model = england_data["GMM"]

    lo_threshold = np.sum([model.predict(0)[0] == model.predict(i)[0] for i in
                           range(0, int(max(outbreak_sizes)))])


    n = 5

    test = outbreak_sizes.groupby(outbreak_sizes.values // n).count() / \
           (n*len(outbreak_sizes))
    test2 = test[test.index >= lo_threshold//n + 1/2]

    fig = plt.figure(figsize=(6, 3))
    gs = GridSpec(1, 1)
    axes = np.array([plt.subplot(gs[0, 0])])
    ax = axes[0]
    ax.scatter(test.index*n + n/2, np.log10(test),
               label="Small outbreaks", color=h.pal[0])
    ax.scatter(test2.index*n + n/2, np.log10(test2),
               label="Large outbreaks", color=h.pal[2])
    ax.plot(np.log10(np.e)*model.log_probability(np.array([i for i in range(700)])),
            color=h.pal[1],
            label="GMM fit")
    ax.axvline(lo_threshold, linestyle="--", color="grey")
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.set_ylim(-5,-1)
    ax.set_xlim(-10,700)
    ax.set_xlabel("2004-2005 outbreak size")
    ax.set_ylabel("Log$_{10}$ probability")
    ax.legend()
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    print("lo_threshold:", lo_threshold)
    print("number zero:", outbreak_sizes.value_counts()[0])
    print("number above threshold:",
          outbreak_sizes[(outbreak_sizes > lo_threshold)].shape[0])

    return fig, axes

# Plot the relationship between outbreaksize and lead time
def plot_lead_time_vs_outbreaksize(england_data):

    def lead_time(x):
        return x[~x].index[-1]

    lo = england_data["large_outbreaks"].columns
    outbreak_sizes = england_data["outbreak_sizes"]
    lt = england_data["la_pred"].loc[:2005].apply(lead_time)

    test = pd.concat([lt,outbreak_sizes], axis=1)
    test["lo"] = [1 if i in lo else 0 for i in test.index]
    test = test[test[1] > 0]

    test2 = test[test[0] < 2004]

    tau1, p_value1 = stats.spearmanr(test[0], test[1])
    tau2, p_value2 = stats.spearmanr(test2[0], test2[1])

    def p_value_text(x):
        if np.round(x, 3) == 0:
            return "$p < 0.001$"
        else:
            return "$p = $" + str(np.round(x, 3))

    print("Detection time pre-2005: rho = ",str(np.round(tau1, 2)), " ",
          p_value_text(p_value1))
    print("Only pre-2004: rho = ",str(np.round(tau2, 2)), " ",
          p_value_text(p_value2))

    fig,ax = plt.subplots(nrows=1)

    # ax.text(1997.5, 60, "Detection time pre-2005: $\\tau = $"
    #         + str(np.round(tau1, 2)) +
    #         "; " + p_value_text(p_value1), color="black")
    # ax.text(1997.5, 15, "Only pre-2004: $\\tau = $" + str(np.round(tau2, 2))
    #         + "; " + p_value_text(p_value2), color="black")

    ax.scatter(test[test["lo"] == 1][0], test[test["lo"] == 1][1],
               color=h.pal[2],
               label="Local authorities with large outbreaks")
    ax.scatter(test[test["lo"] == 0][0], test[test["lo"] == 0][1],
               color=h.pal[0], label="Local authorities with small outbreaks")

    ax.legend(frameon=False)
    ax.set_xlabel("Detection time")
    ax.set_ylabel("Outbreak size")
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    return fig, ax

# Correlation plot
def plot_correlation_with_time(england_data):
    region_er = england_data["regions_er"].copy()
    region_er = region_er.dropna()
    region_er["Time"] = region_er.index
    k_corr = region_er.corr(method="kendall")["Time"]
    s_corr = region_er.corr(method="spearman")["Time"]
    p_corr = region_er.corr(method="pearson")["Time"]

    fig, ax = plt.subplots(1)
    ax.plot(k_corr[k_corr.index != "Time"], marker="o", label="Kendall")
    ax.plot(s_corr[s_corr.index != "Time"], marker="o",label = "Spearman")
    ax.plot(p_corr[p_corr.index != "Time"], marker="o", label="Pearson")
    ax.legend(frameon=False)
    ax.set_xlabel("Region")
    ax.set_ylabel("Correlation")
    sns.despine(offset=5, trim=False, right=True, left=False)

    return fig, ax

# Plot the relationship between outbreaksize and Emergence risk
def plot_er_vs_outbreaksize(england_data, year=2004):


    lo = england_data["large_outbreaks"].columns
    outbreak_sizes = england_data["outbreak_sizes"]

    er = england_data["la_er"].loc[year] #.mean(axis=0)


    test = pd.concat([er,outbreak_sizes], axis=1)
    test.columns = ["er","os"]
    test["lo"] = [1 if i in lo else 0 for i in test.index]
    test = test[test["os"] > 0]
    # test["df"] = logit(test["er"])
    test = test.dropna()

    tau1, p_value1 = stats.spearmanr(test["er"], test["os"])


    def p_value_text(x):
        if np.round(x, 3) == 0:
            return "$p < 0.001$"
        else:
            return "$p = $" + str(np.round(x, 3))

    print("Emergence risk (2004): rho = ",str(np.round(tau1, 2)), " ",
          p_value_text(p_value1))

    fig,ax = plt.subplots(nrows=1)

    ax.scatter(test[test["lo"] == 1]["er"], (test[test["lo"] == 1]["os"]),
               color=h.pal[2],
               label="Local authorities with large outbreaks")
    ax.scatter(test[test["lo"] == 0]["er"], (test[test["lo"] == 0]["os"]),
               color=h.pal[0], label="Local authorities with small outbreaks")

    ax.legend(frameon=False)
    ax.set_xlabel("Emergence risk (" + str(year) + ")")
    ax.set_ylabel("Outbreak size")
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    return fig, ax

# Plot the relationship between outbreaksize and Delta Emergence risk
def plot_delta_er_vs_outbreaksize(england_data, year=2004):


    lo = england_data["large_outbreaks"].columns
    outbreak_sizes = england_data["outbreak_sizes"]

    er = england_data["la_er"].diff(1).loc[year]# .loc[1998:2004].mean(axis=0)


    test = pd.concat([er,outbreak_sizes], axis=1)
    test.columns = ["er","os"]
    test["lo"] = [1 if i in lo else 0 for i in test.index]
    test = test[test["os"] > 0]
    # test["df"] = logit(test["er"])

    tau1, p_value1 = stats.kendalltau(test["er"], test["os"])

    def p_value_text(x):
        if np.round(x, 3) == 0:
            return "$p < 0.001$"
        else:
            return "$p = $" + str(np.round(x, 3))

    fig,ax = plt.subplots(nrows=1)

    ax.scatter(test[test["lo"] == 1]["er"], (test[test["lo"] == 1]["os"]),
               color=h.long_pal[3],
               label="Local authorities with large outbreaks")
    ax.scatter(test[test["lo"] == 0]["er"], (test[test["lo"] == 0]["os"]),
               color=h.long_pal[2], label="Remaining local authorities")

    ax.legend(frameon=False)
    ax.set_xlabel("Emergence risk (" + str(year) + ")")
    ax.set_ylabel("Outbreak size")
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    return fig, ax


# Example figure 3:
def plot_england_mumps2(england_data, threshold, simulated_data=None):

    pc = h.plot_colours
    axes_label_fontsize = 8
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})



    ed = england_data
    lo_columns = ed["large_outbreaks"].columns
    pal = h.pal#["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]

    # reorder pal for this figure
    pal = [pal[3], pal[2], pal[0]]

    fig = plt.figure(figsize=(3.6, 4.5))
    gs = GridSpec(4, 1)
    axes = np.array([plt.subplot(gs[0]), plt.subplot(gs[1]),
                     plt.subplot(gs[2]),# plt.subplot(gs[2, 1]),
                     plt.subplot(gs[3])
                     #, plt.subplot(gs[6:10, 1])
                     ])
    # if simulated_data is not None:
    #     axes = np.concatenate((axes, np.array([plt.subplot(gs[4:6, 1])])))
    #

    # Get time series for panels b to d
    la_max = ed["la_cases"][2004:2006].sum(axis=0).idxmax()
    region_max = ed["regions_cases"][2004:2006].sum(axis=0).idxmax()
    df_list = [ed["country_cases"][1], # ed["regions_cases"][region_max],
               ed["la_cases"][la_max]]
    ews_list = [ed["country_er"], # ed["regions_er"][region_max],
                ed["la_er"][la_max]]
    titles = ["a) England", # "c) "+region_max,
              "b) "+la_max[0]]
    ts_max = [200, # ??
              20]
    yticks = [[0,50,100,150,200],
              [0,5,10,15,20]]

    for i, df in enumerate(df_list):
        # df = df_list[i]
        ews_df = ews_list[i]
        # ts_max = df[1990:2005].max()/3
        ews_df = ews_df[ews_df.index < 2004]
        ax = axes[i]
        ax.plot(df.index, df, color=pal[i])
        ax.set_title(titles[i], fontsize=axes_label_fontsize,
                     loc="left", color=pc["axis_text"])
        ax.set_ylim(0, ts_max[i])
        ax.set_yticks(yticks[i])
        ax.set_xlim(1990, 2005)
        ax.set_xticks([1990, 1995, 2000, 2005])
        is_emerging = np.array(
            [j > threshold for j in ews_df])
        ax.fill_between(ews_df.index, ax.get_ylim()[0],
                        is_emerging * ax.get_ylim()[1],
                        color=pal[i], alpha=0.3, edgecolor="black")
        ax.set_ylabel("Monthly\ncases", fontsize=axes_label_fontsize,
                      color=pc["axis_text"])

        ax.tick_params(labelbottom=False)


        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)

        ax1 = ax.twinx()
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([0, 1])
        ax1.tick_params(right=True, labelright=True)
        sns.despine(ax=ax1, offset=5, trim=True, right=False, left=True)
        ax1.axhline(threshold, color=pal[i])
        ax1.plot(ews_df, "-", color=pc["axis"])
        ax1.set_ylabel("Emergence\n risk", fontsize=axes_label_fontsize,
                       color=pc["axis_text"])
        ax1.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)

    ax = axes[2]
    df_positive = pd.DataFrame({"country": ed["country_pred"],
                                "regions": ed["regions_pred"].mean(axis=1),
                                "la_lo": ed["la_pred"][lo_columns].mean(axis=1),
                                "la_no_lo": ed["la_pred"]
                                .drop(lo_columns, axis=1).mean(axis=1)})

    df_positive = df_positive[df_positive.index < 2004]

    # ax.plot(df_list[0]/df_list[0].max(), color="black",
    #        label="Normalised cases", alpha=0.2 )
    ax.plot(df_positive["country"], color=pal[0], label="Whole country")
    # ax.plot(df_positive["regions"], label="Regions ", color=pal[1])
    ax.plot(df_positive["la_lo"],
            label="Large outbreak LAs",# (" +
                  #str(ed["la_pred"][lo_columns].shape[1])+" total)",
            color=pal[1])
    ax.plot(df_positive["la_no_lo"],
            label="Small outbreak LAs", # (" + str(ed["la_pred"].shape[1]) + " total)",
            color=pal[2])
    ax.set_ylabel("Fraction above\nthreshold", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    # ax.set_xlabel("Year", fontsize=axes_label_fontsize,
    #               color=pc["axis_text"])
    ax.set_title("c) Mumps in England", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlim(1990,2005)
    ax.set_xticks([1990, 1995, 2000, 2005])
    ax.tick_params(labelbottom=False)
    ax.set_ylim(0,1.03)
    ax.set_yticks([0,0.25,0.5,0.75,1])

    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    ax.legend(fontsize=6, frameon=False, loc=(0.0, 0.3))

    ax = axes[3]

    simulated_data = simulated_data[simulated_data.index < 2004]

    #ax.plot(simulated_data["regions"], label="Regions", color=h.pal[1])
    ax.plot(simulated_data["country"], label="Whole country", color=pal[0])
    ax.plot(simulated_data["la_lo"], label="LAs with large\noutbreaks",
            color=pal[1])
    ax.plot(simulated_data["la_no_lo"], label="Remaining LAs", color=pal[2])
    ax.set_xlim(1990, 2005)
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_xticks([1990, 1995, 2000, 2005])
    ax.set_xticklabels([1990, 1995, 2000, 2005])
    ax.set_title("d) Simulated data", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlabel("Year",fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_ylabel("Fraction above\nthreshold", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)

    fig.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=0.1)
    # fig.tight_layout()
    #fig_eng.subplots_adjust(left=0.08, right=0.93, top=0.96, bottom=0.08,
    fig.subplots_adjust(left=0.2, bottom=0.1, right=0.85, top=0.95,
                        wspace=None, hspace=0.7)

    return fig, axes

def plot_la_comparison(england_data):

    ed = england_data
    lo_columns = ed["large_outbreaks"].columns
    non0_columns = ed["la_cases"].loc[:,ed["la_cases"][1990:2004].sum() > 0].columns
    ed_lo = ed["la_pred"].loc[:,lo_columns] #drop(no_columns, axis=1)
    ed_lo_f = ed["la_pred"].loc[:,lo_columns.intersection(non0_columns)] #drop(no_columns, axis=1)
    ed_no_lo = ed["la_pred"].drop(lo_columns, axis=1)
    ed_no_lo_f = ed_no_lo.loc[:,non0_columns]
    print(ed_lo.shape[1], ed_no_lo.shape[1], ed_no_lo_f.shape[1])


    df_positive = pd.DataFrame({"country": ed["country_pred"],
                                "regions": ed["regions_pred"].mean(axis=1),
                                "la_lo": ed_lo.mean(axis=1),
                                "la_lo_f": ed_lo_f.mean(axis=1),
                                "la_no_lo": ed_no_lo.mean(axis=1),
                                "la_no_lo_f": ed_no_lo_f.mean(axis=1)})

    df_positive = df_positive[df_positive.index < 2004]

    pal = h.pal  # ["#11a1b7", "#11b74c", "#f75a5b", "#a64ca6",   "#fee9ce"]
    # reorder pal for this figure
    pal = [pal[3], pal[2], pal[0], pal[1], pal[4],pal[5]]
    pc = h.plot_colours
    axes_label_fontsize = 8
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    fig, axes = plt.subplots(2,1, figsize=(7.2, 4))

    ax = axes[0]
    # ax.plot(df_list[0]/df_list[0].max(), color="black",
    #        label="Normalised cases", alpha=0.2 )
    ax.plot(df_positive["la_lo"],
            label="Large outbreak LAs",
            color=pal[1])
    ax.plot(df_positive["la_no_lo"],
            label="Small outbreak LAs",
            color=pal[2])
    ax.plot(df_positive["la_lo_f"],
            label="Large outbreak LAs (only non-zero LAs)",
            color=pal[4])
    ax.plot(df_positive["la_no_lo_f"],
            label="Small outbreak LAs (only non-zero LAs)",
            color=pal[3])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of LAs above\nthreshold", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    ax.set_title("a)", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlim(1990, 2005)
    ax.set_xticks([1990, 1995, 2000, 2005])
    #ax.tick_params(labelbottom=False)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    ax.legend(fontsize=8, frameon=False, loc=(0.0, 0.3))



    ax = axes[1]
    # ax.plot(df_list[0]/df_list[0].max(), color="black",
    #        label="Normalised cases", alpha=0.2 )
    ax.plot(df_positive["la_lo"] * ed_lo.shape[1],
            label="Large outbreak LAs",
            color=pal[1])
    ax.plot(df_positive["la_no_lo"] * ed_no_lo.shape[1],
            label="Small outbreak LAs",
            color=pal[2])
    ax.plot(df_positive["la_no_lo"] * ed_no_lo.shape[1] +
            df_positive["la_lo"] * ed_lo.shape[1],
            label="All LAs",
            color=pal[5])

    ax.set_ylabel("Number of LAs above\nthreshold", fontsize=axes_label_fontsize,
                  color=pc["axis_text"])
    # ax.set_xlabel("Year", fontsize=axes_label_fontsize,
    #               color=pc["axis_text"])
    ax.set_title("b)", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_xlim(1990, 2005)
    ax.set_xticks([1990, 1995, 2000, 2005])
    ax.set_xlabel("Year", fontsize=axes_label_fontsize, color=pc["axis_text"])
    #ax.tick_params(labelbottom=False)
    sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
    ax.tick_params(axis='both', color=pc["axis"],
                   labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    return fig

#21
df_c_dict["Emergence risk"] = h.lr_emergence_risk(df_c_dict["all"])
dt = h.lr_emergence_risk(df_c_dict["all"])
s = "Emergence risk"
dt = df_c_dict[s]
england_data = get_mumps_england(er_threshold=dt,coefs=coefs_best,
                                 return_signal=s)
df_sim_england = get_simulated_england(england_data,
                                       threshold=dt, coefs=coefs_best,
                                       return_signal=s, rng_seed=93475)

signals = h.signals()[3:6]
p_coef = pd.read_csv(folder+"/pertussis_coefficients_incidence.csv",
                     index_col=0, header=None, squeeze=True)
dt_p = h.lr_emergence_risk(p_coef["c"])
england_data_pertussis = get_mumps_england(er_threshold=dt_p,coefs=p_coef,
                                 return_signal="Emergence risk")
df_sim_pertussis = get_simulated_england(england_data_pertussis,
                                       threshold=dt_p, coefs=p_coef,
                                       return_signal=s, rng_seed=93475)
signals = h.signals()

fig_eng, _ = plot_england_mumps2(england_data, threshold=dt, simulated_data=df_sim_england)

fig_eng_pertussis, _ = plot_england_mumps2(england_data_pertussis,
                                           threshold=dt_p,
                                           simulated_data=df_sim_pertussis)

fig_la_comp = plot_la_comparison(england_data)


fig_la_comp.savefig("./fig_la_comparison_high_dt.png")

for i in np.arange(0,9):
    fig_la, _ = plot_local_authority(england_data, i, 18, threshold=dt)

    fig_la.savefig("./fig_mumps_local_authorities_"+str(i+1)+".png")



fig_eng.savefig("./fig_mumps_england2.png", threshold=dt, dpi=600)
fig_eng.savefig("./fig_mumps_england2.pdf", threshold=dt, dpi=600)

fig_eng_pertussis.savefig("./fig_mumps_england_pertussis.png",
                          threshold=dt, dpi=600)

fig_lt, _ = plot_lead_time_vs_outbreaksize(england_data)
fig_lt.savefig("./fig_lead_time_vs_outbreaksize.png")


fig_er, _ = plot_er_vs_outbreaksize(england_data, year=2004)
fig_er.savefig("./fig_er_vs_outbreaksize.png")


fig_os, _ = plot_outbreak_sizes(england_data)
fig_os.savefig("./fig_gmm_outbreak_size.png")



def get_auc(las_dict):
    def get_la_auc(s):
        if signals != "all":
            dft = pd.DataFrame(dict(
                zip(las_dict.keys(),
                    [las_dict[column][s] for column in las_dict.keys()])))
        else:
            dft = england_data["la_er"]

        # Create dataframe of LAs with (=1) and without (=0) large outbreaks
        df = pd.DataFrame(columns=dft.columns,
                          index=["is_test"])

        for column in df.columns:
            if column in england_data["large_outbreaks"].columns:
                df[column] = 1
            else:
                df[column] = 0

        # Calculate AUC through time
        def f(row):

            er = row.dropna()
            is_test = df[er.index]
            if (er.shape[0] == 0):
                return np.nan
            elif (is_test.loc["is_test"].value_counts().shape[0] == 1):
                return np.nan
            else:
                return metrics.roc_auc_score(is_test.values[0], er)

        auc = np.array(
            [[i, f(row)] for i, row in dft.iterrows()])
        return auc


    auc_dict = {}
    for s in signals + ["Emergence risk"]:
        auc_dict[s] = get_la_auc(s)

    return auc_dict


ed_s = dict(zip(signals,
            [get_mumps_england(er_threshold=df_c_dict[s], coefs=coefs_best,
                               return_signal=s)
             for s in signals]))
ed_s["Emergence risk"] = england_data


def get_sum_stats(ed_s):

    def auc_func(x,is_test):
        y = x[~x.isna()]
        is_test2 = is_test[y.index]
        try:
            a = metrics.roc_auc_score(is_test2,y)
        except ValueError:
            a = np.nan
        return a

    def get_stats_single_signal(s):
        lo = ed_s[s]["large_outbreaks"].columns
        fp = ed_s[s]["la_pred"].drop(lo, axis=1).mean(axis=1)[:2004]
        tp = ed_s[s]["la_pred"][lo].mean(axis=1)[:2004]
        is_test = ed_s[s]['la_er'].apply(lambda x: x.name in lo)
        auc =  ed_s[s]['la_er'].apply(lambda x: auc_func(x,is_test),
                                      axis=1)
        return pd.DataFrame({"signal": s, "fp": fp, "tp": tp,
                             "auc": auc}, index=fp.index)

    return pd.concat([get_stats_single_signal(s) for s in
                      signals+["Emergence risk"]])

ed_sum_s = get_sum_stats(ed_s)


def plot_ews_performance(ed_sum_s):

    sig_cols = h.ews_colours
    sig_labs = h.sig_labels()
    sig_cols["Emergence risk"] = sig_cols['Best fit to simulated data']
    sig_labs["Emergence risk"] = "Best fit to simulated data"

    pc = h.plot_colours
    axes_label_fontsize = 8
    sns.set_style('ticks', {'axes.edgecolor': pc["axis"]})

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(5,7))

    for s, g in ed_sum_s.groupby("signal"):

        ax = axes[0]
        ax.plot(g["tp"], label=sig_labs[s], c=sig_cols[s])

        ax = axes[1]
        ax.plot(g["fp"], label=sig_labs[s], c=sig_cols[s])

        ax = axes[2]
        ax.plot(g["tp"]-g["fp"], label=sig_labs[s], c=sig_cols[s])

        ax = axes[4]
        ax.plot(np.abs(g["auc"]-0.5), label=sig_labs[s], c=sig_cols[s])

        if s not in ["coefficient_of_variation", "kurtosis", "skewness"]:
            ax = axes[3]
            ax.plot(g["tp"] - g["fp"], label=s, c=sig_cols[s])

    axes[0].legend(ncol=1, fontsize=axes_label_fontsize, loc="upper left",
                   bbox_to_anchor=(1, 0))

    for ax in axes[:3]:
        ax.set_ylim(0,1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(1990,2005)
        ax.set_xticks([1992,1996,2000,2004])

    ax = axes[3]
    ax.set_xlim(1998,2005)
    ax.set_xticks([1998, 2000, 2002, 2004])

    for ax in axes[2:4]:
        ax.set_xlabel("Year", fontsize=axes_label_fontsize)
        ax.set_ylim(0,0.65)
        ax.set_yticks([0, 0.2, 0.4, 0.6])

    ax = axes[4]
    ax.set_ylim(0,0.5)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlim(1990,2005)
    ax.set_xticks([1992,1996,2000,2004])

    ax = axes[3]
    ax.set_xlim(1998,2005)
    ax.set_xticks([1998, 2000, 2002, 2004])

    ylabels = ["True positive\nrate", "False positive\nrate",
               "Difference in\npositive rates","Difference in\npositive rates",
               "$|\mathrm{AUC} - 0.5|$"]

    titles = ["a)", "b)", "c)", "d)", "e)"]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], loc="left",fontsize=axes_label_fontsize)
        ax.set_ylabel(ylabels[i], fontsize=axes_label_fontsize)
        ax.set_xlabel("Year", fontsize=axes_label_fontsize)
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False)
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)

    fig.subplots_adjust(left=0.15, bottom=0.08, right=0.65, top=0.97,
                               wspace=0.4, hspace=1.0)

    return fig


fig_ews_perform = plot_ews_performance(ed_sum_s)
fig_ews_perform.savefig("./fig_mumps_ews_performance.png", dpi=600)
