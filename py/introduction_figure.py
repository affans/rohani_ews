from subprocess import call
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import py.helper as h

rng_seed = 2478
def intro_sim_args(fsim):
    # load default simulator arguments

    sim_args = {}#h.simulator_args(folder=fsim)
    sim_args["folder"] = fsim

    sim_args["runs"] = 1
    sim_args["T"] = 40
    sim_args["R0_rs"] = 15
    sim_args["R0_rd"] = 20
    sim_args["R0_i"] = 0.2
    sim_args["N_i"] = 1E5
    sim_args["N_f"] = sim_args["N_i"]
    sim_args["rp_i"] = 0.25
    sim_args["rp_f"] = sim_args["rp_i"]
    sim_args["bb_a"] = 2
    sim_args["seed"] = rng_seed

    sim_args["R0_ramp"] = "bb"
    sim_args["R0_f"] = 3.
    sim_args["R0_u"] = sim_args["R0_f"]
    sim_args["eta_i"] = 1 / 7
    sim_args["eta_f"] = 1 / 7

    sim_args["v_i"] = 0
    sim_args["v_f"] = 0
    sim_args["v_rs"] = 10
    sim_args["v_rd"] = 0


    # sim_args["rp_f"] = 0.6

    return sim_args

simulator = "./SEIR-Simulator-0.2.5/seir_simulator_gamma"

e_sim_args = intro_sim_args("./data/intro/")

pl = [k + '=' + str(e_sim_args[k]) for k in e_sim_args.keys()]
comm = [simulator] + pl

#+ pl
call(comm)

df = pd.read_csv("./data/intro/epi_data.csv")
df["N"] = df[["S","I","R","V"]].sum(axis=1)
df["Reff"] = (df["S"]/df["N"])*df["R0"]
df["Reff_df"] = (1-df["V"]/df["N"])*df["R0"]
df["month"] = df["time"] // 28

df_m = pd.DataFrame(df.groupby("month")["reported_cases"].sum(axis=1))
df_m["Reff"] = df.groupby("month")["Reff"].mean()
df_m["R0"] = df.groupby("month")["R0"].mean()

df_m.index /= 13
df_m.index -= df_m[df_m["R0"] < 1].index[-1]


df_england = pd.read_csv("./data/mumps_england_4week.csv", index_col=0,
                         header=None)



tdf = pd.read_csv("./data/pertussis.51.12.csv", header=0, sep=",")
states_list = tdf.columns[2:]
tdf["time"] = tdf.YEAR + tdf.MONTH / 12.
tdf = (tdf.fillna(method="ffill") + tdf.fillna(method="bfill")) / 2
tdf = tdf.set_index("time")

#tdf = tdf.groupby("YEAR").sum(axis=0)

tdf["country"] = tdf.iloc[:,2:].sum(axis=1)


def plot_introduction():

    fig, axes = plt.subplots(3, 1, sharex=False, sharey=False,
                             figsize=(3.6, 3.))

    pc = h.plot_colours
    axes_label_fontsize = 8
    ts_colour = h.long_pal[7]
    ts_colour2 = h.long_pal[2]
    r0_colour = h.long_pal[8]

    sns.set_style('ticks', {'axes.edgecolor':pc["axis"]})


    ax = axes[0]
    ax.set_xlim(-12,4)
    ax.set_xticks([-12,-8,-4,0,4])

    # Log10
    # ax.set_ylim(0,3)
    # yticks = np.array([0, 1, 2, 3])
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(["$10^" + str(i) + "$" for i in yticks])

    # Sqrt
    ax.set_ylim(0,27)
    yticks = np.array([0, 9, 18, 27])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i**2) for i in yticks])


    ax.set_ylabel("Monthy cases", color=pc["axis_text"],
                  fontsize=axes_label_fontsize)
    ax.set_xlabel("Lead time", color=pc["axis_text"],
                  fontsize=axes_label_fontsize)
    ax.set_title("a) simulation of emergence", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])


    axR = ax.twinx()
    axR.set_ylim(-0.5, 1.7)
    axR.set_yticks([0., 0.5, 1,1.5])
    axR.set_ylabel("$R_{\mathrm{eff}}$            ", color=pc["axis_text"],
                  fontsize=axes_label_fontsize)
    axR.tick_params(right=True, labelright=True)
    sns.despine(ax=axR, offset=2, trim=True, bottom=True, right=False,
                left=True)
    axR.axvline(0,  color=pc["axis"], linestyle="--", alpha=1,
                linewidth=1)

    ax.annotate("critical transition $R_{\mathrm{eff}}=1$", xy=(-0.1, 8), xytext=(-2, 18),
                fontSize=axes_label_fontsize, color=r0_colour,
                arrowprops=dict(facecolor=r0_colour, shrink=0.5,
                                width=2, headwidth=4),
                horizontalalignment='right'
                )

    axR.plot(df_m["R0"], color=r0_colour, alpha=0.7)

    ax.plot(np.sqrt(df_m.loc[:0, "reported_cases"]), color=ts_colour,
            linewidth=1)
    ax.plot(np.sqrt(df_m.loc[0:, "reported_cases"]), color=ts_colour2,
            linewidth=1)



    ax = axes[1]
    ax.plot(np.sqrt(df_england), color=ts_colour, linewidth=1)
    ax.set_xlim(1992,2008)
    ax.set_xticks([1992,1996,2000,2004,2008])

    ax.set_ylim(0,27)
    yticks = np.array([0, 9, 18, 27])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i**2) for i in yticks])

    ax.set_title("b) mumps in England", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_ylabel("Monthy cases", color=pc["axis_text"],
                  fontsize=axes_label_fontsize)

    ax = axes[2]
    ax.plot(tdf["New.York"], color=ts_colour, linewidth=1)
    ax.set_ylim(-6,350)
    ax.set_xlim(1968,2006)
    ax.set_xticks([1968,1978,1988,1998,2008])
    ax.set_yticks([0,100,200,300])
    ax.set_title("c) pertussis in New York state", fontsize=axes_label_fontsize,
                 loc="left", color=pc["axis_text"])
    ax.set_ylabel("Monthy cases", color=pc["axis_text"],
                  fontsize=axes_label_fontsize)

    for ax in axes:
        ax.tick_params(axis='both', color=pc["axis"],
                       labelcolor=pc["axis_text"],
                       which='both', labelsize=axes_label_fontsize)
        sns.despine(ax=ax, offset=5, trim=True, right=True, left=False,
                    bottom=False)

    axR.tick_params(axis='both', color=pc["axis"], labelcolor=pc["axis_text"],
                   which='both', labelsize=axes_label_fontsize)

    fig.subplots_adjust(left=0.19, bottom=0.1, right=0.86, top=0.94,
                        wspace=None, hspace=1.5)

    return fig, axes

fig_intro, _ = plot_introduction()
#plt.show()
fig_intro.savefig("./fig_introduction_sqrt.png", dpi=600)

fig_intro.savefig("./fig_introduction_sqrt.svg", dpi=600)
fig_intro.savefig("./fig_introduction_sqrt.pdf", dpi=600)


