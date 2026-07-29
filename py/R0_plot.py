import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import py.helper as h
import matplotlib.lines as mpl


folder = "./data/backup-01-14" # Change from zenodo

# Read data
params_df = pd.read_csv(folder + "/simulation_parameters.csv", index_col=0)


params_df = pd.concat([params_df[:20], params_df[-20:]])

df = pd.concat(
    [pd.read_csv(folder + "/data_" + str(i) + ".csv")
        .assign(model=i).assign(
        is_test=int(params_df.loc[i, "R0_f"] == 1.0))
     for i in params_df.index.values])


df.index = df["time"]/365 - 20
# need to aggregate into months?


def r0_convexity_plot(df, params_df, figsize=(8, 3)):

    is_convex = params_df["bb_a"].apply(lambda x: x > 1)


    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(2, 2)
    axes = np.array([plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]),
                     plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])])
    colours = [h.pal[0]] + [h.pal[2]]
    custom_lines = [mpl.Line2D([0], [0], color=colours[1], lw=1.5),
                    mpl.Line2D([0], [0], color=colours[0], lw=1.5)]
    dfg = df.groupby(["model", "is_test"])

    for i, g in dfg:
        if i[1] == 1:
            axes[0].plot(g.iloc[:-1]["R0"], color=colours[int(is_convex[i[0]])],
                         alpha=0.4, label="_nolegend_")
            axes[2].plot(g["reported_cases"],
                         color=colours[int(is_convex[i[0]])],
                         alpha=0.4, label="_nolegend_")
        else:
            axes[1].plot(g.iloc[:-1]["R0"], color='0.2',
                         alpha=0.4, label="_nolegend_")
            axes[3].plot(g["reported_cases"], color='0.2',
                         alpha=0.4, label="_nolegend_")

    ax = axes[0]
    ax.set_ylabel("$R_0$")
    ax.set_ylim(0., 1)
    ax.set_title("Emerging")
    ax.legend(custom_lines, ['Convex', 'Concave'], ncol=2, frameon=False,
              loc=(0.3,-0.1))

    ax = axes[1]
    ax.set_ylim(0., 1)
    ax.set_title("Not emerging")

    ax = axes[2]
    ax.set_ylabel("Reported cases")
    ax.set_xlabel("Lead time")
    ax.set_ylim(0., 20)


    ax = axes[3]
    ax.set_xlabel("Lead time")
    ax.set_ylim(0., 20)

    for ax in axes:
        ax.set_xlim(-10,0)
        sns.despine(ax=ax, offset=2, trim=True, right=True, left=False)


    fig.tight_layout(rect=(0, 0.1, 1, 1))

    return fig, axes


figr0, _ = r0_convexity_plot(df, params_df)

figr0.savefig("./fig_r0_convexity.png")
