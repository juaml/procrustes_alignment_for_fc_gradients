"""Plot figure 3 of the paper."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seabornxt import stripboxplot
from scipy.stats import pearsonr


def load_summary_transformation_matrices():
    path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "transmats_sum_vals.csv"
    )
    data = pd.read_csv(path)
    data = data.query("dataset == 'camcan'")
    data = data.query("sparsity == 0.9")
    data = data.query("parcellation == 'Schaefer400'")
    # data = data.query("alignment == 'procrustes'")
    data = data.query("kernel == 'normalized_angle'")
    data = data.query("approach == 'dm'")

    data = data.rename(
        columns={
            "mixedness": "correspondence",
        }
    )

    return data


def get_typicality_data(dataset):
    typicality_path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / f"connectome_typicality_{dataset}.csv"
    )
    typicality_df = pd.read_csv(typicality_path)
    return typicality_df.query("parcellation == 'Schaefer400'")


def get_corr_df(transmat_val="transmat_abs_sum"):
    data = load_summary_transformation_matrices()

    n_componentses = range(2, 21)
    datasets = ["camcan"]
    result = {
        "dataset": [],
        "Pearson's r (FD)": [],
        "Pearson's r (TFC)": [],
        "n of gradients used\nin alignment": [],
    }
    for n_components, dataset in product(n_componentses, datasets):

        typicality = get_typicality_data(dataset)
        path_fd = (
            Path("..")
            / ".."
            / "data"
            / f"{dataset}_framewise_displacement.csv"
        )
        fd_df = pd.read_csv(path_fd, index_col=0)
        subset = data.query(f"n_components == {n_components}")
        subset = subset.query(f"dataset == '{dataset}'").set_index("subject")
        fd_df_subset = fd_df.loc[subset.index]

        r_FD, _ = pearsonr(fd_df_subset["FD"], subset[transmat_val])

        subset_fct = typicality.set_index("subject").loc[subset.index]

        r_FCT, _ = pearsonr(subset_fct["typicality"], subset[transmat_val])

        result["dataset"].append(dataset)
        result["Pearson's r (FD)"].append(r_FD)
        result["Pearson's r (TFC)"].append(r_FCT)
        result["n of gradients used\nin alignment"].append(n_components)

    return pd.DataFrame(result)


def plot_correlations_fd(ax, y, remove_legend):
    result_df = get_corr_df("transmat_abs_sum")
    plot = sns.lineplot(
        data=result_df,
        x="n of gradients used\nin alignment",
        y=y,
        ax=ax,
        linewidth=0.6,
        markers=["x"],
        style="dataset",
        markeredgecolor="k",
        markeredgewidth=1,
        dashes=False,
    )
    ax.get_legend().remove()

    return ax


def plot_correlation_timeseries(ax):

    path = (
        Path("..")
        / ".."
        / "results"
        / "fd_correlations"
        / "camcan_Schaefer400_correlations_with_FD.csv"
    )
    data = pd.read_csv(path, index_col=0)

    data = data.reset_index().melt(
        value_vars=data.columns.to_list(),
        value_name="Pearson's r",
        id_vars="index",
    )

    plot = sns.histplot(
        data,
        x="Pearson's r",
        kde=True,
        bins=20,
        ax=ax,
        stat="count",
    )
    return ax


def plot_correspondence(ax):
    data = load_summary_transformation_matrices()
    # data = data.query("n_components < 11")
    data = data.rename(
        columns={
            "n_components": "n of gradients used\nin alignment",
        }
    )
    plot = stripboxplot(
        data=data,
        x="n of gradients used\nin alignment",
        y="correspondence",
        # linewidth=0.1,
        # showfliers=False,
        strip_kwargs={"jitter": True, "alpha": 0.4, "color": "k", "s": 1.2},
        box_kwargs={
            "showfliers": False,
        },
        ax=ax,
    )
    plot.set(ylim=(0, 1))

    xticks = plot.get_xticks()
    xticklabels = plot.get_xticklabels()

    new_xticklabels = [
        label if i % 2 == 0 else "" for i, label in enumerate(xticklabels)
    ]

    plot.set(xticks=xticks, xticklabels=new_xticklabels)
    return ax


def main():
    cm = 1 / 2.54

    with plt.style.context("default.mplstyle"):
        plt.rcParams["text.latex.preamble"] = r"\boldmath"
        # dpi = 300
        fig = plt.figure(figsize=(13 * cm, 13 * cm))
        grid = fig.add_gridspec(2, 2)

        ax_a = fig.add_subplot(grid[0, 0])
        ax_a = plot_correspondence(ax_a)
        ax_a.set_title("a", fontweight="bold", fontsize=12, loc="left")

        ax_b = fig.add_subplot(grid[0, 1])
        ax_b = plot_correlation_timeseries(ax_b)
        ax_b.set_title("b", fontweight="bold", fontsize=12, loc="left")

        ax_c = fig.add_subplot(grid[1, 0])
        ax_c = plot_correlations_fd(ax_c, "Pearson's r (FD)", True)
        ax_c.set_title("c", fontweight="bold", fontsize=12, loc="left")
        ax_c.set_ylim(-0.6, 0.1)

        ax_d = fig.add_subplot(grid[1, 1])
        ax_d = plot_correlations_fd(ax_d, "Pearson's r (TFC)", False)
        ax_d.set_title("d", fontweight="bold", fontsize=12, loc="left")
        ax_d.set_ylim(-0.6, 0.1)

        outpath = Path("..") / ".." / "figures" / "paper"
        fig.savefig(outpath / "paper_figure_5.png")
        fig.savefig(outpath / "paper_figure_5.svg")
        fig.savefig(outpath / "paper_figure_5.pdf")


if __name__ == "__main__":
    main()
