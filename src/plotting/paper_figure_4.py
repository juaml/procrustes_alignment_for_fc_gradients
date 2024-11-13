"""Plot figure 3 of the paper."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from seabornxt import stripboxplot


def load_summary_transformation_matrices():
    path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "transmats_sum_vals.csv"
    )
    datasets_aomic = ["aomic_piop1", "aomic_piop2"]
    data = pd.read_csv(path)

    data = data.query("dataset in @datasets_aomic")
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

    data["dataset"] = data["dataset"].replace(
        {"aomic_piop1": "PIOP1", "aomic_piop2": "PIOP2"}
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

    dataset_names = {"PIOP1": "aomic_piop1", "PIOP2": "aomic_piop2"}
    n_componentses = range(2, 21)
    datasets = ["PIOP1", "PIOP2"]

    result = {
        "dataset": [],
        "Pearson's r (FD)": [],
        "Pearson's r (TFC)": [],
        "n of gradients used\nin alignment": [],
    }
    for n_components, dataset in product(n_componentses, datasets):
        dataset_name = dataset_names[dataset]
        typicality = get_typicality_data(dataset_name)
        path_fd = (
            Path("..")
            / ".."
            / "data"
            / f"{dataset_name}_framewise_displacement.csv"
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
        hue="dataset",
        ax=ax,
        hue_order=["PIOP1", "PIOP2"],
        linewidth=0.6,
        markers=["x", "x"],
        style="dataset",
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    if remove_legend:
        ax.get_legend().remove()
    else:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize=8)
    return ax


def plot_correlations_timeseries(ax, dataset):
    dataset_colors = {
        "aomic_piop1": (230 / 255, 75 / 255, 53 / 255, 1),
        "aomic_piop2": (77 / 255, 187 / 255, 213 / 255, 1),
    }
    dataset_names = {
        "aomic_piop1": "PIOP1",
        "aomic_piop2": "PIOP2",
    }
    path = (
        Path("..")
        / ".."
        / "results"
        / "fd_correlations"
        / f"{dataset}_Schaefer400_correlations_with_FD.csv"
    )
    data = pd.read_csv(path, index_col=0)
    data = data.reset_index().melt(
        value_vars=data.columns.to_list(),
        value_name="Pearson's r",
        id_vars="index",
    )
    data["dataset"] = [dataset_names[dataset] for _ in range(data.shape[0])]
    plot = sns.histplot(
        data,
        x="Pearson's r",
        kde=True,
        hue="dataset",
        bins=20,
        palette=[dataset_colors[dataset]],
        ax=ax,
        stat="count",
    )
    plot.set(xlim=(-0.2, 0.2))
    ax.lines[0].set_color("darkslategrey")
    ax.get_legend().remove()
    ax.annotate(
        dataset_names[dataset],
        (10, 100),
        xycoords="axes points",
        c=dataset_colors[dataset],
    )
    mean = data["Pearson's r"].mean().round(2)
    sd = data["Pearson's r"].std().round(2)
    ax.annotate(
        f"M = {mean}",
        (10, 90),
        xycoords="axes points",
        c=dataset_colors[dataset],
    )
    ax.annotate(
        f"SD = {sd}",
        (10, 80),
        xycoords="axes points",
        c=dataset_colors[dataset],
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
        hue="dataset",
        # linewidth=0.1,
        # showfliers=False,
        strip_kwargs={"jitter": True, "alpha": 0.4, "color": "k", "s": 1.2},
        box_kwargs={
            "hue_order": ["PIOP1", "PIOP2"],
            "showfliers": False,
        },
        ax=ax,
    )
    plot.set(ylim=(0, 1))
    # plot.legend_.set_title(None)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize=8)
    return ax


def main():
    cm = 1 / 2.54

    with plt.style.context("default.mplstyle"):
        plt.rcParams["text.latex.preamble"] = r"\boldmath"
        # dpi = 300
        fig = plt.figure(figsize=(16 * cm, 18 * cm))
        grid = fig.add_gridspec(3, 2)

        ax_a = fig.add_subplot(grid[0, :])
        ax_a = plot_correspondence(ax_a)
        ax_a.set_title("a", fontweight="bold", fontsize=12, loc="left")

        ax_b = fig.add_subplot(grid[1, 0])
        ax_b = plot_correlations_fd(ax_b, "Pearson's r (FD)", True)
        ax_b.set_title("b", fontweight="bold", fontsize=12, loc="left")
        ax_b.set_ylim(-0.6, 0.1)

        ax_c = fig.add_subplot(grid[1, 1])
        ax_c = plot_correlations_fd(ax_c, "Pearson's r (TFC)", False)
        ax_c.set_title("c", fontweight="bold", fontsize=12, loc="left")
        ax_c.set_ylim(-0.6, 0.1)

        ax_d = fig.add_subplot(grid[2, 0])
        ax_d = plot_correlations_timeseries(ax_d, "aomic_piop1")
        ax_d.set_title("d", fontweight="bold", fontsize=12, loc="left")

        ax_e = fig.add_subplot(grid[2, 1])
        ax_e = plot_correlations_timeseries(ax_e, "aomic_piop2")
        ax_e.set_title("e", fontweight="bold", fontsize=12, loc="left")

        outpath = Path("..") / ".." / "figures" / "paper"
        fig.savefig(outpath / "paper_figure_4.png")
        fig.savefig(outpath / "paper_figure_4.svg")
        fig.savefig(outpath / "paper_figure_4.pdf")


if __name__ == "__main__":
    main()
