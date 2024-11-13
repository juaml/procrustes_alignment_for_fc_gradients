"""Plot supplementary figure 1 of the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_identification_results():
    path = (
        Path("..") / ".." / "results" / "concatenated" / "identification.csv"
    )

    data = pd.read_csv(path)
    data = data.query("icafix == True")
    data = data.query("parcellation == 'Schaefer400'")
    data = data.query("alignment == 'procrustes'")
    data = data.query("kernel == 'normalized_angle'")
    data = data.query("sparsity == 0.9").reset_index(drop=True)

    return data


def load_summary_transformation_matrices():
    path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "transmats_sum_vals.csv"
    )
    data = pd.read_csv(path)

    data = data.query("dataset == 'hcp_ya_ica_fix'")

    data = data.query("sparsity == 0.9")
    data = data.query("parcellation == 'Schaefer400'")
    data = data.query("kernel == 'normalized_angle'")
    data = data.query("approach == 'dm'")
    data = data.query("n_components < 21")
    data = data.rename(
        columns={
            "mixedness": "correspondence",
        }
    )
    return data


def plot_within(ax):
    data = load_identification_results()
    iacc_vars = [
        x for x in data.columns if "mean_within_subject_correlation" in x
    ]
    select = ["n_components", "approach"] + iacc_vars
    data_melt = data[select].melt(
        id_vars=["n_components", "approach"],
        value_vars=iacc_vars,
        value_name="mean within-subject corr.",
    )
    data_melt = data_melt.rename(
        columns={"n_components": "n of gradients used\nin alignment"}
    )

    plot = sns.lineplot(
        data_melt,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="mean within-subject corr.",
        hue="approach",
        hue_order=["dm", "le", "pca"],
        linewidth=2,
        errorbar="sd",
        markers=["x", "x", "x"],
        style="approach",
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    plot.legend_.remove()
    plot.set(ylim=(0.5, 1))
    return ax


def plot_between(ax):
    data = load_identification_results()
    idiff_vars = [
        x for x in data.columns if "mean_between_subject_correlation" in x
    ]
    select = ["n_components", "approach"] + idiff_vars
    data_melt = data[select].melt(
        id_vars=["n_components", "approach"],
        value_vars=idiff_vars,
        value_name="mean between-subject corr.",
    )
    data_melt = data_melt.rename(
        columns={"n_components": "n of gradients used\nin alignment"}
    )

    plot = sns.lineplot(
        data_melt,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="mean between-subject corr.",
        hue="approach",
        hue_order=["dm", "le", "pca"],
        linewidth=2,
        errorbar="sd",
        markers=["x", "x", "x"],
        style="approach",
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    plot.set(ylim=(0.5, 1))
    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1, 1),
        title=None,
    )

    return ax


def main():
    cm = 1 / 2.54
    with plt.style.context("default.mplstyle"):
        plt.rcParams["text.latex.preamble"] = r"\boldmath"
        # dpi = 300
        fig = plt.figure(figsize=(16 * cm, 10 * cm))
        grid = fig.add_gridspec(2, 2)

        ax_a = fig.add_subplot(grid[0, 0])
        ax_a = plot_within(ax_a)
        ax_a.set_title("a", fontweight="bold", fontsize=12, loc="left")

        ax_b = fig.add_subplot(grid[0, 1])
        ax_b = plot_between(ax_b)
        ax_b.set_title("b", fontweight="bold", fontsize=12, loc="left")

        outpath = (
            Path("..")
            / ".."
            / "figures"
            / "supplementary"
            / "supplementary_figure_1"
        )
        fig.savefig(f"{outpath}.png")
        fig.savefig(f"{outpath}.svg")
        fig.savefig(f"{outpath}.pdf")


if __name__ == "__main__":
    main()
