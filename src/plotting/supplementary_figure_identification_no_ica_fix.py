"""Plot figure 1 of the paper but without ICA FIX data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seabornxt import stripboxplot


def load_identification_results():
    path = (
        Path("..") / ".." / "results" / "concatenated" / "identification.csv"
    )

    data = pd.read_csv(path)
    data = data.query("parcellation == 'Schaefer400'")
    data = data.query("icafix == False")
    data = data.query("alignment == 'procrustes'")
    data = data.query("kernel == 'normalized_angle'")
    data = data.query("sparsity == 0.9").reset_index(drop=True)

    data["approach"] = data["approach"].replace(
        {
            "dm": "Diffusion\nMaps",
            "le": "Laplacian\nEigenmaps",
            "pca": "Principal\nComponents",
        }
    )

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

    data = data.query("dataset == 'hcp_ya_non_ica_fix'")

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


def plot_IAcc(ax):
    data = load_identification_results()
    iacc_vars = [x for x in data.columns if "IAcc" in x]
    select = ["n_components", "approach"] + iacc_vars
    data_melt = data[select].melt(
        id_vars=["n_components", "approach"],
        value_vars=iacc_vars,
        value_name="Identification Accuracy",
    )
    data_melt = data_melt.rename(
        columns={"n_components": "n of gradients used\nin alignment"}
    )

    plot = sns.lineplot(
        data_melt,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="Identification Accuracy",
        hue="approach",
        hue_order=[
            "Diffusion\nMaps",
            "Laplacian\nEigenmaps",
            "Principal\nComponents",
        ],
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
    return ax


def plot_IDiff(ax):
    data = load_identification_results()

    idiff_vars = [
        x for x in data.columns if "IDiff" in x and "r_to_z" not in x
    ]
    select = ["n_components", "approach"] + idiff_vars
    data_melt = data[select].melt(
        id_vars=["n_components", "approach"],
        value_vars=idiff_vars,
        value_name="Differential Identifiability",
    )
    data_melt = data_melt.rename(
        columns={"n_components": "n of gradients used\nin alignment"}
    )

    plot = sns.lineplot(
        data_melt,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="Differential Identifiability",
        hue="approach",
        hue_order=[
            "Diffusion\nMaps",
            "Laplacian\nEigenmaps",
            "Principal\nComponents",
        ],
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
    return ax


def plot_IDiff_r_to_z(ax):
    data = load_identification_results()

    idiff_vars = [x for x in data.columns if "IDiff_r_to_z" in x]
    select = ["n_components", "approach"] + idiff_vars
    data_melt = data[select].melt(
        id_vars=["n_components", "approach"],
        value_vars=idiff_vars,
        value_name="Differential Identifiability\n(with r-to-z)",
    )
    data_melt = data_melt.rename(
        columns={"n_components": "n of gradients used\nin alignment"}
    )

    plot = sns.lineplot(
        data_melt,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="Differential Identifiability\n(with r-to-z)",
        hue="approach",
        hue_order=[
            "Diffusion\nMaps",
            "Laplacian\nEigenmaps",
            "Principal\nComponents",
        ],
        linewidth=2,
        errorbar="sd",
        markers=["x", "x", "x"],
        style="approach",
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    sns.move_legend(
        ax,
        # "center",
        ncol=1,
        fontsize=8,
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )
    return ax


def plot_correspondence(ax):
    data = load_summary_transformation_matrices()
    data = data.rename(
        columns={
            "n_components": "n of gradients used\nin alignment",
        }
    )

    plot = stripboxplot(
        data=data,
        x="n of gradients used\nin alignment",
        y="correspondence",
        hue="session",
        # linewidth=0.1,
        # showfliers=False,
        strip_kwargs={
            "jitter": True,
            "alpha": 0.4,
            "color": "k",
            "s": 0.8,
            "hue_order": ["REST1LR", "REST1RL", "REST2LR", "REST2RL"],
        },
        box_kwargs={
            "hue_order": ["REST1LR", "REST1RL", "REST2LR", "REST2RL"],
            "showfliers": False,
        },
        ax=ax,
    )
    plot.set(ylim=(0, 1))
    # plot.legend_.set_title(None)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize=4)
    return ax


def main():
    cm = 1 / 2.54
    with plt.style.context("default.mplstyle"):
        plt.rcParams["text.latex.preamble"] = r"\boldmath"
        # dpi = 300
        fig = plt.figure(figsize=(17 * cm, 10 * cm))
        grid = fig.add_gridspec(2, 3)

        ax_a = fig.add_subplot(grid[0, 0])
        ax_a = plot_IAcc(ax_a)
        ax_a.set_title("a", fontweight="bold", fontsize=12, loc="left")

        ax_b = fig.add_subplot(grid[0, 1])
        ax_b = plot_IDiff(ax_b)
        ax_b.set_title("b", fontweight="bold", fontsize=12, loc="left")

        ax_c = fig.add_subplot(grid[0, 2])
        ax_c = plot_IDiff_r_to_z(ax_c)
        ax_c.set_title("c", fontweight="bold", fontsize=12, loc="left")

        ax_d = fig.add_subplot(grid[1, :])
        ax_d = plot_correspondence(ax_d)
        ax_d.set_title("d", fontweight="bold", fontsize=12, loc="left")

        outpath = Path("..") / ".." / "figures" / "supplementary"
        fig.savefig(outpath / "supplementary_figure_identification_no_ica_fix.png")
        fig.savefig(outpath / "supplementary_figure_identification_no_ica_fix.svg")
        fig.savefig(outpath / "supplementary_figure_identification_no_ica_fix.pdf")


if __name__ == "__main__":
    main()
