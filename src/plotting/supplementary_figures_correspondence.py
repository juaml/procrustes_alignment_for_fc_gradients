from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seabornxt import stripboxplot
from scipy.stats import pearsonr

cm = 1 / 2.54


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
    data = data.query("n_components < 21")
    data = data.rename(
        columns={
            "mixedness": "correspondence",
        }
    )
    data = data.drop(
        columns=["dataset", "principal_gradient_was_in_right_place"]
    )
    data["kernel"] = data["kernel"].fillna("No kernel")
    return data


def main(parcellation):
    data = load_summary_transformation_matrices()
    data = data.query(f"parcellation == '{parcellation}'")
    data = data.rename(
        columns={
            "n_components": "n gradients",
        }
    )
    kernels = [
        "spearman",
        "pearson",
        "normalized_angle",
        "gaussian",
        "cosine",
        "No kernel",
    ]
    approaches = ["dm", "le", "pca"]

    with plt.style.context("default.mplstyle"):
        fig = plt.figure(figsize=(17 * cm, 17 * cm))
        grid = fig.add_gridspec(6, 4)

        for (row, kernel), (col, approach) in product(
            enumerate(kernels), enumerate(approaches)
        ):

            subset = data.copy().query(
                f"(approach == '{approach}') and (kernel == '{kernel}')"
            )

            ax = fig.add_subplot(grid[row, col])

            plot = sns.lineplot(
                data=subset,
                x="n gradients",
                y="correspondence",
                hue="session",
                hue_order=["REST1LR", "REST1RL", "REST2LR", "REST2RL"],
                ax=ax,
                linewidth=0.6,
                markers=["x", "x", "x", "x"],
                style="session",
                markersize=1.6,
                markeredgecolor="k",
                markeredgewidth=0.2,
                dashes=False,
                errorbar="sd",
            )

            plot.set(ylim=(0.2, 0.9))

            if row == 0 and col == 2:
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(1, 1),
                    title=None,
                )
            else:
                ax.get_legend().remove()

            pad = 3
            if row == 0:
                ax.annotate(
                    f"approach = {approach}",
                    xy=(0.5, 1),
                    xytext=(0, pad),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    size=6,
                    ha="center",
                    va="baseline",
                )

            if col != 0:
                plot.set(ylabel=None, yticklabels=[])
            else:
                ax.annotate(
                    f"kernel = {kernel}",
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size=6,
                    ha="right",
                    va="center",
                )

            if row != 5:
                plot.set(xlabel=None, xticklabels=[])

        outpath = Path("..") / ".." / "figures" / "supplementary"
        outpath.mkdir(exist_ok=True, parents=True)
        outname = outpath / f"parc-{parcellation}_supplementary_correspondence"
        fig.savefig(f"{outname}.pdf")
        fig.savefig(f"{outname}.png")
        fig.savefig(f"{outname}.svg")


if __name__ == "__main__":
    for parcellation in [100, 200, 400]:
        main(f"Schaefer{parcellation}")
