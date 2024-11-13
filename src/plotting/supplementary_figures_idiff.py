"""Plot differetial identifiability for different parameters."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cm = 1 / 2.54


def main(parcellation, alignment):
    """Plot differential identifiability for different parameters."""
    path = (
        Path("..") / ".." / "results" / "concatenated" / "identification.csv"
    )

    data = pd.read_csv(path, index_col=0)
    data = data.query(f"parcellation == '{parcellation}'")
    data = data.query("icafix == True")

    iacc_vars = [x for x in data.columns if "IDiff" in x and "r_to_z" not in x]
    select = ["n_components", "approach", "kernel", "sparsity"] + iacc_vars
    data_melt = data[select].melt(
        id_vars=["n_components", "approach", "kernel", "sparsity"],
        value_vars=iacc_vars,
        value_name="Diff. Ident.",
    )
    data_melt = data_melt.rename(columns={"n_components": "n gradients"})
    data_melt["kernel"] = data_melt["kernel"].fillna("No kernel")

    with plt.style.context("default.mplstyle"):
        fig = plt.figure(figsize=(17 * cm, 17 * cm))
        grid = fig.add_gridspec(6, 4)

        sparsities = [0, 0.3, 0.6, 0.9]
        kernels = [
            "spearman",
            "pearson",
            "normalized_angle",
            "gaussian",
            "cosine",
            "No kernel",
        ]
        for (row, kernel), (col, sparsity) in product(
            enumerate(kernels), enumerate(sparsities)
        ):
            subset = data_melt.copy().query(
                f"(sparsity == {sparsity}) and (kernel == '{kernel}')"
            )

            ax = fig.add_subplot(grid[row, col])

            plot = sns.lineplot(
                subset,
                ax=ax,
                x="n gradients",
                y="Diff. Ident.",
                hue="approach",
                hue_order=["dm", "le", "pca"],
                linewidth=0.6,
                markers=["x", "x", "x"],
                style="approach",
                markersize=1.6,
                markeredgecolor="k",
                markeredgewidth=0.2,
                dashes=False,
                palette="husl",
            )
            if row == 0 and col == 3:
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
                    f"sparsity = {sparsity}",
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

            plot.set(ylim=(0, 18))
            if row != 5:
                plot.set(xlabel=None, xticklabels=[])

        outpath = Path("..") / ".." / "figures" / "supplementary"
        outpath.mkdir(exist_ok=True, parents=True)
        outname = outpath / f"parc-{parcellation}_supplementary_idiff"
        fig.savefig(f"{outname}.pdf")
        fig.savefig(f"{outname}.png")
        fig.savefig(f"{outname}.svg")


if __name__ == "__main__":
    parcellations = [f"Schaefer{x}" for x in [100, 200, 400]]
    alignments = ["minimal", "procrustes"]
    for parcellation, alignment in product(parcellations, alignments):
        main(parcellation, alignment)
