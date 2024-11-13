from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cm = 1 / 2.54


def main():
    """Create the main figure."""

    # Load the main data to be used for plotting here
    path = (
        Path("..")
        / ".."
        / "results"
        / "fd_correlations"
        / "hcp_ya_non_ica_fix_Schaefer400_correlations_with_FD.csv"
    )
    data = pd.read_csv(path, index_col=0)
    dfs = []
    sessions, phase_encodings = ["REST1", "REST2"], ["LR", "RL"]
    for session, phase_encoding in product(sessions, phase_encodings):
        subset = (
            data.filter(regex=f".*{session}_{phase_encoding}")
            .stack()
            .reset_index()
            .rename(columns={0: "correlation"})
        )
        subset["Session"] = [
            f"{session}_{phase_encoding}" for _ in range(len(subset))
        ]
        dfs.append(subset)
    concatenated = pd.concat(dfs)
    concatenated["Session"] = concatenated["Session"].replace(
        {
            "REST1_LR": "REST1LR",
            "REST1_RL": "REST1RL",
            "REST2_LR": "REST2LR",
            "REST2_RL": "REST2RL",
        }
    )
    # import the style sheet
    with plt.style.context("default.mplstyle"):
        # set the figure size here
        fig = plt.figure(figsize=(16 * cm, 12 * cm))

        # make a grid of plots to make multiple panels
        # the example here will give you axes for 4 subplots in a 2x2 grid
        rows, cols = 2, 2
        grid = fig.add_gridspec(rows, cols)

        sessions = ["REST1LR", "REST1RL", "REST2LR", "REST2RL"]
        titles = ["a", "b", "c", "d"]
        colors = [
            (230 / 255, 75 / 255, 53 / 255, 1),
            (77 / 255, 187 / 255, 213 / 255, 1),
            (0, 160 / 255, 135 / 255, 1),
            (60 / 255, 84 / 255, 136 / 255, 1),
        ]
        # iterate over every position/sub-plot in the figure
        for index, (row, col) in enumerate(product(range(rows), range(cols))):
            title = titles[index]
            session = sessions[index]
            color = colors[index]
            subset = concatenated.query(f"Session == '{session}'")
            ax = fig.add_subplot(grid[row, col], aspect="auto")
            plot = sns.histplot(
                subset,
                x="correlation",
                kde=True,
                hue="Session",
                bins=40,
                palette=[color],
                ax=ax,
                stat="count",
            )
            plot.set(xlim=(-0.2, 0.2))
            ax.lines[0].set_color("darkslategrey")
            ax.get_legend().remove()
            ax.set_title(title, fontweight="bold", fontsize=12, loc="left")
            ax.annotate(session, (10, 100), xycoords="axes points", c=color)
            mean = subset["correlation"].mean().round(2)
            sd = subset["correlation"].std().round(2)
            ax.annotate(
                f"M = {mean}", (10, 90), xycoords="axes points", c=color
            )
            ax.annotate(
                f"SD = {sd}", (10, 80), xycoords="axes points", c=color
            )

        # define an outpath where to save your figures
        outpath = Path("..") / ".." / "figures" / "supplementary"
        outpath.mkdir(exist_ok=True, parents=True)
        outname = (
            outpath
            / "supplementary_figure_correlation_histograms_hcp_no_ica_fix"
        )
        fig.savefig(f"{outname}.pdf")
        fig.savefig(f"{outname}.png")
        fig.savefig(f"{outname}.svg")


if __name__ == "__main__":
    main()
