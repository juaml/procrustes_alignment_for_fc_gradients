"""Plot figure 2 of the paper."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seabornxt import stripboxplot
from scipy.stats import pearsonr


def get_typicality_data():
    typicality_path_ica_fix = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "connectome_typicality_hcp_ya_ica_fix.csv"
    )

    typicality_path_no_ica_fix = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "connectome_typicality_hcp_ya_non_ica_fix.csv"
    )

    typicality_df_ica_fix = pd.read_csv(typicality_path_ica_fix)
    typicality_df_ica_fix["session"] = (
        typicality_df_ica_fix["task"] + typicality_df_ica_fix["phase_encoding"]
    )
    typicality_df_ica_fix["Preprocessing"] = [
        "ICA-FIX" for _ in range(typicality_df_ica_fix.shape[0])
    ]

    typicality_df_no_ica_fix = pd.read_csv(typicality_path_no_ica_fix)
    typicality_df_no_ica_fix["session"] = (
        typicality_df_no_ica_fix["task"]
        + typicality_df_no_ica_fix["phase_encoding"]
    )
    typicality_df_no_ica_fix["Preprocessing"] = [
        "Minimal" for _ in range(typicality_df_no_ica_fix.shape[0])
    ]

    typicality_df = pd.concat(
        [typicality_df_ica_fix, typicality_df_no_ica_fix]
    )
    return typicality_df.query("parcellation == 'Schaefer400'")


def load_summary_transformation_matrices(dataset):
    path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "transmats_sum_vals.csv"
    )

    data = pd.read_csv(path)
    data = data.query("sparsity == 0.9")
    data = data.query("parcellation == 'Schaefer400'")
    # data = data.query("alignment == 'procrustes'")
    data = data.query("kernel == 'normalized_angle'")
    data = data.query("approach == 'dm'")
    data = data.query(f"dataset == '{dataset}'")

    data = data.rename(
        columns={
            "mixedness": "correspondence",
        }
    )
    return data


def get_corr_df(transmat_val="transmat_abs_sum"):
    data = {
        "Minimal": load_summary_transformation_matrices(
            "hcp_ya_non_ica_fix"
        ),
        "ICA-FIX": load_summary_transformation_matrices("hcp_ya_ica_fix"),
    }

    typicality = get_typicality_data()
    typicality["session"] = typicality["task"] + typicality["phase_encoding"]
    path_fd = (
        Path("..")
        / ".."
        / "data"
        / "hcp_ya_framewise_displacement_by_phase_encoding.csv"
    )
    fd_df = pd.read_csv(path_fd, index_col=0)
    n_componentses = range(2, 21)
    sessions = ["REST1LR", "REST1RL", "REST2LR", "REST2RL"]

    result = {
        "session": [],
        "Pearson's r (FD)": [],
        "Pearson's r (TFC)": [],
        "n of gradients used\nin alignment": [],
        "Preprocessing": [],
    }
    preprocessings = ["ICA-FIX", "Minimal"]
    for n_components, session, preprocessing in product(
        n_componentses, sessions, preprocessings
    ):
        subset = data[preprocessing].query(f"n_components == {n_components}")
        subset = subset.query(f"session == '{session}'").set_index("subject")
        fd_df_subset = fd_df.loc[subset.index]
        session_fd = session[:5] + "_" + session[-2:]
        r_FD, _ = pearsonr(fd_df_subset[session_fd], subset[transmat_val])

        subset_fct = (
            typicality.query(f"session == '{session}'")
            .query(f"Preprocessing == '{preprocessing}'")
            .set_index("subject")
            .loc[subset.index]
        )

        r_FCT, _ = pearsonr(subset_fct["typicality"], subset[transmat_val])

        result["session"].append(session)
        result["Pearson's r (FD)"].append(r_FD)
        result["Pearson's r (TFC)"].append(r_FCT)
        result["n of gradients used\nin alignment"].append(n_components)
        result["Preprocessing"].append(preprocessing)

    return pd.DataFrame(result)


def plot_correlation_all(ax, y):
    result_df = get_corr_df("transmat_abs_sum")

    plot = sns.lineplot(
        data=result_df,
        x="n of gradients used\nin alignment",
        y=y,
        hue="Preprocessing",
        ax=ax,
        hue_order=["ICA-FIX", "Minimal"],
        linewidth=0.6,
        markers=["x", "x"],
        style="Preprocessing",
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    # sns.move_legend(
    #    ax, "center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, 0.9)
    # )
    ax.get_legend().remove()
    # plot.set(ylim=(0, 0.3))
    # plot.legend_.set_title(None)
    return ax


def plot_correlation_first_column(ax, y, remove_legend):
    result_df = get_corr_df("transmat_abs_sum_first_column")

    plot = sns.lineplot(
        data=result_df,
        x="n of gradients used\nin alignment",
        y=y,
        hue="Preprocessing",
        ax=ax,
        hue_order=["ICA-FIX", "Minimal"],
        linewidth=0.6,
        markers=["x", "x"],
        style="Preprocessing",
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    # plot.set(ylim=(0, 0.3))
    if not remove_legend:
        sns.move_legend(
            ax,
            # "center",
            ncol=1,
            fontsize=8,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )
    else:
        ax.get_legend().remove()
    # plot.legend_.set_title(None)
    return ax


def main():
    cm = 1 / 2.54
    with plt.style.context("default.mplstyle"):
        plt.rcParams["text.latex.preamble"] = r"\boldmath"
        # dpi = 300
        fig = plt.figure(figsize=(14 * cm, 12 * cm))
        grid = fig.add_gridspec(2, 2)

        ax_a = fig.add_subplot(grid[0, 0], aspect="auto")
        ax_a = plot_correlation_all(ax_a, y="Pearson's r (FD)")
        ax_a.set_title("a", fontweight="bold", fontsize=12, loc="left")
        ax_a.set_ylim(-0.1, 0.4)

        ax_b = fig.add_subplot(grid[0, 1], aspect="auto")
        ax_b = plot_correlation_first_column(
            ax_b, y="Pearson's r (FD)", remove_legend=False
        )
        ax_b.set_title("b", fontweight="bold", fontsize=12, loc="left")
        ax_b.set_ylim(-0.1, 0.4)

        ax_c = fig.add_subplot(grid[1, 0], aspect="auto")
        ax_c = plot_correlation_all(ax_c, y="Pearson's r (TFC)")
        ax_c.set_title("c", fontweight="bold", fontsize=12, loc="left")
        ax_c.set_ylim(-0.8, 0)

        ax_d = fig.add_subplot(grid[1, 1], aspect="auto")
        ax_d = plot_correlation_first_column(
            ax_d, y="Pearson's r (TFC)", remove_legend=True
        )
        ax_d.set_title("d", fontweight="bold", fontsize=12, loc="left")
        ax_d.set_ylim(-0.8, 0)

        outpath = Path("..") / ".." / "figures" / "paper"
        fig.savefig(outpath / f"paper_figure_2.png")
        fig.savefig(outpath / f"paper_figure_2.svg")
        fig.savefig(outpath / f"paper_figure_2.pdf")


if __name__ == "__main__":
    main()
