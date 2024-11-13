"""Plot regression scores (figure 6 of gradient alignment paper)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl


cm = 1 / 2.54


def format_correlations(r):
    r_str = f"{r:.2f}"
    reached_dot = False
    new_str = []
    if "-" in r_str:
        new_str.append("-")
    for s in r_str:
        if s == ".":
            reached_dot = True

        if reached_dot:
            new_str.append(s)

    return "".join(new_str)


def get_correlation_matrix_camcan():
    data_path = Path("..") / ".." / "data"
    concatenated_results_path = Path("..") / ".." / "results" / "concatenated"
    cattel_path = data_path / "camcan_cattel.txt"

    camcan_subjects = list(
        np.loadtxt(data_path / "camcan_analysis_subjects.txt", dtype=str)
    )

    camcan_fd = (
        pd.read_csv(data_path / "camcan_framewise_displacement.csv")
        .set_index("subject")
        .loc[camcan_subjects]
    )
    camcan_participants = (
        pd.read_csv(data_path / "camcan_participants.tsv", sep="\t")
        .set_index("participant_id")
        .loc[camcan_subjects]
    )
    camcan_tfc = (
        pd.read_csv(
            concatenated_results_path / "connectome_typicality_camcan.csv"
        )
        .query("parcellation == 'Schaefer400'")
        .set_index("subject")
        .loc[camcan_subjects]
    ).drop(columns="parcellation")

    camcan_transformations = (
        pd.read_csv(concatenated_results_path / "transmats_sum_vals.csv")
        .query("dataset == 'camcan'")
        .query("approach == 'dm'")
        .query("kernel == 'normalized_angle'")
        .query("sparsity == 0.9")
        .query("n_components == 10")
        .query("parcellation == 'Schaefer400'")
        .set_index("subject")
        .loc[camcan_subjects]
    ).drop(columns="parcellation")

    with open(cattel_path, "r") as f:
        lines = f.read().split("\n")

        adjusted = lines[8:-10]
        df_cattel = pd.DataFrame([sub[:-1].split("\t") for sub in adjusted])
        df_cattel.columns = df_cattel.iloc[0]
        df_cattel = df_cattel.drop(index=0)
        df_cattel = df_cattel.set_index("CCID")
        df_cattel.columns = [f"cattel_{x}" for x in df_cattel]

    correlates = {
        "age": "Age",
        "tiv_cubicmm": "TIV ($mm^3$)",
        "typicality": "TFC",
        "FD": "FD",
        "cattel_TotalScore": "CFIT Score",
        "transmat_abs_sum": "Transformation",
        "mixedness": "Correspondence",
    }
    camcan_joined = (
        camcan_participants.join(camcan_tfc)
        .join(camcan_fd)
        .join(df_cattel)
        .join(camcan_transformations)
    )[correlates.keys()]

    camcan_joined = camcan_joined.rename(columns=correlates)
    return camcan_joined.corr()


def load_regression_intelligence(parcellation, model):
    path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "regression_intelligence.csv"
    )

    data = pd.read_csv(path)
    data = (
        data.query(f"parcellation == '{parcellation}'")
        .query(f"model == '{model}'")
        .query("target != 'CogTotalComp_Unadj'")
        .query("approach == 'dm'")
        .query("kernel == 'normalized_angle'")
        .query("sparsity == 0.9")
    )

    data["confounds"] = data["confounds"].fillna("No confounds")

    data = data[
        [
            "dataset",
            "session",
            "target",
            "confounds",
            "n_components",
            "test_r2",
            "repeat",
            "fold",
        ]
    ]
    return (
        data.groupby(
            [
                "dataset",
                "session",
                "target",
                "confounds",
                "n_components",
                "repeat",
            ]
        )
        .mean()
        .reset_index()
    )


def load_camcan_age_prediction_results(parcellation, model):
    path = (
        Path("..")
        / ".."
        / "results"
        / "concatenated"
        / "camcan_age_prediction.csv"
    )
    data = pd.read_csv(path)
    data = (
        data.query(f"parcellation == '{parcellation}'")
        .query(f"model == '{model}'")
        .query("approach == 'dm'")
        .query("kernel == 'normalized_angle'")
        .query("sparsity == 0.9")
    )

    data = data[
        [
            "removeconfounds",
            "n_components",
            "test_r2",
            "repeat",
            "fold",
        ]
    ]
    return (
        data.groupby(
            [
                "removeconfounds",
                "n_components",
                "repeat",
            ]
        )
        .mean()
        .reset_index()
    )


def lineplot(data, ax):
    plot = sns.lineplot(
        data,
        ax=ax,
        x="            n of gradients used in alignment",
        y="$R^2$",
        hue="confounds",
        hue_order=["No confounds", "FD", "FD+Age"],
        linewidth=0.6,
        markers=["x", "x", "x"],
        style="confounds",
        markersize=1.6,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
        # palette="husl",
    )
    return plot


def lineplot_diff(data, ax):
    plot = sns.lineplot(
        data,
        ax=ax,
        x="            n of gradients used in alignment",
        y="Derivative ($R^2$)",
        hue="confounds",
        hue_order=["No confounds", "FD", "FD+Age"],
        linewidth=0.6,
        markers=["x", "x", "x"],
        style="confounds",
        markersize=1.6,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
        # palette="husl",
    )
    return plot


def lineplot_age_prediction(data, ax):

    plot = sns.lineplot(
        data,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="$R^2$",
        hue="confounds",
        hue_order=["FD not removed", "FD removed"],
        linewidth=0.6,
        markers=["x", "x"],
        style="confounds",
        markersize=1.6,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
        # palette="husl",
    )
    return plot


def lineplot_diff_age_prediction(data, ax):

    plot = sns.lineplot(
        data,
        ax=ax,
        x="n of gradients used\nin alignment",
        y="Derivative ($R^2$)",
        hue="confounds",
        hue_order=["FD not removed", "FD removed"],
        linewidth=0.6,
        markers=["x", "x"],
        style="confounds",
        markersize=1.6,
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
        # palette="husl",
    )
    return plot


def get_diff(data):
    confounds = ["No confounds", "FD", "FD+Age"]
    diff_df_list = []
    for confound in confounds:
        df = (
            data.query(f"confounds == '{confound}'")
            .groupby("            n of gradients used in alignment")
            .mean(numeric_only=True)
            .reset_index()
        )
        df["Derivative ($R^2$)"] = df["$R^2$"].diff()

        df["confounds"] = [confound for x in range(df.shape[0])]

        df = df[~df["Derivative ($R^2$)"].isna()]
        diff_df_list.append(df)

    return pd.concat(diff_df_list, axis=0)


def get_diff_age_prediction(data):
    confounds = ["FD removed", "FD not removed"]
    diff_df_list = []
    for confound in confounds:
        df = (
            data.query(f"confounds == '{confound}'")
            .groupby("n of gradients used\nin alignment")
            .mean(numeric_only=True)
            .reset_index()
        )
        df["Derivative ($R^2$)"] = df["$R^2$"].diff()

        df["confounds"] = [confound for x in range(df.shape[0])]

        df = df[~df["Derivative ($R^2$)"].isna()]
        diff_df_list.append(df)

    return pd.concat(diff_df_list, axis=0)


def plot_hcp_ya(data, ax_scores, ax_diff):

    data = data.query("dataset == 'hcp_ya_ica_fix' & session == 'REST1_LR'")
    assert data.shape[0] == 570  # 3 * 19 * 10 results
    plot = lineplot(data, ax_scores)
    ax_scores.get_legend().remove()
    plot.set(ylim=(-0.15, 0.2), xlim=(1, 21), xlabel="", xticklabels=[])

    diff_df = get_diff(data)
    plot_diff = lineplot_diff(diff_df, ax_diff)
    plot_diff.set(
        ylim=(-0.02, 0.05),
        xlim=(1, 21),
        xlabel="",
        # xticklabels=[],
    )
    ax_diff.get_legend().remove()

    return plot, plot_diff


def plot_aomic_piop1(data, ax_scores, ax_diff):

    data = data.query("dataset == 'aomic_piop1'")
    assert data.shape[0] == 570  # 3 * 19 * 10 results
    plot = lineplot(data, ax_scores)
    ax_scores.get_legend().remove()
    plot.set(
        ylim=(-0.15, 0.2),
        xlim=(1, 21),
        xlabel="",
        xticklabels=[],
        ylabel="",
        yticklabels=[],
    )

    diff_df = get_diff(data)
    plot_diff = lineplot_diff(diff_df, ax_diff)
    plot_diff.set(
        ylim=(-0.02, 0.05),
        xlim=(1, 21),
        ylabel="",
        yticklabels=[],
    )
    ax_diff.get_legend().remove()

    return plot, plot_diff


def plot_aomic_piop2(data, ax_scores, ax_diff):

    data = data.query("dataset == 'aomic_piop2'")
    assert data.shape[0] == 570  # 3 * 19 * 10 results
    plot = lineplot(data, ax_scores)
    ax_scores.get_legend().remove()
    plot.set(
        ylim=(-0.15, 0.2),
        xlim=(1, 21),
        xlabel="",
        xticklabels=[],
        ylabel="",
        yticklabels=[],
    )

    diff_df = get_diff(data)
    plot_diff = lineplot_diff(diff_df, ax_diff)
    plot_diff.set(
        ylim=(-0.02, 0.05),
        xlim=(1, 21),
        ylabel="",
        xlabel="",
        yticklabels=[],
    )
    ax_diff.get_legend().remove()

    return plot, plot_diff


def plot_camcan(data, ax_scores, ax_diff):

    data = data.query("dataset == 'camcan'")

    assert data.shape[0] == 570  # 3 * 19 * 10 results
    plot = lineplot(data, ax_scores)
    # ax_scores.get_legend().remove()
    plot.set(
        ylim=(-0.15, 0.2),
        xlim=(1, 21),
        xlabel="",
        xticklabels=[],
        ylabel="",
        yticklabels=[],
    )

    diff_df = get_diff(data)
    plot_diff = lineplot_diff(diff_df, ax_diff)
    plot_diff.set(
        ylim=(-0.02, 0.05),
        xlim=(1, 21),
        xlabel="",
        ylabel="",
        yticklabels=[],
    )
    ax_diff.get_legend().remove()

    return plot, plot_diff


def plot_camcan_age_prediction(data, ax_scores, ax_diff):

    plot = lineplot_age_prediction(data, ax_scores)
    ax_scores.get_legend().remove()
    plot.set(
        # ylim=(-0.15, 0.2),
        xlim=(1, 21),
        xlabel="",
        xticklabels=[],
        # ylabel="",
        # yticklabels=[],
    )

    diff_df = get_diff_age_prediction(data)

    plot_diff = lineplot_diff_age_prediction(diff_df, ax_diff)
    plot_diff.set(
        # ylim=(-0.02, 0.05),
        xlim=(1, 21),
        # ylabel="",
        # yticklabels=[],
    )
    ax_diff.get_legend().remove()

    return plot, plot_diff


def main():
    parcellation = "Schaefer400"
    model = "ridge"
    regression_results = load_regression_intelligence(parcellation, model)
    regression_results = regression_results.rename(
        columns={
            "test_r2": "$R^2$",
            "n_components": "            n of gradients used in alignment",
        }
    )
    assert regression_results.shape[0] == 3990

    camcan_age_prediction = load_camcan_age_prediction_results(
        parcellation, model
    ).rename(
        columns={
            "test_r2": "$R^2$",
            "n_components": "n of gradients used\nin alignment",
            "removeconfounds": "confounds",
        }
    )

    camcan_age_prediction["confounds"] = camcan_age_prediction[
        "confounds"
    ].replace({True: "FD removed", False: "FD not removed"})

    correlation_matrix = get_correlation_matrix_camcan()

    outpath = Path("..") / ".." / "figures" / "paper" / "paper_figure_6"

    with plt.style.context("default.mplstyle"):
        fig = plt.figure(figsize=(14 * cm, 16 * cm))
        grid = fig.add_gridspec(5, 4, height_ratios=[15, 25, 25, 25, 25])

        # HCP YA
        ax_a_scores = fig.add_subplot(grid[1, 0])
        ax_a_scores.set_title("a HCP-YA", loc="left", fontweight="bold")

        ax_b_diff = fig.add_subplot(grid[2, 0])
        ax_b_diff.set_title("e", loc="left", fontweight="bold")
        plot_hcp_ya(regression_results, ax_a_scores, ax_b_diff)

        # AOMIC PIOP1
        ax_c_scores = fig.add_subplot(grid[1, 1])
        ax_c_scores.set_title("b PIOP1", loc="left", fontweight="bold")

        ax_d_diff = fig.add_subplot(grid[2, 1])
        ax_d_diff.set_title("f", loc="left", fontweight="bold")
        plot_aomic_piop1(regression_results, ax_c_scores, ax_d_diff)

        # AOMIC PIOP2
        ax_e_scores = fig.add_subplot(grid[1, 2])
        ax_e_scores.set_title("c PIOP2", loc="left", fontweight="bold")

        ax_f_diff = fig.add_subplot(grid[2, 2])
        ax_f_diff.set_title("g", loc="left", fontweight="bold")
        plot_aomic_piop2(regression_results, ax_e_scores, ax_f_diff)

        # CAMCAN
        ax_g_scores = fig.add_subplot(grid[1, 3])
        ax_g_scores.set_title("d Cam-CAN", loc="left", fontweight="bold")

        ax_h_diff = fig.add_subplot(grid[2, 3])
        ax_h_diff.set_title("h", loc="left", fontweight="bold")
        plot_camcan(regression_results, ax_g_scores, ax_h_diff)

        handles, labels = ax_g_scores.get_legend_handles_labels()
        ax_g_scores.get_legend().remove()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1),
            title="Confounds",
        )

        # CAMCAN age prediction
        ax_i_scores = fig.add_subplot(grid[3, 0])
        ax_i_scores.set_title("i Cam-CAN", loc="left", fontweight="bold")

        ax_j_diff = fig.add_subplot(grid[4, 0])
        ax_j_diff.set_title("j", loc="left", fontweight="bold")
        plot_camcan_age_prediction(
            camcan_age_prediction, ax_i_scores, ax_j_diff
        )

        ax_k = fig.add_subplot(grid[3:, 2:])
        ax_k.set_title("Cam-CAN", loc="left", fontweight="bold")

        ax_cb = fig.add_subplot(grid[3:, 1])
        # ax_cb.set_title("k", loc="left", fontweight="bold")
        ax_cb.axis("off")

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax_cb)
        cax = divider.append_axes("right", size="10%", pad=0.1)
        cax.set_title("k", loc="left", fontweight="bold")

        annotations = correlation_matrix.map(format_correlations)
        mask = np.tril(np.ones(correlation_matrix.shape, dtype=bool))
        heatmap = sns.heatmap(
            ax=ax_k,
            data=correlation_matrix,
            cbar=False,
            mask=mask,
            cmap="bwr",
            vmin=-1,
            vmax=1,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 8},
        )
        ax_k.yaxis.tick_right()
        ax_k.set_yticklabels(ax_k.get_yticklabels(), rotation=300, fontsize=6)

        ax_k.set_xticklabels(ax_k.get_xticklabels(), fontsize=6)

        cmap = plt.get_cmap("bwr")
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cb = mpl.colorbar.ColorbarBase(
            cax,
            cmap=cmap,
            norm=norm,
            orientation="vertical",
            ticklocation="left",
        )

    fig.savefig(f"{outpath}.pdf")
    fig.savefig(f"{outpath}.png")
    fig.savefig(f"{outpath}.svg")


if __name__ == "__main__":
    main()
