from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hcp_ya(data, ax, scoring_metric):
    data = data.query("dataset == 'hcp_ya_ica_fix'")
    plot = sns.lineplot(
        data=data,
        x="n of gradients used\nin alignment",
        y=scoring_metric,
        hue="session",
        style="session",
        errorbar="sd",
        ax=ax,
        linewidth=0.6,
        markers=["x", "x", "x", "x"],
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    if scoring_metric == "Accuracy":
        ax.get_legend().remove()
    else:
        sns.move_legend(
            ax,
            # "center",
            ncol=1,
            fontsize=8,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )
    plot.set(xticklabels=[], xlabel="")
    return ax


def plot_aomic(data, ax, scoring_metric):
    data = data.query("dataset == 'aomic_piop1' or dataset == 'aomic_piop2'")
    data["dataset"] = data["dataset"].replace(
        {"aomic_piop1": "PIOP1", "aomic_piop2": "PIOP2"}
    )
    plot = sns.lineplot(
        data=data,
        x="n of gradients used\nin alignment",
        y=scoring_metric,
        hue="dataset",
        style="dataset",
        errorbar="sd",
        ax=ax,
        linewidth=0.6,
        markers=["x", "x"],
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    if scoring_metric == "Accuracy":
        ax.get_legend().remove()
    else:
        sns.move_legend(
            ax,
            # "center",
            ncol=1,
            fontsize=8,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )
    plot.set(xticklabels=[], xlabel="")
    return ax


def plot_camcan(data, ax, scoring_metric):
    data = data.query("dataset == 'camcan'")

    plot = sns.lineplot(
        data=data,
        x="n of gradients used\nin alignment",
        y=scoring_metric,
        errorbar="sd",
        ax=ax,
        linewidth=0.6,
        style="dataset",
        markers=["x"],
        markeredgecolor="k",
        markeredgewidth=0.2,
        dashes=False,
    )
    ax.get_legend().remove()
    return ax


def load_classification_results():
    data_path = (
        Path("..") / ".." / "results" / "concatenated" / "classification.csv"
    )

    classification_results_df = pd.read_csv(data_path).query(
        "parcellation == 'Schaefer400' &"
        " kernel == 'normalized_angle' &"
        " sparsity == 0.9 &"
        " approach == 'dm' &"
        " model == 'ridge'"
    )

    classification_results_df = classification_results_df.rename(
        columns={
            "n_components": "n of gradients used\nin alignment",
            "test_accuracy": "Accuracy",
            "test_roc_auc": "AUC",
        }
    )
    select_cols = [
        "dataset",
        "session",
        "n of gradients used\nin alignment",
        "Accuracy",
        "AUC",
        "repeat",
        "fold",
    ]
    return (
        classification_results_df[select_cols]
        .groupby(
            [
                "dataset",
                "session",
                "n of gradients used\nin alignment",
                "repeat",
            ]
        )
        .mean(numeric_only=True)
        .reset_index()
    )


def main():
    classification_results_df = load_classification_results()

    cm = 1 / 2.54
    with plt.style.context("default.mplstyle"):
        fig = plt.figure(figsize=(17 * cm, 17 * cm))
        rows, cols = 3, 2
        grid = fig.add_gridspec(rows, cols)

        ax_a = fig.add_subplot(grid[0, 0])
        ax_a.set_title("a", loc="left")
        ax_a = plot_hcp_ya(
            classification_results_df, ax=ax_a, scoring_metric="Accuracy"
        )
        ax_a.set_ylim(0.4, 0.7)
        ax_a.axhline(y=0.5, linewidth=2, color="black", ls=":")

        ax_b = fig.add_subplot(grid[0, 1])
        ax_b.set_title("b", loc="left")
        ax_b = plot_hcp_ya(
            classification_results_df, ax=ax_b, scoring_metric="AUC"
        )
        ax_b.set_ylim(0.4, 0.7)
        ax_b.axhline(y=0.5, linewidth=2, color="black", ls=":")

        ax_c = fig.add_subplot(grid[1, 0])
        ax_c.set_title("c", loc="left")
        ax_c = plot_aomic(
            classification_results_df, ax=ax_c, scoring_metric="Accuracy"
        )
        ax_c.set_ylim(0.4, 0.7)
        ax_c.axhline(y=0.5, linewidth=2, color="black", ls=":")

        ax_d = fig.add_subplot(grid[1, 1])
        ax_d.set_title("d", loc="left")
        ax_d = plot_aomic(
            classification_results_df, ax=ax_d, scoring_metric="AUC"
        )
        ax_d.set_ylim(0.4, 0.7)
        ax_d.axhline(y=0.5, linewidth=2, color="black", ls=":")

        ax_e = fig.add_subplot(grid[2, 0])
        ax_e.set_title("e", loc="left")
        ax_e = plot_camcan(
            classification_results_df, ax=ax_e, scoring_metric="Accuracy"
        )
        ax_e.set_ylim(0.4, 0.7)
        ax_e.axhline(y=0.5, linewidth=2, color="black", ls=":")

        ax_f = fig.add_subplot(grid[2, 1])
        ax_f.set_title("f", loc="left")
        ax_f = plot_camcan(
            classification_results_df, ax=ax_f, scoring_metric="AUC"
        )
        ax_f.set_ylim(0.4, 0.7)
        ax_f.axhline(y=0.5, linewidth=2, color="black", ls=":")

    fig.savefig(Path("..") / ".." / "figures" / "paper" / "paper_figure_7.png")
    fig.savefig(Path("..") / ".." / "figures" / "paper" / "paper_figure_7.pdf")
    fig.savefig(Path("..") / ".." / "figures" / "paper" / "paper_figure_7.svg")


if __name__ == "__main__":
    main()
