"""Plot supplementary boxplots of fd."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from seabornxt import stripboxplot


def load_fd():
    data_path = Path("..") / ".." / "data"

    fd_list = []
    for dataset in ["aomic_piop1", "aomic_piop2", "camcan"]:

        fd_data = pd.read_csv(
            data_path / f"{dataset}_framewise_displacement.csv"
        )
        fd_data = fd_data.assign(
            dataset=[dataset for _ in range(fd_data.shape[0])]
        )
        fd_list.append(fd_data)


    hcp_data = pd.read_csv(
        data_path / "hcp_ya_framewise_displacement_by_phase_encoding.csv"
    )

    sessions = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]
    hcp_data = hcp_data[["subjects"] + sessions]

    hcp_data = hcp_data.melt(
        id_vars="subjects",
        value_vars=sessions,
        var_name="dataset",
        value_name="FD",
    ).rename(columns={"subjects": "subject"})

    fd_list.append(hcp_data)
    return pd.concat(fd_list)


def main():

    fd = load_fd()

    cm = 1 / 2.54
    with plt.style.context("default.mplstyle"):
        plt.rcParams["text.latex.preamble"] = r"\boldmath"
        # dpi = 300
        fig = plt.figure(figsize=(16 * cm, 16 * cm))
        grid = fig.add_gridspec(1, 1)

        ax_a = fig.add_subplot(grid[0, 0])
        ax_a.set_title("a", fontweight="bold", fontsize=12, loc="left")

        
        stripboxplot(
            data=fd,
            x="dataset",
            y="FD",
            # hue="dataset",
            # linewidth=0.1,
            # showfliers=False,
            strip_kwargs={
                "jitter": True,
                "alpha": 0.4,
                "color": "k",
                "s": 0.8,
            },
            box_kwargs={
                "showfliers": False,
            },
            ax=ax_a,
        )

        outpath = (
            Path("..")
            / ".."
            / "figures"
            / "supplementary"
            / "supplementary_boxplots_fd"
        )
        fig.savefig(f"{outpath}.png")
        fig.savefig(f"{outpath}.svg")
        fig.savefig(f"{outpath}.pdf")


if __name__ == "__main__":
    main()
