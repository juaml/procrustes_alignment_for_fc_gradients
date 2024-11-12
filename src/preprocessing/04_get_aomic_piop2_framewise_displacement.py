import os
import tempfile
from pathlib import Path

import datalad.api as dl
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    """Run main program."""
    subject_list_path = (
        Path("..") / ".." / "data" / "aomic_piop2_subject_list.txt"
    )
    subject_list = list(np.loadtxt(subject_list_path, dtype=str))
    framewise_displacement_list = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        start_dir = os.getcwd()
        os.chdir(tmp)
        dl.clone("https://github.com/OpenNeuroDatasets/ds002790.git", tmp)
        print("Iterating over subjects to get framewise displacements")
        for sub in tqdm(subject_list):
            confounds_file = (
                tmp
                / "derivatives"
                / "fmriprep"
                / f"sub-{sub}"
                / "func"
                / f"sub-{sub}_task-restingstate_acq-seq_desc-confounds_regressors.tsv"
            )
            dl.get(confounds_file)
            confounds = pd.read_csv(confounds_file, sep="\t")

            framewise_displacement_list.append(
                pd.DataFrame(confounds["framewise_displacement"]).rename(
                    columns={"framewise_displacement": sub}
                )
            )

        os.chdir(start_dir)
        framewise_displacement_df = pd.concat(
            framewise_displacement_list, axis=1
        ).fillna(0)

        framewise_displacement_df_mean = pd.DataFrame(
            framewise_displacement_df.mean()
        ).rename(columns={0: "FD"})
        framewise_displacement_df_mean.index.name = "subject"
        framewise_displacement_df_mean.to_csv(
            Path("..")
            / ".."
            / "data"
            / "aomic_piop2_framewise_displacement.csv"
        )

        framewise_displacement_df.to_csv(
            Path("..") / ".." / "data" / "aomic_piop2_fd_timeseries.csv",
        )


if __name__ == "__main__":
    main()
