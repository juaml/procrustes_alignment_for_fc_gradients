#!/usr/bin/env python


from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    sessions, phase_encodings = ["REST1", "REST2"], ["LR", "RL"]
    subject_list_path = Path("..") / ".." / "data" / "hcp_ya_subject_list.txt"
    outpath = Path("..") / ".." / "data" / "hcp_ya_fd_timeseries.csv"
    subject_list = np.loadtxt(subject_list_path, dtype=str)
    path_to_confdir = (
        Path("/data")
        / "group"
        / "appliedml"
        / "data"
        / "HCP1200_Confounds_tsv_withFD"
    )
    result = pd.DataFrame()
    for subject, session, phase_encoding in product(
        subject_list, sessions, phase_encodings
    ):
        path_element = (
            path_to_confdir
            / subject
            / "MNINonLinear"
            / "Results"
            / f"rfMRI_{session}_{phase_encoding}"
            / f"Confounds_{subject}.tsv"
        )
        confounds_data = pd.read_csv(path_element, sep="\t")

        result[f"{subject}_{session}_{phase_encoding}"] = confounds_data[
            "FD"
        ].copy()

    result.to_csv(outpath)


if __name__ == "__main__":
    main()
