"""Calculate correlations between BOLD and FD time series."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from junifer.storage import HDF5FeatureStorage
from tqdm import tqdm


def load_bold_timeseries(dataset, parcellation):
    """Load BOLD time series for a given parcellation."""
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"
    # ugly fix but necessary
    if "hcp_ya" in dataset:
        data_subj = "hcp_ya"
    else:
        data_subj = dataset

    subjects_main_analysis = list(
        np.loadtxt(path / f"{data_subj}_analysis_subjects.txt", dtype=str)
    )
    subjects_holdout = list(
        np.loadtxt(path / f"{data_subj}_holdout_subjects.txt", dtype=str)
    )

    storage = HDF5FeatureStorage(
        path / dataset / f"{dataset}.hdf5",
        single_output=False,
    )

    marker = f"BOLD_parccortical-{parcellation}x7FSLMNI_marker-timeseries"
    marker_df = storage.read_df(marker)

    assert marker_df.isna().sum().sum() == 0, "NaN values in the time series."
    assert not np.isinf(
        marker_df.values
    ).any(), "Inf values in the time series."
    return marker_df


def main(dataset):
    """Run the main program."""
    # ugly fix but necessary
    if "hcp_ya" in dataset:
        data_subj = "hcp_ya"
    else:
        data_subj = dataset

    subjects = np.loadtxt(
        Path("..") / ("..") / "data" / f"{data_subj}_subject_list.txt",
        dtype=str,
    )
    fd_timeseries_path = (
        Path("..") / ("..") / "data" / f"{data_subj}_fd_timeseries.csv"
    )
    fd_timeseries = pd.read_csv(fd_timeseries_path, index_col=0)

    outpath = Path("..") / ".." / "results" / "fd_correlations"

    parcellations = ["Schaefer400"]

    if "hcp_ya" in dataset:
        sessions = ["REST1", "REST2"]
        phase_encodings = ["LR", "RL"]

        for parcellation in parcellations:
            bold_timeseries = load_bold_timeseries(dataset, parcellation)
            all_correlations_list = []
            column_names = []
            print(f"Running dataset: {dataset}; parcellation: {parcellation}")
            for subject, session, phase_encoding in tqdm(
                product(subjects, sessions, phase_encodings)
            ):
                subset = (
                    bold_timeseries.query(f"subject == '{subject}'")
                    .query(f"task == '{session}'")
                    .query(f"phase_encoding == '{phase_encoding}'")
                    .copy()
                )

                corr = subset.reset_index(drop=True).corrwith(
                    fd_timeseries[
                        f"{subject}_{session}_{phase_encoding}"
                    ].reset_index(drop=True)
                )

                column_names.append(f"{subject}_{session}_{phase_encoding}")
                all_correlations_list.append(corr)

            all_correlations_df = pd.concat(all_correlations_list, axis=1)
            all_correlations_df.columns = column_names
            all_correlations_df.to_csv(
                outpath / f"{dataset}_{parcellation}_correlations_with_FD.csv"
            )

    elif dataset in ["aomic_piop1", "aomic_piop2"]:
        for parcellation in parcellations:
            print(f"Running dataset: {dataset}; parcellation: {parcellation}")
            all_correlations_list = []
            bold_timeseries = load_bold_timeseries(dataset, parcellation)
            for subject in tqdm(subjects):
                subset = (
                    bold_timeseries.query(f"subject == '{subject}'")
                    .query("task == 'restingstate'")
                    .copy()
                )
                corr = pd.DataFrame(
                    subset.reset_index(drop=True).corrwith(
                        fd_timeseries[subject].reset_index(drop=True)
                    )
                )
                corr.columns = [subject]
                all_correlations_list.append(corr)

            all_correlations_df = pd.concat(all_correlations_list, axis=1)

            all_correlations_df.to_csv(
                outpath / f"{dataset}_{parcellation}_correlations_with_FD.csv"
            )

    elif dataset == "camcan":
        for parcellation in parcellations:
            print(f"Running dataset: {dataset}; parcellation: {parcellation}")
            all_correlations_list = []
            bold_timeseries = load_bold_timeseries(dataset, parcellation)
            for subject in tqdm(subjects):
                subset = (
                    bold_timeseries.query(f"subject == '{subject}'")
                    .query("task == 'rest'")
                    .copy()
                )
                corr = pd.DataFrame(
                    subset.reset_index(drop=True).corrwith(
                        fd_timeseries[subject].reset_index(drop=True)
                    )
                )
                corr.columns = [subject]
                all_correlations_list.append(corr)

            all_correlations_df = pd.concat(all_correlations_list, axis=1)

            all_correlations_df.to_csv(
                outpath / f"{dataset}_{parcellation}_correlations_with_FD.csv"
            )


if __name__ == "__main__":
    for dataset in [
        "camcan",
        "hcp_ya_non_ica_fix",
        "hcp_ya_ica_fix",
        "aomic_piop1",
        "aomic_piop2",
    ]:
        main(dataset)
