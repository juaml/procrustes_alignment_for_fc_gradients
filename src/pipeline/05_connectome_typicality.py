"""Calculate connectome typicality."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from junifer.storage import HDF5FeatureStorage
from tqdm import tqdm


def typicality(subject_gradient, template, correlation_method="pearson"):
    """Calculate typicality of each subject gradient."""
    correlation = subject_gradient.corr(
        pd.Series(template), method=correlation_method
    )
    return (1 + correlation) / 2


def load_connectomes(dataset, parcellation):
    """Load connectomes for a given parcellation."""
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

    marker = f"BOLD_parccortical-{parcellation}x7FSLMNI_marker-empiricalFC"
    marker_df = storage.read_df(marker)

    # filter diagonal
    non_diagonal = []
    for col in marker_df:
        roi1, roi2 = col.split("~")
        if roi1 != roi2:
            non_diagonal.append(col)

    marker_df = marker_df[non_diagonal]

    # select main analysis subjects and holdout subjects
    connectomes_main_analysis = marker_df.query(
        "subject in @subjects_main_analysis"
    )
    holdout_connectomes = marker_df.query("subject in @subjects_holdout")

    holdout_connectome = (
        holdout_connectomes.apply(np.arctanh, axis=1).mean().apply(np.tanh)
    )

    assert (
        connectomes_main_analysis.isna().sum().sum() == 0
    ), "NaN values in the connectomes."
    assert (
        holdout_connectomes.isna().sum().sum() == 0
    ), "NaN values in the connectomes."
    assert not np.isinf(
        connectomes_main_analysis.values
    ).any(), "Inf values in the connectomes."
    assert not np.isinf(
        holdout_connectome.values
    ).any(), "Inf values in the connectomes."

    return (connectomes_main_analysis, holdout_connectome)


def main(dataset):
    """Run the main program."""
    outpath_typicality = Path("..") / ".." / "results" / "concatenated"
    outname_typicality = f"connectome_typicality_{dataset}.csv"
    assert outpath_typicality.is_dir(), "Outpath doesn't exist."

    all_dfs = []
    sessions = ["REST1", "REST2"]
    phase_encodings = ["LR", "RL"]
    parcellations = ["Schaefer100", "Schaefer200", "Schaefer400"]

    if "hcp_ya" in dataset:
        print(f"Running dataset: {dataset}")
        for session, phase_encoding, parcellation in tqdm(
            product(sessions, phase_encodings, parcellations)
        ):
            connectomes_rest, holdout = load_connectomes(dataset, parcellation)

            subset = (
                connectomes_rest.query(f"task == '{session}'")
                .query(f"phase_encoding == '{phase_encoding}'")
                .copy()
            )

            n_subs = subset.shape[0]
            assert n_subs == 296, f"Should be 296 subjects, {n_subs}"

            # typicality analsis
            typicality_series = subset.apply(
                typicality,
                axis=1,
                template=holdout,
                correlation_method="pearson",
            )
            typicality_df = pd.DataFrame(
                typicality_series, columns=["typicality"]
            )

            # format results
            params = {
                "parcellation": parcellation,
            }

            params = {
                key: [value for _ in range(n_subs)]
                for key, value in params.items()
            }
            typicality_df = typicality_df.assign(**params)
            all_dfs.append(typicality_df)
    else:
        if "aomic" in dataset:
            session = "restingstate"
        elif dataset == "camcan":
            session = "rest"

        print(f"Running dataset: {dataset}")

        for parcellation in tqdm(parcellations):
            connectomes_rest, holdout = load_connectomes(dataset, parcellation)
            connectomes_rest = connectomes_rest.query(f"task == '{session}'")

            # typicality analsis
            n_subs = connectomes_rest.shape[0]
            typicality_series = connectomes_rest.apply(
                typicality,
                axis=1,
                template=holdout,
                correlation_method="pearson",
            )
            typicality_df = pd.DataFrame(
                typicality_series, columns=["typicality"]
            )
            # format results
            params = {
                "parcellation": parcellation,
            }

            params = {
                key: [value for _ in range(n_subs)]
                for key, value in params.items()
            }
            typicality_df = typicality_df.assign(**params)
            all_dfs.append(typicality_df)

    final_df = pd.concat(all_dfs)

    final_df.to_csv(outpath_typicality / outname_typicality)


if __name__ == "__main__":
    for dataset in [
        "camcan",
        "hcp_ya_non_ica_fix",
        "hcp_ya_ica_fix",
        "aomic_piop1",
        "aomic_piop2",
    ]:
        main(dataset)
