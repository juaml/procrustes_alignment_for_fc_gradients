"""Run a basic gradient identification pipeline."""

from argparse import ArgumentParser
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from brainspace.gradient.gradient import GradientMaps
from identification import get_idiff, get_idiff_matrix, identify
from junifer.storage import HDF5FeatureStorage
from nilearn.connectome import vec_to_sym_matrix


def parse_args():
    """Parse pipeline arguments."""
    parser = ArgumentParser(description="Run main identification pipeline.")
    parser.add_argument(
        "parcellation", choices=[f"Schaefer{x}" for x in [100, 200, 400]]
    )
    parser.add_argument("n_components", type=int)
    parser.add_argument("approach", choices=["dm", "le", "pca"])
    parser.add_argument(
        "kernel",
        choices=[
            "pearson",
            "spearman",
            "cosine",
            "normalized_angle",
            "gaussian",
            "None",
        ],
    )
    parser.add_argument("sparsity", type=float)
    parser.add_argument("--icafix", action="store_true")
    return parser.parse_args()


def make_holdout_gradient(connectome, grad_kwargs, fit_kwargs):
    """Construct holdout fitted GradientMaps object.."""
    mat = vec_to_sym_matrix(connectome)
    rows, cols = mat.shape
    assert rows == cols, "Connectome not NxN."

    # nilearn squares the diagonal so make sure they are one
    np.fill_diagonal(mat, np.ones(rows))

    # extract gradients
    gm = GradientMaps(**grad_kwargs)
    gm.fit(mat, **fit_kwargs)

    # return the principal gradient for the connectome
    return gm


def connectome_to_gradient(connectome, reference, grad_kwargs, fit_kwargs):
    """Convert vectorised connectome to principal component."""
    mat = vec_to_sym_matrix(connectome)
    rows, cols = mat.shape
    assert rows == cols, "Connectome not NxN."

    # nilearn squares the diagonal so make sure they are one
    np.fill_diagonal(mat, np.ones(rows))

    # extract gradients
    gm = GradientMaps(**grad_kwargs)
    gm.fit(mat, reference=reference, **fit_kwargs)

    # return the principal gradient for the connectome
    return gm.aligned_[:, 0]


def load_connectomes(parcellation, icafix=False):
    """Load connectomes for a given parcellation.

    Parameters
    ----------
    parcellation : str
        Which parcellation to use for the marker {'Schaefer100', 'Schaefer200',
        'Schaefer400'}
    icafix : bool
        Whether data with icafix and motion regression applied should be
        used or not.
    """
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"

    subjects_main_analysis = list(
        np.loadtxt(path / "hcp_ya_analysis_subjects.txt", dtype=str)
    )

    subjects_holdout = list(
        np.loadtxt(path / "hcp_ya_holdout_subjects.txt", dtype=str)
    )

    if icafix:
        storage_name = "hcp_ya_ica_fix"
    else:
        storage_name = "hcp_ya_non_ica_fix"

    storage = HDF5FeatureStorage(
        path / storage_name / f"{storage_name}.hdf5", single_output=False
    )

    marker = f"BOLD_parccortical-{parcellation}x7FSLMNI_marker-empiricalFC"
    marker_df = storage.read_df(marker)

    # select main analysis subjects and holdout subjects
    connectomes_main_analysis = marker_df.query(
        "subject in @subjects_main_analysis"
    )
    holdout_connectomes = marker_df.query("subject in @subjects_holdout")

    connectomes_dict = {}
    for i, (task, phase_encoding) in enumerate(
        product(["REST1", "REST2"], ["LR", "RL"])
    ):
        session_connectomes = (
            connectomes_main_analysis.query(
                f"(task == '{task}') & (phase_encoding == '{phase_encoding}')"
            )
            .reset_index(drop=False)
            .drop(columns=["task", "phase_encoding"])
            .set_index("subject")
        )
        if i == 0:
            index_orientation = session_connectomes.index.copy()
        else:
            session_connectomes = session_connectomes.loc[index_orientation]

        connectomes_dict[f"{task}_{phase_encoding}"] = session_connectomes

    holdout_connectome = (
        holdout_connectomes.apply(np.arctanh, axis=1).mean().apply(np.tanh)
    )
    assert (
        holdout_connectomes.isna().sum().sum() == 0
    ), "NaN values in the connectomes."
    assert not np.isinf(
        holdout_connectome.values
    ).any(), "Inf values in the connectomes."

    for connectomes in connectomes_dict.values():
        assert (
            connectomes.isna().sum().sum() == 0
        ), "NaN values in the connectomes."

        assert not np.isinf(
            connectomes.values
        ).any(), "Inf values in the connectomes."

    return (connectomes_dict, holdout_connectome)


def main():
    """Run the main program."""
    args = parse_args()
    if args.kernel == "None":
        args.kernel = None

    connectomes_dict, holdout = load_connectomes(
        args.parcellation, args.icafix
    )

    outpath = Path("..") / ".." / "results" / "identification"
    assert outpath.is_dir(), "Outpath doesn't exist."

    # random state was determined interactively using:
    # randint(0, 10000)
    grad_kwargs = {
        "n_components": args.n_components,
        "approach": args.approach,
        "kernel": args.kernel,
        "random_state": 6372,
    }
    fit_kwargs = {"sparsity": args.sparsity}

    holdout_gm = make_holdout_gradient(holdout, grad_kwargs, fit_kwargs)
    grad_kwargs["alignment"] = "procrustes"

    gradients_dict = {}
    for key, connectomes in connectomes_dict.items():
        # hard coded number of subjects to avoid surprises
        assert connectomes.shape[0] == 296, "Should be 296 subjects"
        gradients_dict[key] = connectomes.apply(
            connectome_to_gradient,
            axis=1,
            reference=holdout_gm.gradients_,
            grad_kwargs=grad_kwargs,
            fit_kwargs=fit_kwargs,
            result_type="expand",
        )

    # prepare results dictionary
    results = {
        "icafix": args.icafix,
        "parcellation": args.parcellation,
        **grad_kwargs,
        **fit_kwargs,
    }
    for session_one, session_two in combinations(gradients_dict.keys(), 2):
        # Identification Accuracy
        # one way
        acc_sess_one_sess_two = identify(
            gradients_dict[session_one],
            gradients_dict[session_two],
            metric="spearman",
        )
        # and then the other way; they are not equivalent
        acc_sess_two_sess_one = identify(
            gradients_dict[session_two],
            gradients_dict[session_one],
            metric="spearman",
        )

        # Differential Identifiability
        idiff_matrix = get_idiff_matrix(
            gradients_dict[session_one],
            gradients_dict[session_two],
            metric="spearman",
        )
        idiff_value, within, between = get_idiff(
            idiff_matrix, return_mean_correlations=True
        )
        idiff_value *= 100

        r_to_z_idiff, _, _ = np.tanh(
            get_idiff(np.arctanh(idiff_matrix), return_mean_correlations=True)
        )
        r_to_z_idiff *= 100

        # add results
        results[f"IAcc_{session_one}_{session_two}"] = acc_sess_one_sess_two
        results[f"IAcc_{session_two}_{session_one}"] = acc_sess_two_sess_one
        results[
            f"{session_one}_{session_two}_mean_within_subject_correlation"
        ] = within
        results[
            f"{session_one}_{session_two}_mean_between_subject_correlation"
        ] = between
        results[f"{session_one}_{session_two}_IDiff"] = idiff_value
        results[f"{session_one}_{session_two}_IDiff_r_to_z"] = r_to_z_idiff

    results = {key: [value] for key, value in results.items()}
    outfile = outpath / (
        f"parc-{args.parcellation}_kernel-{args.kernel}"
        f"_approach-{args.approach}_sparsity-{args.sparsity}"
        f"_ncomps-{args.n_components}_icafix-{args.icafix}.csv"
    )
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
