"""Extract the transformation matrices used in alignment."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
from brainspace.gradient.gradient import GradientMaps
from junifer.storage import HDF5FeatureStorage
from nilearn.connectome import vec_to_sym_matrix


def parse_args():
    """Parse pipeline arguments."""
    parser = ArgumentParser(description="Run main identification pipeline.")
    parser.add_argument(
        "dataset",
        choices=[
            "hcp_ya_non_ica_fix",
            "hcp_ya_ica_fix",
            "aomic_piop1",
            "aomic_piop2",
            "camcan",
        ],
    )
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

    return parser.parse_args()


def make_holdout_gradient(connectome, grad_kwargs, fit_kwargs):
    """Construct holdout fitted GradientMaps object.."""
    mat = vec_to_sym_matrix(np.array(connectome))
    rows, cols = mat.shape
    assert rows == cols, "Connectome not NxN."

    # nilearn squares the diagonal so make sure they are one
    np.fill_diagonal(mat, np.ones(rows))

    # extract gradients
    gm = GradientMaps(**grad_kwargs)
    gm.fit(mat, **fit_kwargs)

    # return the principal gradient for the connectome
    return gm


def get_transformation_matrix(connectome, reference, grad_kwargs, fit_kwargs):
    """Get Procrustes transformation matrix for these gradient settings."""

    mat = vec_to_sym_matrix(connectome)
    rows, cols = mat.shape
    assert rows == cols, "Connectome not NxN."

    # nilearn squares the diagonal so make sure they are one
    np.fill_diagonal(mat, np.ones(rows))

    # extract gradients
    gm = GradientMaps(**grad_kwargs)
    gm.fit(mat, reference=reference, **fit_kwargs)

    # return the transformation matrix
    t = gm.t_[0]
    return t


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


def main():
    """Run the main program."""
    args = parse_args()
    if args.kernel == "None":
        args.kernel = None

    connectomes_rest, holdout = load_connectomes(
        args.dataset, args.parcellation
    )
    outpath = Path("..") / ".." / "results" / "transmats"
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

    if "hcp_ya" in args.dataset:
        for session, phase_encoding in product(
            ["REST1", "REST2"], ["LR", "RL"]
        ):
            outfile = (
                f"dataset-{args.dataset}_session-{session}{phase_encoding}_"
                f"parcellation-{args.parcellation}_ncomponents-{args.n_components}"
                f"_approach-{args.approach}_kernel-{args.kernel}_"
                f"sparsity-{args.sparsity}_alignmentaffinematrix.npz"
            )
            print(session, phase_encoding)
            subset = (
                connectomes_rest.query(f"task == '{session}'")
                .query(f"phase_encoding == '{phase_encoding}'")
                .copy()
            )

            n_subs = subset.shape[0]
            assert n_subs == 296, f"Should be 296 subjects, {n_subs}"

            result_dict = {}
            for connectome in subset.iterrows():
                subject = connectome[0][1]
                transformation_matrix = get_transformation_matrix(
                    connectome[1],
                    reference=holdout_gm.gradients_,
                    grad_kwargs=grad_kwargs,
                    fit_kwargs=fit_kwargs,
                )
                result_dict[subject] = transformation_matrix

            np.savez(outpath / outfile, **result_dict)

    else:
        if "aomic" in args.dataset:
            session = "restingstate"
        elif args.dataset == "camcan":
            session = "rest"

        outfile = (
            f"dataset-{args.dataset}_parcellation-{args.parcellation}_"
            f"ncomponents-{args.n_components}_approach-{args.approach}_"
            f"kernel-{args.kernel}_sparsity-{args.sparsity}_"
            f"alignmentaffinematrix.npz"
        )

        connectomes_rest = connectomes_rest.query(f"task == '{session}'")
        result_dict = {}
        for (subject, task), connectome in connectomes_rest.iterrows():
            transformation_matrix = get_transformation_matrix(
                connectome,
                reference=holdout_gm.gradients_,
                grad_kwargs=grad_kwargs,
                fit_kwargs=fit_kwargs,
            )

            result_dict[subject] = transformation_matrix

        np.savez(outpath / outfile, **result_dict)


if __name__ == "__main__":
    main()
