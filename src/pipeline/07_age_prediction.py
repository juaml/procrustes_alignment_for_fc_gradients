"""Run a basic gradient-based age prediction pipeline."""

import copy
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from brainspace.gradient.gradient import GradientMaps
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.scoring import register_scorer
from julearn.utils import configure_logging
from junifer.storage import HDF5FeatureStorage
from nilearn.connectome import vec_to_sym_matrix
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RepeatedKFold

sys.path.append("../lib")
from modelgrids import get_model_grid


def corr(y_true, y_pred):
    """Get Pearson's r."""
    return pearsonr(y_true, y_pred)[0]


def parse_args():
    """Parse pipeline arguments."""
    parser = ArgumentParser(description="Run main pipeline to predict age.")
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
    parser.add_argument(
        "model",
        type=str,
        choices=["ridge", "linear_svm", "rbf_svm", "rf"],
    )
    parser.add_argument("--remove_confounds", action="store_true")

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


def load_connectomes(dataset, parcellation, session):
    """Load connectomes for a given parcellation."""
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"

    subjects_main_analysis = list(
        np.loadtxt(path / f"{dataset}_analysis_subjects.txt", dtype=str)
    )
    subjects_holdout = list(
        np.loadtxt(path / f"{dataset}_holdout_subjects.txt", dtype=str)
    )
    storage = HDF5FeatureStorage(
        path / dataset / f"{dataset}.hdf5",
        single_output=False,
    )

    marker = f"BOLD_parccortical-{parcellation}x7FSLMNI_marker-empiricalFC"
    marker_df = storage.read_df(marker)
    marker_df = marker_df.query(f"task == '{session}'")

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

    return (
        connectomes_main_analysis,
        holdout_connectome,
    )


def load_age(dataset, session):
    """Load the HCP phenotypes for the specified subjects."""
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"

    phenotypes = pd.read_csv(
        path / f"{dataset}_participants.tsv",
        index_col="participant_id",
        dtype={"participant_id": str},
        sep="\t",
    )

    subjects_main_analysis = list(
        np.loadtxt(path / f"{dataset}_analysis_subjects.txt", dtype=str)
    )

    phenotypes.index = phenotypes.index.astype(str)

    return phenotypes.loc[subjects_main_analysis]["age"]


def load_fd(dataset, session):
    """Load the HCP phenotypes for the specified subjects."""
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"

    phenotypes = pd.read_csv(
        path / f"{dataset}_framewise_displacement.csv",
        index_col="subject",
        dtype={"subject": str},
    )

    subjects_main_analysis = list(
        np.loadtxt(path / f"{dataset}_analysis_subjects.txt", dtype=str)
    )

    phenotypes.index = phenotypes.index.astype(str)

    return phenotypes.loc[subjects_main_analysis]


def main():
    """Run the main program."""
    # For now hard code the problem type here
    configure_logging(level="INFO")
    sklearn_scorer = make_scorer(corr)
    register_scorer("corr", sklearn_scorer)
    problem_type = "regression"
    args = parse_args()

    if args.kernel == "None":
        args.kernel = None

    dataset = "camcan"
    session = "rest"

    # load connectome data
    connectomes, holdout = load_connectomes(
        dataset, args.parcellation, session
    )

    connectomes = (
        connectomes.reset_index().drop(columns=["task"]).set_index("subject")
    )
    outpath = Path("..") / ".." / "results" / "camcan_age_prediction"
    assert outpath.is_dir(), "Outpath doesn't exist."

    # prepare the gradients as features
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
    gradients = connectomes.apply(
        connectome_to_gradient,
        axis=1,
        reference=holdout_gm.gradients_,
        grad_kwargs=grad_kwargs,
        fit_kwargs=fit_kwargs,
        result_type="expand",
    )
    features = list(gradients.columns.astype(str))
    gradients.columns = features

    X_types = {"features": copy.deepcopy(features)}
    if args.remove_confounds:
        X_types["confounds"] = ["fd"]
        gradients["fd"] = load_fd(dataset, session).loc[gradients.index]
        features += ["fd"]

    # load phenotypes to add the target
    gradients["age"] = load_age(dataset, session).loc[gradients.index]

    # Create the model
    model_instance, grid = get_model_grid(
        args.model, problem_type=problem_type
    )
    model = PipelineCreator(problem_type=problem_type)
    if args.remove_confounds:
        model.add(
            "confound_removal", apply_to="features", confounds="confounds"
        )
    model.add(model_instance, **grid, apply_to="*")

    # configure CV
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=6372)
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=6372)
    scoring = [
        "neg_mean_absolute_error",
        "corr",
        "neg_root_mean_squared_error",
        "r2",
    ]

    assert gradients.isna().sum().sum() == 0, "NaNs in prediction data."
    print(gradients)

    # run
    scores, final_julearn_model, inspector = run_cross_validation(
        X=features,
        # hard code target in this script
        y="age",
        data=gradients,
        X_types=X_types,
        model=model,
        seed=6372,
        scoring=scoring,
        cv=outer_cv,
        n_jobs=1,
        search_params={
            "cv": inner_cv,
            "n_jobs": 1,
            "verbose": 10,
        },
        verbose=10,
        return_estimator="all",
        return_inspector=True,
    )

    # format results
    results = {
        "dataset": dataset,
        "session": session,
        "target": "age",
        "model": args.model,
        "parcellation": args.parcellation,
        "removeconfounds": args.remove_confounds,
        **grad_kwargs,
        **fit_kwargs,
    }
    results = {
        key: [value for _ in range(50)] for key, value in results.items()
    }

    results = pd.concat([pd.DataFrame(results), scores], axis=1)
    outfile = outpath / (
        f"dataset-{dataset}_session-{session}_"
        f"parc-{args.parcellation}_kernel-{args.kernel}"
        f"_approach-{args.approach}_sparsity-{args.sparsity}"
        f"_ncomps-{args.n_components}_model-{args.model}_removeconfounds-{args.remove_confounds}"
        f"_target-age"
    )
    results.to_csv(f"{outfile}.csv", index=False)
    # joblib.dump(final_julearn_model, outpath / f"{outfile}_model.joblib")
    # joblib.dump(inspector, outpath / f"{outfile}_inspector.joblib")


if __name__ == "__main__":
    main()
