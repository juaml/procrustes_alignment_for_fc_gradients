"""Concatenate predictions using the Inspector."""

from itertools import product
from pathlib import Path

import joblib
import pandas as pd


def main():
    path = Path("..") / "results" / "regression"
    datasets = ["hcp_ya_non_ica_fix", "hcp_ya_ica_fix"]
    sessions = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]

    prediction_dfs = []
    for dataset, session in product(datasets, sessions):
        f = (
            f"dataset-{dataset}_session-{session}_parc-Schaefer400_"
            "kernel-normalized_angle_approach-dm_sparsity-0.9_ncomps"
            "-10_model-rbf_svm_target-FD.csv_inspector.joblib"
        )
        inspector = joblib.load(path / f)
        predictions = inspector.folds.predict()
        predictions = predictions.assign(
            **{
                "dataset": [dataset for _ in range(predictions.shape[0])],
                "session": [session for _ in range(predictions.shape[0])],
            }
        )
        prediction_dfs.append(predictions)

    outpath = Path("..") / "results" / "concatenated" / "predictions.csv"
    pd.concat(prediction_dfs).reset_index(drop=True).to_csv(
        outpath, index=False
    )


if __name__ == "__main__":
    main()
