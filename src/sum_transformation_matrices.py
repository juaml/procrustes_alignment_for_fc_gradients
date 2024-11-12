"""Correlate the sum(abs(transformation)) with FD."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    transmats_path = Path("..") / "results" / "transmats"

    outpath = (
        Path("..") / "results" / "concatenated" / "transmats_sum_vals.csv"
    )
    parcellations = [f"Schaefer{x}" for x in [100, 200, 400]]
    datasets = [
        "camcan",
        "hcp_ya_non_ica_fix",
        "aomic_piop1",
        "aomic_piop2",
        "hcp_ya_ica_fix",
    ]
    n_componentses = range(2, 21)
    approaches = ["dm", "pca", "le"]
    kernels = [
        "pearson",
        "spearman",
        "cosine",
        "normalized_angle",
        "gaussian",
        "None",
    ]
    sparsities = [0.9]  # np.linspace(0, 0.9, 4)

    results = {
        "dataset": [],
        "subject": [],
        "session": [],
        "parcellation": [],
        "n_components": [],
        "approach": [],
        "kernel": [],
        "sparsity": [],
        "transmat_abs_sum": [],
        "transmat_abs_sum_first_column": [],
        "principal_gradient_was_in_right_place": [],
        "mixedness": [],
    }

    for dataset in datasets:
        # ugly fix but necessary
        if "hcp_ya" in dataset:
            data_subj = "hcp_ya"
        else:
            data_subj = dataset

        subject_list = list(
            np.loadtxt(
                Path("..") / "data" / f"{data_subj}_analysis_subjects.txt",
                dtype=str,
            )
        )

        if "hcp_ya" in dataset:
            sessions = ["REST1LR", "REST2LR", "REST1RL", "REST2RL"]
        elif "aomic" in dataset:
            sessions = ["restingstate"]
        elif dataset == "camcan":
            sessions = ["rest"]

        print("\n", f"Running dataset: {dataset}")
        for (
            session,
            parcellation,
            n_components,
            approach,
            kernel,
            sparsity,
        ) in tqdm(
            product(
                sessions,
                parcellations,
                n_componentses,
                approaches,
                kernels,
                sparsities,
            )
        ):
            if "hcp_ya" in dataset:
                file_path = transmats_path / (
                    f"dataset-{dataset}_session-{session}_"
                    f"parcellation-{parcellation}_ncomponents-{n_components}_"
                    f"approach-{approach}_kernel-{kernel}_sparsity-{sparsity}_"
                    f"alignmentaffinematrix.npz"
                )
            else:
                file_path = transmats_path / (
                    f"dataset-{dataset}_"
                    f"parcellation-{parcellation}_ncomponents-{n_components}_"
                    f"approach-{approach}_kernel-{kernel}_sparsity-{sparsity}_"
                    f"alignmentaffinematrix.npz"
                )

            transmat_npz = np.load(file_path)

            for subject in subject_list:
                transmat_df = pd.DataFrame(transmat_npz[str(subject)])

                mixedness = (
                    transmat_df.iloc[0, :].abs().max()
                    / transmat_df.iloc[:, 0].abs().sum()
                )
                sum_val = transmat_df.abs().sum().sum()
                abs_sum_first_col = transmat_df.abs().iloc[:, 0].sum()
                was_principal_gradient_in_right_place = (
                    1 if transmat_df.iloc[0, :].idxmax() == 0 else 0
                )

                results["dataset"].append(dataset)
                results["subject"].append(subject)
                results["session"].append(session)
                results["parcellation"].append(parcellation)
                results["n_components"].append(n_components)
                results["approach"].append(approach)
                results["kernel"].append(kernel)
                results["sparsity"].append(sparsity)
                results["transmat_abs_sum"].append(sum_val)
                results["transmat_abs_sum_first_column"].append(
                    abs_sum_first_col
                )
                results["principal_gradient_was_in_right_place"].append(
                    was_principal_gradient_in_right_place
                )
                results["mixedness"].append(mixedness)

    results_df = pd.DataFrame(results)

    print(results_df)
    results_df.to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
