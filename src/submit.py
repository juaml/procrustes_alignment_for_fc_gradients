"""Generate a basic condor submit string."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

PREAMBLE = """executable = /usr/bin/bash
transfer_executable = False
initial_dir= ./pipeline
universe = vanilla
getenv = True
request_cpus = 1
request_memory = 25GB
request_disk = 5GB

"""

JOB_TEMPLATE = """arguments = {arguments}
log = {logs_dir}/{log_string}_$(Cluster).$(Process).log
output = {logs_dir}/{log_string}_$(Cluster).$(Process).out
error = {logs_dir}/{log_string}_$(Cluster).$(Process).err
Queue

"""


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser(description="Submit a batch of jobs to HTCondor.")
    parser.add_argument(
        "pipeline",
        choices=[
            "identification",
            "transmats",
            "classification",
            "camcan_age_prediction",
            "regression_intelligence",
        ],
    )
    return parser.parse_args()


def main():
    """Run main loop over all combinations of arguments."""
    args = parse_args()
    logs_dir = (Path("..") / "logs" / args.pipeline).resolve()
    assert logs_dir.is_dir(), f"{logs_dir} is not a directory!"
    parcellations = [f"Schaefer{x}" for x in [100, 200, 400]]
    n_componentses = [1] + list(range(5, 55, 5))
    kernels = [
        "pearson",
        "spearman",
        "cosine",
        "normalized_angle",
        "gaussian",
        "None",
    ]
    approaches = ["dm", "le", "pca"]
    sparsities = np.linspace(0, 0.9, 4)
    icafixes = [True, False]

    print(PREAMBLE)

    if args.pipeline == "identification":
        for (
            parc,
            n_components,
            kernel,
            approach,
            sparsity,
            icafix,
        ) in product(
            parcellations,
            n_componentses,
            kernels,
            approaches,
            sparsities,
            icafixes,
        ):
            format_log_string = f"{parc}_{n_components}_{kernel}_{approach}_{sparsity}_{icafix}"
            if icafix:
                flag = " --icafix"
            else:
                flag = ""

            format_arguments = (
                f"./run_conda.sh 02_identification.py"
                f" {parc} {n_components} {approach} {kernel} "
                f"{sparsity}{flag}"
            )
            print(
                JOB_TEMPLATE.format(
                    arguments=format_arguments,
                    log_string=format_log_string,
                    logs_dir=logs_dir.as_posix(),
                )
            )

    elif args.pipeline == "transmats":
        datasets = [
            "hcp_ya_ica_fix",
            "hcp_ya_non_ica_fix",
            "aomic_piop1",
            "aomic_piop2",
            "camcan",
        ]
        sparsities = [0.9]
        n_componentses = range(2, 21)
        for (
            parc,
            n_components,
            kernel,
            approach,
            sparsity,
            dataset,
        ) in product(
            parcellations,
            n_componentses,
            kernels,
            approaches,
            sparsities,
            datasets,
        ):
            format_log_string = (
                f"{parc}_{n_components}_{kernel}_{approach}_{sparsity}"
            )
            format_arguments = (
                f"./run_conda.sh "
                f"03_extract_transformation_matrix_per_session.py {dataset}"
                f" {parc} {n_components} {approach} "
                f"{kernel} {sparsity}"
            )
            print(
                JOB_TEMPLATE.format(
                    arguments=format_arguments,
                    log_string=format_log_string,
                    logs_dir=logs_dir.as_posix(),
                )
            )

    elif args.pipeline == "classification":
        n_componentses = range(2, 21)
        datasets = ["hcp_ya_ica_fix", "aomic_piop1", "aomic_piop2", "camcan"]
        models = ["rf", "rbf_svm", "ridge"]

        kernels = ["normalized_angle"]
        approaches = ["dm"]
        for (
            dataset,
            parc,
            n_components,
            approach,
            kernel,
            sparsity,
            model,
        ) in product(
            datasets,
            parcellations,
            n_componentses,
            approaches,
            kernels,
            sparsities,
            models,
        ):
            if dataset == "hcp_ya_ica_fix":
                sessions = [
                    "REST1_LR",
                    "REST1_RL",
                    "REST2_LR",
                    "REST2_RL",
                ]
            elif dataset == "camcan":
                sessions = ["rest"]
            elif "aomic" in dataset:
                sessions = ["restingstate"]

            for session in sessions:
                format_log_string = (
                    f"{dataset}_{parc}_{session}_{n_components}_{kernel}_"
                    f"{approach}_{sparsity}_{model}"
                )

                format_arguments = (
                    f"./run_conda.sh 06_classification.py"
                    f" {dataset} {parc} {session} {n_components} {approach} "
                    f"{kernel} {sparsity} {model}"
                )

                print(
                    JOB_TEMPLATE.format(
                        arguments=format_arguments,
                        log_string=format_log_string,
                        logs_dir=logs_dir.as_posix(),
                    )
                )

    elif args.pipeline == "camcan_age_prediction":
        models = ["rbf_svm", "ridge", "linear_svm"]
        confound_removal = [True, False]
        n_componentses = range(2, 21)
        sparsity = 0.9
        kernels = ["normalized_angle"]
        approaches = ["dm"]

        for (
            confounds,
            parc,
            n_components,
            approach,
            kernel,
            model,
        ) in product(
            confound_removal,
            parcellations,
            n_componentses,
            approaches,
            kernels,
            models,
        ):
            if confounds:
                format_log_string = (
                    f"{parc}_{n_components}_{kernel}_"
                    f"{approach}_{sparsity}_{model}_remove_confounds"
                )

                format_arguments = (
                    f"./run_conda.sh 07_age_prediction.py"
                    f" {parc} {n_components} {approach} "
                    f"{kernel} {sparsity} {model} --remove_confounds"
                )

                print(
                    JOB_TEMPLATE.format(
                        arguments=format_arguments,
                        log_string=format_log_string,
                        logs_dir=logs_dir.as_posix(),
                    )
                )

            else:
                format_log_string = (
                    f"{parc}_{n_components}_{kernel}_"
                    f"{approach}_{sparsity}_{model}"
                )

                format_arguments = (
                    f"./run_conda.sh 07_age_prediction.py"
                    f" {parc} {n_components} {approach} "
                    f"{kernel} {sparsity} {model}"
                )

                print(
                    JOB_TEMPLATE.format(
                        arguments=format_arguments,
                        log_string=format_log_string,
                        logs_dir=logs_dir.as_posix(),
                    )
                )

    elif args.pipeline == "regression_intelligence":
        models = ["rbf_svm", "ridge", "linear_svm"]
        n_componentses = range(2, 21)
        sparsity = 0.9
        kernels = ["normalized_angle"]
        approaches = ["dm"]
        confounds = ["None", "FD", "FD+Age"]
        datasets = ["hcp_ya_ica_fix", "hcp_ya_non_ica_fix"]

        for (
            dataset,
            confound_level,
            parc,
            n_components,
            approach,
            kernel,
            model,
        ) in product(
            datasets,
            confounds,
            parcellations,
            n_componentses,
            approaches,
            kernels,
            models,
        ):
            if "hcp_ya" in dataset:
                sessions = [
                    "REST1_LR",
                    "REST1_RL",
                    "REST2_LR",
                    "REST2_RL",
                ]
            elif dataset == "camcan":
                sessions = ["rest"]
            elif "aomic" in dataset:
                sessions = ["restingstate"]

            for session in sessions:
                format_log_string = (
                    f"{dataset}_{parc}_{session}_{n_components}_{kernel}_"
                    f"{approach}_{sparsity}_{model}_{confound_level}"
                )

                format_arguments = (
                    f"./run_conda.sh 08_regression_intelligence.py"
                    f" {dataset} {parc} {session} {n_components} {approach} "
                    f"{kernel} {sparsity} {model} {confound_level}"
                )

                print(
                    JOB_TEMPLATE.format(
                        arguments=format_arguments,
                        log_string=format_log_string,
                        logs_dir=logs_dir.as_posix(),
                    )
                )


if __name__ == "__main__":
    main()
