"""Get a hold-out subject list to construct group level gradient."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = ArgumentParser(
        description=(
            "Split subjects into holdout and main analysis "
            "datasets for different samples."
        )
    )
    parser.add_argument(
        "sample",
        help="Which sample to split into holdout and analysis datasets.",
        choices=["hcp_ya", "aomic_piop1", "aomic_piop2", "camcan"],
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help=(
            "If active, saves the split subjects into two separate text files."
        ),
    )
    return parser.parse_args()


def main():
    """Get a hold-out subject list to construct group level gradient."""
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"
    args = parse_args()
    subjects = list(
        np.loadtxt(path / f"{args.sample}_subject_list.txt", dtype=str)
    )

    if args.sample == "hcp_ya":
        sex = "Gender"
        demographics = pd.read_csv(
            path / "hcp_ya_unrestricted.csv",
            index_col=0,
            dtype={"Subject": str},
        ).loc[subjects]
    elif "aomic" in args.sample:
        # aomic dataset has different coding for gender/sex
        sex = "sex"
        subjects_as_idx = [f"sub-{x}" for x in subjects]
        demographics = pd.read_csv(
            path / f"{args.sample}_participants.tsv", sep="\t", index_col=0
        ).loc[subjects_as_idx]
        # junifer outputs subject id's without prefix `sub-`
        demographics.index = subjects
    elif args.sample == "camcan":
        sex = "gender_code"
        demographics = pd.read_csv(
            path / f"{args.sample}_participants.tsv", sep="\t", index_col=0
        ).loc[subjects]
        demographics.index = subjects

    # random state was determined interactively using:
    # randint(0, 1000000000000)
    main_subjects, holdout = train_test_split(
        subjects,
        stratify=demographics[sex],
        shuffle=True,
        random_state=140542490,
    )
    dem_main = demographics.loc[main_subjects]
    dem_hold = demographics.loc[holdout]
    print("Analysis subjects gender counts:")
    print("-" * 80)
    print(dem_main[sex].value_counts())
    print("-" * 80)
    print("\n")

    print("Holdout subjects gender counts:")
    print("-" * 80)
    print(dem_hold[sex].value_counts())
    print("-" * 80)
    print("\n")

    if args.save:
        analysis_filename = path / f"{args.sample}_analysis_subjects.txt"
        holdout_filename = path / f"{args.sample}_holdout_subjects.txt"
        print("Saving the split subjects group in \n")
        print(f"{analysis_filename}\n")
        print(f"{holdout_filename}\n")
        np.savetxt(analysis_filename, main_subjects, fmt="%s")
        np.savetxt(holdout_filename, holdout, fmt="%s")
    else:
        print(
            "Not saving the subject list. Use '-s, --save' "
            "option to save lists as txt files."
        )


if __name__ == "__main__":
    main()
