"""Describe the main demographics of our sample."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser(description="Describe a sample used in this study")
    parser.add_argument(
        "sample",
        help="A sample used in this study",
        choices=["hcp_ya", "aomic_piop1", "aomic_piop2", "camcan"],
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help=("If active, saves the sample description as a markdown table."),
    )
    return parser.parse_args()


def main():
    """Describe the sample used in this study."""

    args = parse_args()
    path = Path(__file__).parent.resolve() / ".." / ".." / "data"

    samples = {
        "sample_main_analysis": list(
            np.loadtxt(
                path / f"{args.sample}_analysis_subjects.txt", dtype=str
            )
        ),
        "sample_holdout": list(
            np.loadtxt(path / f"{args.sample}_holdout_subjects.txt", dtype=str)
        ),
    }
    samples["whole_sample"] = (
        samples["sample_main_analysis"] + samples["sample_holdout"]
    )

    if args.sample == "hcp_ya":
        sex = "Gender"
        restricted = path / f"{args.sample}_restricted.csv"
        unrestricted = path / f"{args.sample}_unrestricted.csv"

        vars_of_interest = ["Gender", "Age_in_Yrs"]
        dems_unsampled = pd.concat(
            [
                pd.read_csv(restricted, index_col=0),
                pd.read_csv(unrestricted, index_col=0),
            ],
            axis=1,
        )[vars_of_interest]

        dems_unsampled.index = dems_unsampled.index.astype(str)
    elif "aomic" in args.sample:
        sex = "sex"
        vars_of_interest = ["sex", "age"]
        dems_unsampled = pd.read_csv(
            path / f"{args.sample}_participants.tsv",
            sep="\t",
        )
        subject_ids = [
            str(x.split("-")[-1]) for x in dems_unsampled["participant_id"]
        ]
        dems_unsampled.index = subject_ids
        dems_unsampled = dems_unsampled[vars_of_interest]
    elif args.sample == "camcan":
        sex = "gender_text"
        vars_of_interest = ["gender_text", "age"]
        dems_unsampled = pd.read_csv(
            path / f"{args.sample}_participants.tsv", sep="\t", index_col=0
        )
        # subject_ids = [
        #    str(x.split("-")[-1]) for x in dems_unsampled["participant_id"]
        # ]
        # dems_unsampled.index = subject_ids
        dems_unsampled = dems_unsampled[vars_of_interest]

    all_sample_list = []
    for name, sample in samples.items():
        dems = dems_unsampled.loc[sample]
        grouped = dems.groupby(sex).describe()
        grouped.columns = grouped.columns.droplevel(0)
        all = dems.describe().T
        all.index = ["Both"]
        complete = pd.concat([grouped, all])
        complete["sample"] = [name for _ in range(complete.shape[0])]
        all_sample_list.append(complete)

    all_sample_description = (
        pd.concat(all_sample_list)
        .reset_index()
        .rename(columns={"index": sex})
        .set_index(["sample", sex])
    )
    if args.save:
        filename = path / f"{args.sample}_descriptive_statistics.md"
        print(f"Saving result table in {filename}")
        all_sample_description.reset_index().to_markdown(filename)
    else:
        print(all_sample_description)


if __name__ == "__main__":
    main()
