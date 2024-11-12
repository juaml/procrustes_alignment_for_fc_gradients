"""Concatenate results."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args():
    """Parse arguments."""
    parser = ArgumentParser(description="Concatenate results.")
    parser.add_argument(
        "experiment",
        choices=[
            "identification",
            "regression_intelligence",
            "typicality",
            "icc",
            "classification",
            "camcan_age_prediction",
            "camcan_age_prediction_extended_confounds",
        ],
    )
    return parser.parse_args()


def main():
    """Run main program to concatenate results."""
    args = parse_args()
    path = Path("..") / "results" / args.experiment
    result = pd.concat(
        [pd.read_csv(x) for x in tqdm(path.glob("*.csv"))]
    ).reset_index(drop=True)

    outpath = (
        Path("..") / "results" / "concatenated" / f"{args.experiment}.csv"
    )
    result.to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
