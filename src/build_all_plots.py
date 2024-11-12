#!/usr/bin/env python3
"""Provide script to build all the plots."""

import contextlib
import os
import subprocess
from pathlib import Path
import sys
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


@contextlib.contextmanager
def working_directory(path):
    """Change working directory and return to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def build_plot(py_file):
    """Build a plot."""
    if "mricrogl" in py_file:
        subprocess.run(["MRIcroGL", py_file], check=True)
    else:
        subprocess.run(["python3", "-W ignore", py_file], check=True)


def main():
    """Build all plots."""
    path = Path("plotting")
    with working_directory(path):
        all_py_files = sorted([x for x in Path(".").glob("*.py")])

        # Number of parallel jobs, adjust as needed
        num_files = len(all_py_files)
        num_jobs = num_files if num_files <= 16 else 16
        # Use Parallel to run the build_plot function in parallel
        Parallel(n_jobs=num_jobs, verbose=51)(
            delayed(build_plot)(str(py_file)) for py_file in all_py_files
        )


if __name__ == "__main__":
    main()
