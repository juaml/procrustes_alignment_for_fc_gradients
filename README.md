# Procrustes Alignment in Gradient-based Individual-level Analyses

All code for this project can be found in the `src` directory. Let's go through
each component of the source code of this study individually:

## `preprocessing`

The `src/preprocessing` directory contains all code related to preprocessing the
functional connectomes used in this study. This preprocessing was performed
using [junifer](https://juaml.github.io/junifer/main/index.html). The `src/preprocessing/yamls`
directory contains all the junifer yaml files used to define the individual preprocessing
pipelines, and the `src/preprocessing/junifer_jobs` directory contains all
configuration, submit, and log files related to each pipeline that was created and run.

## `pipeline`

The `src/pipeline` directory contains each of the individual pipeline components
used to get the main results of this study. The contents of this directory 
are essentially python scripts that accept a number of arguments for configuration
of the pipeline component. The filenames are numbered in the order in which the
pipeline components should be executed. Each script is meant to be run using 
`src/pipeline` as a working directory and may not work if run from elsewhere.
Each of them use the [ArgumentParser](https://docs.python.org/3/library/argparse.html)
to parse command line arguments so you can run the `-h, --help` option to learn
more about how to use them.

## `plotting`

Unsurprisingly, the `src/plotting` directory contains all scripts that create
plots used to present the results of this project. Similarly to the `src/pipeline`,
the scripts in `src/plotting` are roughly numbered in order of appearance.

## Data Directories

The directories `results`, `logs`, `data`, and `figures` contain all study input
and output data, not all of which are pushed to the repository to protect confidential
data.

## Environment

The conda environment used in this project can be recreated using the
`environment.yaml` file in the root of the project.
