#!/usr/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate gradient_identification
python3 $@
