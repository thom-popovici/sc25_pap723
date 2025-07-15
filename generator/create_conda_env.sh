#!/bin/bash

ENV_NAME="sc2025_pap723"
PYTHON_VERSION="3.11"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

