#!/bin/bash

FROM="dimage-dft-ae-package-list.txt"
source env-name.sh
CMD="conda create --name ${DIMAGE_DFT_ENV} --file ${FROM}"
$CMD

pip install z3-solver
