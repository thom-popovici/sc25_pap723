#!/bin/bash

binconda=`env | grep 'bin/conda'`
etcpath=`echo $binconda | sed -e 's/conda$//g' | sed -e 's/^.*=//g'`
srcpath=`find . $etcpath../etc -name conda.sh`
ls -l $srcpath
source $srcpath
source env-name.sh
conda activate $DIMAGE_DFT_ENV
