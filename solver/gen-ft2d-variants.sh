#!/bin/bash

## This script is part of the artifact of the paper:
## "Automatic Generation of Mappings for Distributed Fourier Operations"
## accepted for publication to SC'25.

## This script generates all 2D-FT DAGs, on 2D and 1D grids.
## The 16GB parameter can be changed to a larger GPU memory.
## However, given the internal handling of NCCL and GPU-aware
## runtimes, it is recommended to use at most 80% of the GPU
## memory.

source ./dimage-setup-conda.sh

STARTT=`date`


modes="1d"
procs="512 256 128 64 32 16"
pblist="3 4"

GRID=1
bmlist="fb f"
for bm in $bmlist; do
  for pbs in $pblist; do
    for pp in $procs; do
      for mam in $modes; do
        dagtemp="ft2d-${bm}.rels.template"
        ./test-with-capacity.sh $dagtemp $GRID $pp $pbs $mam 16GB
      done
    done
  done
done

MIDT1=`date`


GRID=2
bmlist="fb-mm f-mm fb-batch f-batch"
#bmlist="f-batch"
#bmlist="fb-mm f-mm"
for bm in $bmlist; do
  for pbs in $pblist; do
    for pp in $procs; do
      for mam in $modes; do
        dagtemp="ft2d-${bm}.rels.template"
        ./test-with-capacity.sh $dagtemp $GRID $pp $pbs $mam 16GB
      done
    done
  done
done

ENDT=`date`

LOGFILE="times-2d.txt"
echo "Start time: $STARTT" > $LOGFILE
echo "Midpoint 1: $MIDT1" >> $LOGFILE
echo "End time  : $ENDT" >> $LOGFILE
cat $LOGFILE
mv $LOGFILE regenerated/

