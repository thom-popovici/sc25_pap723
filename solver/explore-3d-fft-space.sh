#!/bin/bash


## This script is part of the artifact of the paper:
## "Automatic Generation of Mappings for Distributed Fourier Operations"
## accepted for publication to SC'25.

## This script generates specific 3D-FT DAGs variants, on 3D, 2D and 1D grids,
## across varying number of GPUs (16 to 512 in most cases).
## The 16GB parameter can be changed to a larger GPU memory.
## However, given the internal handling of NCCL and GPU-aware
## runtimes, it is recommended to use at most 80% of the GPU.
## Selected GPU capacities are specific to LBL's Perlmutter.

if [ $# -ne 3 ]; then
  echo "explore-3d-fft-space.sh: Require input template name, use batch (0:No, 1:Yes), use-auto-sched (0=fixed; 1=auto)"
  exit 42
fi

dagtemp=$1
batch=$2
schedmodestr=$3

## Problem sizes: 4=XL, 3=L, 2=M(edium), 1=S(mall)
PBS=4


if [ ! -f $dagtemp ]; then
  echo "Input template $dagtemp not found. Aborting ..."
  exit 42
fi

schedmode=1
if [ "x$schedmodestr" == "xauto" ]; then
 schedmode=1 
fi
if [ "x$schedmodestr" == "xfixed" ]; then
 schedmode=0 
fi

MODEFLAG=""

#modes="2d 1d m4 m5 m3"
modes="2d 1d"
procs="512 256 128 64 32 16"
procs="512 256 128 64"

## Unbatched variants.
if [ $batch -eq 0 ]; then

  procs="512 256 128 64 32 16"

  GRID=2
  modes="2d"
  for pp in $procs; do
    for mam in $modes; do
      MODEFLAG="2d"
      if [ $schedmode -eq 1 ]; then
        MODEFLAG="''"
      fi
      ./test-one.sh $dagtemp $GRID $pp $PBS $MODEFLAG
    done
  done

  GRID=1
  modes="1d"
  for pp in $procs; do
    for mam in $modes; do
      ./test-one.sh $dagtemp $GRID $pp $PBS $mam fast
    done
  done
fi

## Re-insert end-if here.

## Batched variants without matmul.
if [ $batch -eq 1 ]; then
  GRID=3
  modes="2d"
  for pp in $procs; do
    for mam in $modes; do
      MODEFLAG="2d"
      if [ $schedmode -eq 1 ]; then
        MODEFLAG="''"
      fi
      ./test-one.sh $dagtemp $GRID $pp $PBS $MODEFLAG
    done
  done

  GRID=2
  modes="1d"
  for pp in $procs; do
    for mam in $modes; do
      ./test-one.sh $dagtemp $GRID $pp $PBS $mam
    done
  done
fi

## Batched variants composed with matmul.
if [ $batch -eq 2 ]; then

  GRID=3
  modes="2d"
  for pp in $procs; do
    for mam in $modes; do
      MODEFLAG="2d"
      if [ $schedmode -eq 1 ]; then
        MODEFLAG="''"
      fi
      cap=16
      #if [ $pp -lt 128 ]; then
      #  cap=16
      #fi
      ./test-with-capacity.sh $dagtemp $GRID $pp $PBS $MODEFLAG "${cap}GB"
    done
  done

  GRID=2
  modes="1d"
  cap="16GB"
  for pp in $procs; do
    for mam in $modes; do
      ./test-with-capacity.sh $dagtemp $GRID $pp $PBS $mam $cap
    done
  done
fi


exit 0




