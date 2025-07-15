#!/bin/bash

## This script is part of the artifact of the paper:
## "Automatic Generation of Mappings for Distributed Fourier Operations"
## accepted for publication to SC'25.

## Description.
## This script invoked DiMage DFT with specific capacity constraints (last argument).
## Argument: 
## @1: .rels.template file
## @2: grid shape 1D, 2D or 3D
## @3: Number of processes.
## @4: Problem size. Mostly 4 (extra large) but also '3' (used in 2D DFT DAGs)
## @5: Special sub-space mode: 1d, 2d, m2--m5.
## @6: Memory capacity constraint, in GB or MB units. Used 16GB or close to it for ASPLOS'25 submission.


if [ $# -ne 6 ]; then
  echo "Require: dagname, grid_dim (1,2,3), procs (int), pbs (0:small,1:medium,2:large), special flag (nmodes=, 1d, 2d, m2-m5, max, all), capacity (e.g., 32GB, 16GB, 512MB)"
  exit 42
fi

daglist="ft3d-f ft3d-fb ft3d-f-batch ft3d-fb-batch ft3d-f-mm ft3d-fb-mm ft2d-f ft2d-fb ft2d-f-batch ft2d-fb-batch ft2d-f-mm ft2d-fb-mm ft4d-f ft4d-fb ft4d-f-batch ft4d-fb-batch"
dagid=0

dagarg=$1
DIMGRID=$2
PC=$3
ACC="full"
CAP=$6

ACCOK=1


curr=0
for dag in $daglist; do
  curr=$(($curr+1))
  if [ "${dag}.rels.template" == $dagarg ]; then
    dagid=$curr
    break
  fi
done

if [ $dagid -eq 0 ]; then
  echo "DAG $dagarg not found. Aborting ..."
  exit 42
else
  echo "Found DAGID $dagid for $dagarg ..."
fi

PRFLAG=""

./prep-file.sh $dagid $dagarg $DIMGRID ${PC} $4

if [ $? -ne 0 ]; then
  echo "Found problems. Aborting ..."
  exit
fi

MFLAG=""
if [ "x$5" == "xm5" ]; then
  MFLAG="-dft-n-modes=5"
fi
if [ "x$5" == "xm4" ]; then
  MFLAG="-dft-n-modes=4"
fi
if [ "x$5" == "xm3" ]; then
  MFLAG="-dft-n-modes=3"
fi
if [ "x$5" == "xm2" ]; then
  MFLAG="-dft-n-modes=2"
fi
if [ "x$5" == "x1d" ]; then
  MFLAG="-dft-1d"
fi
if [ "x$5" == "x2d" ]; then
  MFLAG="-dft-2d"
fi
if [ "x$5" == "xmax" ]; then
  MFLAG="-dft-maxdim"
fi
if [ "x$5" == "xsub" ]; then
  MFLAG="-dft-subdim"
fi
if [ "x$5" == "xall" ]; then
  MFLAG="-dft-allmodes"
fi

FASTFLAG=''
if [ "$ACC" == "fast" ]; then
  FASTFLAG=" -fast -max-tries=200 "
fi

cmd="python dimage-fft.py current.rels -procs=${DIMGRID}D,${PC}p -nocodegen $MFLAG $PRFLAG -timeout=240 -memcap=${CAP} -grid-nondesc  $FASTFLAG "

echo "Command: [$cmd]"

$cmd

res=$?

if [ $res -ne 0 ]; then
  exit 42
fi

PBS=M
if [ $4 -eq 2 ]; then
  PBS=L
fi
if [ $4 -eq 4 ]; then
  PBS=X
fi
if [ $4 -eq 3 ]; then
  PBS=B
fi
if [ $4 -eq 0 ]; then
  PBS=S
fi
tmp=`echo $dagarg | sed -e 's/\.rels\.template//g'`
PCSTR="$PC"
if [ $PC -lt 10 ]; then
  PCSTR="000${PC}"
fi
if [ $PC -ge 10 -a $PC -lt 100 ]; then
  PCSTR="00${PC}"
fi
if [ $PC -ge 100 -a $PC -lt 1000 ]; then
  PCSTR="0${PC}"
fi


TGTDIR="${tmp}_${DIMGRID}D-${PCSTR}p-${PBS}-${MFLAG}-${CAP}"

BIGDIR="regenerated"

if [ ! -d $BIGDIR ]; then
  mkdir $BIGDIR
fi


if [ -d $TGTDIR ]; then
  rm $TGTDIR/*
else
  mkdir $TGTDIR
fi
count=`ls -l $BIGDIR | grep ${TGTDIR} | wc -l`
mv current* $TGTDIR/
date > $TGTDIR/timestamp.txt
mv $TGTDIR $BIGDIR/${TGTDIR}-$count

exit 0


