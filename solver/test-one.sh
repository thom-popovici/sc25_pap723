#!/bin/bash

## This script is part of the artifact of the paper:
## "Automatic Generation of Mappings for Distributed Fourier Operations"
## accepted for publication to SC'25.

if [ $# -lt 5 -o $# -gt 6 ]; then
  echo "Require: dagname, grid_dim (1,2,3), procs (int), pbs (0:small,1:medium,2:large), special flag (nmodes=, 1d, 2d, max, all), [accelerate ('full' or 'fast')]"
  exit 42
fi

daglist="ft3d-f ft3d-fb ft3d-f-batch ft3d-fb-batch ft3d-f-mm ft3d-fb-mm ft2d-f ft2d-fb ft2d-f-batch ft2d-fb-batch ft2d-f-mm ft2d-fb-mm ft4d-f ft4d-fb ft4d-f-batch ft4d-fb-batch"
dagid=0

dagarg=$1
DIMGRID=$2
PC=$3

DIMAGE_EF="dimage.error"

ACC="full"
if [ $# -eq 6 ]; then
  ACC=$6
fi

ACCOK=0
if [ "x$ACC" == "xfull" ]; then
  ACCOK=1
fi
if [ "x$ACC" == "xfast" ]; then
  ACCOK=1
fi

if [ $ACCOK -ne 1 ]; then
  echo "[ERROR] Script requires acceleration mode 'mm' or 'full' (arg#6). Given -$ACC- . Aborting ..."
  exit 42
fi

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

hasmm=`echo $dagarg | grep mm`


PRFLAG=""
if [ "x$hasmm" == "x" ]; then
  PRFLAG=""
fi


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

cmd="python dimage-fft.py current.rels -procs=${DIMGRID}D,${PC}p -nocodegen $MFLAG $PRFLAG -timeout=240 -memcap=16GB $FASTFLAG "

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

BIGDIR="regenerated"

if [ ! -d $BIGDIR ]; then
  mkdir $BIGDIR
fi

TGTDIR="${tmp}_${DIMGRID}D-${PCSTR}p-${PBS}-${MFLAG}"

if [ -d $TGTDIR ]; then
  rm $TGTDIR/*
else
  mkdir $TGTDIR
fi
count=`ls -l $BIGDIR | grep ${TGTDIR} | wc -l`
mv current* $TGTDIR/
date > $TGTDIR/timestamp.txt
if [ -f $DIMAGE_EF ]; then
  mv $DIMAGE_EF $TGTDIR/
fi
mv $TGTDIR $BIGDIR/${TGTDIR}-$count

exit 0


