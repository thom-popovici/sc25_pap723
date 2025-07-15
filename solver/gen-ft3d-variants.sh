#!/bin/bash

## This script is part of the artifact of the paper:
## "Automatic Generation of Mappings for Distributed Fourier Operations"
## accepted for publication to SC'25.

## This script generates all 3D-FT DAGs, on 3D, 2D and 1D grids.
## A secondary script is used to handle some internal differences,
## such as the GPU memory.

#source ./env-name.sh
#shellname=`echo $SHELL | sed -e 's/^.*\///g'`
##conda init $shellname
#conda activate $DIMAGE_DFT_ENV

source ./dimage-setup-conda.sh

STARTT=`date`


#ft3d-fb.rels.template
#ft3d-f.rels.template
bmlist="f fb"
for bm in $bmlist; do
  dag="ft3d-${bm}.rels.template"
  ./explore-3d-fft-space.sh  $dag 0 fixed
done

MIDT1=`date`

#ft3d-fb-mm.rels.template
#ft3d-f-mm.rels.template
bmlist="f fb"
for bm in $bmlist; do
  dag="ft3d-${bm}-mm.rels.template"
  ./explore-3d-fft-space.sh  $dag 2 fixed
done


MIDT2=`date`

#ft3d-f-batch.rels.template
#ft3d-fb-batch.rels.template
bmlist="f fb"
for bm in $bmlist; do
  dag="ft3d-${bm}-batch.rels.template"
  ./explore-3d-fft-space.sh  $dag 1 fixed
done


ENDT=`date`

LOGFILE="times-3d.txt"
echo "Start time: $STARTT" > $LOGFILE
echo "Midpoint 1: $MIDT1" >> $LOGFILE
echo "Midpoint 2: $MIDT2" >> $LOGFILE
echo "End time  : $ENDT" >> $LOGFILE
cat $LOGFILE
mv $LOGFILE regenerated/

