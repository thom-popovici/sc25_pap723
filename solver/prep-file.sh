#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Require: dagID (1:14), dagname, grid_dim (1,2,3), procs (int), pbs (0:small,1:medium,2:large)"
  exit 42
fi

echo "[INFO] Calling script sc24-prep-file.sh ... Will instantiate a .rels file from template ..."

DAGID=$1
DAGNAME=$2
GRID=$3
PROCS=$4
PBS=$5

LARGE=2
MEDIUM=1
SMALL=0
BIG=3
XLG=4

T=1
B=1
M=1

##1: ft3d-f
##2: ft3d-fb
if [ $DAGID -eq 1 -o $DAGID -eq 2 ]; then
  if [ $PBS -eq $LARGE ]; then
    T=256
  fi
  if [ $PBS -eq $XLG ]; then
    T=256
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=32
  fi
  if [ $PBS -eq $SMALL ]; then
    T=16
  fi
fi

##3: ft3d-f-batch
##5: ft3d-f-mm
if [ $DAGID -eq 3 -o $DAGID -eq 5 ]; then
  if [ $PBS -eq $XLG ]; then
    T=256
    B=1024
  fi
  if [ $PBS -eq $BIG ]; then
    T=128
    B=1024
  fi
  if [ $PBS -eq $LARGE ]; then
    T=128
    B=256
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=32
    B=64
  fi
  if [ $PBS -eq $SMALL ]; then
    T=16
    B=32
  fi
  M=$(($T*$T*$T))
fi
 
##4: ft3d-fb-batch
##6: ft3d-fb-mm
if [ $DAGID -eq 4 -o $DAGID -eq 6 ]; then
  if [ $PBS -eq $XLG ]; then
    T=256
    B=1024
  fi
  if [ $PBS -eq $BIG ]; then
    T=128
    B=1024
  fi
  if [ $PBS -eq $LARGE ]; then
    T=64
    B=1024
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=32
    B=128
  fi
  if [ $PBS -eq $SMALL ]; then
    T=16
    B=32
  fi
  M=$(($T*$T*$T))
fi



##7: ft2d-f
##8: ft2d-fb
if [ $DAGID -eq 7 -o $DAGID -eq 8 ]; then
  if [ $PBS -eq $XLG ]; then
    T=16384
  fi
  if [ $PBS -eq $BIG ]; then # BIG = 3
    T=16384
  fi
  if [ $PBS -eq $LARGE ]; then
    T=4096
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=256
  fi
  if [ $PBS -eq $SMALL ]; then
    T=64
  fi
fi

##9: ft2d-f-batch
##11: ft2d-f-mm
if [ $DAGID -eq 9 -o $DAGID -eq 11 ]; then
  if [ $PBS -eq $XLG ]; then
    T=4096
    B=32
  fi
  if [ $PBS -eq $BIG ]; then # BIG = 3
    T=1024
    B=512
  fi
  if [ $PBS -eq $LARGE ]; then
    T=1024
    B=512
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=256
    B=32
  fi
  if [ $PBS -eq $SMALL ]; then
    T=64
    B=16
  fi
  M=$(($T*$T))
fi

##10: ft2d-fb-batch
##12: ft2d-fb-mm
if [ $DAGID -eq 10 -o $DAGID -eq 12 ]; then
  if [ $PBS -eq $XLG ]; then
    T=4096
    B=32
  fi
  if [ $PBS -eq $BIG ]; then # BIG = 3
    T=1024
    B=512
  fi
  if [ $PBS -eq $LARGE ]; then
    T=1024
    B=512
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=256
    B=128
  fi
  if [ $PBS -eq $SMALL ]; then
    T=64
    B=16
  fi
  M=$(($T*$T))
fi

##13: ft4d-f
##14: ft4d-fb
##15: ft4d-f-batch
##16: ft4d-fb-batch
if [ $DAGID -ge 13 -a $DAGID -le 16 ]; then
  if [ $PBS -eq $XLG ]; then
    T=64
    B=32
  fi
  if [ $PBS -eq $LARGE ]; then
    T=64
    B=32
  fi
  if [ $PBS -eq $MEDIUM ]; then
    T=16
    B=8
  fi
  if [ $PBS -eq $SMALL ]; then
    T=8
    B=4
  fi
  if [ $DAGID -eq 13 ]; then
    T=$(($T*2))
  fi
  M=$(($T*$T*$T*$T))
fi



SRC=$DAGNAME

echo "Using=DIMS: T=$T; M=$M; B=$B; File=$SRC;"
echo "Using=DIMS: T=$T; M=$M; B=$B; File=$SRC;" > config.txt
if [ -f $SRC ]; then
  echo "Template file found: $SRC"
else
  echo "NO Template found: $SRC"
  exit 42
fi

RF=current.rels
echo $RF

cat $SRC | sed -e "s/#TDIM#/$T/g" -e "s/#BDIM#/$B/g" -e "s/#MDIM#/$M/g" > $RF

exit 0
