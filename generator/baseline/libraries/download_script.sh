#!/bin/bash

git clone https://github.com/af-ayala/heffte.git
git clone --recursive https://github.com/eth-cscs/COSMA
git clone https://github.com/bcumming/semiprof.git

cp heffte_helper/Makefile heffte/
cp heffte_helper/install.sh heffte/

cp COSMA_helper/*.cpp COSMA/miniapp/
cp COSMA_helper/CMakeLists.txt COSMA/miniapp/
cp COSMA_helper/install.sh COSMA/

cp semiprof_helper/install.sh semiprof/
cp semiprof_helper/install.sh semiprof/