#!/bin/bash

mkdir semiprof_install

mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=/global/homes/t/thom13/Repos/codegen/generator/baseline/libraries/semiprof/semiprof_install/ ..
make -j16
make install 

cd ../