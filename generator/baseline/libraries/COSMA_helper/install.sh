#!/bin/bash

mkdir cosma_install

mkdir build
cd build

export CXX=CC

export semiprof_DIR=/global/homes/t/thom13/Repos/codegen/generator/baseline/libraries/semiprof/semiprof_install/

cmake -DCOSMA_BLAS=CUDA -DCOSMA_SCALAPACK=CRAY_LIBSCI -DCOSMA_WITH_GPU_AWARE_MPI=ON -DCOSMA_WITH_PROFILING=ON -DCMAKE_INSTALL_PREFIX=../cosma_install ..
make -j16

cd ../