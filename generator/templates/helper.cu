#include "helper.h"

// This code is part of the artifact of the paper:
// "Automatic Generation of Mappings for Distributed Fourier Operations"
// accepted for publication to SC'25.

#define THREADS 256
#define BLOCKS 108

__global__ void MPI_Filter_dev(double *input, double *output, int size, int rank) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  double2 *ind = (double2*) input;
  double2 *outd = (double2*) output;

  for(int i = tid; i < size; i += gridDim.x * blockDim.x) {
    *(outd + i) = *(ind + size * rank + i);
  }
}

void MPI_Filter(double *input, int in_size, double *output, int out_size, int rank)
{
  MPI_Filter_dev<<<2 * BLOCKS, THREADS>>>(input, output, out_size, rank);
}
