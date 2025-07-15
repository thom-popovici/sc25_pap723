#include "local_data.h"

// This code is part of the artifact of the paper:
// "Automatic Generation of Mappings for Distributed Fourier Operations"
// accepted for publication to SC'25.

void __global__ transpose_device(int rows, int cols, double *input, double *output)
{
  double2 *inx = (double2 *)input;
  double2 *outx = (double2 *)output;

  for (int ij = threadIdx.x + blockDim.x * blockIdx.x; ij < (rows * cols); ij += blockDim.x * gridDim.x)
  {
    int i = ij % cols;
    int j = ij / cols;

    *(outx + ij) = *(inx + i * rows + j);
  }
}
void transpose(int rows, int cols, double *input, double *output)
{
  transpose_device<<<216, 256>>>(rows, cols, input, output);
}

void __global__ pack_device(int lm0, int m1, int n, double *input, double *output)
{
  double2 *inx = (double2 *)input;
  double2 *outx = (double2 *)output;

  for (int ij = threadIdx.x + blockDim.x * blockIdx.x; ij < (lm0 * m1 * n); ij += blockDim.x * gridDim.x)
  {
    int i = ij % (lm0);
    int j = (ij / (lm0)) % m1;
    int k = ij / (lm0 * m1);

    *(outx + (lm0 * n) * j + lm0 * k + i) = *(inx + ij);
  }
}
void pack_data(int lm0, int m1, int n, int in_size, double *input, int out_size, double *output)
{
  pack_device<<<216, 256>>>(lm0, m1, n, input, output);
}

void __global__ unpack_device(int lm, int n0, int n1, double *input, double *output)
{
  double2 *inx = (double2 *)input;
  double2 *outx = (double2 *)output;

  for (int ij = threadIdx.x + blockDim.x * blockIdx.x; ij < (lm * n0 * n1); ij += blockDim.x * gridDim.x)
  {
    int i = ij % (lm);
    int j = (ij / (lm)) % n0;
    int k = ij / (lm * n0);

    *(outx + (lm * n1) * j + (lm) * k + i) = *(inx + ij);
  }
}
void unpack_data(int lm, int n0, int n1, int in_size, double *input, int out_size, double *output)
{
  unpack_device<<<216, 256>>>(lm, n0, n1, input, output);
}
