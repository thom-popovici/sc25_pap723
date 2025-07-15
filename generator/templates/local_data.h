#pragma once

// This code is part of the artifact of the paper:
// "Automatic Generation of Mappings for Distributed Fourier Operations"
// accepted for publication to SC'25.

void transpose(int rows, int cols, double *input, double *output);

void pack_data(int lm0, int m1, int n, int in_size, double *input, int out_size, double *output);

void unpack_data(int lm, int n0, int n1, int in_size, double *input, int out_size, double *output);