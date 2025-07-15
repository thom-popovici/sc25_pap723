#pragma once

// This code is part of the artifact of the paper:
// "Automatic Generation of Mappings for Distributed Fourier Operations"
// accepted for publication to SC'25.

void MPI_Filter(double *input, int in_size, double *output, int out_size, int rank);