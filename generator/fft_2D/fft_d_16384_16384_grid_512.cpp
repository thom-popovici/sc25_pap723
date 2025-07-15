#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include <complex>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include "templates/local_data.h"
#include "templates/helper.h"

__attribute__((always_inline)) inline void fft_compute(int fft_type, cufftHandle plan, int l, int m, int n, int size_in, std::complex<double> *input, int size_out, std::complex<double> *output)
{
	std::complex<double> *in_ptr = input;
	std::complex<double> *out_ptr = output;

	if(l != 1)
	{
		transpose(l, m * n, (double*) in_ptr, (double*) out_ptr);
	}
	else
	{
		in_ptr = output;
		out_ptr = input;
	}

	cufftExecZ2Z(plan, (cufftDoubleComplex *)out_ptr, (cufftDoubleComplex *)in_ptr, fft_type);

	if(l != 1)
	{
		transpose(m * n, l, (double*) in_ptr, (double*) out_ptr);
	}
}

// 	[0, 1]	[1, 0]
void function(int cold_runs, int hot_runs)
{
	// create grid
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm comm0 = MPI_COMM_WORLD;

	// create arrays
	std::complex<double> *A0_host = NULL, *B0_host = NULL;
	A0_host = (std::complex<double>*) malloc(524288 * sizeof(std::complex<double>));	
	B0_host = (std::complex<double>*) malloc(524288 * sizeof(std::complex<double>));	

	for(int i = 0; i < 524288; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(A0_host + i) = value;
	}

	for(int i = 0; i < 524288; ++i)
	{
		std::complex<double> value(0.0, 0.0);
		*(B0_host + i) = value;
	}

	// create fft_plans
	int inx[1], inembed[1], onembed[1];

	cufftHandle fft_plan0;
	inx[0] = inembed[0] = onembed[0] = 16384;
	cufftPlanMany(&fft_plan0, 1, inx, inembed, 1,  16384, onembed, 1,  16384, CUFFT_Z2Z, 32);

	cufftHandle fft_plan1;
	inx[0] = inembed[0] = onembed[0] = 16384;
	cufftPlanMany(&fft_plan1, 1, inx, inembed, 1,  16384, onembed, 1,  16384, CUFFT_Z2Z, 32);

	// create device arrays
	std::complex<double> *A0 = NULL, *B0 = NULL, *temp = NULL;
	cudaMalloc((void**)&A0, 524288 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&B0, 524288 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&temp, 1048576 * sizeof(std::complex<double>));	
	std::complex<double> *temp0 = (temp + 0 * 524288);
	std::complex<double> *temp1 = (temp + 1 * 524288);

	// copy data from arrays to device arrays
	cudaMemcpy(A0, A0_host, 524288 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	
	cudaMemcpy(B0, B0_host, 524288 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	

	// create measuring parameters
	cudaEvent_t counters[5];
	for(int c = 0; c < 5; ++c)
	{
		cudaEventCreate(&counters[c]);
	}

	double t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;
	for(int r = 0; r < cold_runs + hot_runs; ++r)
	{
		// add measuring things
		cudaEventRecord(counters[0]);
		fft_compute(CUFFT_FORWARD, fft_plan0, 1, 16384, 32, 524288, A0, 524288, temp0);
		// add measuring things
		cudaEventRecord(counters[1]);
		pack_data(32, 512, 32, 524288, (double*) temp0, 524288, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[2]);
		MPI_Alltoall((double*) temp1, 524288/512, MPI_DOUBLE_COMPLEX, (double*) temp0, 524288/512,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[3]);
		fft_compute(CUFFT_FORWARD, fft_plan1, 32, 16384, 1, 524288, temp0, 524288, B0);
		// add measuring things
		cudaEventRecord(counters[4]);

		cudaDeviceSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);

		if(r >= cold_runs)
		{
			float milliseconds;

			cudaEventElapsedTime(&milliseconds, counters[0], counters[1]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[1], counters[2]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[2], counters[3]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[3], counters[4]);
			t_comp += milliseconds;
		}
	}
	// copy data from device arrays to device
	cudaMemcpy(B0_host, B0, 524288 * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);	

	// get measuerments
	MPI_Allreduce(MPI_IN_PLACE, &t_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if(rank == 0)
		printf("16384 16384\tD\t512\tComputation\t%lf\tPacking\t%lf\tCommunication\t%lf\tTotal\t%lf\n", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);

	// destroy fft_plans
	cufftDestroy(fft_plan0);
	cufftDestroy(fft_plan1);

	// destroy dev arrays
	cudaFree(A0);
	cudaFree(B0);
	cudaFree(temp);

	// destroy measuring parameters
	for(int c = 0; c < 5; ++c)
	{
		cudaEventDestroy(counters[c]);
	}

	// destroy arrays
	free(A0_host);
	free(B0_host);
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int cold_runs = atoi(argv[1]);
	int hot_runs = atoi(argv[2]);
	function(cold_runs, hot_runs);
	MPI_Finalize();
	return 0;
}
