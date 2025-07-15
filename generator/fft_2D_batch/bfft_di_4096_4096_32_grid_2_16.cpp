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

// 	[1, 0, 2]	[1, 0, 2]
void function(int cold_runs, int hot_runs)
{
	// create grid
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm comm0, comm1;
	MPI_Comm_split(MPI_COMM_WORLD, rank / 2, rank, &comm0);
	MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &comm1);

	// create arrays
	std::complex<double> *A0_host = NULL, *C0_host = NULL;
	A0_host = (std::complex<double>*) malloc(16777216 * sizeof(std::complex<double>));	
	C0_host = (std::complex<double>*) malloc(16777216 * sizeof(std::complex<double>));	

	for(int i = 0; i < 16777216; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(A0_host + i) = value;
	}

	for(int i = 0; i < 16777216; ++i)
	{
		std::complex<double> value(0.0, 0.0);
		*(C0_host + i) = value;
	}

	// create fft_plans
	int inx[1], inembed[1], onembed[1];

	cufftHandle fft_plan0;
	inx[0] = inembed[0] = onembed[0] = 4096;
	cufftPlanMany(&fft_plan0, 1, inx, inembed, 1,  4096, onembed, 1,  4096, CUFFT_Z2Z, 4096);

	cufftHandle fft_plan1;
	inx[0] = inembed[0] = onembed[0] = 4096;
	cufftPlanMany(&fft_plan1, 1, inx, inembed, 1,  4096, onembed, 1,  4096, CUFFT_Z2Z, 4096);

	cufftHandle fft_plan2;
	inx[0] = inembed[0] = onembed[0] = 4096;
	cufftPlanMany(&fft_plan2, 1, inx, inembed, 1,  4096, onembed, 1,  4096, CUFFT_Z2Z, 4096);

	cufftHandle fft_plan3;
	inx[0] = inembed[0] = onembed[0] = 4096;
	cufftPlanMany(&fft_plan3, 1, inx, inembed, 1,  4096, onembed, 1,  4096, CUFFT_Z2Z, 4096);

	// create device arrays
	std::complex<double> *A0 = NULL, *C0 = NULL, *temp = NULL;
	cudaMalloc((void**)&A0, 16777216 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&C0, 16777216 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&temp, 33554432 * sizeof(std::complex<double>));	
	std::complex<double> *temp0 = (temp + 0 * 16777216);
	std::complex<double> *temp1 = (temp + 1 * 16777216);

	// copy data from arrays to device arrays
	cudaMemcpy(A0, A0_host, 16777216 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	
	cudaMemcpy(C0, C0_host, 16777216 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	

	// create measuring parameters
	cudaEvent_t counters[11];
	for(int c = 0; c < 11; ++c)
	{
		cudaEventCreate(&counters[c]);
	}

	double t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;
	for(int r = 0; r < cold_runs + hot_runs; ++r)
	{
		// add measuring things
		cudaEventRecord(counters[0]);
		fft_compute(CUFFT_FORWARD, fft_plan0, 2048, 4096, 2, 16777216, A0, 16777216, temp0);
		// add measuring things
		cudaEventRecord(counters[1]);
		pack_data(4194304, 2, 2, 16777216, (double*) temp0, 16777216, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[2]);
		MPI_Alltoall((double*) temp1, 16777216/2, MPI_DOUBLE_COMPLEX, (double*) temp0, 16777216/2,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[3]);
		unpack_data(2048, 4096, 2, 16777216, (double*) temp0, 16777216, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[4]);
		fft_compute(CUFFT_FORWARD, fft_plan1, 1, 4096, 4096, 16777216, temp1, 16777216, temp0);
		// add measuring things
		cudaEventRecord(counters[5]);
		fft_compute(CUFFT_INVERSE, fft_plan2, 1, 4096, 4096, 16777216, temp0, 16777216, temp1);
		// add measuring things
		cudaEventRecord(counters[6]);
		pack_data(2048, 2, 4096, 16777216, (double*) temp1, 16777216, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[7]);
		MPI_Alltoall((double*) temp0, 16777216/2, MPI_DOUBLE_COMPLEX, (double*) temp1, 16777216/2,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[8]);
		unpack_data(4194304, 2, 2, 16777216, (double*) temp1, 16777216, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[9]);
		fft_compute(CUFFT_INVERSE, fft_plan3, 2048, 4096, 2, 16777216, temp0, 16777216, C0);
		// add measuring things
		cudaEventRecord(counters[10]);

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
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[4], counters[5]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[5], counters[6]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[6], counters[7]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[7], counters[8]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[8], counters[9]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[9], counters[10]);
			t_comp += milliseconds;
		}
	}
	// copy data from device arrays to device
	cudaMemcpy(C0_host, C0, 16777216 * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);	

	// get measuerments
	MPI_Allreduce(MPI_IN_PLACE, &t_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if(rank == 0)
		printf("4096 4096 32\tDI\t2 16\tComputation\t%lf\tPacking\t%lf\tCommunication\t%lf\tTotal\t%lf\n", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);

	// destroy fft_plans
	cufftDestroy(fft_plan0);
	cufftDestroy(fft_plan1);
	cufftDestroy(fft_plan2);
	cufftDestroy(fft_plan3);

	// destroy dev arrays
	cudaFree(A0);
	cudaFree(C0);
	cudaFree(temp);

	// destroy measuring parameters
	for(int c = 0; c < 11; ++c)
	{
		cudaEventDestroy(counters[c]);
	}

	// destroy arrays
	free(A0_host);
	free(C0_host);
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
