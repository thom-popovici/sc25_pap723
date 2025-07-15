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

// 	[3, 2, 0, 1]	[0, 3, 2, 1]
void function(int cold_runs, int hot_runs)
{
	// create grid
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm comm0, comm1, comm2;
	MPI_Comm_split(MPI_COMM_WORLD, rank / 2, rank, &comm0);
	MPI_Comm_split(MPI_COMM_WORLD, rank % 2 + 2 * (rank / (2 * 4)), rank, &comm1);
	MPI_Comm_split(MPI_COMM_WORLD, rank % (2 * 4), rank, &comm2);

	// create arrays
	std::complex<double> *A0_host = NULL, *C0_host = NULL;
	A0_host = (std::complex<double>*) malloc(67108864 * sizeof(std::complex<double>));	
	C0_host = (std::complex<double>*) malloc(67108864 * sizeof(std::complex<double>));	

	for(int i = 0; i < 67108864; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(A0_host + i) = value;
	}

	for(int i = 0; i < 67108864; ++i)
	{
		std::complex<double> value(0.0, 0.0);
		*(C0_host + i) = value;
	}

	// create fft_plans
	int inx[1], inembed[1], onembed[1];

	cufftHandle fft_plan0;
	inx[0] = inembed[0] = onembed[0] = 256;
	cufftPlanMany(&fft_plan0, 1, inx, inembed, 1,  256, onembed, 1,  256, CUFFT_Z2Z, 262144);

	cufftHandle fft_plan1;
	inx[0] = inembed[0] = onembed[0] = 256;
	cufftPlanMany(&fft_plan1, 1, inx, inembed, 1,  256, onembed, 1,  256, CUFFT_Z2Z, 262144);

	cufftHandle fft_plan2;
	inx[0] = inembed[0] = onembed[0] = 256;
	cufftPlanMany(&fft_plan2, 1, inx, inembed, 1,  256, onembed, 1,  256, CUFFT_Z2Z, 262144);

	cufftHandle fft_plan3;
	inx[0] = inembed[0] = onembed[0] = 256;
	cufftPlanMany(&fft_plan3, 1, inx, inembed, 1,  256, onembed, 1,  256, CUFFT_Z2Z, 262144);

	cufftHandle fft_plan4;
	inx[0] = inembed[0] = onembed[0] = 256;
	cufftPlanMany(&fft_plan4, 1, inx, inembed, 1,  256, onembed, 1,  256, CUFFT_Z2Z, 262144);

	cufftHandle fft_plan5;
	inx[0] = inembed[0] = onembed[0] = 256;
	cufftPlanMany(&fft_plan5, 1, inx, inembed, 1,  256, onembed, 1,  256, CUFFT_Z2Z, 262144);

	// create device arrays
	std::complex<double> *A0 = NULL, *C0 = NULL, *temp = NULL;
	cudaMalloc((void**)&A0, 67108864 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&C0, 67108864 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&temp, 134217728 * sizeof(std::complex<double>));	
	std::complex<double> *temp0 = (temp + 0 * 67108864);
	std::complex<double> *temp1 = (temp + 1 * 67108864);

	// copy data from arrays to device arrays
	cudaMemcpy(A0, A0_host, 67108864 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	
	cudaMemcpy(C0, C0_host, 67108864 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	

	// create measuring parameters
	cudaEvent_t counters[19];
	for(int c = 0; c < 19; ++c)
	{
		cudaEventCreate(&counters[c]);
	}

	double t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;
	for(int r = 0; r < cold_runs + hot_runs; ++r)
	{
		// add measuring things
		cudaEventRecord(counters[0]);
		fft_compute(CUFFT_FORWARD, fft_plan0, 512, 256, 512, 67108864, A0, 67108864, temp0);
		// add measuring things
		cudaEventRecord(counters[1]);
		pack_data(4096, 32, 512, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[2]);
		MPI_Alltoall((double*) temp1, 67108864/32, MPI_DOUBLE_COMPLEX, (double*) temp0, 67108864/32,  MPI_DOUBLE_COMPLEX, comm2);
		// add measuring things
		cudaEventRecord(counters[3]);
		unpack_data(8, 262144, 32, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[4]);
		fft_compute(CUFFT_FORWARD, fft_plan1, 1, 256, 262144, 67108864, temp1, 67108864, temp0);
		// add measuring things
		cudaEventRecord(counters[5]);
		pack_data(64, 4, 262144, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[6]);
		MPI_Alltoall((double*) temp1, 67108864/4, MPI_DOUBLE_COMPLEX, (double*) temp0, 67108864/4,  MPI_DOUBLE_COMPLEX, comm1);
		// add measuring things
		cudaEventRecord(counters[7]);
		unpack_data(4096, 4096, 4, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[8]);
		fft_compute(CUFFT_FORWARD, fft_plan2, 64, 256, 4096, 67108864, temp1, 67108864, temp0);
		// add measuring things
		cudaEventRecord(counters[9]);
		fft_compute(CUFFT_INVERSE, fft_plan3, 64, 256, 4096, 67108864, temp0, 67108864, temp1);
		// add measuring things
		cudaEventRecord(counters[10]);
		pack_data(512, 32, 4096, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[11]);
		MPI_Alltoall((double*) temp0, 67108864/32, MPI_DOUBLE_COMPLEX, (double*) temp1, 67108864/32,  MPI_DOUBLE_COMPLEX, comm2);
		// add measuring things
		cudaEventRecord(counters[12]);
		unpack_data(4096, 512, 32, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[13]);
		fft_compute(CUFFT_INVERSE, fft_plan4, 512, 256, 512, 67108864, temp0, 67108864, temp1);
		// add measuring things
		cudaEventRecord(counters[14]);
		pack_data(32768, 4, 512, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[15]);
		MPI_Alltoall((double*) temp0, 67108864/4, MPI_DOUBLE_COMPLEX, (double*) temp1, 67108864/4,  MPI_DOUBLE_COMPLEX, comm1);
		// add measuring things
		cudaEventRecord(counters[16]);
		unpack_data(64, 262144, 4, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[17]);
		fft_compute(CUFFT_INVERSE, fft_plan5, 1, 256, 262144, 67108864, temp0, 67108864, C0);
		// add measuring things
		cudaEventRecord(counters[18]);

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
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[6], counters[7]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[7], counters[8]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[8], counters[9]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[9], counters[10]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[10], counters[11]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[11], counters[12]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[12], counters[13]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[13], counters[14]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[14], counters[15]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[15], counters[16]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[16], counters[17]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[17], counters[18]);
			t_comp += milliseconds;
		}
	}
	// copy data from device arrays to device
	cudaMemcpy(C0_host, C0, 67108864 * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);	

	// get measuerments
	MPI_Allreduce(MPI_IN_PLACE, &t_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if(rank == 0)
		printf("256 256 256 1024\tDI\t2 4 32\tComputation\t%lf\tPacking\t%lf\tCommunication\t%lf\tTotal\t%lf\n", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);

	// destroy fft_plans
	cufftDestroy(fft_plan0);
	cufftDestroy(fft_plan1);
	cufftDestroy(fft_plan2);
	cufftDestroy(fft_plan3);
	cufftDestroy(fft_plan4);
	cufftDestroy(fft_plan5);

	// destroy dev arrays
	cudaFree(A0);
	cudaFree(C0);
	cudaFree(temp);

	// destroy measuring parameters
	for(int c = 0; c < 19; ++c)
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
