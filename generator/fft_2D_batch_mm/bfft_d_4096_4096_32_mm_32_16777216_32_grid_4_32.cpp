#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include <complex>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cublas_v2.h>

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

__attribute__((always_inline)) inline void gemm_compute(cublasHandle_t handle, int m, int k, int n, int sizeA, std::complex<double> *A, int sizeB, std::complex<double> *B, int sizeC, std::complex<double> *C)
{
	cuDoubleComplex alpha, beta;
	alpha.x = 1.0;
	alpha.y = 1.0;
	beta.x = 0.0;
	beta.y = 0.0;

	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (cuDoubleComplex *)B, n, (cuDoubleComplex *)A, k, &beta, (cuDoubleComplex *)C, n);
}

// 	[0, 1]	[0, 1, 2]
void function(int cold_runs, int hot_runs)
{
	// create grid
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm comm0, comm1;
	MPI_Comm_split(MPI_COMM_WORLD, rank / 4, rank, &comm0);
	MPI_Comm_split(MPI_COMM_WORLD, rank % 4, rank, &comm1);

	// create arrays
	std::complex<double> *A0_host = NULL, *B0_host = NULL, *C0_host = NULL;
	A0_host = (std::complex<double>*) malloc(134217728 * sizeof(std::complex<double>));	
	B0_host = (std::complex<double>*) malloc(4194304 * sizeof(std::complex<double>));	
	C0_host = (std::complex<double>*) malloc(1024 * sizeof(std::complex<double>));	

	for(int i = 0; i < 134217728; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(A0_host + i) = value;
	}

	for(int i = 0; i < 4194304; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(B0_host + i) = value;
	}

	for(int i = 0; i < 1024; ++i)
	{
		std::complex<double> value(0.0, 0.0);
		*(C0_host + i) = value;
	}

	// create fft_plans
	int inx[1], inembed[1], onembed[1];

	cufftHandle fft_plan0;
	inx[0] = inembed[0] = onembed[0] = 4096;
	cufftPlanMany(&fft_plan0, 1, inx, inembed, 1,  4096, onembed, 1,  4096, CUFFT_Z2Z, 1024);

	cufftHandle fft_plan1;
	inx[0] = inembed[0] = onembed[0] = 4096;
	cufftPlanMany(&fft_plan1, 1, inx, inembed, 1,  4096, onembed, 1,  4096, CUFFT_Z2Z, 1024);

	// create gemm_plans
	cublasHandle_t gemm_plan;
	cublasCreate(&gemm_plan);

	// create device arrays
	std::complex<double> *A0 = NULL, *B0 = NULL, *C0, *temp = NULL, *aux = NULL;
	cudaMalloc((void**)&A0, 134217728 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&B0, 4194304 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&C0, 1024 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&temp, 8388608 * sizeof(std::complex<double>));	
	std::complex<double> *temp0 = (temp + 0 * 4194304);
	std::complex<double> *temp1 = (temp + 1 * 4194304);
	cudaMalloc((void**)&aux, 4194304 * sizeof(std::complex<double>));	
	std::complex<double>* Tx0 = temp;

	// copy data from arrays to device arrays
	cudaMemcpy(A0, A0_host, 134217728 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	
	cudaMemcpy(B0, B0_host, 134217728 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	

	// create measuring parameters
	cudaEvent_t counters[9];
	for(int c = 0; c < 9; ++c)
	{
		cudaEventCreate(&counters[c]);
	}

	double t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;
	for(int r = 0; r < cold_runs + hot_runs; ++r)
	{
		// add measuring things
		cudaEventRecord(counters[0]);
		fft_compute(CUFFT_FORWARD, fft_plan0, 1, 4096, 1024, 4194304, B0, 4194304, temp0);
		// add measuring things
		cudaEventRecord(counters[1]);
		pack_data(1024, 4, 1024, 4194304, (double*) temp0, 4194304, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[2]);
		MPI_Alltoall((double*) temp1, 4194304/4, MPI_DOUBLE_COMPLEX, (double*) temp0, 4194304/4,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[3]);
		fft_compute(CUFFT_FORWARD, fft_plan1, 1024, 4096, 1, 4194304, temp0, 4194304, temp1);
		// add measuring things
		cudaEventRecord(counters[4]);
		MPI_Alltoall((double*) temp1, 4194304/4, MPI_DOUBLE_COMPLEX, (double*) temp0, 4194304/4,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[5]);
		unpack_data(1024, 1024, 4, 4194304, (double*) temp0, 4194304, (double*) Tx0);
		// add measuring things
		cudaEventRecord(counters[6]);
		gemm_compute(gemm_plan, 32, 4194304, 1, 134217728, A0, 4194304, Tx0, 32, temp0);
		// add measuring things
		cudaEventRecord(counters[7]);
		MPI_Allreduce((double*) temp0, (double*) C0, 32, MPI_DOUBLE_COMPLEX, MPI_SUM, comm0);
		// add measuring things
		cudaEventRecord(counters[8]);

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
			cudaEventElapsedTime(&milliseconds, counters[4], counters[5]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[5], counters[6]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[6], counters[7]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[7], counters[8]);
			t_comm += milliseconds;
		}
	}
	// copy data from device arrays to device
	cudaMemcpy(C0_host, C0, 1024 * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);	

	// get measuerments
	MPI_Allreduce(MPI_IN_PLACE, &t_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if(rank == 0)
		printf("32 16777216\tD\t4 32\tComputation\t%lf\tPacking\t%lf\tCommunication\t%lf\tTotal\t%lf\n", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);

	// destroy fft_plans
	cufftDestroy(fft_plan0);
	cufftDestroy(fft_plan1);

	cublasDestroy(gemm_plan);
	// destroy dev arrays
	cudaFree(A0);
	cudaFree(B0);
	cudaFree(C0);
	cudaFree(temp);
	cudaFree(aux);

	// destroy measuring parameters
	for(int c = 0; c < 9; ++c)
	{
		cudaEventDestroy(counters[c]);
	}

	// destroy arrays
	free(A0_host);
	free(B0_host);
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
