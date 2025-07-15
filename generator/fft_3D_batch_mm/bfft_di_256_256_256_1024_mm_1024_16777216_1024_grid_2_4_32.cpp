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

// 	[1, 3]	[3, 0, 1, 2]
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
	std::complex<double> *A0_host = NULL, *B0_host = NULL, *C0_host = NULL;
	A0_host = (std::complex<double>*) malloc(268435456 * sizeof(std::complex<double>));	
	B0_host = (std::complex<double>*) malloc(67108864 * sizeof(std::complex<double>));	
	C0_host = (std::complex<double>*) malloc(1048576 * sizeof(std::complex<double>));	

	for(int i = 0; i < 268435456; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(A0_host + i) = value;
	}

	for(int i = 0; i < 67108864; ++i)
	{
		double real_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		double imag_part = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
		std::complex<double> value(real_part, imag_part);
		*(B0_host + i) = value;
	}

	for(int i = 0; i < 1048576; ++i)
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

	// create gemm_plans
	cublasHandle_t gemm_plan;
	cublasCreate(&gemm_plan);

	// create device arrays
	std::complex<double> *A0 = NULL, *B0 = NULL, *C0, *temp = NULL, *aux = NULL;
	cudaMalloc((void**)&A0, 268435456 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&B0, 67108864 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&C0, 1048576 * sizeof(std::complex<double>));	
	cudaMalloc((void**)&temp, 268435456 * sizeof(std::complex<double>));	
	std::complex<double> *temp0 = (temp + 0 * 134217728);
	std::complex<double> *temp1 = (temp + 1 * 134217728);
	cudaMalloc((void**)&aux, 134217728 * sizeof(std::complex<double>));	
	std::complex<double>* Tx0 = temp;

	// copy data from arrays to device arrays
	cudaMemcpy(A0, A0_host, 268435456 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	
	cudaMemcpy(B0, B0_host, 268435456 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);	

	// create measuring parameters
	cudaEvent_t counters[26];
	for(int c = 0; c < 26; ++c)
	{
		cudaEventCreate(&counters[c]);
	}

	double t_comp = 0.0, t_pack = 0.0, t_comm = 0.0;
	for(int r = 0; r < cold_runs + hot_runs; ++r)
	{
		// add measuring things
		cudaEventRecord(counters[0]);
		fft_compute(CUFFT_FORWARD, fft_plan0, 8, 256, 32768, 67108864, B0, 67108864, temp0);
		// add measuring things
		cudaEventRecord(counters[1]);
		pack_data(1024, 2, 32768, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[2]);
		MPI_Alltoall((double*) temp1, 67108864/2, MPI_DOUBLE_COMPLEX, (double*) temp0, 67108864/2,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[3]);
		unpack_data(131072, 256, 2, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[4]);
		fft_compute(CUFFT_FORWARD, fft_plan1, 1024, 256, 256, 67108864, temp1, 67108864, temp0);
		// add measuring things
		cudaEventRecord(counters[5]);
		pack_data(8192, 32, 256, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[6]);
		MPI_Alltoall((double*) temp1, 67108864/32, MPI_DOUBLE_COMPLEX, (double*) temp0, 67108864/32,  MPI_DOUBLE_COMPLEX, comm2);
		// add measuring things
		cudaEventRecord(counters[7]);
		unpack_data(8, 262144, 32, 67108864, (double*) temp0, 67108864, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[8]);
		fft_compute(CUFFT_FORWARD, fft_plan2, 1, 256, 262144, 67108864, temp1, 67108864, temp0);
		// add measuring things
		cudaEventRecord(counters[9]);
		fft_compute(CUFFT_INVERSE, fft_plan3, 1, 256, 262144, 67108864, temp0, 67108864, temp1);
		// add measuring things
		cudaEventRecord(counters[10]);
		pack_data(128, 2, 262144, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[11]);
		MPI_Alltoall((double*) temp0, 67108864/2, MPI_DOUBLE_COMPLEX, (double*) temp1, 67108864/2,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[12]);
		unpack_data(16384, 2048, 2, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[13]);
		fft_compute(CUFFT_INVERSE, fft_plan4, 128, 256, 2048, 67108864, temp0, 67108864, temp1);
		// add measuring things
		cudaEventRecord(counters[14]);
		pack_data(1024, 32, 2048, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[15]);
		MPI_Alltoall((double*) temp0, 67108864/32, MPI_DOUBLE_COMPLEX, (double*) temp1, 67108864/32,  MPI_DOUBLE_COMPLEX, comm2);
		// add measuring things
		cudaEventRecord(counters[16]);
		unpack_data(8192, 256, 32, 67108864, (double*) temp1, 67108864, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[17]);
		fft_compute(CUFFT_INVERSE, fft_plan5, 1024, 256, 256, 67108864, temp0, 67108864, temp1);
		// add measuring things
		cudaEventRecord(counters[18]);
		MPI_Allgather((double*) temp1, 67108864, MPI_DOUBLE_COMPLEX, (double*) temp0, 134217728,  MPI_DOUBLE_COMPLEX, comm0);
		// add measuring things
		cudaEventRecord(counters[19]);
		unpack_data(128, 524288, 2, 134217728, (double*) temp0, 134217728, (double*) temp1);
		// add measuring things
		cudaEventRecord(counters[20]);
		pack_data(16384, 32, 256, 134217728, (double*) temp1, 134217728, (double*) temp0);
		// add measuring things
		cudaEventRecord(counters[21]);
		MPI_Alltoall((double*) temp0, 134217728/32, MPI_DOUBLE_COMPLEX, (double*) temp1, 134217728/32,  MPI_DOUBLE_COMPLEX, comm2);
		// add measuring things
		cudaEventRecord(counters[22]);
		unpack_data(2048, 2048, 32, 134217728, (double*) temp1, 134217728, (double*) Tx0);
		// add measuring things
		cudaEventRecord(counters[23]);
		gemm_compute(gemm_plan, 512, 524288, 256, 268435456, A0, 134217728, Tx0, 131072, temp0);
		// add measuring things
		cudaEventRecord(counters[24]);
		MPI_Allreduce((double*) temp0, (double*) C0, 131072, MPI_DOUBLE_COMPLEX, MPI_SUM, comm2);
		// add measuring things
		cudaEventRecord(counters[25]);

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
			cudaEventElapsedTime(&milliseconds, counters[18], counters[19]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[19], counters[20]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[20], counters[21]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[21], counters[22]);
			t_comm += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[22], counters[23]);
			t_pack += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[23], counters[24]);
			t_comp += milliseconds;
			cudaEventElapsedTime(&milliseconds, counters[24], counters[25]);
			t_comm += milliseconds;
		}
	}
	// copy data from device arrays to device
	cudaMemcpy(C0_host, C0, 1048576 * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);	

	// get measuerments
	MPI_Allreduce(MPI_IN_PLACE, &t_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &t_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if(rank == 0)
		printf("1024 16777216\tDI\t2 4 32\tComputation\t%lf\tPacking\t%lf\tCommunication\t%lf\tTotal\t%lf\n", t_comp / hot_runs, t_pack / hot_runs, t_comm / hot_runs, (t_comp + t_pack + t_comm) / hot_runs);

	// destroy fft_plans
	cufftDestroy(fft_plan0);
	cufftDestroy(fft_plan1);
	cufftDestroy(fft_plan2);
	cufftDestroy(fft_plan3);
	cufftDestroy(fft_plan4);
	cufftDestroy(fft_plan5);

	cublasDestroy(gemm_plan);
	// destroy dev arrays
	cudaFree(A0);
	cudaFree(B0);
	cudaFree(C0);
	cudaFree(temp);
	cudaFree(aux);

	// destroy measuring parameters
	for(int c = 0; c < 26; ++c)
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
