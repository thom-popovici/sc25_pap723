NVCC = nvcc
CXX = CC
CXXFLAGS = -O3 -std=c++17 -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/include/ -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/lib64/ -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/lib64/ -lcufft -lcublas -lcudart
NVCCFLAGS = -O3 --ptxas-options=-v -arch=sm_80 -gencode=arch=compute_80,code=compute_80

CUDA_SRCS = templates/helper.cu templates/local_data.cu
CUDA_OBJS = templates/helper.o templates/local_data.o

FOLDERS = fft_2D fft_2D_batch fft_2D_batch_mm fft_3D fft_3D_batch fft_3D_batch_mm
CPP_SRCS = $(wildcard $(foreach d,$(FOLDERS),$(d)/*.cpp))
EXES = $(CPP_SRCS:.cpp=.x)

all: $(CUDA_OBJS) $(EXES)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.x: %.cpp $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) $< $(CUDA_OBJS) -o $@

clean:
	rm -f $(EXES) $(CPP_SRCS) $(CUDA_OBJS)