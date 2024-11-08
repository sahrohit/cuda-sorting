# Paths and compiler flags
cudaInclude = /usr/local/cuda-10.0/targets/x86_64-linux/include/
CUDAFLAGS = -O3 -I $(cudaInclude)
OMPFLAGS = -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
LDFLAGS = -lgomp

# Source files
CUDA_SRC = main.cu
OMP_SRC = ompmain.cu

# Targets
cuda: $(CUDA_SRC)
	nvcc $(CUDAFLAGS) $(CUDA_SRC) ${OMPFLAGS} ${LDFLAGS} -o cuda

omp: $(OMP_SRC)
	nvcc $(CUDAFLAGS) $(OMP_SRC) $(OMPFLAGS) $(LDFLAGS) -o omp

# Phony target for cleaning up generated files
clean:
	rm -f cuda omp *.o
