# Paths and compiler flags
cudaInclude = /usr/local/cuda-12.4/targets/x86_64-linux/include/
CUDAFLAGS = -O3 -I $(cudaInclude)
OMPFLAGS = -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
LDFLAGS = -lgomp

# Source files
CUDA_SRC = main.cpp
OMP_SRC = ompmain.cpp

# Targets
cuda: $(CUDA_SRC)
    g++ $(CUDAFLAGS) $(CUDA_SRC) -o cuda

omp: $(OMP_SRC)
    g++ $(CUDAFLAGS) $(OMPFLAGS) $(OMP_SRC) $(LDFLAGS) -o omp

# Phony target for cleaning up generated files
clean:
    rm -f cuda omp *.o
