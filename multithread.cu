#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;

#define CUDA_CHECK_ERROR
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err,
  const char * file,
    const int line) {
  #ifdef CUDA_CHECK_ERROR
  #pragma warning(push)
  #pragma warning(disable: 4127)
  do {
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }
  } while (0);
  #pragma warning(pop)
  #endif
  return;
}

inline void __cudaCheckError(const char * file,
  const int line) {
  #ifdef CUDA_CHECK_ERROR
  #pragma warning(push)
  #pragma warning(disable: 4127)

  do {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }

    err = cudaThreadSynchronize();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }
  } while (0);

  #pragma warning(pop)
  #endif
  return;
}

int * makeRandArray(const int size,
  const int seed) {
  srand(seed);
  int * array = new int[size];
  for (int i = 0; i < size; i++) {
    array[i] = std::rand() % size;
  }
  return array;
}

__device__ void swap(int * a, int * b) {
  int tmp = * a;
  * a = * b;
  * b = tmp;
}

__global__ void bubbleSort(int * array, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < n; i++) {
    int offset = i % 2;
    int left = 2 * id + offset;
    int right = left + 1;

    if (right < n) {
      if (array[left] > array[right]) {
        swap( & array[left], & array[right]);
      }
    }
    __syncthreads();
  }
}

int main(int argc, char * argv[]) {
  int * array;
  int size, seed;
  bool printSorted = false;
  if (argc < 4) {
    std::cerr << "usage: " <<
      argv[0] <<
      " [amount of random nums to generate] [seed value for rand]" <<
      " [1 to print sorted array, 0 otherwise]" <<
      std::endl;
    exit(-1);
  } {
    std::stringstream ss1(argv[1]);
    ss1 >> size;
  } {
    std::stringstream ss1(argv[2]);
    ss1 >> seed;
  } {
    int sortPrint;
    std::stringstream ss1(argv[3]);
    ss1 >> sortPrint;
    if (sortPrint == 1)
      printSorted = true;
  }
  array = makeRandArray(size, seed);

  cudaEvent_t startTotal, stopTotal;
  float timeTotal;
  cudaEventCreate( & startTotal);
  cudaEventCreate( & stopTotal);
  cudaEventRecord(startTotal, 0);

  int * d_array;
  CudaSafeCall(cudaMalloc( & d_array, size * sizeof(int)));
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));
  CudaCheckError();

  dim3 grdDim;
  dim3 blkDim;

  cudaDeviceProp dev_prop;
  cudaGetDeviceProperties( & dev_prop, 0);
  int maxThreads = dev_prop.maxThreadsDim[0];
  if (size / 2 < maxThreads) {
    blkDim = dim3(size / 2, 1, 1);
    grdDim = dim3(1, 1, 1);
  } else {
    blkDim = dim3(maxThreads, 1, 1);
    grdDim = dim3(ceil((size / (2.0 * maxThreads))), 1, 1);

  }

  bubbleSort << < grdDim, blkDim >> > (d_array, size);
  CudaCheckError();

  CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));
  CudaCheckError();

  cudaEventRecord(stopTotal, 0);
  cudaEventSynchronize(stopTotal);
  cudaEventElapsedTime( & timeTotal, startTotal, stopTotal);
  cudaEventDestroy(startTotal);
  cudaEventDestroy(stopTotal);

  std::cerr << "elapsed time: " << timeTotal / 1000.0 << std::endl;

  if (printSorted) {
    for (int i = 0; i < size; i++) {
      cout << array[i] << " ";
    }
  }

  cudaFree(d_array);
  free(array);

  return 0;
}