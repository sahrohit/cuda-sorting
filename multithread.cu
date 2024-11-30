#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thrust/sort.h>
using namespace std;

#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaSafeCall(cudaError err,
	const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 ) 

	do
	{
		if (cudaSuccess != err)
		{
			fprintf(stderr,
				"cudaSafeCall() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}
	} while (0);
#pragma warning( pop )
#endif 
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 )

	do
	{
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			fprintf(stderr,
				"cudaCheckError() failed at %s:%i : %s.\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}

		err = cudaThreadSynchronize();
		if (cudaSuccess != err)
		{
			fprintf(stderr,
				"cudaCheckError() with sync failed at %s:%i : %s.\n",
				file, line, cudaGetErrorString(err));
			exit(-1);
		}
	} while (0);
#pragma warning( pop )
#endif 
	return;
}

int * makeRandArray(const int size, const int seed) {
	srand(seed);
	int * array = new int[size];
	for (int i = 0; i < size; i++) {
		array[i] = std::rand() % size;
	}
	return array;
}

__device__ inline void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

__device__ inline int partition(int *a, int l, int h)
{
	int x = a[h];
	int i = (l - 1);

	for (int j = l; j <= h - 1; j++)
	{
		if (a[j] <= x)
		{
			i++;
			swap(&a[i], &a[j]);
		}
	}
	swap(&a[i + 1], &a[h]);
	return (i + 1);
}

__global__ static void quickSort(int *arr, int size) {
	extern __shared__ int sh[];
	const unsigned int tid = threadIdx.x;
	sh[tid] = arr[tid];
	__syncthreads();

	int* s = new int[size];

	int l = 0;
	int h = size - 1;

	int t = -1;

	s[++t] = l;
	s[++t] = h;

	while (t >= 0) {
		h = s[t--];
		l = s[t--];

		int p = partition(sh, l, h);

		if (p - 1 > l) {
			s[++t] = l;
			s[++t] = p - 1;
		}


		if (p + 1 < h)
		{
			s[++t] = p + 1;
			s[++t] = h;
		}
	}
	arr[tid] = sh[tid];
}

int main(int argc, char* argv[])
{
	int *array;//, *d_array;
	int size, seed;
	bool printSorted = false;

	if (argc < 4) {
		std::cerr << "usage: "
			<< argv[0]
			<< " [amount of random nums to generate] [seed value for rand]"
			<< " [1 to print sorted array, 0 otherwise]"
			<< std::endl;
		exit(-1);
	}

	{
		std::stringstream ss1(argv[1]);
		ss1 >> size;
	}

	{
		std::stringstream ss1(argv[2]);
		ss1 >> seed;
	}

	{
		int sortPrint;
		std::stringstream ss1(argv[3]);
		ss1 >> sortPrint;
		if (sortPrint == 1)
			printSorted = true;
	}

	array = (int*)malloc(size * sizeof(int));

	array = makeRandArray(size, seed);

	cudaEvent_t startTotal, stTotal;
	float timeTotal;
	cudaEventCreate(&startTotal);
	cudaEventCreate(&stTotal);
	cudaEventRecord(startTotal, 0);
	int *d_array;
	CudaSafeCall(cudaMalloc((void**)&d_array, size * sizeof(int)));

	cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);

	quickSort <<< 1, size, sizeof(int) * size * 2 >>> (d_array, size);

	cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stTotal, 0);
	cudaEventSynchronize(stTotal);
	cudaEventElapsedTime(&timeTotal, startTotal, stTotal);
	cudaEventDestroy(startTotal);
	cudaEventDestroy(stTotal);

	cudaFree(d_array);

	std::cerr << "Total time in seconds: "
		<< timeTotal / 1000.0 << std::endl;

	if (printSorted) {
		for (int i = 0; i < size; i++)
		{
			cout << array[i] << " ";
		}
	}
}

