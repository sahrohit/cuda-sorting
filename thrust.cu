#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <time.h>
#include <thrust/sort.h>
using namespace std;

#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaSafeCall(cudaError err,const char *file, const int line)
{
	#ifdef CUDA_CHECK_ERROR
	#pragma warning( push )
	#pragma warning( disable: 4127 ) 
	do
		{
			if (cudaSuccess != err)
			{
				fprintf(stderr,"cudaSafeCall() failed at %s:%i : %s\n",file, line, cudaGetErrorString(err));
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
			fprintf(stderr,"cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
			exit(-1);
		}

		err = cudaThreadSynchronize();
		if (cudaSuccess != err)
		{
			fprintf(stderr,"cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
			exit(-1);
		}
	} while (0);
	
	#pragma warning( pop )
	#endif 
	return;
}

int * makeRandArray(const int size, const int seed)
{
	srand(seed);
	int * array = new int[size];
	for (int i = 0; i < size; i++)
	{
		array[i] = std::rand() % size;
	}
	return array;
}

__global__ void matavgKernel()
{
}

int main(int argc, char* argv[])
{
	int * array;
	int size, seed;
	bool printSorted = false;

	if (argc < 4) 
	{
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

	array = makeRandArray(size, seed);

	cudaEvent_t startTotal, stopTotal;
	float timeTotal;
	cudaEventCreate(&startTotal);
	cudaEventCreate(&stopTotal);
	cudaEventRecord(startTotal, 0);

	thrust::sort(array, array + size);

	cudaEventRecord(stopTotal, 0);
	cudaEventSynchronize(stopTotal);
	cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
	cudaEventDestroy(startTotal);
	cudaEventDestroy(stopTotal);

	std::cerr << "Total time in seconds: "
		<< timeTotal / 1000.0 << std::endl;

	if (printSorted) {
		for (int i =0; i<size; i++)
		{
			cout<<array[i]<<" ";
		}
	}
}
