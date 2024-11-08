#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>

// Single-threaded Bubble Sort kernel
__global__ void singleThreadBubbleSort(int *data, int size)
{
    // Only one thread will do the sorting in this kernel
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < size - 1; i++)
        {
            for (int j = 0; j < size - i - 1; j++)
            {
                if (data[j] > data[j + 1])
                {
                    // Swap if elements are out of order
                    int temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0] << " [amount of random nums to generate] [seed value for rand] [1 to print sorted array, 0 otherwise]" << std::endl;
        return -1;
    }

    int size = atoi(argv[1]);
    int seed = atoi(argv[2]);
    bool printSorted = atoi(argv[3]);

    // Allocate and initialize host data
    int *h_data = new int[size];
    srand(seed);
    for (int i = 0; i < size; i++)
    {
        h_data[i] = rand() % 1000000;
    }

    // Allocate device data
    int *d_data;
    CudaSafeCall(cudaMalloc(&d_data, size * sizeof(int)));
    CudaSafeCall(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    singleThreadBubbleSort<<<1, 1>>>(d_data, size);
    CudaCheckError();

    // Copy result back to host
    CudaSafeCall(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CudaSafeCall(cudaFree(d_data));

    // Output sorted array if requested
    if (printSorted)
    {
        for (int i = 0; i < size; i++)
        {
            std::cout << h_data[i] << " ";
        }
        std::cout << std::endl;
    }

    delete[] h_data;
    return 0;
}
