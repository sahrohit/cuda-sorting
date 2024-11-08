#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define MAX_BITS 32 // We assume 32-bit integers

// Device function to count 0s and 1s for each bit position
__global__ void countBits(int *data, int *bitCounts, int bit, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        int bitValue = (data[index] >> bit) & 1; // Extract the bit at 'bit' position
        atomicAdd(&bitCounts[bitValue], 1);      // Update count of 0s or 1s
    }
}

// Exclusive scan (prefix sum) for computing positions based on bit counts
__global__ void exclusiveScan(int *input, int *output, int size)
{
    extern __shared__ int temp[];
    int index = threadIdx.x;

    if (index < size)
        temp[index] = input[index];
    __syncthreads();

    for (int offset = 1; offset < size; offset *= 2)
    {
        int value = 0;
        if (index >= offset)
            value = temp[index - offset];
        __syncthreads();
        temp[index] += value;
        __syncthreads();
    }
    output[index] = (index == 0) ? 0 : temp[index - 1];
}

// Kernel to sort by each bit position based on scanned indices
__global__ void sortStep(int *data, int *output, int *scan, int bit, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        int bitValue = (data[index] >> bit) & 1;
        int pos = scan[bitValue] + index;
        output[pos] = data[index];
    }
}

// Host function to perform parallel radix sort
void parallelRadixSort(int *h_data, int size)
{
    int *d_data, *d_output, *d_bitCounts, *d_scan;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_bitCounts, 2 * sizeof(int)); // Only 0s and 1s for each bit
    cudaMalloc(&d_scan, 2 * sizeof(int));

    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    for (int bit = 0; bit < MAX_BITS; ++bit)
    {
        cudaMemset(d_bitCounts, 0, 2 * sizeof(int)); // Reset bit counts

        // Count 0s and 1s for the current bit position
        int blocks = (size + 255) / 256;
        countBits<<<blocks, 256>>>(d_data, d_bitCounts, bit, size);

        // Perform exclusive scan on bit counts to get positions
        exclusiveScan<<<1, 2, 2 * sizeof(int)>>>(d_bitCounts, d_scan, 2);

        // Sort by current bit based on scanned positions
        sortStep<<<blocks, 256>>>(d_data, d_output, d_scan, bit, size);

        // Swap pointers
        int *temp = d_data;
        d_data = d_output;
        d_output = temp;
    }

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_bitCounts);
    cudaFree(d_scan);
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

    int *h_data = new int[size];
    srand(seed);
    for (int i = 0; i < size; i++)
    {
        h_data[i] = rand() % 1000000;
    }

    // Perform parallel radix sort
    parallelRadixSort(h_data, size);

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
