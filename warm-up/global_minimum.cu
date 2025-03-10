#include <iostream>
#include <array>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <vector>

__device__ int global_min;

__global__ void find_minimum(const int* vet, size_t size)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size)
  {
    atomicMin(&global_min, vet[i]);
  }
}


int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <array_size>" << std::endl;
    return -1;
  }
  // Read vector from file
  std::ifstream input_file(argv[1]);
  if (!input_file) {
    std::cerr << "Error opening file" << std::endl;
    return -1;
  }

  size_t N = std::stoul(argv[2]);
  std::vector<int> array(N);

  for (size_t i = 0; i < N; i++) {
    input_file >> array[i];
  }

  int* d_array;
  int min_value = INT_MAX;
  
  size_t size = N * sizeof(int);

  auto start = std::chrono::steady_clock::now();

  cudaMalloc(&d_array, size);
  cudaMemcpy(d_array, array.data(), size, cudaMemcpyHostToDevice);

  // Initialize global_min on the device
  cudaMemcpyToSymbol(global_min, &min_value, sizeof(int));

  const int blocks = (N + 1023) / 1024;
  const int threads = 1024;
  find_minimum<<<blocks, threads>>>(d_array, N);

  // Copy the result back to the host
  cudaMemcpyFromSymbol(&min_value, global_min, sizeof(int));
  
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

std::cout << "GPU - GLOBAL MINIMUM ALGORITHM" << std::endl;
  std::cout << " Array size: " << N << std::endl;
  std::cout << " Min = " << min_value << std::endl;
  std::cout << " Time: " << duration.count()  << " us" << std::endl;

  // Print time to file
  std::ofstream output_file("time_analysis.csv", std::ios::app);
  if (!output_file) {
    std::cerr << "Error opening time file" << std::endl;
    return -1;
  }
  output_file << "atomic," << N << "," << duration.count() << std::endl;

  cudaFree(d_array);
  
  return 0;
}