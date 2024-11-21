#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <climits>

__device__ int global_min;

__global__ void find_minimum_fixed_point(const int* data, size_t size)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size)
  {
    int local_min = data[i];
    if (local_min < global_min)
    {
      global_min = local_min;
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <array_size>" << std::endl;
    return -1;
  }

  std::ifstream input_file(argv[1]);
  if (!input_file) {
    std::cerr << "Error opening file" << std::endl;
    return -1;
  }

  size_t N = std::stoul(argv[2]);
  std::vector<int> array(N);

  for (size_t i = 0; i < N && input_file; ++i) {
    input_file >> array[i];
  }

  input_file.close();

  int* d_array;
  int min_value = INT_MAX;
  
  size_t size = N * sizeof(int);

  auto start = std::chrono::steady_clock::now();

  cudaMalloc(&d_array, size);
  cudaMemcpy(d_array, array.data(), size, cudaMemcpyHostToDevice);

  const int blocks = (N + 1023) / 1024;
  const int threads = 1024;

  int old_min_value;
  do {
    old_min_value = min_value;
    cudaMemcpyToSymbol(global_min, &min_value, sizeof(int));
    find_minimum_fixed_point<<<blocks, threads>>>(d_array, N);
    cudaMemcpyFromSymbol(&min_value, global_min, sizeof(int));
  } while (old_min_value != min_value);

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "GPU - OPTIMISTIC ALGORITHM" << std::endl;
  std::cout << " Array size: " << N << std::endl;
  std::cout << " Min = " << min_value << std::endl;
  std::cout << " Time: " << duration.count()  << " us" << std::endl;

  // Print time to file
  std::ofstream output_file("time_analysis.csv", std::ios::app);
  if (!output_file) {
    std::cerr << "Error opening time file" << std::endl;
    return -1;
  }
  output_file << "optimistic," << N << "," << duration.count() <<  std::endl;  

  cudaFree(d_array);
  return 0;
}