#include <iostream>
#include <vector>
#include <array>
#include <cuda_runtime.h>



__global__ void find_minimum(const int* vet, int* min_value, size_t size)
{
  int i = threadIdx.x;
  if(i < size)
  {
    if (min_value > vet[i])
      min_value = vet[i];
  } 
}

constexpr size_t N = 5;

int find_min(const std::array<int, N> &vet)
{
  int minimum = vet[0];
  for(auto element : vet)
  {
    if (element < minimum)
      minimum = element;
  }
  return minimum;
}
int main()
{
  std::array<int, N> array = {10, 20, 2, 3, 4};
  int* d_array;
  int* d_min_value, min_value;

  size_t size = N * sizeof(int);
  cudaMalloc(&d_array, size);
  cudaMalloc(&d_min_value, sizeof(int));
  cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice);  const int blocks = 1;
  const int threads = 1024;
  find_minimum<<<blocks, threads>>>(d_array, d_min_value, size);
  cudaMemcpy(min_value, d_min_value, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Min = " << *min_value << std::endl;
  cudaFree(d_array);
  return 0;
}