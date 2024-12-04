#include <cuda_runtime.h>
#include "parser.hpp"

// CUDA kernel to check singleton values
__global__ void checkSingletonKernel(bool* d_domains, const int* d_offset, const int* d_domain_upperbounds, bool* d_singleton, int* d_singleton_values, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  int val = -1;
  if (idx < N) {
    for (int i = 0; i <= d_domain_upperbounds[idx]; i++) {
      if(d_domains[d_offset[idx] + i]) {
        count++;
        val = i;
      }
    }
  }
  if (idx < N && count == 1) {
    d_singleton[idx] = true;
    d_singleton_values[idx] = val;
  }
  else {
    d_singleton[idx] = false;
  }
}

// CUDA kernel to update domains
__global__ void updateDomainKernel(bool* domains, const int* d_offset, int* d_domain_upperbounds, bool* singleton, int* singleton_values, int N, int var, const int* d_C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N && singleton[idx]) {
    for (int i = var + 1; i < N; i++) {
      if (singleton_values[idx] <= d_domain_upperbounds[i] && d_C[var * N + i] == 1) {
        domains[d_offset[i] + singleton_values[idx]] = false;
      }
    }
  }
}

// CUDA kernel to check for solution with fixed point
__global__ void checkForSolution(bool* d_found_solution, bool* d_singleton, int var, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > var &&  idx < N && !d_singleton[idx]) {
    *d_found_solution = false;
  }
}