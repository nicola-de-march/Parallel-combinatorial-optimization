#include <cuda_runtime.h>
#include "parser.hpp"

// CUDA kernel to check singleton values
__global__ void checkSingletonKernel(bool* d_domains, int* d_offset, int* d_domain_upperbounds, bool* d_singleton, int* d_singleton_values, int N, int var) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("idx: %d\n", idx);
}

// CUDA kernel to update domains
// __global__ void updateDomainKernel(bool* domains, int* domain_upperbounds, int* singleton, int* singleton_values, int N, int var, const Data* data) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < N && singleton[idx]) {
//     for (int i = var + 1; i < N; i++) {
//       if (data->get_C_at(var, i) == 1) {
//         domains[i * (domain_upperbounds[i] + 1) + singleton_values[idx]] = false;
//       }
//     }
//   }
// }