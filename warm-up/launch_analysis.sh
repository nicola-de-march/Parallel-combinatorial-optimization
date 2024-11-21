#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -J warm-up
#SBATCH -o time_analysis.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH -p gpu

module load lib/UCX/1.9.0-GCCcore-10.2.0-CUDA-11.1.1

# Compile the codes
nvcc -o build/global_minimum global_minimum.cu
nvcc -o build/optimistic_min optimistic_min.cu
nvcc -o build/find_minimum find_minimum.cpp

# Run the codes
for size in 100 500 1000 5000 10000 100000 500000 1000000 10000000 100000000; do
  for i in {1..3}; do
    ./build/optimistic_min arrays/array_${size}.txt ${size}
    ./build/global_minimum arrays/array_${size}.txt ${size}
    ./build/find_minimum arrays/array_${size}.txt ${size}
  done
done
