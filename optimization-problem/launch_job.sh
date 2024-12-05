#!/usr/bin/bash --login

#SBATCH --job-name=gpu_example
#SBATCH --output=output.out
#SBATCH --error=error.err

### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicola.demarch.001@student.uni.lu

### Submit to the `gpu` partition of Iris
#SBATCH -p gpu
#SBATCH --qos=normal

# Load any necessary modules
module load lib/UCX/1.9.0-GCCcore-10.2.0-CUDA-11.1.1 

# Compile
nvcc -x cu -o build/exercise_GPU exercise_GPU.cu
nvcc -x cu -o build/exercise1_3_fixed_point exercise1_3_fixed_point.cpp
nvcc -x cu -o build/exercise1_2 exercise1_2.cpp

for i in {1..30}
do
  for j in 3 4 5
  do
    # Run the compiled program
    srun build/exercise_GPU tests/pco_${j}.txt
    srun build/exercise1_3_fixed_point tests/pco_${j}.txt
    srun build/exercise1_2 tests/pco_${j}.txt
  done
done