#!/bin/bash

#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -G 64
#SBATCH -q regular
#SBATCH -J distrib_fft
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --output=results_64_%j.out
#SBATCH --error=error_64_%j.out
#SBATCH -A mXXXX
#SBATCH -t 00:30:00

srun -n 64 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ../libraries/COSMA/build/miniapp/cosma_miniapp -m 1024 -n 1024 -k 16777216 -r 10 -t zdouble"
srun -n 64 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ../libraries/COSMA/build/miniapp/cosma_miniapp -m 1024 -n 1024 -k 16777216 -r 10 -t zdouble -x 8 -y 8"
