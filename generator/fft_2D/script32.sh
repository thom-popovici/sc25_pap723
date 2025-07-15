#!/bin/bash

#SBATCH -N 8
#SBATCH -C gpu
#SBATCH -G 32
#SBATCH -q regular
#SBATCH -J distrib_fft
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --output=results_32_%j.out
#SBATCH --error=error_32_%j.out
#SBATCH -A mXXXX
#SBATCH -t 00:30:00

srun -n 32 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft_d_16384_16384_proc_32_grid_1D.x 5 5"
srun -n 32 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft_di_16384_16384_proc_32_grid_1D.x 5 5"
