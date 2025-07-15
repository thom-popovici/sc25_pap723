#!/bin/bash

#SBATCH -N 64
#SBATCH -C gpu
#SBATCH -G 256
#SBATCH -q regular
#SBATCH -J distrib_fft
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --output=results_256_%j.out
#SBATCH --error=error_256_%j.out
#SBATCH -A mXXXX
#SBATCH -t 00:30:00

srun -n 256 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft_d_1024_1024_512_proc_256_grid_2D.x 5 5"
srun -n 256 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft_di_1024_1024_512_proc_256_grid_2D.x 5 5"
srun -n 256 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft_d_4096_4096_32_proc_256_grid_2D.x 5 5"
srun -n 256 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft_di_4096_4096_32_proc_256_grid_2D.x 5 5"