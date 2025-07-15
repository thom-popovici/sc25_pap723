#!/bin/bash

#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -G 16
#SBATCH -q regular
#SBATCH -J distrib_fft
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=email@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH --output=results_16_%j.out
#SBATCH --error=error_16_%j.out
#SBATCH -A mXXXX
#SBATCH -t 00:30:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/t/thom13/Repos/codegen/generator/baseline/libraries/heffte/lib/
srun -n 16 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft2d_batch.x 1024 1024 512 4 -reorder -a2a -mps"
srun -n 16 -c 32 --cpu_bind=cores bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ./fft2d_batch.x 4096 4096 32 4 -reorder -a2a -mps"
