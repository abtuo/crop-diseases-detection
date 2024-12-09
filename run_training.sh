#!/usr/bin/bash
#SBATCH --account=lasti
#SBATCH -J crop_detection
#SBATCH -o ./slurm_output/out_%j.log
#SBATCH -e ./slurm_output/err_%j.log
#SBATCH --partition=gpu40G,classicgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01-00:00:00 # 1 day
#SBATCH --mem=64G

echo "=== start ==="
date

source ~/mambaforge/etc/profile.d/conda.sh

conda activate od


python train.py


echo "=== end ==="
date
