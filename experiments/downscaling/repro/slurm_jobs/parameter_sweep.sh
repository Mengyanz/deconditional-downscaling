#!/bin/bash
#SBATCH --job-name=cloud-sweep
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --mem=4G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o

pyenv activate deconditioning
bash /data/ziz/not-backed-up/bouabid/repos/ContBagGP/experiments/downscaling/repro/run_parameter_sweep.sh
echo "Job Completed"
