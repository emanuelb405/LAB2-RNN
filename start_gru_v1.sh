#!/bin/bash

#SBATCH --job-name="gru_v1"

#SBATCH --workdir=/home/nct01/nct01068/Lab2

#SBATCH --output=err_gru%j.out

#SBATCH --error=err_gru%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=01:00:00

#SBATCH--partition=debug

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python gru_v1.py
