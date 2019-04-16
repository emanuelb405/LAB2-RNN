#!/bin/bash

#SBATCH --job-name="gru_4_2"

#SBATCH --workdir=/home/nct01/nct01068/Lab2

#SBATCH --output=err_gru_4_2%j.out

#SBATCH --error=err_gru4_2%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=01:00:00
#SBATCH--partition=debug
module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python gru_4_2.py
