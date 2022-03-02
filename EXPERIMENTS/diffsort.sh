#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -t 3:00:00

ml py-pytorch/1.8
ml py-matplotlib/3.4

python3 diffsort.py --k 64 --method anchor