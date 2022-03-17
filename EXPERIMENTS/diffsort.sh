#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
#SBATCH -t 3:00:00

ml py-pytorch/1.8
ml py-matplotlib/3.4

python3 diffsort.py --anchor 1024 --k 1024 --method plane

# python3 diffsort.py --anchor 128 --k 128 --method plane --top topk --r 10 --folder topk