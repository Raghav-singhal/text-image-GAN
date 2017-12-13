#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --mem=20000
#SBATCH -t24:00:00


#python main.py --cuda --ngpu 2 --dataset mnist #--dataroot ./mnist_data --outf mnist_out

#python main.py --cuda --ngpu 2 --dataset mnist #--dataroot ./mnist_data

python main.py --cuda --ngpu 3 --dataset folder --dataroot ./flower_data --outf flower_out_300 --niter 300

