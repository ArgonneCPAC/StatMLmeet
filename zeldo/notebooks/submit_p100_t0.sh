#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment

#source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf_gpu
#export KERAS_BACKEND=tensorflow
#srun -p cp100 /homes/nramachandra/anaconda3/envs/tf_gpu_14/bin/python VAE_wideBottleneck_new.py
srun -p cp100 python VAE_t0_wideBottle.py

echo [$SECONDS] End job 
