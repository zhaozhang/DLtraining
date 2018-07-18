#!/bin/bash
#SBATCH -J Cifar10-1node    # job name
#SBATCH -o Cifar10-1node.out         # output and error file name (%j expands to jobID)
#SBATCH -N 1              # total number of nodes
#SBATCH -n 1
#SBATCH -p flat-quadrant           # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 4 hours
#SBATCH -A Intel-TensorFlow    #TG-CCR150011    # project name

module load phdf5

mkdir -p /tmp/keras
cp /home1/apps/keras/data/datasets.tar /tmp/keras/datasets.tar 
tar xf /tmp/keras/datasets.tar -C /tmp/keras

ibrun -np 1 python cifar10_resnet_keras.py
