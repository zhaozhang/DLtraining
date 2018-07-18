#!/bin/bash
#SBATCH -J Cifar10-4node    # job name
#SBATCH -o Cifar10-4node.out         # output and error file name (%j expands to jobID)
#SBATCH -N 4              # total number of nodes
#SBATCH -n 4
#SBATCH -p flat-quadrant           # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 4 hours
#SBATCH -A Intel-TensorFlow    #TG-CCR150011    # project name

ibrun -np 4 mkdir /tmp/keras
/home1/apps/dl-tools/bin/broadcast-mpi.sh /home1/apps/keras/data/datasets.tar /tmp/keras/datasets.tar 4
ibrun -np 4 tar xf /tmp/keras/datasets.tar -C /tmp/keras

ibrun -np 4 python cifar10_resnet_horovod.py
