#!/bin/sh
#SBATCH -J qm9chm
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos p70700_a100dual
#SBATCH --gres=gpu:1
octave driver.m
