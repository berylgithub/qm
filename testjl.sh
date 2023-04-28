#!/bin/sh
#SBATCH -J juliamt
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos p70700_a100dual
#SBATCH --gres=gpu:1
#octave test.m
julia --threads 36 caller.jl
