#!/bin/bash

set -x

# Incase you need to load custom CUDA versions. Not really necessary for standard implementations.
#module load cuda/9.0
#module unload cuda/8.0
#module load cuda/9.0 cudnn/7.1.3_cuda-9.0
#!nvcc --version
# Edit the main code to enable running on CPU. You can use a flag for gpu=True/False
# gpu=True

gpuid=-1
model_type="vgg16face"
model_name="vgg16_pt"
#! augment=False # Update main to include data augmentation code. Implement experiment code with and without data augmentation
name_job="ConFAR_withoutgpu"
echo $name_job
sbatch -J $name_job -o cpu_out_logs_${name_job}.out.log -e cpu_err_${name_job}.err.log 02_slurm_script.script ${gpuid} ${model_type} ${model_name} 
