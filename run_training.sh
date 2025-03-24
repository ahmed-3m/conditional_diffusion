#!/bin/bash

# Set environment variables for better GPU performance
export CUDA_VISIBLE_DEVICES=3,4  

# Run training with desired parameters
python /system/user/studentwork/mohammed/diffusion-classifier/main.py \
    --batch_size 64 \
    --num_workers 4 \
    --binary_classes \
    --num_classes 2 \
    --embed_dim 128 \
    --max_epochs 801 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --scheduler_gamma 0.99 \
    --classification_interval 5 \
    --gpus 2 \
    --precision "16-mixed" \
    --seed 42 \
    --project_name "diffusion-cifar" \
    --run_name "binary-classifier_2nd" \
    --checkpoint_dir "/system/user/studentwork/mohammed/diffusion-classifier/checkpoints" \
    --resume_from_checkpoint "/system/user/studentwork/mohammed/diffusion-classifier/checkpoints/last.ckpt"\
    --storage_dir "/system/user/studentwork/mohammed/diffusion-classifier/" \
    --test_classification

# Run with debug mode (smaller dataset, faster iterations)
# Uncomment the following line for debugging
# python /system/user/studentwork/mohammed/diffusion-classifier/main.py --debug --fast_dev_run --batch_size 16 --max_epochs 2