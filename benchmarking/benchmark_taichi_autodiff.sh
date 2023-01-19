#!/bin/bash

export ROOT_DIR=/home/loyot/workspace/code/ngp_pl_gui/Synthetic_NeRF

python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name Lego \
    --hash_type taichi --dir_type taichi --rendering_ad taichi \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips
