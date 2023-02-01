#!/bin/bash

export ROOT_DIR=./Synthetic_NeRF

# python train.py \
#     --root_dir $ROOT_DIR/Chair \
#     --exp_name Chair \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/Drums \
#     --exp_name Drums \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/Ficus \
#     --exp_name Ficus \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/Hotdog \
#     --exp_name Hotdog \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


NGP_BACKEND=taichi python train.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name Lego \
    --hash_type cuda --dir_type cuda --rendering_ad cuda \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips --no_save_test


# python train.py \
#     --root_dir $ROOT_DIR/Materials \
#     --exp_name Materials \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


# python train.py \
#     --root_dir $ROOT_DIR/Mic \
#     --exp_name Mic \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


# python train.py \
#     --root_dir $ROOT_DIR/Ship \
#     --exp_name Ship \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips




# python train.py \
#     --root_dir $ROOT_DIR/Chair \
#     --exp_name Chair \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/Drums \
#     --exp_name Drums \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/Ficus \
#     --exp_name Ficus \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips

# python train.py \
#     --root_dir $ROOT_DIR/Hotdog \
#     --exp_name Hotdog \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


# python train.py \
#     --root_dir $ROOT_DIR/Lego \
#     --exp_name Lego \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


# python train.py \
#     --root_dir $ROOT_DIR/Materials \
#     --exp_name Materials \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


# python train.py \
#     --root_dir $ROOT_DIR/Mic \
#     --exp_name Mic \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips


# python train.py \
#     --root_dir $ROOT_DIR/Ship \
#     --exp_name Ship \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda \
#     --num_epochs 20 --batch_size 8192 --lr 1e-2 --eval_lpips