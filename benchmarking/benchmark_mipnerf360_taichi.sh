export ROOT_DIR="/home/loyot/workspace/Datasets/NeRF/360_v2/"
export DOWNSAMPLE=0.25 # to avoid OOM

# python train.py \
#     --root_dir $ROOT_DIR/bicycle --dataset_name colmap \
#     --exp_name bicycle --downsample $DOWNSAMPLE \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 16.0 --eval_lpips 


# python train.py \
#     --root_dir $ROOT_DIR/bonsai --dataset_name colmap \
#     --exp_name bonsai --downsample $DOWNSAMPLE \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 16.0 --eval_lpips 

# python train.py \
#     --root_dir $ROOT_DIR/counter --dataset_name colmap \
#     --exp_name counter --downsample $DOWNSAMPLE  \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 16.0 --eval_lpips 

# python train.py \
#     --root_dir $ROOT_DIR/kitchen --dataset_name colmap \
#     --exp_name kitchen --downsample $DOWNSAMPLE  \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 4.0 --eval_lpips 

python train.py \
    --root_dir $ROOT_DIR/stump --dataset_name colmap \
    --exp_name stump --downsample $DOWNSAMPLE  \
    --hash_type cuda --dir_type cuda --rendering_ad cuda --lr 1e-2 \
    --num_epochs 20 --batch_size 8192 --scale 64.0 --eval_lpips 

# python train.py \
#     --root_dir $ROOT_DIR/room --dataset_name colmap \
#     --exp_name room --downsample $DOWNSAMPLE  \
#     --hash_type cuda --dir_type cuda --rendering_ad cuda --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 4.0 --eval_lpips 


# python train.py \
#     --root_dir $ROOT_DIR/flowers --dataset_name colmap \
#     --exp_name flowers --downsample $DOWNSAMPLE  \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 16.0 --eval_lpips 

# python train.py \
#     --root_dir $ROOT_DIR/treehill --dataset_name colmap \
#     --exp_name treehill --downsample $DOWNSAMPLE  \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 64.0 --eval_lpips 

# python train.py \
#     --root_dir $ROOT_DIR/bicycle --dataset_name colmap \
#     --exp_name bicycle --downsample $DOWNSAMPLE  \
#     --hash_type taichi --dir_type taichi --rendering_ad taichi --lr 1e-2 \
#     --num_epochs 20 --batch_size 8192 --scale 16.0 --eval_lpips 