from models.taichi_modules import (
    HashEncoder, DirEncoder, HashEmbedder, SHEncoder, 
    VolumeRendererTaichi
)
from models.custom_functions import VolumeRenderer
import tinycudann as tcnn
import taichi as ti
import torch

ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
device = torch.device('cuda')
L=16; F=2; log2_T=19; N_min=16; b=1.3195079565048218

taichi_hash_encoder = HashEncoder(cuda_hash_encoder.params, b, 8192).to(device)

# check forward
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    r2 = taichi_hash_encoder(dump_position)
print('pytorch forward\n', prof.key_averages(group_by_stack_n=5).table(
    sort_by='self_cuda_time_total', row_limit=5))

# check backward
# a strange loss for better verification
loss2 = ((r2 * r2) - torch.tanh(r2)).sum()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    loss2.backward()
print('pytorch backward\n', prof.key_averages(group_by_stack_n=5).table(
    sort_by='self_cuda_time_total', row_limit=5))