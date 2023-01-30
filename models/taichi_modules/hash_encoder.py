import os
import torch 
import numpy as np
import taichi as ti
import tinycudann as tcnn

from typing import Dict, Tuple
from taichi.math import uvec3, vec3
from torch.cuda.amp import custom_fwd, custom_bwd
from .utils import (
    torch2ti_grad, ti2torch_grad, ti2torch, torch2ti,
    data_type, torch_type
)

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

@ti.kernel
def ti_copy(data1 : ti.template(), data2 : ti.template()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]

@ti.kernel
def ti_copy_array(data1 : ti.types.ndarray(), data2 : ti.types.ndarray()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]

@ti.kernel
def ti_copy_field_array(data1 : ti.template(), data2 : ti.types.ndarray()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]

            
@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
    # primes = uvec3(ti.uint32(1), ti.uint32(1958374283), ti.uint32(2654435761))
    primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861))
    for i in ti.static(range(3)):
        result ^= ti.uint32(pos_grid_local[i]) * primes[i]
    return result

@ti.func
def under_hash(pos_grid_local, resolution):
    result = ti.uint32(0)
    stride = ti.uint32(1)
    for i in ti.static(range(3)):
        result += ti.uint32(pos_grid_local[i] * stride)
        stride *= resolution
    return result

@ti.func
def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
    hash_result = ti.uint32(0)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, resolution)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size

# per_level_scale = 0.
ivec8 = ti.types.vector(8, dtype=ti.i32)
fvec8 = ti.types.vector(8, dtype=ti.f16)
@ti.kernel
def hash_encode_kernel(
                xyzs: ti.template(),
               table: ti.template(),
      xyzs_embedding: ti.template(),
  hash_map_indicator: ti.template(),
hash_map_sizes_field: ti.template(),
             offsets: ti.template(),
                   B: ti.i32,
     per_level_scale: ti.f32
    ):

    # get hash table embedding
    ti.loop_config(block_dim=16)
    for i, level in ti.ndrange(B, 16):
        # normalize to [0, 1], before is [-0.5, 0.5]
        xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])
        # xyz = xyzs[i]

        scale = 16 * ti.exp(level*ti.log(per_level_scale)) - 1.0
        resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

        offset = offsets[level] * 2

        pos = xyz * scale + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
        pos -= pos_grid_uint
        # pos_grid_uint = ti.cast(pos_grid, ti.uint32)

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0
        # init_0 = 0.0
        # init_1 = 0.0
        # init_w = 1.0
        # local_feature_0 = ti.f16(init_0)
        # local_feature_1 = ti.f16(init_1)

        for idx in ti.static(range(8)):
            # idx_uint = ti.cast(idx, ti.uint32)
            w = 1.
            # w = ti.f16(init_w)
            pos_grid_local = uvec3(0)

            for d in ti.static(range(3)):
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    # w *= ti.f16(1 - pos[d])
                    w *= 1 - pos[d]
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    # w *= ti.f16(pos[d])
                    w *= pos[d]

            index = grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size)
            index_table = offset+index*2
            index_table_int = ti.cast(index_table, ti.int32)
            # local_feature_0 += w * ti.f16(table[index_table_int])
            # local_feature_1 += w * ti.f16(table[index_table_int+1])
            local_feature_0 += w * table[index_table_int]
            local_feature_1 += w * table[index_table_int+1]

        # xyzs_embedding[i, level*2] = data_type(local_feature_0)
        # xyzs_embedding[i, level*2+1] = data_type(local_feature_1)
        xyzs_embedding[i, level*2] = local_feature_0
        xyzs_embedding[i, level*2+1] = local_feature_1

class HashEncoder(torch.nn.Module):
    def __init__(self, tcnn_params, b=1.3195079565048218, batch_size=8192):
        super(HashEncoder, self).__init__()

        self.per_level_scale = b

        # per_level_scale = 1.3195079565048218
        print("per_level_scale: ", b)
        self.offsets = ti.field(ti.i32, shape=(16,))
        self.hash_map_sizes_field = ti.field(ti.uint32, shape=(16,))
        self.hash_map_indicator = ti.field(ti.i32, shape=(16,))
        base_res = 16
        max_params = 2 ** 19
        offset_ = 0
        hash_map_sizes = []
        for i in range(16):
            resolution = int(np.ceil(base_res * np.exp(i*np.log(self.per_level_scale)) - 1.0)) + 1
            params_in_level = resolution ** 3
            params_in_level = int(resolution ** 3) if params_in_level % 8 == 0 else int((params_in_level + 8 - 1) / 8) * 8
            params_in_level = min(max_params, params_in_level)
            self.offsets[i] = offset_
            hash_map_sizes.append(params_in_level)
            self.hash_map_indicator[i] = 1 if resolution ** 3 <= params_in_level else 0
            offset_ += params_in_level
        print("offset_: ", offset_)
        size = np.uint32(np.array(hash_map_sizes))
        self.hash_map_sizes_field.from_numpy(size) 

        self.total_hash_size = offset_*2

        self.hash_table = torch.nn.Parameter(
            torch.zeros(self.total_hash_size, dtype=torch_type), requires_grad=True
        )
        # random_initialize(self.hash_table)
        ti_copy_array(self.hash_table, tcnn_params.to(torch.float32))

        self.parameter_fields = ti.field(data_type, shape=(self.total_hash_size,), needs_grad=True)
        self.input_fields = ti.field(dtype=data_type, shape=(batch_size*1024, 3), needs_grad=True)
        self.output_fields = ti.field(dtype=data_type, shape=(batch_size*1024, 32), needs_grad=True)

        class _module_function(torch.autograd.Function):
            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, params):
                # If no output gradient is provided, no need to
                # automatically materialize it as torch.zeros.
                # ctx.set_materialize_grads(False) # maybe not needed

                output_embedding = torch.zeros(input_pos.shape[0], 32, dtype=torch_type, device=input_pos.device)
                # ti.sync()
                torch2ti(self.input_fields, input_pos.contiguous())
                torch2ti(self.parameter_fields, params.contiguous())
                hash_encode_kernel(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.offsets,
                    input_pos.shape[0],
                    self.per_level_scale,
                )
                ti2torch(self.output_fields, output_embedding)
                # ti.sync()

                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                grad = torch.zeros(self.total_hash_size, dtype=torch_type, device=doutput.device)
                self.zero_grad()
                # ti.sync()
                # doutput *= 128
                # print("doutput max", doutput_scale.mean())
                torch2ti_grad(self.output_fields, doutput.contiguous())
                hash_encode_kernel.grad(
                    self.input_fields, 
                    self.parameter_fields, 
                    self.output_fields, 
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.offsets,
                    doutput.shape[0],
                    self.per_level_scale,
                )
                ti2torch_grad(self.parameter_fields, grad)
                # ti.sync()
                return None, grad

        self._module_function = _module_function
        self.save = False

    def zero_grad(self):
        self.parameter_fields.grad.fill(0.)
        # self.input_fields.grad.fill(0.)
        # self.output_fields.grad.fill(0.)

    def forward(self, positions):
        if self.save ==True:
            os.makedirs('./test_data', exist_ok=True)
            torch.save(positions, './test_data/positions.t')
            self.save = False
        return self._module_function.apply(positions, self.hash_table)