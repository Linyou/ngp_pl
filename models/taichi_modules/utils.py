import torch 
import taichi as ti

from taichi.math import uvec3
taichi_block_size = 128

data_type = ti.f32
torch_type = torch.float32


MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
SQRT3 = 1.7320508075688772
SQRT3_MAX_SAMPLES = SQRT3/1024
SQRT3_2 = 1.7320508075688772*2

@ti.func
def scalbn(x, exponent):
    return x * ti.math.pow(2, exponent)

@ti.func
def calc_dt(t, exp_step_factor, grid_size, scale):
    return ti.math.clamp(t*exp_step_factor, SQRT3_MAX_SAMPLES, SQRT3_2*scale/grid_size)

@ti.func
def frexp(x, base):
    exponent = 0
    start = base
    for i in range(6):
        # start = start * (2**i)
        start_next = start * 2
        if start <= x and x < start_next:
            exponent = i+1
            break
        start = start_next
    return exponent

@ti.func
def frexp_bit(x):
    exponent = 0.0
    if x != 0.0:
        bits = ti.bit_cast(x, ti.u32)
        exponent = ti.i32(bits >> 23 & 0xff) - 0x7f
    return exponent

# float frexp(float x, int* exp) {
#     if (x == 0.0f) {
#         *exp = 0;
#         return 0.0f;
#     }

#     std::uint32_t bits = *reinterpret_cast<std::uint32_t*>(&x);
#     int e = (int)(bits >> 23 & 0xff) - 0x7f;
#     *exp = e;
#     bits &= 0x7fffff;
#     bits |= 0x3f800000;
#     return *reinterpret_cast<float*>(&bits) * (x < 0 ? -1.0f : 1.0f);
# }

@ti.func
def mip_from_pos(xyz, cascades):
    mx = ti.abs(xyz).max()
    exponent = frexp_bit(mx)
    return ti.min(exponent, cascades-1)

@ti.func
def mip_from_dt(dt, grid_size, cascades):
    exponent = frexp_bit(dt*grid_size)
    return ti.min(exponent, cascades-1)

@ti.func
def __expand_bits(v):
    v = (v * ti.uint32(0x00010001)) & ti.uint32(0xFF0000FF)
    v = (v * ti.uint32(0x00000101)) & ti.uint32(0x0F00F00F)
    v = (v * ti.uint32(0x00000011)) & ti.uint32(0xC30C30C3)
    v = (v * ti.uint32(0x00000005)) & ti.uint32(0x49249249)
    return v


@ti.func
def __morton3D(xyz):
    xyz = __expand_bits(xyz)
    return xyz[0] | (xyz[1] << 1) | (xyz[2] << 2)

@ti.func
def __morton3D_invert(x):
    x = x & (0x49249249)
    x = (x | (x >> 2)) & ti.uint32(0xc30c30c3)
    x = (x | (x >> 4)) & ti.uint32(0x0f00f00f)
    x = (x | (x >> 8)) & ti.uint32(0xff0000ff)
    x = (x | (x >> 16)) & ti.uint32(0x0000ffff)
    return ti.int32(x)

@ti.kernel
def morton3D_invert_kernel(
    indices: ti.types.ndarray(field_dim=1),
     coords: ti.types.ndarray(field_dim=2)):
    for i in indices:
        ind = ti.uint32(indices[i])
        coords[i, 0] = __morton3D_invert(ind >> 0)
        coords[i, 1] = __morton3D_invert(ind >> 1)
        coords[i, 2] = __morton3D_invert(ind >> 2)

def morton3D_invert(indices):
    coords = torch.zeros(indices.size(0), 3, device=indices.device, dtype=torch.int32)
    morton3D_invert_kernel(indices.contiguous(), coords)
    ti.sync()
    return coords


@ti.kernel
def morton3D_kernel(
       xyzs: ti.types.ndarray(field_dim=2),
    indices: ti.types.ndarray(field_dim=1)):
    for s in indices:
        xyz = uvec3([xyzs[s, 0], xyzs[s, 1], xyzs[s, 2]])
        indices[s] = ti.cast(__morton3D(xyz), ti.int32)

def morton3D(coords1):
    indices = torch.zeros(coords1.size(0), device=coords1.device, dtype=torch.int32)
    morton3D_kernel(coords1.contiguous(), indices)
    ti.sync()
    return indices


@ti.kernel           
def packbits(
         density_grid: ti.types.ndarray(field_dim=1),
    density_threshold: float,
     density_bitfield: ti.types.ndarray(field_dim=1)):

    for n in density_bitfield:
        bits = ti.uint8(0)

        for i in ti.static(range(8)):
            bits |= (ti.uint8(1)<<i) if (density_grid[8*n+i]>density_threshold) else ti.uint8(0)

        density_bitfield[n] = bits



@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]

@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]

@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]

@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]