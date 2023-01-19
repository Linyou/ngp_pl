import taichi as ti

taichi_block_size = 256



@ti.kernel
def sigma_1(B: ti.i32):
    # run network: 1-sigma, 2-rgb
    # sigma layer - 1 weight 32 x 64 --> 64
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j, la in ti.ndrange(B, 64, 16):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for ls in ti.static(range(2)):
                l = la * 2 + ls
                local_val += self.sigma_weights_1[l+j*32] * self.xyzs_embedding[sn, l]
            # relu
            self.hid_sum_temp[sn, j, la] = local_val

@ti.kernel
def sigma_1_sum(self, B: ti.i32):
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j in ti.ndrange(B, 64):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for l in ti.static(range(16)):
                local_val += self.hid_sum_temp[sn, j, l]
            # relu
            self.hid_temp[sn, j] = ti.max(0.0, local_val)      


@ti.kernel
def sigma_2(self, B: ti.i32):
    # sigma layer - 2 weight 64 x 16 --> 16
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j, la in ti.ndrange(B, 16, 16):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for ls in ti.static(range(4)):
                l = la * 4 + ls
                local_val += self.sigma_weights_2[l+j*64] * self.hid_temp[sn, l]
            self.hid_sum_temp[sn, j, la] = local_val

@ti.kernel
def sigma_2_sum(self, B: ti.i32):
    # sigma layer - 2 weight 64 x 16 --> 16
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j in ti.ndrange(B, 16):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for l in ti.static(range(16)):
                local_val += self.hid_sum_temp[sn, j, l]
            self.final_embedding[sn, 16+j] = local_val              

@ti.kernel
def sigma_3(self, B: ti.i32):
    # sigma out write the first element to out_1 with exp activation
    ti.loop_config(block_dim=taichi_block_size)
    for sn in ti.ndrange(B):
        if self.run_model_ind[sn]:
            self.out_1[sn] = ti.exp(self.final_embedding[sn, 16])

@ti.kernel
def rgb_1(self, B: ti.i32):
    # rgb layer - 1 weight 32 x 64 --> 64
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j, la in ti.ndrange(B, 64, 16):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for ls in ti.static(range(2)):
                l = la * 2 + ls
                local_val += self.rgb_weights_1[l+j*32] * self.final_embedding[sn, l]
            self.hid_sum_temp[sn, j, la] = local_val

@ti.kernel
def rgb_1_sum(self, B: ti.i32):
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j in ti.ndrange(B, 64):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for l in ti.static(range(16)):
                local_val += self.hid_sum_temp[sn, j, l]
            # relu
            self.hid_temp[sn, j] = ti.max(0.0, local_val)

@ti.kernel
def rgb_2(self, B: ti.i32):
    # rgb layer - 2 weight 64 x 64 --> 64
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j, la in ti.ndrange(B, 64, 16):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for ls in ti.static(range(4)):
                l = la * 4 + ls
                local_val += self.rgb_weights_2[l+j*64] * self.hid_temp[sn, l]
            self.hid_sum_temp[sn, j, la] = local_val    

@ti.kernel
def rgb_2_sum(self, B: ti.i32):
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j in ti.ndrange(B, 64):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for l in ti.static(range(16)):
                local_val += self.hid_sum_temp[sn, j, l]
            # relu
            self.hid_temp[sn, j] = ti.max(0.0, local_val)

@ti.kernel
def rgb_3(self, B: ti.i32):
    # rgb layer - 3 weight 64 x 3 --> 3
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j, la in ti.ndrange(B, 3, 16):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for ls in ti.static(range(4)):
                l = la * 4 + ls
                local_val += self.rgb_weights_3[l+j*64] * self.hid_temp[sn, l]
            self.hid_sum_temp[sn, j, la] = local_val

@ti.kernel
def rgb_3_sum(self, B: ti.i32):
    ti.loop_config(block_dim=taichi_block_size)
    for sn, j in ti.ndrange(B, 3):
        if self.run_model_ind[sn]:
            local_val = 0.0
            for l in ti.static(range(16)):
                local_val += self.hid_sum_temp[sn, j, l]
            # sigmoid
            self.out_3[sn, j] = 1 / (1 + ti.exp(-local_val))