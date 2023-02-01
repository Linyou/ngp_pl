import taichi as ti
import torch 
from torch.cuda.amp import custom_fwd, custom_bwd
from .utils import calc_dt, mip_from_pos, mip_from_dt

@ti.kernel
def composite_train_fw(
           sigmas: ti.types.ndarray(field_dim=1),
             rgbs: ti.types.ndarray(field_dim=2),
           deltas: ti.types.ndarray(field_dim=1),
               ts: ti.types.ndarray(field_dim=1),
           rays_a: ti.types.ndarray(field_dim=2),
      T_threshold: float,
    total_samples: ti.types.ndarray(field_dim=1),
          opacity: ti.types.ndarray(field_dim=1),
            depth: ti.types.ndarray(field_dim=1),
              rgb: ti.types.ndarray(field_dim=2),
               ws: ti.types.ndarray(field_dim=1),):

    for n in opacity:
        ray_idx = rays_a[n, 0]
        start_idx = rays_a[n, 1]
        N_samples = rays_a[n, 2]

        T = 1.0
        samples = 0
        while samples<N_samples:
            s = start_idx + samples
            a = 1.0 - ti.exp(-sigmas[s]*deltas[s])
            w = a*T

            rgb[ray_idx, 0] += w*rgbs[s, 0]
            rgb[ray_idx, 1] += w*rgbs[s, 1]
            rgb[ray_idx, 2] += w*rgbs[s, 2]
            depth[ray_idx] += w*ts[s]
            opacity[ray_idx] += w
            ws[s] = w
            T *= 1.0-a

            if T<T_threshold:
                break
            samples += 1

        total_samples[ray_idx] = samples


@ti.kernel
def composite_train_bw(
        dL_dopacity: ti.types.ndarray(field_dim=1),
          dL_ddepth: ti.types.ndarray(field_dim=1),
            dL_drgb: ti.types.ndarray(field_dim=2),
             dL_dws: ti.types.ndarray(field_dim=1),
             sigmas: ti.types.ndarray(field_dim=1),
               rgbs: ti.types.ndarray(field_dim=2),
    dL_dws_times_ws: ti.types.ndarray(field_dim=1),
                 ws: ti.types.ndarray(field_dim=1),
             deltas: ti.types.ndarray(field_dim=1),
                 ts: ti.types.ndarray(field_dim=1),
             rays_a: ti.types.ndarray(field_dim=2),
            opacity: ti.types.ndarray(field_dim=1),
              depth: ti.types.ndarray(field_dim=1),
                rgb: ti.types.ndarray(field_dim=2),
        T_threshold: float,
         dL_dsigmas: ti.types.ndarray(field_dim=1),
           dL_drgbs: ti.types.ndarray(field_dim=2),):

    for n in dL_dws_times_ws:
        dL_dws_times_ws[n] = dL_dws[n] * ws[n]

    for n in opacity:
        ray_idx = rays_a[n, 0]
        start_idx = rays_a[n, 1]
        N_samples = rays_a[n, 2]

        samples = 0
        R = rgb[ray_idx, 0]
        G = rgb[ray_idx, 1]
        B = rgb[ray_idx, 2]

        O = opacity[ray_idx]
        D = depth[ray_idx]

        T = 1.0
        r = 0.0
        g = 0.0
        b = 0.0
        d = 0.0

        dL_dws_times_ws_sum = 0
        for i in range(N_samples):
            dL_dws_times_ws_sum += int(dL_dws_times_ws[start_idx+i])
            dL_dws_times_ws[start_idx+i] = dL_dws_times_ws_sum

        while samples < N_samples:
            s = start_idx + samples
            a = 1.0 - ti.exp(-sigmas[s]*deltas[s])
            w = a*T

            r += w*rgbs[s, 0]
            g += w*rgbs[s, 1]
            b += w*rgbs[s, 2]
            d += w*ts[s]
            T *= 1.0-a

            # compute gradients by math...
            dL_drgbs[s, 0] = dL_drgb[ray_idx, 0] * w
            dL_drgbs[s, 1] = dL_drgb[ray_idx, 1] * w
            dL_drgbs[s, 2] = dL_drgb[ray_idx, 2] * w

            dL_dsigmas[s] = deltas[s] * (
                dL_drgb[ray_idx, 0] * (rgbs[s, 0]*T-(R-r)) +
                dL_drgb[ray_idx, 1] * (rgbs[s, 1]*T-(G-g)) +
                dL_drgb[ray_idx, 2] * (rgbs[s, 2]*T-(B-b)) +
                dL_dopacity[ray_idx] * (1.0-O) +
                dL_ddepth[ray_idx] * (ts[s]*T-(D-d)) +
                T*dL_dws[s]-(dL_dws_times_ws_sum-dL_dws_times_ws[s])
            )

            if T<T_threshold:
                break
            samples += 1


class VolumeRenderer(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray
    Used in training only

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        total_samples: int, total effective samples
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
        ws: (N) sample point weights
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        opacity = torch.zeros(rays_a.size(0), device=sigmas.device)
        depth = torch.zeros(rays_a.size(0), device=sigmas.device)
        rgb = torch.zeros(rays_a.size(0), 3, device=sigmas.device)
        ws = torch.zeros_like(ts)
        total_samples = torch.zeros(rays_a.size(0), device=sigmas.device)

        # rays_a = rays_a.contiguous()
        # sigmas = sigmas.contiguous()
        # rgbs = rgbs.contiguous()
        # deltas = deltas.contiguous()
        # ts = ts.contiguous()
        composite_train_fw(
                sigmas, 
                rgbs, 
                deltas, 
                ts, 
                rays_a, 
                T_threshold, total_samples, opacity, depth, rgb, ws)
        # ti.sync()
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays_a,
                              opacity, depth, rgb, ws)
        ctx.T_threshold = T_threshold
        return total_samples.sum(), opacity, depth, rgb, ws


    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth, dL_drgb, dL_dws):
        sigmas, rgbs, deltas, ts, rays_a, \
        opacity, depth, rgb, ws = ctx.saved_tensors

        dL_dsigmas = torch.zeros_like(sigmas)
        dL_drgbs = torch.zeros_like(rgbs)

        dL_dws_times_ws = torch.zeros_like(dL_dws)

        composite_train_bw(dL_dopacity, dL_ddepth, dL_drgb, dL_dws,
                                    sigmas, rgbs, dL_dws_times_ws, ws, deltas, ts,
                                    rays_a,
                                    opacity, depth, rgb, 
                                    ctx.T_threshold, 
                                    dL_dsigmas, dL_drgbs)
        # ti.sync()
        return dL_dsigmas, dL_drgbs, None, None, None, None