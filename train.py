import torch
from torch import nn
from opt import get_opts
import os
import time
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler
from nnprofiler import LayerProf
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union

import taichi as ti
# ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
ti.init(arch=ti.cuda, device_memory_GB=4)

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP, TaichiNGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def print_stats(times, layer_name):
    forw_t = 0
    back_t = 0
    for name in times:
        forw_t += times[name][0]
        back_t += times[name][1]

    forw_t, back_t = forw_t/1000, back_t/1000
    print(f"{layer_name}: forward: {forw_t}, backward: {back_t}")

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        # self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        self.model = TaichiNGP(self.hparams, scale=self.hparams.scale, rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

        self.tic = 0.0
        self.test_id = 0
        self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.hparams.hash_type}/training'
        os.makedirs(self.val_dir, exist_ok=True)


    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        if self.hparams.dataset_name == 'colmap':
            self.test_dataset_traj = dataset(split='test_traj', **kwargs)
            self.test_saving_training = 20000 // len(self.test_dataset_traj)
        else:
            self.test_saving_training = 20000 // len(self.test_dataset)

        self.test_saving_training = 5

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = torch.optim.Adam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [torch.optim.Adam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

        self.tic = time.time()

    def on_train_end(self):
        self.elapsed_time = time.time() - self.tic

    def training_step(self, batch, batch_nb, *args):
        # if self.global_step == 100:
        #     self.model.render_func.save = True
        #     self.model.hash_encoder.save = True
        #     self.model.dir_encoder.save = True
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        # if not hparams.no_save_test and \
        #     (self.global_step%self.test_saving_training== 0):
        #     if self.hparams.dataset_name == 'colmap':
        #         training_vis_size = len(self.test_dataset_traj)
        #     else:
        #         training_vis_size = len(self.test_dataset)

        #     if self.test_id < training_vis_size:
        #         self.eval()
        #         with torch.no_grad():
        #             if self.hparams.dataset_name=='colmap':
        #                 batch_val = self.test_dataset_traj[self.test_id]
        #             else:
        #                 batch_val = self.test_dataset[self.test_id]

        #             for k, v in batch_val.items():
        #                 if isinstance(v, torch.Tensor):
        #                     batch_val[k] = v.to(self.device)

        #             self.val_on_training(batch_val)
        #             self.test_id += 1
        #         self.train()
        return loss

    def on_validation_start(self):
        # if not hparams.no_save_test:
        # # saving training
        #     imgs = sorted(glob.glob(os.path.join(system.val_dir, 'rgb_*.png')))
        #     imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
        #                     [imageio.imread(img) for img in imgs],
        #                     fps=24, macro_block_size=2)
        #     imgs = sorted(glob.glob(os.path.join(system.val_dir, 'depth_*.png')))
        #     imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
        #                     [imageio.imread(img) for img in imgs],
        #                     fps=24, macro_block_size=2)

        self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.hparams.hash_type}/rendering'
        os.makedirs(self.val_dir, exist_ok=True)

        self.eval()
        for batch_val in self.test_dataset_traj:
            for k, v in batch_val.items():
                if isinstance(v, torch.Tensor):
                    batch_val[k] = v.to(self.device)
            self.val_on_training(batch_val)

        imgs = sorted(glob.glob(os.path.join(system.val_dir, 'rgb_*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs],
                        fps=24, macro_block_size=2)
        imgs = sorted(glob.glob(os.path.join(system.val_dir, 'depth_*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs],
                        fps=24, macro_block_size=2)

        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.hparams.hash_type}'
            os.makedirs(self.val_dir, exist_ok=True)

    def val_on_training(self, batch):
        if self.hparams.dataset_name=='colmap':
            results = self(batch, split='test_traj')
        else:
            results = self(batch, split='test')
        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)

        idx = batch['img_idxs']
        rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        rgb_pred = (rgb_pred*255).astype(np.uint8)
        depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(self.val_dir, f'rgb_{idx:03d}.png'), rgb_pred)
        imageio.imsave(os.path.join(self.val_dir, f'depth_{idx:03d}.png'), depth)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'rgb_{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'depth_{idx:03d}.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams).to(torch.device('cuda'))

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16,
                    #   profiler=PyTorchProfiler(dirpath='./results/', filename='cuda_profiler', row_limit=-1, sort_by_key="cuda_time_total",
                    #   profiler_kwargs={
                    #     'on_trace_ready': torch.profiler.tensorboard_trace_handler('./results', worker_name='worker1'),
                    #     "record_shapes": True
                    #   })
                      )
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     with_modules=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./results', worker_name='worker0'),
    # ) as p:

    # def print_stats(times, layer_name):
    #     forw_t = 0
    #     back_t = 0
    #     for name in times:
    #         forw_t += times[name][0]
    #         back_t += times[name][1]

    #     forw_t, back_t = forw_t/1000, back_t/1000
    #     print(f"{layer_name}: forward: {forw_t}, backward: {back_t}")


    # with LayerProf(system.model.xyz_encoder) as sigma, LayerProf(system.model.rgb_net) as rgb_layer, LayerProf(system.model.hash_encoder) as hash_encoder, LayerProf(system.model.render_func) as render_func:
    trainer.fit(system, ckpt_path=hparams.ckpt_path)
    print(f"total training time: {system.elapsed_time:.2f}")
        # sigma.layerwise_summary()
        # print_stats(sigma._counter, "sigma")
        # rgb_layer.layerwise_summary()
        # print_stats(rgb_layer._counter, "rgb")
        # hash_encoder.layerwise_summary()
        # print_stats(hash_encoder._counter, "hash")
        # render_func.layerwise_summary()
        # print_stats(render_func._counter, "render_func")


    # print(summary_str)
    # trainer.fit(system, ckpt_path=hparams.ckpt_path)
    # print(f"total training time: {system.elapsed_time:.2f}")
    # ti.profiler.print_kernel_profiler_info()
    # print(p.key_averages().table(
        # sort_by="self_cuda_time_total", row_limit=-1))

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, 'rgb_*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs],
                        fps=24, macro_block_size=1)
        imgs = sorted(glob.glob(os.path.join(system.val_dir, 'depth_*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs],
                        fps=24, macro_block_size=1)
