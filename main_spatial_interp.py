import os
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spokes', type=int, metavar='', required=True)
parser.add_argument('-g', '--gpu', type=int, metavar='', required=True)
parser.add_argument('-t', '--tv_weight', type=float, metavar='', required=False, default=0.02)
parser.add_argument('-l', '--lr_weight', type=float, metavar='', required=False, default=0.0002)
parser.add_argument('-st', '--stv_weight', type=float, metavar='', required=False, default=0) # Just in case
parser.add_argument('-n', '--neuron', type=int, metavar='', required=False, default=128)
parser.add_argument('-ly', '--layers', type=int, metavar='', required=False, default=5)
parser.add_argument('-hs', '--log2_hashmap_size', type=int, metavar='', required=False, default=24)
parser.add_argument('-ls', '--per_level_scale', type=float, metavar='', required=False, default=1.8)
parser.add_argument('-e', '--epochs', type=int, metavar='', required=False, default=1600)
parser.add_argument('-m', '--mask', action='store_true', required=False)
parser.add_argument('-r', '--relL2', action='store_true', required=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch
import datetime
import h5py
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torch.utils.tensorboard import SummaryWriter
from utils import fftnc, ifftnc, coil_combine, path_checker, visual_mag, visual_err_mag, gen_traj, NUFFT
from scipy import io
import sigpy.mri as mr
from model import INR

params = {
    'n_levels': 16,#16
    "n_features_per_level": 2,
    "log2_hashmap_size": 23,#args.log2_hashmap_size,
    "base_resolution": 16,
    "per_level_scale": args.per_level_scale,
    'lr': 0.001,
    "n_neurons": 128,#args.neuron,
    "n_hidden_layers": 3,#args.layers,
    "tv_weight": args.tv_weight,
    "lr_weight": args.lr_weight,
    "stv_weight": args.stv_weight,
    "epochs": args.epochs, 
    "mask": args.mask,
    "relL2": args.relL2
}
print(params)

# Important Constants
GA = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))  # GoldenAngle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 1e-3
summary_epoch = 50
spoke_num = args.spokes
epoch = params['epochs']
relL2_eps = 1e-4
scale = 1.5


log_path = './log_cmr/spoke{}_{}'.format(spoke_num, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S')))
path_checker(log_path)
writer = SummaryWriter(log_path)

# Import and Preprocess Data
mat_path = './test_cardiac.mat'
with h5py.File(mat_path, 'r') as f:
    img_full = f['img'][:]
    smap_full = f['smap'][:]
img_full = torch.as_tensor(img_full).to(device)
smap_full = torch.as_tensor(smap_full).to(device)
frames = img_full.shape[0]
coil_num = img_full.shape[1]
grid_size_full = img_full.shape[-1]
grid_size = int(grid_size_full//scale)
spoke_length = grid_size * 2
offset = int((grid_size_full-grid_size)//2)
img_fft = fftnc(img_full, (-2, -1))[..., offset:offset+grid_size, offset:offset+grid_size]
img = ifftnc(img_fft, (-2, -1))
smap = mr.app.EspiritCalib(torch.mean(img_fft, dim=0).cpu().numpy(),crop=0).run()
smap = torch.as_tensor(smap).to(device)
img_gt = coil_combine(img, smap)
scale_factor = torch.abs(img_gt).max()
img_gt /= scale_factor # Normalization
img_gt_full = coil_combine(img_full, smap_full)
scale_factor = torch.abs(img_gt_full).max()
img_gt_full /= scale_factor
import scipy.io as sio
import matplotlib.pyplot as plt

# Speichern des Ground Truth als .mat
sio.savemat(os.path.join(log_path, 'ground_truth.mat'), {
    'img_gt_full': img_gt_full.cpu().numpy()
})
sio.savemat(os.path.join(log_path, 'downsized_img.mat'), {
    'img_gt': img_gt.cpu().numpy()
})

gt_mag_full = torch.abs(img_gt_full).cpu().numpy().squeeze()
gt_mag_cropped = torch.abs(img_gt).cpu().numpy().squeeze()

# Create subfolders
frame_folder_full = os.path.join(log_path, 'ground_truth_frames')
frame_folder_cropped = os.path.join(log_path, 'ground_truth_frames_cropped')
os.makedirs(frame_folder_full, exist_ok=True)
os.makedirs(frame_folder_cropped, exist_ok=True)

# Save all frames from both versions
for i in range(gt_mag_full.shape[0]):
    # Full-size frame
    frame_full = gt_mag_full[i].squeeze()
    save_path_full = os.path.join(frame_folder_full, f'frame_{i:02d}.png')
    plt.imsave(save_path_full, frame_full, cmap='gray')

    # Cropped frame
    frame_cropped = gt_mag_cropped[i].squeeze()
    save_path_cropped = os.path.join(frame_folder_cropped, f'frame_{i:02d}.png')
    plt.imsave(save_path_cropped, frame_cropped, cmap='gray')

ktraj = gen_traj(GA, spoke_length, frames * spoke_num).reshape(2, frames, -1).transpose(1, 0)
dcomp = torch.abs(torch.linspace(-1, 1, spoke_length)).repeat([spoke_num, 1]).to(device)
nufft_op = NUFFT(ktraj, dcomp, smap)
kdata = nufft_op.forward(img_gt).reshape([frames, coil_num, spoke_num, spoke_length])

# Build Model and Loss
inr = INR(nufft_op, params, lr, relL2_eps)
pos, pos_dense_s = inr.build_ssr_pos(grid_size, frames, scale)


psnr = 0.0
ssim = 0.0
time_usage = 0.0
epoch_loop = tqdm(range(epoch), total=epoch, leave=True)
for e in epoch_loop:

    # Training
    intensity, delta_time = inr.train(pos, kdata, e)
    time_usage += delta_time
    epoch_loop.set_description("[Train] [Lr:{:5e}]".format(inr.scheduler.get_last_lr()[0]))
    epoch_loop.set_postfix(dc_loss=inr.dc_loss.item(), tv_loss=inr.tv_loss.item(), max=torch.abs(intensity).max().item(),
                           lowrank_loss=inr.lowrank_loss.item())
    writer.add_scalar('loss_train', inr.loss_train, e + 1)

    # Infering
    if (e + 1) % summary_epoch == 0:
        with torch.no_grad():
            intensity, psnr_tmp, ssim_tmp = inr.infer(pos_dense_s, img_gt_full, smap_full, sscale=scale)
            if (e + 1) == params['epochs']:
                frame0 = torch.abs(intensity[0]).cpu().numpy().squeeze()
                plt.imsave(os.path.join(log_path, 'final_frame0_epoch1600.png'), frame0, cmap='gray')

                err0 = torch.abs(intensity[0] - img_gt_full[0]).cpu().numpy().squeeze()
                plt.imsave(os.path.join(log_path, 'final_frame0_epoch1600_err.png'), err0, cmap='hot')

        io.savemat(log_path + '/proposed_{}.mat'.format(e+1),
                    {'img_proposed': intensity.cpu().numpy()})
        visual_mag(intensity,
            log_path + '/proposed_{}_{}_abs_{}.png'.format(spoke_num, frames, e+1))
        visual_err_mag(intensity, img_gt_full, log_path + '/proposed_{}_{}_abs_err_{}.png'.format(spoke_num, frames, e+1))
        writer.add_scalar('psnr', psnr_tmp, e + 1)
        writer.add_scalar('ssim', ssim_tmp, e + 1)
        if psnr_tmp > psnr:
            psnr = psnr_tmp

# Summary
print('Best PSNR: {:.4f}'.format(psnr))
print('Time Consumption: {:.2f}s'.format(time_usage))