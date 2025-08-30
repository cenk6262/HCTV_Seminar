import torch
import numpy as np
import cv2
import h5py

# ========= PARAMETERS ==========
# Path to your .mat or .h5 file containing img_gt_full
mat_path = './test_cardiac.mat'  # adjust to your file
video_path = 'cardiac_one_coil_per_frame.mp4'
fps = 5  # Adjust playback speed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# ===============================

# ========= LOAD DATA ==========
# If saved as .pt or .pth file
# img_gt_full = torch.load(mat_path)

# Example for .mat or .h5
with h5py.File(mat_path, 'r') as f:
    img_gt_full = f['img'][:]  # modify key to match your dataset

# Convert to torch tensor if loaded from h5py
img_gt_full = torch.from_numpy(img_gt_full)

# Confirm shape
print(f"Loaded data shape: {img_gt_full.shape}")
frames, coils, h, w = img_gt_full.shape

# ========= PREPROCESS ==========
assert frames <= coils, "Frames exceed coil count; adjust indexing if needed."

imgs_norm = []

for i in range(frames):
    for c in range(coils):
        img = torch.abs(img_gt_full[i, c]).cpu().numpy()
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype(np.uint8)
        imgs_norm.append(img)

# Write video as before
out = cv2.VideoWriter('cardiac_all_coils.mp4', fourcc, fps, (w, h), isColor=False)

for img in imgs_norm:
    out.write(img)  # No conversion to BGR

out.release()
print(f"Video saved to {video_path}")
