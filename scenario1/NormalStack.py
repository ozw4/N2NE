#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 08:32:49 2023

@author: aiuser4
"""
import sys
sys.path.append('../')
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from util import load_seismic_data, reshape_traces, normalization, calculate_psnr

# Constants and hyperparameters setup
sweep_n = 2
noise_level = 0.02
codename = os.path.basename(__file__)
name = f"{codename[:-3]}_{noise_level}_{sweep_n}"

# Loading Marmousi data from the SEGY file
segy_file_path = '../data/marmousi_csg.sgy'
traces, attributes = load_seismic_data(segy_file_path)
traces = reshape_traces(traces, attributes['ffid'], attributes['trcnum_within_filedrec'])
original_traces = traces.copy()

# Adding noise to seismic traces
sweep_noise = np.load('../data/sweep_noise.npy')
traces = original_traces.copy()
traces = np.stack([traces] * sweep_n)
traces += sweep_noise[:sweep_n] * noise_level 
traces = np.transpose(traces, (1, 0, 2, 3))

# Averaging traces to create a filtered stack
flt_stack = np.mean(traces, axis=1)

# Calculating metrics
psnr = []
ssim_score = []
for i, (ori, flt) in tqdm(enumerate(zip(original_traces, flt_stack)), desc='calculate metric'):
    psnr.append(calculate_psnr(ori, flt))
    ssim_score.append(ssim(ori, flt, data_range=flt.max() - flt.min()))

average_psnr = np.mean(psnr)
average_ssim = np.mean(ssim_score)
mse = np.mean((original_traces - flt_stack) ** 2)

print(f'Noise Level: {noise_level} Sweep_N: {sweep_n}')
print(f'MSE: {mse}')
print(f'PSNR: {average_psnr}')
print(f'SSIM: {average_ssim}')

plt_idx=400
fig, ax = plt.subplots(1,3, figsize=(12,8))
ax[0].imshow(original_traces[plt_idx].T, vmin=-3, vmax=3, cmap='seismic',  aspect=0.5, interpolation='None')
ax[1].imshow(traces[plt_idx,0].T, vmin=-3, vmax=3, cmap='seismic',  aspect=0.5, interpolation='None')
ax[2].imshow(flt_stack[plt_idx].T, vmin=-3, vmax=3, cmap='seismic',  aspect=0.5, interpolation='None')
ax[0].set_title('Original Shot')
ax[1].set_title('Noisy Shot')
ax[2].set_title('Filtered Shot')
for i in range(3):
    ax[i].xaxis.set_tick_params(length=0)
    ax[i].yaxis.set_tick_params(length=0)
    ax[i].set_ylabel('Time(sample)')
    ax[i].set_xlabel('Trace number')    
plt.tight_layout()
plt.savefig(f'../image/{name}.png')