#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 08:32:49 2023

@author: Mitsuyuki Ozawa (mitsuyuki.ozawa@jgi.co.jp)
"""

import sys
sys.path.append('../')
import os
import numpy as np
import matplotlib.pyplot as plt
import segyio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from util import load_seismic_data, reshape_traces, torch_fix_seed, UNet, S2RS_Dataset, calculate_psnr, train, valid

codename = os.path.basename(__file__)

# Constants and hyperparameters setup
CONVENTIONAL_METHOD='FXDECON' # select "FXDECON", "MEDIANFILT", "BLINDTRACENET"
TRAIN = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 8
epoch = 200
noise_level = 0.08
early_stop_tolerance = 5
loss_fn = nn.L1Loss()
learning_rate = 1e-4

# Loading Marmousi data from the SEGY file
segy_file_path = '../data/marmousi_csg.sgy'
traces, attributes = load_seismic_data(segy_file_path)
traces = reshape_traces(traces, attributes['ffid'], attributes['trcnum_within_filedrec'])
original_traces = traces.copy()

sweep_noise = np.load('../data/sweep_noise.npy')

torch_fix_seed(0)
name=codename[:-3]+f'_{CONVENTIONAL_METHOD}_{noise_level}'

traces += sweep_noise[0] * noise_level
traces = np.pad(traces, ((0, 0), (7, 8), (6, 7)))
traces = np.expand_dims(traces, 1) 

# Loading pseudo-denoised traces
if CONVENTIONAL_METHOD=='FXDECON':
    segy_denoised_path=f'../data/marmousi_csg_flt_fxdecon_fieldnoise{noise_level}.sgy'
    with segyio.open(segy_denoised_path, ignore_geometry=True, endian='big') as f:
        pseudo_denoised_traces = np.array([x.copy() for x in tqdm(f.trace, desc='Load SEGY')]).astype(np.float32)
elif CONVENTIONAL_METHOD =='MEDIANFILT':
        pseudo_denoised_traces = np.load(f'../data/MedianFilter_{noise_level}_ilineksize3_xlineksize3.npy')
elif CONVENTIONAL_METHOD =='BLINDTRACENET':
        pseudo_denoised_traces = np.load(f'../data/BlindTraceNet_{noise_level}_ratio0.6.npy')
else:
    raise ValueError(" CONVENTIONAL_METHOD is not 'FXDECON','MEDIANFILT'  or 'BLINDTRACENET'.")
    
pseudo_denoised_traces=pseudo_denoised_traces.reshape(original_traces.shape)
pseudo_denoised_traces=np.pad(pseudo_denoised_traces, ((0,0),  (7,8), (6,7)))
pseudo_denoised_traces=np.expand_dims(pseudo_denoised_traces,1)   

traces_train, traces_valid = train_test_split(traces, test_size=0.1, random_state=1)
pseudo_denoised_traces_train, pseudo_denoised_traces_valid = train_test_split(pseudo_denoised_traces, test_size=0.1, random_state=1)
    
train_set = S2RS_Dataset(traces_train, pseudo_denoised_traces_train, flip=True)
valid_set = S2RS_Dataset(traces_valid, pseudo_denoised_traces_valid, flip=False)

train_iter = DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=0)
valid_iter = DataLoader(valid_set, batch_size=batchsize, shuffle=False, num_workers=0)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
best_valid_loss = 1e10
early_stop_count = 0

report_itera=len(traces_train)//batchsize

if TRAIN:
    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_iter, model, optimizer, loss_fn, device)
        valid_loss = valid(valid_iter, model, loss_fn, device)

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"../model/{name}.pth")  
            best_epoch = t
            early_stop_count = 0
        if early_stop_count >= early_stop_tolerance:
            print('early stop')
            break 
        early_stop_count += 1
            
    print(f'Finish Training best_epoch:{best_epoch}')
    
model.load_state_dict(torch.load(f'../model/{name}.pth', map_location=device))
model.eval()

flt_stack=np.zeros_like(traces[:,0, 7:-8, 6:-7])

for i, x in enumerate(traces):
    inp = torch.Tensor(x).to(device)
    inp = torch.unsqueeze(inp, 0) 
    with torch.no_grad():
        output = model(inp).cpu().numpy()[0,0, 7:-8, 6:-7]
    flt_stack[i] = output

# Calculating metrics
psnr=[]
ssim_score=[]
for i, (ori, flt) in tqdm(enumerate(zip(original_traces, flt_stack)), desc='calculate metric'):
    psnr.append(calculate_psnr(ori, flt))
    ssim_score.append(ssim(ori, flt, data_range=flt.max() - flt.min()))
average_psnr = np.mean(psnr)
average_ssim = np.mean(ssim_score)
mse = np.mean((original_traces - flt_stack) ** 2)

# Printing results
print(f'Noise Level:{noise_level}')
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
