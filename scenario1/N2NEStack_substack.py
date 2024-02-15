#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 08:32:49 2023

@author: Mitsuyuki Ozawa (mitsuyuki.ozawa@jgi.co.jp)
"""

# Importing necessary libraries and modules
import sys
sys.path.append('../')
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from util import load_seismic_data, reshape_traces, torch_fix_seed, UNet, substack_dataload, calculate_psnr
from tqdm import tqdm

codename=os.path.basename(__file__)

#Constants and hyperparameters setup
TRAIN = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 8
maxepoch = 200
sweep_n = 16
substack_n = 8
noise_level = 0.16
early_stop_tolerance = 5
loss_fn = nn.L1Loss()
learning_rate =1e-4

torch_fix_seed(0)
name=codename[:-3]+f'_{noise_level}_{sweep_n}'

# Loading Marmousi data from the SEGY file
segy_file_path = '../data/marmousi_csg.sgy'
traces, attributes = load_seismic_data(segy_file_path)
traces = reshape_traces(traces, attributes['ffid'], attributes['trcnum_within_filedrec'])
original_traces = traces.copy()

sweep_noise = np.load('../data/sweep_noise.npy')

flt_stacks=[]

torch_fix_seed(0)
name=codename[:-3]+f'_{noise_level}_{substack_n}'

traces = original_traces.copy()
traces = np.stack([traces] * sweep_n)
traces += sweep_noise[:sweep_n] * noise_level
traces = np.pad(traces, ((0, 0), (0, 0), (7, 8), (6, 7)))
traces = np.transpose(traces, (1, 0, 2, 3))
traces = np.expand_dims(traces, 2)
traces_train, traces_valid = train_test_split(traces, test_size = 0.1, random_state=1)
x_valid, t_valid = substack_dataload(traces_valid, substack_n, len(traces_valid))

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if TRAIN:
    train_loss = 0
    best_valid_loss = 1e10
    early_stop_count = 0
    report_itera = len(traces_train) // batchsize
    scaler = GradScaler()
    itera = 0
    epoch = 0
    
    while True:
        itera+=1
        x_train, t_train = substack_dataload(traces_train, substack_n,  batchsize)
        if np.random.rand()>0.5: 
            x_data = np.flip(x_train, axis=2).copy()   
            t_data = np.flip(t_train, axis=2).copy()   
        x_train = torch.tensor(x_train).to(device)
        t_train = torch.tensor(t_train).to(device)        
        
        model.train()
        optimizer.zero_grad()
        with autocast():
            outputs = model(x_train) 
            loss = loss_fn(outputs, t_train) 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * x_train.size(0)
        
        if itera % report_itera == 0:
            epoch+=1
            print(f"epoch: {epoch}")

            model.eval()
            valid_loss = 0
            outputs_valid=[]
            
            x_valid = torch.tensor(x_valid).to(device)
            for x, t in zip(x_valid, t_valid):
                x,t = torch.tensor(x).to(device), torch.tensor(t).to(device)
                x,t = torch.unsqueeze(x, 0), torch.unsqueeze(t, 0)
                with torch.no_grad():
                    output = model(x)
                loss = loss_fn(output, t)
                valid_loss += loss
                outputs_valid.append(output.cpu().numpy())
            outputs_valid = np.array(outputs_valid)
            
            train_loss /= len(traces_train)            
            valid_loss /= len(x_valid)

            print(f"train loss: {train_loss:>7f}")
            print(f"valid loss: {valid_loss:>7f}")
                     
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"../model/{name}.pth")  
                best_epoch = epoch
                early_stop_count = 0
            if early_stop_count >= early_stop_tolerance:
                print('early stop')
                break 
            early_stop_count += 1
            
            x_valid = x_valid.cpu().numpy()

            if epoch==200:
                break
               
    print(f'Finish Training best_epoch:{best_epoch}')
    
model.load_state_dict(torch.load(f'../model/{name}.pth', map_location=device))
model.eval()

flt_stack = np.zeros_like(original_traces)

for n in range(sweep_n):
    for i, x in enumerate(traces):
        ind = np.arange(sweep_n)
        ind = np.delete(ind, n)
        rand_ind = random.sample(list(ind), substack_n - 1)
        rand_ind.append(n)
        x = np.mean(x[rand_ind], axis=0)

        inp = torch.Tensor(x).to(device)
        inp = torch.unsqueeze(inp, 0)
        with torch.no_grad():
            output = model(inp).cpu().numpy()[0, 0, 7:-8, 6:-7]
        flt_stack[i] += output
flt_stack /= sweep_n
flt_stacks.append(flt_stack)

psnr = []
ssim_score = []
for i, (ori, flt) in tqdm(enumerate(zip(original_traces, flt_stack)), desc='calculate metric'):
    psnr.append(calculate_psnr(ori, flt))
    ssim_score.append(ssim(ori, flt, data_range=flt.max() - flt.min()))
average_psnr = np.mean(psnr)
average_ssim = np.mean(ssim_score)

mse = np.mean((original_traces - flt_stack) ** 2)

# Printing results
print(f'Noise Level:{noise_level} Sweep_N:{sweep_n}')
print(f'MSE: {mse}')
print(f'PSNR: {average_psnr}')
print(f'SSIM: {average_ssim}')

traces = traces[:, :, 0, 7:-8, 6:-7]

plt_idx = 400
fig, ax = plt.subplots(1, 3, figsize=(12, 8))
ax[0].imshow(original_traces[plt_idx].T, vmin=-3, vmax=3, cmap='seismic', aspect=0.5, interpolation='None')
ax[1].imshow(traces[plt_idx, 0].T, vmin=-3, vmax=3, cmap='seismic', aspect=0.5, interpolation='None')
ax[2].imshow(flt_stack[plt_idx].T, vmin=-3, vmax=3, cmap='seismic', aspect=0.5, interpolation='None')
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
