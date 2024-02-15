#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 08:32:49 2023
This code is based on the https://ieeexplore.ieee.org/document/10286279
@author: Mitsuyuki Ozawa (mitsuyuki.ozawa@jgi.co.jp)
"""

import sys
sys.path.append('../')
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import  GradScaler, autocast
import random
from util import load_seismic_data, reshape_traces, torch_fix_seed, UNet, calculate_psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

codename=os.path.basename(__file__)

# Constants and hyperparameters setup
TRAIN = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 8
maxepoch = 200
noise_level = 0.08
early_stop_tolerance = 5
learning_rate = 1e-4
gamma1 = 2
gamma2 = 0.25
loss_fn = nn.L1Loss()
name=codename[:-3]+f'_{noise_level}_{gamma1}_{gamma2}'

# Loading Marmousi data from the SEGY file
segy_file_path = '../data/marmousi_csg.sgy'
traces, attributes = load_seismic_data(segy_file_path)
traces = reshape_traces(traces, attributes['ffid'], attributes['trcnum_within_filedrec'])
original_traces = traces.copy()

sweep_noise = np.load('../data/sweep_noise.npy')
 
torch_fix_seed(0)
traces += sweep_noise[0]*noise_level

traces = np.pad(traces, ((0,0),  (7,8), (6,7)))
traces = np.expand_dims(traces,1)   
    
traces_train=traces[:721]
traces_valid=traces[721:]
    
traces_valid=np.stack([traces_valid[:-1], traces_valid[1:]])
x_valid, t_valid=traces_valid[0], traces_valid[1]

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loss = 0
best_valid_loss = 1e10
early_stop_count = 0
    
report_itera=len(traces_train)//batchsize

scaler=GradScaler()
        
itera=0
epoch=0
if TRAIN:
    while True:
        itera+=1
        
        ind = np.random.randint(1, len(traces_train)-1, size=batchsize)
        x_train = traces[ind].copy()
        RandAdjInd = random.choice([-1,1]) # R (radius of Ur(xk)) = 1
        t_train = traces[ind+RandAdjInd]    
        if np.random.rand()>0.5: 
            x_train = np.flip(x_train, axis=2).copy()   
            t_train = np.flip(t_train, axis=2).copy()   
        x_train = torch.tensor(x_train).to(device)
        t_train = torch.tensor(t_train).to(device)        
        
        model.train()
        optimizer.zero_grad()
        noise_std=np.std(sweep_noise[0])*noise_level
        
        with autocast():
            #denoising
            denoise_outputs = model(x_train) 
            #add prior_noise
            noise = np.random.poisson(lam=noise_std**2*10000, size=denoise_outputs.shape).astype(np.float32)
            noise /= np.sqrt(10000)
            noise -= np.mean(noise)

            noise = torch.Tensor(noise).to(device)
            addnoise_outputs = denoise_outputs + noise
            #re-denoising
            re_denoise_outputs = model(addnoise_outputs)

            loss_s2s = loss_fn(denoise_outputs, t_train)  #Loss S2S
            loss_rede = loss_fn(re_denoise_outputs, denoise_outputs) #Loss re-de-noising
            loss_stab = loss_fn(re_denoise_outputs, x_train) #Loss stability
            loss = loss_s2s + gamma1*(gamma2*loss_rede + loss_stab)    #  equation(5) 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * x_train.size(0)
        
        if itera % report_itera == 0:
            epoch+=1
            print(f"epoch: {epoch}")

            model.eval()
            valid_loss = 0
            valid_loss_s2s = 0
            valid_loss_rede = 0
            valid_loss_stab = 0
            denoise_outputs_valid=[]
            addnoise_outputs_valid=[]
            re_denoise_outputs_valid=[]            
            
            x_valid = torch.tensor(x_valid).to(device)
            for x, t in zip(x_valid, t_valid):
                x,t = torch.tensor(x).to(device), torch.tensor(t).to(device)
                x,t = torch.unsqueeze(x, 0), torch.unsqueeze(t, 0)
                with torch.no_grad():
                    #denoising
                    denoise_outputs = model(x) 
                    #add prior_noise     
                    noise = np.random.poisson(lam=noise_std**2*10000, size=denoise_outputs.shape).astype(np.float32)
                    noise /= np.sqrt(10000)
                    noise -= np.mean(noise)

                    noise = torch.Tensor(noise).to(device)
                    addnoise_outputs = denoise_outputs + noise
                    #re-denoising
                    re_denoise_outputs = model(addnoise_outputs)    
                    
                    loss_s2s = loss_fn(denoise_outputs, t)  #Loss S2S
                    loss_rede = loss_fn(re_denoise_outputs, denoise_outputs) #Loss re-de-noising
                    loss_stab = loss_fn(re_denoise_outputs, x) #Loss stability
                    loss = loss_s2s + gamma1*(gamma2*loss_rede + loss_stab)    #  equation(5)                            
                                
                valid_loss += loss
                valid_loss_s2s += loss_s2s
                valid_loss_rede += loss_rede
                valid_loss_stab += loss_stab                    
                denoise_outputs_valid.append(denoise_outputs.cpu().numpy())
                addnoise_outputs_valid.append(addnoise_outputs.cpu().numpy())
                re_denoise_outputs_valid.append(re_denoise_outputs.cpu().numpy())
                                    
            denoise_outputs_valid = np.array(denoise_outputs_valid)
            addnoise_outputs_valid = np.array(addnoise_outputs_valid)
            re_denoise_outputs_valid = np.array(re_denoise_outputs_valid)
            train_loss /= len(traces_train)            
            valid_loss /= len(x_valid)
            valid_loss_s2s /= len(x_valid)
            valid_loss_rede /= len(x_valid)
            valid_loss_stab /= len(x_valid)
            print(f"train loss: {train_loss:>5f}")
            print(f"valid loss: {valid_loss:>5f} S2S:{valid_loss_s2s:>5f}  rede:{valid_loss_rede:>5f} stab:{valid_loss_stab:>5f}")
                     
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"../model/{name}.pth")  
                best_epoch = epoch
                early_stop_count = 0
            if early_stop_count >= early_stop_tolerance:
                print('early stop')
                break 
            early_stop_count += 1

            if epoch == maxepoch:
                break

    print(f'Finish Training best_epoch:{best_epoch}')
   
model.load_state_dict(torch.load(f'../model/{name}.pth', map_location=device))
model.eval()

flt_stack=np.zeros_like(traces[:,0,7:-8,6:-7])
   
for i, x in enumerate(traces):
    inp = torch.Tensor(x).to(device)
    inp = torch.unsqueeze(inp, 0)   
    with torch.no_grad():
        output = model(inp).cpu().numpy()
        flt_stack[i, :] = output[0,0,7:-8,6:-7]

# Calculating metrics
psnr=[]
ssim_score=[]
for i, (ori, flt) in tqdm(enumerate(zip(original_traces, flt_stack)), desc='calculate metric'):
    psnr.append(calculate_psnr(ori, flt))
    ssim_score.append(ssim(ori, flt,data_range=flt.max()-flt.min()))
average_psnr=np.mean(psnr)
average_ssim=np.mean(ssim_score)

mse=np.mean((original_traces-flt_stack)**2)

print(f'Noise Level:{noise_level}')
print(f'MSE: {mse}')
print(f'PSNR: {average_psnr}')
print(f'SSIM: {average_ssim}')

traces=traces[:,0,7:-8,6:-7]
plt_idx=400
fig, ax = plt.subplots(1,3, figsize=(12,8))
ax[0].imshow(original_traces[plt_idx].T, vmin=-3, vmax=3, cmap='seismic',  aspect=0.5, interpolation='None')
ax[1].imshow(traces[plt_idx].T, vmin=-3, vmax=3, cmap='seismic',  aspect=0.5, interpolation='None')
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