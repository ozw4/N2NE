#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 08:32:49 2023
This code is based on the https://library.seg.org/doi/10.1190/geo2022-0371.1
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
from util import load_seismic_data, reshape_traces,  calculate_psnr,  torch_fix_seed, UNet, save_sgy
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

def randmask(x, ratio=0.2):
    mask_x = x.copy() 
    h=x.shape[0]
    ind=np.random.randint(0,h, size=int(ratio*h))
    mask_x[ind]=np.random.uniform(-0.8,0.8, size=(int(ratio*h),x.shape[-1]))
    return mask_x, ind   

class MaskDataset(Dataset):
    def __init__(self, t, flip=False, ratio=0.6):
        self.ts = t.copy()
        self.flip = flip
        self.ratio = ratio
    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        seed=np.random.randint(0,2**16)
        ts=self.ts[idx]
        if self.flip:
            if np.random.rand()>0.5:
                ts=np.flip(ts,axis=1).copy()
        xs, mask_idx=randmask(ts[0], ratio=self.ratio)
        xs=xs.reshape(ts.shape)

        return xs, ts, mask_idx


def extract_mask_position(data, mask_idx):
    extract_data=[]
    for i, idx in enumerate(mask_idx):
        extract_data.append(data[i,0, idx])
    extract_data=torch.stack(extract_data)
    return extract_data

def train(dataloader, model,  optimizer):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    train_loss=0
    scaler=GradScaler()

    for batch, (X, y, mask_idx) in enumerate(dataloader):        
       
        X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)    
        optimizer.zero_grad()
        # Compute prediction error
        with autocast():
            output = model(X)    
            y_maskpos=extract_mask_position(y, mask_idx)
            output_maskpos=extract_mask_position(output, mask_idx)
            loss=loss_fn(y_maskpos,output_maskpos)             
   
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss+=loss.detach()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f" [{current:>5d}/{size:>5d}]")
            print(train_loss.detach().cpu().numpy()/(batch+1))
            
    train_loss/=num_batches
    print(f"Train Error: fp:{train_loss:>5f}\n")

def valid(dataloader, model):
    num_batches = len(dataloader)
    model.eval()
    valid_loss= 0
    inputs=[]
    outputs=[]
    targets=[]
    for X, y, mask_idx in dataloader:            
        inputs.append(X)
        targets.append(y)
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
           output = model(X)
           y_maskpos=extract_mask_position(y, mask_idx)
           output_maskpos=extract_mask_position(output, mask_idx)
           valid_loss += loss_fn(y_maskpos,output_maskpos).item()
        outputs.append(output.cpu().numpy())         
        
    inputs=np.concatenate(inputs)
    outputs=np.concatenate(outputs)
    targets=np.concatenate(targets)
    valid_loss /= num_batches

    print(f"valid Error: \n  Avg loss: {valid_loss:>5f} \n")
    return valid_loss, outputs, inputs, targets

codename=os.path.basename(__file__)

# Constants and hyperparameters setup
TRAIN = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 8
epoch = 200
noise_level = 0.08
early_stop_tolerance = 5
learning_rate = 1e-4
noise_trace_ratio = 0.6
loss_fn = nn.L1Loss()
name=codename[:-3]+f'_{noise_level}_ratio{noise_trace_ratio}'

# Loading Marmousi data from the SEGY file
segy_file_path = '../data/marmousi_csg.sgy'
traces, attributes = load_seismic_data(segy_file_path)
traces = reshape_traces(traces, attributes['ffid'], attributes['trcnum_within_filedrec'])
original_traces = traces.copy()

sweep_noise = np.load('../data/sweep_noise.npy')

torch_fix_seed(0)

traces+=sweep_noise[0]*noise_level
traces=np.transpose(traces,(1,0,2)) # csg=>crg
traces=np.pad(traces, ((0,0),  (7,8), (6,7)))

traces=np.expand_dims(traces,1)   

traces_train, traces_valid = train_test_split(traces, test_size=0.1, random_state=1)

train_set=MaskDataset(traces_train, flip=True, ratio=noise_trace_ratio)
valid_set=MaskDataset(traces_valid, flip=False, ratio=noise_trace_ratio)
train_iter = DataLoader(train_set,
                        batch_size=batchsize, 
                        shuffle=True,
                        num_workers=os.cpu_count())
valid_iter = DataLoader(valid_set,
                        batch_size=batchsize, 
                        shuffle=False,
                        num_workers=os.cpu_count())
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss=0
best_valid_loss=1e10
early_stop_count = 0

report_itera=len(traces_train)//batchsize

train_loss = 0.0

if TRAIN:
    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_iter, model, optimizer)
        valid_loss, outputs, inputs, targets=valid(valid_iter, model)

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

flt_stack=np.zeros_like(original_traces)

for i, x in enumerate(traces):
    inp = torch.Tensor(x).to(device)
    inp = torch.unsqueeze(inp, 0)   
    with torch.no_grad():
        output = model(inp)
    output=output.cpu().numpy()[0,0,7:-8, 6:-7]
    flt_stack[i] = output

#crg => csg
traces=np.transpose(traces[:,0,7:-8, 6:-7], (1,0,2))
flt_stack=np.transpose(flt_stack, (1,0,2))

# Calculating metrics
psnr=[]
ssim_score=[]
for i, (ori, flt) in tqdm(enumerate(zip(original_traces, flt_stack)), desc='calculate metric'):
    psnr.append(calculate_psnr(ori, flt))
    ssim_score.append(ssim(ori, flt,data_range=flt.max()-flt.min()))
average_psnr=np.mean(psnr)
average_ssim=np.mean(ssim_score)

mse=np.mean((original_traces-flt_stack)**2)

print(f'Noise Level:{noise_level} ')
print(f'MSE: {mse}')
print(f'PSNR: {average_psnr}')
print(f'SSIM: {average_ssim}')

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

np.save(f'../data/{name}.npy', flt_stack)           

