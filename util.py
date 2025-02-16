#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:28:31 2023

@author: Mitsuyuki.Ozawa (mitsuyuki.ozawa@jgi.co.jp)
"""

import random
import h5py
import numpy as np
import segyio
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import  GradScaler, autocast
from numba import jit
from scipy import signal

def load_seismic_data(segy_file_path, ffid_byte = 21, trcnum_within_filedrec_byte = 25):
    """
    Load seismic traces and attributes from a SEGY file.

    Args:
        segy_file_path (str): Path to the SEGY file to be loaded.
        ffid_byte (int, optional): Byte location for the Field File ID. Defaults to 21.
        trcnum_within_filedrec_byte (int, optional): Byte location for the trace number within the field record. Defaults to 25.

    Returns:
        tuple: A tuple containing two elements:
            - numpy.ndarray: An array of normalized seismic traces as float32.
            - dict: Attributes of the seismic data including 'ffid' (Field File ID), 
                    'trcnum_within_filedrec' (trace number within the field record), and 'dt' (sample interval).

    Note:
        The function normalizes seismic traces and extracts specific attributes from the SEGY file, with options to specify byte locations for 'ffid' and 'trcnum_within_filedrec'. The sample interval 'dt' is calculated as the difference between the first two samples.
    """
    with segyio.open(segy_file_path, ignore_geometry=True, endian='little') as f:
        traces = np.array([normalization(x) for x in f.trace]).astype(np.float32)
        attributes = {
            'ffid': f.attributes(ffid_byte)[:],
            'trcnum_within_filedrec': f.attributes(trcnum_within_filedrec_byte)[:],
            'dt': f.samples[1] - f.samples[0]
        }
    return traces, attributes

def save_sgy(name, data, spec, header):
    """
    Saves seismic data to a SEG-Y file.

    Args:
        name (str): The name of the output SEG-Y file.
        data (numpy.ndarray): The seismic data array to be saved.
        spec (segyio.spec): The specification for the SEG-Y file, defining its layout.
        header (dict): Headers to be associated with each trace in the SEG-Y file.

    This function creates a SEG-Y file with the specified name and specifications, writes the provided seismic data into it,
    and assigns the provided header information to each trace. 
    """
    zn=data.shape[-1]

    with segyio.create(name, spec) as f:
         n=0
         #n1=0
         f.bin[f.bin.keys()[3]]=zn
         f.bin[f.bin.keys()[4]]=zn        
         
         for i in tqdm(range(len(data))): 
            f.trace[i]=data[i]
            f.header[i]=header[i]

def reshape_traces(traces, ffid, trcnum_within_filedrec):
    """Reshape traces into a structured array based on unique FFIDs and trace numbers."""
    nkey1 = len(np.unique(ffid))
    nkey2 = len(np.unique(trcnum_within_filedrec))
    nt = traces.shape[-1]
    return traces.reshape((nkey1, nkey2, nt))

def torch_fix_seed(seed=42):
    """
    Fix the seed for Python random, Numpy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): Seed value for random number generation. Default is 42.

    Returns:
        None
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

@jit
def normalization(trc):
    """
    Normalize a seismic trace by subtracting its mean and dividing by its standard deviation.

    Args:
        trc (numpy.ndarray): Input seismic trace.

    Returns:
        numpy.ndarray: Normalized seismic trace.
    """
    trc = (trc - trc.mean()) / (trc.std() + 1e-8)
    return trc

def sweep_dataload(data, sweep_n, batchsize):
    """
    Generate batch data by randomly selecting seismic sweep.

    Args:
        data (numpy.ndarray): Input data.
        sweep_n (int): Number of sweeps.
        batchsize (int): Batch size.
        traces (numpy.ndarray): Array of seismic traces.

    Returns:
        tuple: Tuple containing x_data (input data) and t_data (target data).
    """
    batch_ind = random.sample(range(data.shape[0]), batchsize)
    sweep_pair = np.array([random.sample(range(sweep_n), 2) for _ in range(batchsize)])
    x_data = data[batch_ind[:], sweep_pair[:, 0]]
    t_data = data[batch_ind[:], sweep_pair[:, 1]]
    return x_data, t_data
    
def calculate_psnr(original_image, compared_image):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        original_image (numpy.ndarray): Original image.
        compared_image (numpy.ndarray): Image to be compared with the original.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((original_image - compared_image) ** 2)
    max_possible_value = np.max(original_image)
    psnr = 10 * np.log10((max_possible_value ** 2) / mse)
    return psnr

def Instantaneous_scaling(data, gatel, desired_rms=1):
    """
    Perform instantaneous scaling on the given data.

    Parameters:
        data (numpy.ndarray): A NumPy array containing the data to apply scaling to.
        gatel (int): An integer representing the length of the gate.
        desired_rms (float, optional): The desired Root Mean Square (RMS) value. Default is 1.

    Returns:
        tuple: A tuple containing the scaled data and the gain function.

    Notes:
        - This function performs instantaneous scaling on the given data based on the specified gate length gatel.
        - It calculates the mean absolute value of instances within the gate and adjusts the gain.
        - If the standard deviation (std) within the gate is smaller than 1e-20, it sets the gain to 1 to prevent division by zero.
        - The gain function is applied to each part of the data, resulting in scaled data.

    Examples:
        # Example usage of the function
        scaled_data, gain_func = Instantaneous_scaling(data, gatel=10)
        # Scaled data and gain function are returned as a tuple.

    """
    halfl = int(gatel / 2)
    shape = data.shape

    gate_func = np.zeros(shape, dtype=np.float32)

    for i in range(halfl, int(shape[-1] - gatel / 2)):
        # Mean absolute value within a specified time gate
        std = np.std(np.abs(data[:, int(i - gatel / 2):int(i + gatel / 2)]), axis=1)

        # Adjust gain based on std, prevent division by zero
        gate_func[std > 1e-20, i] = desired_rms / std[std > 1e-20]
        gate_func[std < 1e-20, i] = 1

    gate_func[:, :halfl] = np.expand_dims(gate_func[:, halfl], 1)
    gate_func[:, -halfl:] = np.expand_dims(gate_func[:, -halfl-1], 1)

    scaled_data = data.copy() * gate_func
    scaled_data = scaled_data
    
    return scaled_data, gate_func

def substack_dataload(data, substack_n, batchsize):
    """
    Loads a substack of data for processing.

    Args:
        data (numpy.ndarray): The full dataset from which substacks will be sampled.
        substack_n (int): The number of sweep to include in each substack.
        batchsize (int): The number of samples to include in each batch.

    Returns:
        tuple: A tuple containing two numpy arrays, `x_data` and `t_data`. Each array represents a batch of substacked data. 
               `x_data` is derived from randomly selected substack indices, and `t_data` is derived from the remaining data after 
               `x_data` extraction, with both having their means calculated along the axis=1.
    """
    batch_ind = random.sample(range(data.shape[0]), batchsize)
    batch=data[batch_ind]

    substack_ind1= np.array([np.random.choice(range(len(sub_array)), substack_n, replace=False) for sub_array in batch])
    x_data= np.array([sub_array[ind] for sub_array, ind in zip(batch, substack_ind1)])
    x_data= np.mean(x_data, axis=1)  
    left_ind=[np.delete(np.arange(len(batch[0])), ind) for ind in substack_ind1]
                              
    substack_ind2= np.array([np.random.choice(left_ind[i], substack_n, replace=False) for i, sub_array in enumerate(batch)])
    t_data= np.array([sub_array[ind] for sub_array, ind in zip(batch, substack_ind2)])
    t_data= np.mean(t_data, axis=1)      
    
    return x_data, t_data



def train(dataloader, model, optimizer, loss_fn, device):
    """
    Trains a model over a single epoch.

    Args:
        dataloader (DataLoader): The DataLoader providing the dataset for training.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_fn (callable): The loss function to use for training.
        device (torch.device): The device on which to train the model.

    Trains the model using the provided dataloader, model, optimizer, and loss function, 
    automatically adjusting gradients with mixed precision.
    """
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    train_loss = 0
    scaler=GradScaler()

    for batch, (X,y) in enumerate(dataloader):        
       
        X,y= X.to(device).to(torch.float32), y.to(device).to(torch.float32)
        optimizer.zero_grad()
        # Compute prediction error
        with autocast():
            output = model(X)   
            loss = loss_fn(output[:,0], y[:,0]).mean() 
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss+=loss.detach()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f" [{current:>5d}/{size:>5d}]")
            print(train_loss.detach().cpu().numpy()/(batch+1))
            
    train_loss/=num_batches
    
    print(f"Train Error: {train_loss:>5f}\n")
            
def valid(data_loader, model, loss_fn,  device):
    """
    Validates the model using a validation dataset.

    Args:
        data_loader (DataLoader): The DataLoader providing the dataset for validation.
        model (torch.nn.Module): The model to be validated.
        loss_fn (callable): The loss function used for validation.
        device (torch.device): The device on which the model and data are located.

    Returns:
        float: The average validation loss over all batches in the validation dataset.
    """
    
    num_batches = len(data_loader)
    model.eval()
    valid_loss= 0
    
    inputs=[]
    outputs=[]
    for batch, (X,y) in enumerate(data_loader):   
        inputs.append(X)
 
        X, y = X.to(device),y.to(device)
        with torch.no_grad():
           output = model(X)
        loss = loss_fn(output[:,0], y[:,0]).mean() 

        valid_loss += loss.item()

        outputs.append(output.cpu().numpy())         

    valid_loss /= num_batches

    print(f"valid Error: {valid_loss:>5f} \n")
    return valid_loss

class S2RS_Dataset(Dataset):
    """
    A dataset class with optional flipping augmentation.

    Args:
        x (numpy.ndarray): Input data).
        y (numpy.ndarray): Target data).
        flip (bool, optional): Whether to randomly flip the data along the second axis as augmentation. Defaults to False.

    This dataset class supports indexing and length querying. If flipping is enabled, each sample (both `x` and `y`) has a 50% chance
    of being flipped along its second axis when retrieved.
    """
    def __init__(self, x, y, flip=False):
        self.xs = x.copy()
        self.ys = y.copy()
        self.flip = flip

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):

        seed=np.random.randint(0,2**16)
        xs=self.xs[idx]
        ys=self.ys[idx]
        if self.flip:
            if np.random.rand()>0.5:
                xs=np.flip(xs,axis=1).copy()
                ys=np.flip(ys,axis=1).copy()
        return xs, ys
    
class SingleConv(nn.Module):
    """
    A single convolutional block module for neural networks.
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        drop (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, in_channels, out_channels, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop),
        )

    def forward(self, x):
        return self.conv(x)

class SingleDownConv(nn.Module):
    """
    A convolutional block with downsampling.
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        drop (float, optional): Dropout probability. Defaults to 0 for no dropout.
    """
    def __init__(self, in_channels, out_channels, drop=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), 2, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """
    Implements a double convolutional block module.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels after the first and second convolutions.
        drop (float, optional): Dropout probability after the first convolutional layer. Defaults to 0 (no dropout).

    The forward pass inputs a tensor and outputs a tensor processed by the two convolutional layers and activations.
    """
    def __init__(self, in_channels, out_channels, drop=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Implements a U-Net architecture for image segmentation.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 1.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        features (list of int, optional): Number of features in each layer of the U-Net. Defaults to [32, 64, 128, 256].
        drop (float, optional): Dropout probability for added regularization. Defaults to 0 (no dropout).

    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], drop=0):
        super().__init__()

        self.encoders = nn.ModuleList()
        for i,feature in enumerate(features):
            self.encoders.append(
                nn.Sequential(
                    SingleConv(in_channels, feature,drop=drop),
                    SingleDownConv(feature, feature,drop=drop)
                )
            )
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1], drop=drop)
        self.decoders = nn.ModuleList()
        for i,feature in enumerate(reversed(features)):
            self.decoders.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.decoders.append(DoubleConv(feature*2, feature//2, drop=drop))
        self.pointwise_conv = nn.Conv2d(features[0]//2, out_channels, kernel_size=1)

    def forward(self, x):
        inp=x.clone()
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoders), 2):
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.pad(x, [0, skip_connection.shape[3] - x.shape[3], 
                                          0, skip_connection.shape[2] - x.shape[2]])
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[i](concat_x)
            x = self.decoders[i+1](x)
        x = self.pointwise_conv(x)
        return x

