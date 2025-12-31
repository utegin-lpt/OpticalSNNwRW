import numpy as np
import torch
import torch.nn as nn
import torch.fft
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def propagation_ASM(u_in, feature_size, wavelength, z,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):
    """Propagates the input field using the angular spectrum method"""
    # resolution of input field
    field_resolution = u_in.size()
    # number of pixels
    num_y, num_x = field_resolution[2], field_resolution[3]
    # sampling interval size
    dy, dx = feature_size
    # size of the field
    y, x = (dy * float(num_y), dx * float(num_x))
    
    if precomped_H is None:
        # frequency coordinates sampling (torch linspace)
        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * y),
                             1 / (2 * dy) - 0.5 / (2 * y), num_y, device=u_in.device, dtype=dtype)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x),
                             1 / (2 * dx) - 0.5 / (2 * x), num_x, device=u_in.device, dtype=dtype)
        # momentum/reciprocal space
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')  # match numpy behavior
    
    if precomped_H is None and precomped_H_exp is None:
        # transfer function
        HH = 2 * math.pi * torch.sqrt(1 / wavelength**2 - (FX**2 + FY**2))
        # create tensor & upload to device
        H_exp = HH.to(dtype=dtype)
        # reshape tensor
        H_exp = H_exp.unsqueeze(0).unsqueeze(0)
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp
        
    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)
        # band-limited ASM
        fy_max = 1 / torch.sqrt(torch.tensor((2 * z * (1 / y))**2 + 1)) / wavelength
        fx_max = 1 / torch.sqrt(torch.tensor((2 * z * (1 / x))**2 + 1)) / wavelength
        H_filter = ((FX.abs() < fx_max) & (FY.abs() < fy_max)).to(dtype=dtype)
        # get real/imag components
        H_real, H_imag = polar_to_rect(H_filter, H_exp)
        H = torch.stack((H_real, H_imag), dim=-1)
        H = ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H
        
    if return_H_exp:
        return H_exp
    if return_H:
        return H
        
    U1 = torch.fft.fft2(ifftshift(u_in), dim=(-2, -1), norm='ortho')
    U2 = H * U1
    u_out = fftshift(torch.fft.ifft2(U2, dim=(-2, -1), norm='ortho'))
    return u_out
