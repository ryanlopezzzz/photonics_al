import numpy as np
import torch
import random

def translate_tensor(tensor,input_size=32, prob=None):
    '''Data augmentation function to enforce periodic boundary conditions. Inputs are arbitrarily translated in each dimension'''
    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            if ndim == 2:
                tensor1 = torch.roll(tensor1,(np.random.choice(input_size),np.random.choice(input_size)),(0,1)) # translate by random no. of units (0-input_size) in each axis
            elif ndim == 3:
                tensor1 = torch.roll(tensor1,(np.random.choice(input_size),np.random.choice(input_size),np.random.choice(input_size)),(0,1,2))
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0) # add back channel dim and batch dim
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def rotate_tensor(tensor, prob=None):
    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            rottimes = np.random.choice(4) # 4-fold rotation; rotate by 0, 90, 280 or 270
            rotaxis = np.random.choice(ndim) # axis to rotate [0,1], [1,0] in 2D (double count negative rot is ok) and [0,1], [1,2], [2,0] in 3D (negative rotation covered by k = 3)
            tensor1 = torch.rot90(tensor1,k=rottimes,dims=[rotaxis,(rotaxis+1)%(ndim)])
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def flip_tensor(tensor, prob=None):
    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            if ndim == 2:
                flipaxis = random.choice([[0],[1],[]]) # flip hor, ver, or None (dont include Diagonals = flip + rot90)
            elif ndim == 3:
                flipaxis = random.choice([[0],[1],[2],[]]) # flip x, y, z or None (dont include Diagonals = flip + rotate)
            tensor1 = torch.flip(tensor1,flipaxis)
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def get_aug(x,translate=False,flip=False,rotate=False,p=1):
    if translate:
        x = translate_tensor(x, prob = p)
    if flip:
        x = flip_tensor(x, prob = p)
    if rotate:
        x = rotate_tensor(x, prob = p)
    return x