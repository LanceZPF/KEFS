import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch

softmax = nn.Softmax(dim=1)

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(w,h)
    
    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

def batch_get_centers(pred_softmax):
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)

def get_variance(part_map, x_c, y_c):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(w,h)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y

def concentration_loss(pred, softmax = False):
    
    if softmax:
        pred_softmax = softmax(pred)[:,1:,:,:]
    else:
        pred_softmax = pred[:,1:,:,:] # I think there no need to do softmax!
        
    B,C,H,W = pred_softmax.shape

    loss = 0
    epsilon = 1e-3
    centers_all = batch_get_centers(pred_softmax)
    for b in range(B):
        centers = centers_all[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:] + epsilon # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map/k
            x_c, y_c = centers[c]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
            
    loss = loss/B
    return loss/B

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss