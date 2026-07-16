import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import scipy.io as sio
from typing import List, Union
import numpy as np
import os

def save_image(epoch, output, image, save_path, th=0.5):
    edge_right = 1.0 - output[:, 0:1, :, :]
    edge_down = 1.0 - output[:, 1:2, :, :]
    
    edge_map = torch.max(edge_right, edge_down) # [B, 1, H, W]
    edge_map = (edge_map >= th).float()
    
    edge_map_rgb = edge_map.repeat(1, 3, 1, 1) # [B, 3, H, W]
    combined = torch.cat([image, edge_map_rgb], dim=3)
    
    vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'), nrow=2, padding=4)
    print(f"saved V4 3x3 grid image: vis_epoch_{epoch}.png")

def load_gt_boundaries(mat_path: Union[str, os.PathLike]) -> List[np.array]:
    """ 
        return a list of ground truth boundaries drawn by all annotators from a .mat file
    """
    mat = sio.loadmat(mat_path)
    gt_array = mat['groundTruth'][0]
    boundaries = [gt[0][0][1].astype(np.float32) for gt in gt_array]
    return boundaries