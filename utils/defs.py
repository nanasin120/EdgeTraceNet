import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os

def save_image(epoch, output, image, save_path):
    edge_map = output.mean(dim=1, keepdim=True) # [B, 1, H, W]

    edge_map = (edge_map >= 0.5).float()
    
    edge_map_rgb = edge_map.repeat(1, 3, 1, 1) # [B, 3, H, W]
    
    combined = torch.cat([image, edge_map_rgb], dim=3)
    
    vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'), nrow=2, padding=4)
    print(f"saved V4 3x3 grid image: vis_epoch_{epoch}.png")