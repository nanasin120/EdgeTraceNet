import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class color_difference_loss(nn.Module): # it makes the edge more
    """
        it checks how similar the colors are between connected pixels
    """
    def __init__(self):
        super(color_difference_loss, self).__init__()

    def forward(self, output, image):
        # output : [B, 2, H, W]
        # image : [B, 3, H, W]

        diff_H = torch.abs(image[..., :-1, :] - image[..., 1:, :]) # [B, 3, H-1, W]
        diff_W = torch.abs(image[..., :, :-1] - image[..., :, 1:]) # [B, 3, H, W-1]

        loss_H = diff_H * output[:, 1:2, :-1, :]
        loss_W = diff_W * output[:, 0:1, :, :-1]

        loss = loss_H.mean() + loss_W.mean()
        
        return loss
    
class feature_difference_loss(nn.Module): # it makes the edge more
    """
        it checkes how similart the features are between connected pixels
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, feature):
        # output : [B, 2, H, W]
        # feature : [B, 3, H, W]

        cos_sim_H = F.cosine_similarity(feature[..., :-1, :], feature[..., 1:, :], dim=1)
        cos_sim_W = F.cosine_similarity(feature[..., :, :-1], feature[..., :, 1:], dim=1)

        diff_H = 1.0 - cos_sim_H
        diff_W = 1.0 - cos_sim_W

        loss_H = diff_H * output[:, 1, :-1, :]
        loss_W = diff_W * output[:, 0, :, :-1]

        loss = loss_H.mean() + loss_W.mean()
        
        return loss
    
class regularization_loss(nn.Module): # it prevents the model makes everthing edge
    """
        it prevents the model from making everything an edge
        it penalizes the model when it disconnects all connections
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        # output : [B, 2, H, W]

        return torch.abs(1.0 - output).mean()
    
class binarization_loss(nn.Module): # it makes the edge more binarized
    """
        it induces the model to output more binarized results, preventing the model outputting 0.5 (gray) and cleary seprating the output into 0 (edge) and 1 (background)
        
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        return (output * (1.0 - output)).mean()

class thick_edge_loss(nn.Module): # it prevents the model making the edge too thick
    """
        it prevents the model making the edge too thick
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        edge_prob = 1 - output
        
        thick_H = edge_prob[:, :, :-1, :] * edge_prob[:, :, 1:, :]
        thick_W = edge_prob[:, :, :, :-1] * edge_prob[:, :, :, 1:]
        
        return thick_H.mean() + thick_W.mean()
    
class smooth_loss(nn.Module): # it isn't used
    """
        it prevents the edge map from being overly fragmented and removes noise or unnecessary textures
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        # output: [B, 2, H, W]
        # it caculates the difference between connnected pixels that predicted by model
        diff_h = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
        diff_w = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
        
        return diff_h.mean() + diff_w.mean()