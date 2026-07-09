import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class color_difference_loss(nn.Module): # 윤곽선을 더 만드려고 함
    """
        연결된 픽셀끼리 색상이 얼마나 닮았는가를 확인
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
    
class feature_difference_loss(nn.Module): # 윤곽선을 더 만드려고 함
    """
        연결된 픽셀끼리 특징이 얼마나 닮았는가를 확인
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
    
class regularization_loss(nn.Module): # 싹다 윤곽선으로 만드는걸 방지함
    """
        모델이 모든 연결을 끊어버리는 꼼수를 방지하기 위한 손실함수
        모델이 연결을 끊을때마다 손실을 준다.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        # output : [B, 2, H, W]

        return torch.abs(1.0 - output).mean()
    
class binarization_loss(nn.Module): # 윤곽선이 0과 1에 가까워지게 함
    """
        모델이 0.5(회색)를 출력하는 것을 막고 0(선) 또는 1(배경)로 확실히 나누도록 유도
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        return (output * (1.0 - output)).mean()

class thick_edge_loss(nn.Module): # 윤곽선이 두꺼워지지 않게 함
    """
        윤곽선이 너무 두꺼워지는걸 방지하기 위한 손실함수
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        edge_prob = 1 - output
        
        thick_H = edge_prob[:, :, :-1, :] * edge_prob[:, :, 1:, :]
        thick_W = edge_prob[:, :, :, :-1] * edge_prob[:, :, :, 1:]
        
        return thick_H.mean() + thick_W.mean()
    
class smooth_loss(nn.Module):
    """
        결과물(윤곽선)이 자잘하게 파편화되는 것을 막고, 
        노이즈나 불필요한 텍스처(키보드 등)를 지우도록 유도하는 TV Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, output):
        # output: [B, 2, H, W]
        # 인접한 출력 픽셀 간의 차이를 계산
        diff_h = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
        diff_w = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
        
        return diff_h.mean() + diff_w.mean()