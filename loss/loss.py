import torch
import torch.nn as nn
import torch.nn.functional as F

class color_difference_loss(nn.Module):
    """
        연결된 픽셀끼리 색상이 얼마나 닮았는가를 확인
    """
    def __init__(self):
        super(color_difference_loss, self).__init__()

    def forward(self, output, image):
        # output : [B, 2, H, W]
        # image : [B, 3, H, W]
        B, _, H, W = image.shape

        diff_W = torch.abs(image[..., :, :W-1] - image[..., :, 1:]) ** 2 # [B, 3, H, W-1]
        diff_H = torch.abs(image[..., :H-1, :] - image[..., 1:, :]) ** 2 # [B, 3, H-1, W]

        loss_H = diff_H * output[:, 1:2, :H-1, :]
        loss_W = diff_W * output[:, 0:1, :, :W-1]

        loss = loss_H.mean() + loss_W.mean()
        
        return loss
    
class regularization_loss(nn.Module):
    """
        모델이 모든 연결을 끊어버리는 꼼수를 방지하기 위한 손실함수
        모델이 연결을 끊을때마다 손실을 준다.
    """
    def __init__(self):
        super(regularization_loss, self).__init__()

    def forward(self, output):
        # output : [B, 2, H, W]

        loss = (1- output).mean()

        return loss