import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    """
        데이터가 들어오면 5개로 정보를 뽑아 출력한다.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4

    def forward(self, x):
        # x : [B, 3, H, W]
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        return [x1, x2, x3, x4, x5]
    
class Decoder(nn.Module):
    """
        정보 5개가 들어오면 픽셀이 오른쪽과 아래 픽셀과 연결되어있을 확률을 출력한다. [B, 2, H, W]으로
    """
    def __init__(self):
        super(Decoder, self).__init__()

        self.up_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ])

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048 + 1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256 + 64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # features : [x1, x2, x3, x4, x5]
        features[0] = self.block(features[0])

        x = features.pop()

        # Upsampling -> skip connection -> Conv
        for i in range(4):
            x = self.up_layers[i](x)
            skip = features.pop()
            x = torch.concat([x, skip], dim=1)
            x = self.conv_layers[i](x)
        
        output = self.final(x)

        return output

class EdgeTraceNet(nn.Module):
    """
        입력 :
            x : [B, 3, H, W]
        출력 : 
            output : [B, 2, H, W]

        이미지가 들어오면 하나의 픽셀이 오른쪽 픽셀, 아래 픽셀과 연결되어있을 확률을 출력 
    """

    def __init__(self):
        super(EdgeTraceNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # x : [B, 3, H, W]
        
        features = self.encoder(x)
        output = self.decoder(features)

        return output