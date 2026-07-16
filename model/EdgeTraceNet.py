import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # output channels : 64
        self.layer2 = resnet.layer2  # output channels: 128
        self.layer3 = resnet.layer3  # output channels: 256
        self.layer4 = resnet.layer4  # output channels: 512

    def forward(self, x):

        x1 = self.layer0(x)
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        return [x1, x2, x3, x4, x5]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        ])

    def forward(self, features, x_shape):
        feats = list(features) 
        x = feats.pop()

        outputs = []
        for i in range(4):
            skip = feats.pop()
            # make sure the feature map size matches the skip connection size
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
            x = torch.concat([x, skip], dim=1)
            x = self.conv_layers[i](x)
            outputs.append(x)

        target_h, target_w = x_shape[2], x_shape[3]
        
        # make sure the output size matches the input size
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
        x = self.conv_layers[4](x)
        outputs.append(x)

        return outputs[1:]

class EdgeHead(nn.Module):
    """
        EdgeHead takes the feature map from the decoder and outputs a 2-channel edge map.
    """
    def __init__(self, in_channel=256):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=in_channel//2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        edge = self.layer(x)

        return edge


class EdgeTraceNet(nn.Module):
    """
        input :
            x : [B, 3, H, W]
        output : 
            output : [B, 2, H, W]

        EdgeTraceNet takes an image as input and outputs a 2-channel edge map
        Each pixel in the edge map represents the probability that the pixel is connected to it's right neighbor and it's bottom neighbor.
    """

    def __init__(self):
        super(EdgeTraceNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.edgeHeads = nn.ModuleList([
            EdgeHead(in_channel=128),
            EdgeHead(in_channel=64),
            EdgeHead(in_channel=32),
            EdgeHead(in_channel=16)
        ])

    def forward(self, x):
        # x : [B, 3, H, W]
        
        features = self.encoder(x)
        outputs = self.decoder(features, x.shape)
        
        edges = self.edgeHeads[-1](outputs[-1])

        return {
            'features' : outputs[-1],
            'edges' : edges
        }