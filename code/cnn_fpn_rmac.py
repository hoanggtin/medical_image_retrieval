# cnn_fpn_rmac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class GeneralizedMeanPoolingP(nn.Module): 
    def __init__(self, norm=3):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * norm)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), (1, 1)).pow(1. / self.p)

class CNN_FPN_RMAC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.EMBED_DIM
        self.rmac_pool = GeneralizedMeanPoolingP(norm=config.GEMP)

        # Backbone CNN (ResNet18)
        backbone = resnet18(pretrained=False)
        self.stage1 = nn.Sequential(*list(backbone.children())[:5])  # conv1 + bn + relu + maxpool + layer1
        self.stage2 = backbone.layer2  # lower resolution
        self.stage3 = backbone.layer3  # lower resolution

        # FPN: đưa về cùng kích thước và chiều sâu
        self.conv1x5 = nn.Conv2d(128, self.embed_dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)

        self.fc = nn.Linear(self.embed_dim * 2, config.NUM_CLASSES)

    def extract_features(self, x):
        B = x.size(0)
        x1 = self.stage1(x)  # B x 64 x H/4 x W/4
        x2 = self.stage2(x1) # B x 128 x H/8 x W/8
        x3 = self.stage3(x2) # B x 256 x H/16 x W/16

        # FPN
        x5 = x3  # deeper
        x4 = F.interpolate(x2, size=x5.shape[2:], mode='nearest')  # align spatial size
        x4 = self.conv1x5(x4)
        x4 = self.conv3x3(x4)

        # RMAC
        pooled_x5 = self.rmac_pool(x5)
        pooled_x4 = self.rmac_pool(x4)

        features = torch.cat((pooled_x4, pooled_x5), 1).squeeze(-1).squeeze(-1)
        return features
        
    def forward(self, x):
        B = x.size(0)
        x1 = self.stage1(x)  # B x 64 x H/4 x W/4
        x2 = self.stage2(x1) # B x 128 x H/8 x W/8
        x3 = self.stage3(x2) # B x 256 x H/16 x W/16

        # FPN
        x5 = x3  # deeper
        x4 = F.interpolate(x2, size=x5.shape[2:], mode='nearest')  # align spatial size
        x4 = self.conv1x5(x4)
        x4 = self.conv3x3(x4)

        # RMAC
        pooled_x5 = self.rmac_pool(x5)
        pooled_x4 = self.rmac_pool(x4)

        features = torch.cat((pooled_x4, pooled_x5), 1).squeeze(-1).squeeze(-1)
        return self.fc(features)
