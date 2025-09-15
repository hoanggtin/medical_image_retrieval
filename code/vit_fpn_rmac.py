
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedMeanPoolingP(nn.Module): 
    def __init__(self, norm=3):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * norm)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), (1, 1)).pow(1. / self.p)

class ViT_FPN_RMAC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.EMBED_DIM
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=config.PATCH_SIZE, stride=config.PATCH_SIZE)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim, 
                nhead=config.NUM_HEADS, 
                dim_feedforward=config.MLP_DIM)
            for _ in range(config.DEPTH)
        ])

        self.conv1x5 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)

        self.rmac_pool = GeneralizedMeanPoolingP(norm=config.GEMP)
        self.fc = nn.Linear(self.embed_dim * 2, config.NUM_CLASSES)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, D, H/patch, W/patch)
        
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        for layer in self.encoder_layers:
            x = layer(x)

        h_w = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h_w, h_w)

        # FPN
        x5 = x
        x4 = F.interpolate(x5, size=(h_w, h_w), mode='nearest')
        x4 = self.conv1x5(x4)
        x4 = self.conv3x3(x4)

        # RMAC
        pooled_x5 = self.rmac_pool(x5)
        pooled_x4 = self.rmac_pool(x4)

        features = torch.cat((pooled_x4, pooled_x5), 1).squeeze(-1).squeeze(-1)
        return self.fc(features)

