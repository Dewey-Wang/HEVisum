# hevisum_model.py

import torch
import torch.nn as nn
from torchvision import models

class SimpleGAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.Parameter(torch.empty(out_dim * 2))
        nn.init.uniform_(self.attn.data, -0.1, 0.1)

    def forward(self, center_feat, neighbor_feats):
        B, N, D = neighbor_feats.shape
        h_c = self.fc(center_feat).unsqueeze(1)       # (B, 1, D)
        h_n = self.fc(neighbor_feats)                 # (B, N, D)
        h_repeat = h_c.expand(-1, N, -1)              # (B, N, D)
        h_concat = torch.cat([h_repeat, h_n], dim=-1) # (B, N, 2D)
        alpha = (h_concat * self.attn).sum(dim=-1)    # (B, N)
        alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)  # (B, N, 1)
        out = torch.sum(alpha * h_n, dim=1)           # (B, D)
        return out

class HEVisumModel(nn.Module):
    def __init__(self, cnn_out_dim=128, output_dim=35):
        super().__init__()
        self.encoder = self.build_encoder(cnn_out_dim)
        self.gat = SimpleGAT(in_dim=cnn_out_dim, out_dim=cnn_out_dim)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def build_encoder(self, out_dim):
        resnet = models.resnet18(weights=None)
        layers = list(resnet.children())[:-2]
        cnn = nn.Sequential(*layers)
        gap = nn.AdaptiveAvgPool2d(1)
        return nn.Sequential(cnn, gap, nn.Flatten(), nn.Linear(512, out_dim))

    def encode_tile(self, x):  # x: (B, 3, H, W)
        return self.encoder(x)  # (B, D)

    def forward(self, center_tile, subtiles, neighbor_tiles):
        B = center_tile.size(0)
        f_center = self.encode_tile(center_tile)  # (B, D)

        subtiles = subtiles.view(B * 9, *subtiles.shape[2:])  # (B*9, 3, h, w)
        f_sub = self.encode_tile(subtiles).view(B, 9, -1).mean(dim=1)  # (B, D)

        neighbor_tiles = neighbor_tiles.view(B * 8, *neighbor_tiles.shape[2:])
        f_neigh = self.encode_tile(neighbor_tiles).view(B, 8, -1)  # (B, 8, D)
        f_neigh_gat = self.gat(f_center, f_neigh)  # (B, D)

        f = torch.cat([f_center, f_sub, f_neigh_gat], dim=1)  # (B, 3D)
        out = self.fc(f)  # (B, 35)
        return out
