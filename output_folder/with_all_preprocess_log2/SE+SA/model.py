import torch
import torch.nn as nn
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Channel-wise attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = x.mean(dim=1, keepdim=True)  # → (B, 1, H, W)
        attn = self.conv(avg_out)              # → (B, 1, H, W)
        attn = self.sigmoid(attn)              # → (B, 1, H, W)
        return x * attn                        # → (B, C, H, W)

class SimpleCNNEncoder(nn.Module):
    def __init__(self, out_dim, in_channels=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),             # Channel-wise Attention ✅
            SpatialAttention(),      # Spatial Attention ✅
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),             # Channel-wise Attention ✅
            SpatialAttention(),      # Spatial Attention ✅
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x




class VisionWithCoord(nn.Module):
    def __init__(self, cnn_out_dim=64, output_dim=35):
        super().__init__()
        self.encoder = SimpleCNNEncoder(out_dim=cnn_out_dim, in_channels=4)
        self.decoder = nn.Sequential(
            nn.Linear(cnn_out_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, tile, coords):
        """
        tile: (B, 4, H, W)
        coords: (B, 2) - normalized x, y
        """
        feat = self.encoder(tile)                # (B, 64)
        x = torch.cat([feat, coords], dim=1)     # (B, 66)
        out = self.decoder(x)                    # (B, 35)
        return out