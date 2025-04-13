import torch
import torch.nn as nn

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
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # ➜ (B, 64, 1, 1)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x)       # (B, 64, 1, 1)
        x = self.flatten(x)   # (B, 64)
        x = self.linear(x)    # (B, out_dim)
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

