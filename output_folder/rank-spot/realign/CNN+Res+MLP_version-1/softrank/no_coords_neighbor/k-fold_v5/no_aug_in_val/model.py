import torch
import torch.nn as nn

# Helper for GroupNorm
def GN(num_channels, num_groups=16):
    return nn.GroupNorm(num_groups, num_channels)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = GN(out_channels)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2   = GN(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                GN(out_channels)
            )
        self.act2 = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        return self.act2(out)

class DeepTileEncoder(nn.Module):
    """Tile branch: multi-scale pooling + deeper MLP"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            GN(32),
            nn.GELU(),
            nn.MaxPool2d(2)  # 78 → 39
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)  # 39 → 19
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2)  # 19 → 9
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256)
        )  # Keeps 9×9

        # Multi-scale pooling: three grid sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((3, 3))
        self.large_pool  = nn.AdaptiveAvgPool2d((9, 9))

        total_dim = 256*1*1 + 256*3*3 + 256*9*9
        # Deeper MLP: total_dim → 8*out_dim → 4*out_dim → 2*out_dim → out_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(total_dim, out_dim*8),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim*8, out_dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*4, out_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        g = self.global_pool(x).reshape(x.size(0), -1)
        m = self.mid_pool(x).reshape(x.size(0), -1)
        l = self.large_pool(x).reshape(x.size(0), -1)
        return self.fc(torch.cat([g, m, l], dim=1))

class SubtileEncoder(nn.Module):
    """Subtile branch: local info + multi-scale attention + deeper MLP"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            GN(32),
            nn.GELU(),
            nn.MaxPool2d(2)  # 26 → 13
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)  # 13 → 6
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )  # Keeps 6×6

        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2, 2))
        self.large_pool  = nn.AdaptiveAvgPool2d((3, 3))

        feat_dim = 128*1*1 + 128*2*2 + 128*3*3
        # Attention to weight subtiles
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.GELU(),
            nn.Linear(feat_dim//2, 1)
        )
        # Deeper MLP
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, out_dim*4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim*4, out_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.GELU()
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.reshape(B*N, C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).reshape(B, N, -1)
        m = self.mid_pool(x).reshape(B, N, -1)
        l = self.large_pool(x).reshape(B, N, -1)
        feats = torch.cat([g, m, l], dim=2)
        scores = self.attn(feats)
        weights = torch.softmax(scores, dim=1)
        pooled  = (feats * weights).sum(dim=1)
        return self.fc(pooled)

class VisionMLP_MultiTask(nn.Module):
    """Fusion model with expanded dimensions"""
    def __init__(self, tile_dim=128, subtile_dim=128, output_dim=35):
        super().__init__()
        self.encoder_tile    = DeepTileEncoder(tile_dim)
        self.encoder_subtile = SubtileEncoder(subtile_dim)
        self.decoder = nn.Sequential(
            nn.Linear(tile_dim + subtile_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, tile, subtiles):
        f_tile = self.encoder_tile(tile)
        f_sub  = self.encoder_subtile(subtiles)
        x = torch.cat([f_tile, f_sub], dim=1)
        return self.decoder(x)

# Instantiate updated model with new dims
model = VisionMLP_MultiTask(tile_dim=128, subtile_dim=128, output_dim=35)
