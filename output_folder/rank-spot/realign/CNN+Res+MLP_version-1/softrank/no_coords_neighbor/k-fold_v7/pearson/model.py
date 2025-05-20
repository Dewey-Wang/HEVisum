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
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return out + identity

class DeepTileEncoder(nn.Module):
    """Tile branch: shallow + two-scale pooling + compact MLP"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        # Reduced depth and channels
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            GN(32), nn.GELU(), nn.MaxPool2d(2)  # 78→39
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2)                     # 39→19
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2)                     # 19→9
        )

        # Two-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((3,3))

        total_dim = 128*1*1 + 128*3*3
        # Compact MLP: total_dim → 4*out_dim → 2*out_dim → out_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(total_dim, out_dim*4), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim*4, out_dim*2), nn.GELU(),
            nn.Linear(out_dim*2, out_dim)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).reshape(x.size(0), -1)
        m = self.mid_pool(x).reshape(x.size(0), -1)
        return self.fc(torch.cat([g, m], dim=1))

class SubtileEncoder(nn.Module):
    """Subtile branch: compact + two-scale pooling + attention MLP"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            GN(32), nn.GELU(), nn.MaxPool2d(2)  # 26→13
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2)                     # 13→6
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128)
        )  # keeps 6×6

        # Two-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))

        feat_dim = 128*1*1 + 128*2*2
        # Attention to weight subtiles
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2), nn.GELU(),
            nn.Linear(feat_dim//2, 1)
        )
        # Compact MLP
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, out_dim*2), nn.GELU(),
            nn.Linear(out_dim*2, out_dim)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.reshape(B*N, C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).reshape(B, N, -1)
        m = self.mid_pool(x).reshape(B, N, -1)
        feats = torch.cat([g, m], dim=2)
        weights = torch.softmax(self.attn(feats), dim=1)
        pooled = (feats * weights).sum(dim=1)
        return self.fc(pooled)

class VisionMLP_MultiTask(nn.Module):
    """Slim Fusion model for limited data"""
    def __init__(self, tile_dim=64, subtile_dim=64, output_dim=35):
        super().__init__()
        self.encoder_tile    = DeepTileEncoder(tile_dim)
        self.encoder_subtile = SubtileEncoder(subtile_dim)
        self.decoder = nn.Sequential(
            nn.Linear(tile_dim + subtile_dim, 128), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, tile, subtiles):
        f_tile = self.encoder_tile(tile)
        f_sub  = self.encoder_subtile(subtiles)
        return self.decoder(torch.cat([f_tile, f_sub], dim=1))

# Instantiate slimmed model
model = VisionMLP_MultiTask(tile_dim=64, subtile_dim=64, output_dim=35)
