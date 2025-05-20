import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.act2 = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return out + identity

class DeepTileEncoder(nn.Module):
    """加深的 Tile 分支：全局信息，多尺度池化 + 三层 MLP"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2)  # 78→39
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)  # 39→19
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2)  # 19→9
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256)
        )  # 保持 9×9

        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((3, 3))
        self.large_pool  = nn.AdaptiveAvgPool2d((9, 9))   # 新增

        total_dim = 256*1*1 + 256*3*3 + 256*9*9           # 改成三个尺度之和
        # 三层 MLP：total_dim → 2*out_dim → out_dim → out_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_dim, out_dim*4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*4, out_dim*2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        g = self.global_pool(x).reshape(x.size(0), -1)   # [B,256]
        m = self.mid_pool(x).reshape(x.size(0), -1)      # [B,256*3*3]
        l = self.large_pool(x).reshape(x.size(0), -1)    # [B,256*9*9]
        return self.fc(torch.cat([g, m, l], dim=1))

class CenterSubtileEncoder(nn.Module):
    """專門處理中心 subtile 的 Encoder"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2)
        )  # 26→13
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64), nn.MaxPool2d(2)
        )  # 13→6
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128)
        )  # 6×6
        
        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))  # 新增
        self.large_pool  = nn.AdaptiveAvgPool2d((3,3))  # 新增

        total_dim = 128*1*1 + 128*2*2 + 128*3*3         # 三尺度之和        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_dim, out_dim*2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.SiLU(),
        )


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).reshape(x.size(0), -1)
        m = self.mid_pool(x).reshape(x.size(0), -1)
        l = self.large_pool(x).reshape(x.size(0), -1)
        return self.fc(torch.cat([g, m, l], dim=1))

class NeighborSubtileEncoder(nn.Module):
    """共享權重，對多個鄰居 subtiles 做 mean pooling"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2)
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64), nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128)
        )
        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))  # 新增
        self.large_pool  = nn.AdaptiveAvgPool2d((3,3))  # 新增

        total_dim = 128*1*1 + 128*2*2 + 128*3*3        # 三尺度之和
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_dim, out_dim*2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.SiLU(),
        )

    def forward(self, subtiles):
        B, N, C, H, W = subtiles.shape
        x = subtiles.view(B*N, C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).reshape(B, N, -1)      # [B,N,feat]
        m = self.mid_pool(x).reshape(B, N, -1)         # [B,N,feat]
        l = self.large_pool(x).reshape(B, N, -1)       # [B,N,feat]
        feats = torch.cat([g, m, l], dim=2)            # [B,N,total_dim]
        # 对每个 neighbor 子块各自做 FC，再取 mean
        f = self.fc(feats.view(B*N, -1)).view(B, N, -1) # [B,N,out_dim]
        return f.mean(dim=1)                           # [B,out_dim]

class VisionMLP_MultiTask(nn.Module):
    """融合 Tile、Center、Neighbor 三路特徵的多任務模型"""
    def __init__(self, tile_dim=64, center_dim=64, neighbor_dim=64, output_dim=35):
        super().__init__()
        self.encoder_tile      = DeepTileEncoder(tile_dim)
        self.encoder_center    = CenterSubtileEncoder(center_dim)
        self.encoder_neighbors = NeighborSubtileEncoder(neighbor_dim)
        fusion_dim = tile_dim + center_dim + neighbor_dim
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, output_dim)
        )

    def forward(self, tile, subtiles):
        f_tile = self.encoder_tile(tile)
        center = subtiles[:, 4]
        f_center = self.encoder_center(center)
        neighbors = torch.cat([subtiles[:, :4], subtiles[:, 5:]], dim=1)
        f_neigh = self.encoder_neighbors(neighbors)
        x = torch.cat([f_tile, f_center, f_neigh], dim=1)
        return self.decoder(x)

# Instantiate and count parameters
model = VisionMLP_MultiTask(tile_dim=64, center_dim=64, neighbor_dim=64, output_dim=35)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
