import torch
import torch.nn as nn

class SimpleConvEncoder(nn.Module):
    """简化版 Conv → Pool → MLP 编码器，适用于 center 和 neighbor subtile"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        # 三层卷积 + 池化
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),    # 26→13 or 78→39

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),    # 13→6 or 39→19

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # 保持尺寸：6×6 或 19×19
        )
        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))
        self.large_pool  = nn.AdaptiveAvgPool2d((3,3))
        total_feat = 128*1*1 + 128*2*2 + 128*3*3

        # MLP 到 out_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_feat, out_dim*2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x)
        g = self.global_pool(x).view(x.size(0), -1)
        m = self.mid_pool(x).view(x.size(0), -1)
        l = self.large_pool(x).view(x.size(0), -1)
        return self.fc(torch.cat([g, m, l], dim=1))


class VisionMLP_MultiTask(nn.Module):
    """只用 Center + Neighbor 两路分支的模型"""
    def __init__(self, center_dim=64, neighbor_dim=64, output_dim=35):
        super().__init__()
        self.encoder_center    = SimpleConvEncoder(center_dim)
        self.encoder_neighbors = SimpleConvEncoder(neighbor_dim)

        fusion_dim = center_dim + neighbor_dim
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, tile, subtiles):
        # tile 参数不再使用
        # center subtile
        center = subtiles[:, 4]                # [B,3,26,26]
        f_center = self.encoder_center(center) # [B, center_dim]

        # 所有 9 张一起编码，作为 neighbors
        # （如果想排除 center，使用 neighbors = torch.cat([...], dim=1)）
        f_neigh = self.encoder_neighbors(
            subtiles.view(-1, *subtiles.shape[2:])  # (B*9,3,26,26)
        ).view(subtiles.size(0), 9, -1).mean(dim=1)  # [B, neighbor_dim]

        x = torch.cat([f_center, f_neigh], dim=1)
        return self.decoder(x)


# 用法示例
model = VisionMLP_MultiTask(center_dim=64, neighbor_dim=64, output_dim=35)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
