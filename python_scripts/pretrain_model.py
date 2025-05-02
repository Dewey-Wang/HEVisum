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
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))
        self.large_pool  = nn.AdaptiveAvgPool2d((3,3))

        total_dim = 128*1*1 + 128*2*2 + 128*3*3
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
        g = self.global_pool(x).view(x.size(0), -1)
        m = self.mid_pool(x).view(x.size(0), -1)
        l = self.large_pool(x).view(x.size(0), -1)
        return self.fc(torch.cat([g, m, l], dim=1))

class NeighborSubtileEncoder(nn.Module):
    """共享權重，對多個鄰居 subtiles 做 mean pooling"""
    def __init__(self, out_dim, in_channels=3):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2)
        )  # h→13
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64), nn.MaxPool2d(2)
        )  # 13→6
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128)
        )  # 6×6

        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))
        self.large_pool  = nn.AdaptiveAvgPool2d((3,3))

        total_dim = 128*1*1 + 128*2*2  + 128*3*3
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
        # subtiles: [B,9,3,26,26]
        B, N, C, H, W = subtiles.shape
        x = subtiles.view(B*N, C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).view(B, N, -1)
        m = self.mid_pool(x).view(B, N, -1)
        l = self.large_pool(x).view(B, N, -1)
        feats = torch.cat([g, m, l], dim=2)  # [B,N,total_dim]
        f = self.fc(feats.view(B*N, -1)).view(B, N, -1)
        return f.mean(dim=1)  # [B,out_dim]
                  # [B,out_dim]
    
    
# ——— 修改 SharedDecoder 构造函数 ———
class SharedDecoder(nn.Module):
    def __init__(self, tile_size=26):
        super().__init__()
        self.tile_size = tile_size
        # 这里只保留最核心的 conv 部分，不用 nn.Unflatten
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 4, 2, 1),  # 8→16
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 16→32
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, full_output: bool = False):
        """
        x: (B, 9*1*8*8)
        """
        B = x.size(0)
        # 先把 x 拆成 (B,9,1,8,8)
        patches = x.view(B, 9, 1, 8, 8)
        # 合并 batch 与 patch 两维 -> (B*9,1,8,8)
        patches = patches.view(B*9, 1, 8, 8)
        # 反卷积到 (B*9,3,32,32)
        recon = self.deconv(patches)
        # 裁剪到 26×26
        _, C, H, W = recon.shape
        h0 = (H - self.tile_size)//2
        w0 = (W - self.tile_size)//2
        recon = recon[:, :, h0:h0+self.tile_size, w0:w0+self.tile_size]  # (B*9,3,26,26)
        # 分回 batch & patch 维度
        recon = recon.view(B, 9, C, self.tile_size, self.tile_size)      # (B,9,3,26,26)

        if full_output:
            return recon
        else:
            return recon[:,4]   # 中心 patch (B,3,26,26)


# ——— 在 AE_Center/AE_AllSubtiles/AE_MaskedMAE 中使用这个 decoder ———
# AE_Center 只输出中心 patch
class AE_Center(nn.Module):
    def __init__(self, center_dim=64, neighbor_dim=64, hidden_dim=128, tile_size=26):
        super().__init__()
        self.enc_center = CenterSubtileEncoder(center_dim)
        self.enc_neigh  = NeighborSubtileEncoder(neighbor_dim)
        fusion_dim = center_dim + neighbor_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        # 9*1*8*8 = 576
        self.fc_dec = nn.Linear(hidden_dim, 9*1*8*8)
        self.decoder = SharedDecoder(tile_size=tile_size)

    def forward(self, tile, subtiles):
        f_c = self.enc_center(subtiles[:,4])
        f_n = self.enc_neigh(subtiles)
        h   = self.fc_enc(torch.cat([f_c, f_n], dim=1))
        x   = self.fc_dec(h)
        return self.decoder(x)  # (B,3,26,26)


# AE_AllSubtiles 重建全部 9 张
class AE_AllSubtiles(nn.Module):
    def __init__(self, center_dim=64, neighbor_dim=64, hidden_dim=128, tile_size=26):
        super().__init__()
        self.enc_center = CenterSubtileEncoder(center_dim)
        self.enc_neigh  = NeighborSubtileEncoder(neighbor_dim)
        fusion_dim = center_dim + neighbor_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.fc_dec = nn.Linear(hidden_dim, 9*1*8*8)
        self.decoder = SharedDecoder(tile_size=tile_size)

    def forward(self, tile, subtiles):
        f_c = self.enc_center(subtiles[:,4])
        f_n = self.enc_neigh(subtiles)
        h   = self.fc_enc(torch.cat([f_c, f_n], dim=1))
        x   = self.fc_dec(h)
        return self.decoder(x, full_output=True)  # (B,9,3,26,26)


# AE_MaskedMAE 随机遮掉部分 subtiles，再重建所有
class AE_MaskedMAE(nn.Module):
    def __init__(self, center_dim=64, neighbor_dim=64, hidden_dim=128,
                 tile_size=26, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ae_all = AE_AllSubtiles(center_dim, neighbor_dim, hidden_dim, tile_size)

    def forward(self, tile, subtiles):
        B = subtiles.size(0)
        mask = (torch.rand(B,9,1,1,1, device=subtiles.device) > self.mask_ratio).float()
        masked = subtiles * mask
        return self.ae_all(tile, masked)  # (B,9,3,26,26)


class PretrainedEncoderRegressor(nn.Module):
    """
    通用：加载任何一个 AE 模型（只取 encoder 部分），
    冻结它（可选），然后加一个新的 head 来输出 output_dim 维。
    """
    def __init__(
        self,
        ae_checkpoint: str,
        ae_type: str = "all",       # "center","all","masked"
        center_dim: int = 64,
        neighbor_dim: int = 64,
        hidden_dim: int = 128,
        tile_size: int = 26,
        output_dim: int = 35,
        freeze_encoder: bool = True # <— 新增参数
    ):
        super().__init__()

        # —— 1) 根据类型实例化并加载 AE
        if ae_type == "center":
            ae = AE_Center(center_dim, neighbor_dim, hidden_dim, tile_size)
        elif ae_type == "all":
            ae = AE_AllSubtiles(center_dim, neighbor_dim, hidden_dim, tile_size)
        elif ae_type == "masked":
            ae = AE_MaskedMAE(center_dim, neighbor_dim, hidden_dim, tile_size)
        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

        ae.load_state_dict(torch.load(ae_checkpoint, map_location="cpu"))

        # —— 2) 按需冻结 AE 参数
        if freeze_encoder:
            for p in ae.parameters():
                p.requires_grad = False

        # —— 3) 挑出它的 encoder
        self.enc_center = ae.enc_center
        self.enc_neigh  = ae.enc_neigh

        # —— 4) 定义你的回归头
        fusion_dim = center_dim + neighbor_dim
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, tile, subtiles):
        f_c = self.enc_center(subtiles[:,4])
        f_n = self.enc_neigh(subtiles)
        x   = torch.cat([f_c, f_n], dim=1)
        return self.decoder(x)

    def unfreeze_encoder(self, lr: float = None, optimizer: torch.optim.Optimizer = None):
        """
        解冻 encoder 部分。如果同时传入 optimizer 和 lr，会把 encoder 参数
        添加到 optimizer.param_groups 并设置学习率 lr。
        """
        # 1) 解冻
        for name, p in self.named_parameters():
            if name.startswith("enc_center") or name.startswith("enc_neigh"):
                p.requires_grad = True

        # 2) 如需热插拔到 optimizer，添加 param_group
        if optimizer is not None:
            params = [
                p for name,p in self.named_parameters()
                if (name.startswith("enc_center") or name.startswith("enc_neigh")) and p.requires_grad
            ]
            group = {"params": params}
            if lr is not None:
                group["lr"] = lr
            optimizer.add_param_group(group)