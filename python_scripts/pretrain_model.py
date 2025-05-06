import torch
from torch import nn
from torch.autograd import Function


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
    def __init__(self, out_dim, in_channels=3, negative_slope=0.01):
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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B,256,1,1]
        self.mid_pool    = nn.AdaptiveAvgPool2d((3, 3))  # [B,256,3,3]

        total_dim = 256*1*1 + 256*3*3
        # 三层 MLP：total_dim → 2*out_dim → out_dim → out_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_dim, out_dim*4),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.1),
            nn.Linear(out_dim*4, out_dim*2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x: [B,256,9,9]
        g = self.global_pool(x).contiguous().reshape(x.size(0), -1)  # [B,256]
        m = self.mid_pool(x).contiguous().reshape(x.size(0), -1)     # [B,256*3*3]

        return self.fc(torch.cat([g, m], dim=1))


class CenterSubtileEncoder(nn.Module):
    """專門處理中心 subtile 的 Encoder"""
    def __init__(self, out_dim, in_channels=3, negative_slope= 0.01):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2)  # 26→13
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)  # 13→6
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )  # 6×6

        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))
        self.large_pool    = nn.AdaptiveAvgPool2d((3,3))

        total_dim = 128*1*1 + 128*2*2 + 128*3*3
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_dim, out_dim*2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        g = self.global_pool(x).contiguous().reshape(x.size(0), -1)
        m = self.mid_pool(x).contiguous().reshape(x.size(0), -1)
        l = self.large_pool(x).contiguous().reshape(x.size(0), -1)

        return self.fc(torch.cat([g, m, l], dim=1)).contiguous()

class NeighborSubtileEncoder(nn.Module):
    def __init__(self, out_dim, in_channels=3, negative_slope=0.01):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2)  # 26→13
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)  # 13→6
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )  # 保持 6×6

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mid_pool    = nn.AdaptiveAvgPool2d((2,2))
        self.large_pool    = nn.AdaptiveAvgPool2d((3,3))

        total_dim = 128*1*1 + 128*2*2 + 128*3*3
        # 两层 MLP：total_dim → out_dim*2 → out_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(total_dim, out_dim*2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.1),
            nn.Linear(out_dim*2, out_dim),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.contiguous().reshape(B*N, C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # g,m: [B*N, feat]
        g = self.global_pool(x).contiguous().reshape(B, N, -1)
        m = self.mid_pool(x).contiguous().reshape(B, N, -1)
        l = self.large_pool(x).contiguous().reshape(B, N, -1)

        # 合并 N 张 subtiles，再 FC
        feat = torch.cat([g, m, l], dim=2).mean(dim=1).contiguous()  # [B, total_dim]
        return self.fc(feat)
    
class SharedDecoder(nn.Module):
    def __init__(self, tile_size=26):
        super().__init__()
        self.tile_size = tile_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, full_output: bool = False):
        B = x.size(0)
        patches = x.contiguous().reshape(B, 9, 1, 8, 8)
        patches = patches.contiguous().reshape(B*9, 1, 8, 8)
        recon = self.deconv(patches)
        _, C, H, W = recon.shape
        h0 = (H - self.tile_size)//2
        w0 = (W - self.tile_size)//2
        recon = recon[:, :, h0:h0+self.tile_size, w0:w0+self.tile_size]
        recon = recon.contiguous().reshape(B, 9, C, self.tile_size, self.tile_size)
        return recon if full_output else recon[:,4]


# ——— 在 AE_Center/AE_AllSubtiles/AE_MaskedMAE 中使用这个 decoder ———
# AE_Center 只输出中心 patch
class AE_Center(nn.Module):
    def __init__(self, tile_dim = 64 , center_dim=64, neighbor_dim=64, hidden_dim=128, tile_size=26):
        super().__init__()
        self.enc_center = CenterSubtileEncoder(center_dim)
        self.enc_neigh  = NeighborSubtileEncoder(neighbor_dim)
        self.enc_tile  = DeepTileEncoder(tile_dim)
        fusion_dim = center_dim + neighbor_dim + tile_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.fc_dec = nn.Linear(hidden_dim, 9*1*8*8)
        self.decoder = SharedDecoder(tile_size)
    def forward(self, tile, subtiles):
        f_c = self.enc_center(subtiles[:,4])
        f_n = self.enc_neigh(subtiles)
        f_t = self.enc_tile(tile)
        fused = torch.cat([f_c, f_n, f_t], dim=1).contiguous()
        h = self.fc_enc(fused)
        x = self.fc_dec(h)
        return self.decoder(x)

# 修改后的 AE_AllSubtiles，AE_MaskedMAE 及 PretrainedEncoderRegressor，实现输入强制 contiguous 并移除所有 view
class AE_AllSubtiles(nn.Module):
    def __init__(self, tile_dim=64, center_dim=64, neighbor_dim=64, hidden_dim=128, tile_size=26):
        super().__init__()
        self.enc_center = CenterSubtileEncoder(center_dim)
        self.enc_neigh  = NeighborSubtileEncoder(neighbor_dim)
        self.enc_tile   = DeepTileEncoder(tile_dim)
        fusion_dim = center_dim + neighbor_dim + tile_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.fc_dec = nn.Linear(hidden_dim, 9 * 1 * 8 * 8)
        self.decoder = SharedDecoder(tile_size=tile_size)

    def forward(self, tile, subtiles):
        tile     = tile.contiguous()
        subtiles = subtiles.contiguous()

        f_c = self.enc_center(subtiles[:, 4])
        f_n = self.enc_neigh(subtiles)
        f_t = self.enc_tile(tile)

        h = self.fc_enc(torch.cat([f_c, f_n, f_t], dim=1).contiguous())
        x = self.fc_dec(h)
        return self.decoder(x, full_output=True)  # [B, 9, 3, tile_size, tile_size]


class AE_MaskedMAE(nn.Module):
    def __init__(self, tile_dim=64, center_dim=64, neighbor_dim=64, hidden_dim=128,
                 tile_size=26, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ae_all = AE_AllSubtiles(tile_dim, center_dim, neighbor_dim, hidden_dim, tile_size)

    def forward(self, tile, subtiles):
        tile     = tile.contiguous()
        subtiles = subtiles.contiguous()
        B = subtiles.size(0)
        # 隨機遮 mask_ratio 的 patches
        mask = (torch.rand(B, 9, 1, 1, 1, device=subtiles.device) > self.mask_ratio).float()
        masked = subtiles * mask
        return self.ae_all(tile, masked)
class PretrainedEncoderRegressor(nn.Module):
    """
    加载任意一個 AE 模型（center, all, masked），提取 encoder 後加上回歸頭。
    支援 DeepTileEncoder。
    """
    def __init__(
        self,
        ae_checkpoint: str,
        ae_type: str = "all",       # "center","all","masked"
        tile_dim: int = 64,
        center_dim: int = 64,
        neighbor_dim: int = 64,
        hidden_dim: int = 128,
        tile_size: int = 26,
        output_dim: int = 35,
        freeze_encoder: bool = True
    ):
        super().__init__()
        # 加载對應的 AE 模型
        if ae_type == "center":
            ae = AE_Center(tile_dim, center_dim, neighbor_dim, hidden_dim, tile_size)
        elif ae_type == "all":
            ae = AE_AllSubtiles(tile_dim, center_dim, neighbor_dim, hidden_dim, tile_size)
        elif ae_type == "masked":
            ae = AE_MaskedMAE(tile_dim, center_dim, neighbor_dim, hidden_dim, tile_size)
        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

        ae.load_state_dict(torch.load(ae_checkpoint, map_location="cpu"))

        if freeze_encoder:
            for p in ae.parameters():
                p.requires_grad = False

        # 提取 encoder 模組
        self.enc_center = ae.enc_center
        self.enc_neigh  = ae.enc_neigh
        self.enc_tile   = ae.enc_tile

        # 計算輸入維度
        fusion_dim = center_dim + neighbor_dim + tile_dim

        # 回歸頭
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, tile, subtiles):
        tile = tile.contiguous()
        subtiles = subtiles.contiguous()
        f_c = self.enc_center(subtiles[:, 4])
        f_n = self.enc_neigh(subtiles)
        f_t = self.enc_tile(tile)
        x = torch.cat([f_c, f_n, f_t], dim=1).contiguous()
        return self.decoder(x)

    def unfreeze_encoder(self, lr: float = None, optimizer: torch.optim.Optimizer = None):
        for name, p in self.named_parameters():
            if name.startswith("enc_center") or name.startswith("enc_neigh") or name.startswith("enc_tile"):
                p.requires_grad = True
        if optimizer is not None:
            params = [
                p for name, p in self.named_parameters()
                if (name.startswith("enc_center") or name.startswith("enc_neigh") or name.startswith("enc_tile"))
                and p.requires_grad
            ]
            group = {"params": params}
            if lr is not None:
                group["lr"] = lr
            optimizer.add_param_group(group)