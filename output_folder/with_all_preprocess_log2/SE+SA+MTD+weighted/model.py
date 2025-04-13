import torch
import torch.nn as nn
class MultiTaskDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=35):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # 每個 cell type 都有一個 Linear 頭部（1 個 scalar output）
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])

    def forward(self, x):
        shared_feat = self.shared(x)  # (B, 64)
        outs = [head(shared_feat) for head in self.heads]  # List[(B, 1)] × 35
        return torch.cat(outs, dim=1)  # (B, 35)

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




class VisionWithCoord_MTL(nn.Module):
    def __init__(self, cnn_out_dim=64, output_dim=35):
        super().__init__()
        self.encoder = SimpleCNNEncoder(out_dim=cnn_out_dim, in_channels=4)
        self.decoder = MultiTaskDecoder(cnn_out_dim + 2, output_dim)

    def forward(self, tile, coords):
        feat = self.encoder(tile)                # (B, 64)
        x = torch.cat([feat, coords], dim=1)     # (B, 66)
        out = self.decoder(x)                    # (B, 35)
        return out


model = VisionWithCoord_MTL()


def get_loss_fn(loss_type="mse", cell_weights=None):
    """
    支援多種 loss function，包含:
    - "mse": Mean Squared Error (預設)
    - "weighted_mse": 根據 cell_weights 做加權的 MSE
    - "mae": Mean Absolute Error
    - "spearman": Spearman loss (非 differentiable，但可以實驗)

    回傳對應的 loss function。
    """
    loss_type = loss_type.lower()

    if loss_type == "mse":
        return nn.MSELoss()
    
    elif loss_type == "mae":
        return nn.L1Loss()

    elif loss_type == "weighted_mse":
        if cell_weights is None:
            raise ValueError("需要提供 cell_weights 才能使用 weighted MSE")

        def weighted_mse(pred, target):
            loss = (pred - target) ** 2  # (B, C)
            loss = loss.mean(dim=0)      # 對 batch 平均 → (C,)
            weighted_loss = (loss * cell_weights.to(pred.device)).mean()
            return weighted_loss

        return weighted_mse

    elif loss_type == "spearman":
        from scipy.stats import spearmanr

        def spearman_loss(pred, target):
            pred = pred.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            rho = np.mean([spearmanr(pred[:, i], target[:, i])[0] for i in range(pred.shape[1])])
            return 1 - rho  # 模擬 loss 越小越好（非 differentiable）
        
        return spearman_loss

    else:
        raise ValueError(f"不支援的 loss_type: {loss_type}")


cell_weights = torch.tensor(mse_per_cell)
cell_weights = cell_weights / cell_weights.mean()  # 均值為 1，總和為 35 左右
cell_weights
tensor([0.3966, 0.8798, 0.4230, 0.4727, 0.5482, 0.7296, 0.8255, 1.6273, 0.4956,
        0.8500, 1.1992, 0.7474, 1.8725, 1.3113, 0.5461, 0.4584, 1.1001, 0.6069,
        1.6320, 1.3040, 1.3606, 1.8043, 0.9306, 0.8514, 1.0419, 1.5623, 0.4745,
        1.3299, 1.1567, 0.9208, 0.7626, 0.8305, 1.8133, 1.3111, 0.8236])