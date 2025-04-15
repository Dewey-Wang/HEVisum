import torch
import torch.nn as nn
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, out_dim, in_channels=3, negative_slope=0.01):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class MultiTaskDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=35, negative_slope=0.01):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=0.3)
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])

    def forward(self, x):
        shared = self.shared(x)
        outs = [head(shared) for head in self.heads]
        return torch.cat(outs, dim=1)  # (B, 35)


class VisionMLP_MultiTask(nn.Module):
    def __init__(self, cnn_out_dim=64, output_dim=35, negative_slope=0.01):
        super().__init__()
        self.encoder_tile = CNNEncoder(cnn_out_dim, in_channels=3, negative_slope=negative_slope)
        self.encoder_neighbors = CNNEncoder(cnn_out_dim, in_channels=3, negative_slope=negative_slope)
        self.encoder_subtiles = CNNEncoder(cnn_out_dim, in_channels=3, negative_slope=negative_slope)

        self.decoder = MultiTaskDecoder(input_dim=cnn_out_dim * 3 + 2,
                                        output_dim=output_dim,
                                        negative_slope=negative_slope)

    def forward(self, tile, subtiles, neighbors, coords):
        B = tile.size(0)

        # tile: [B, 3, 78, 78]
        f_tile = self.encoder_tile(tile)

        # subtiles: [B, 9, 3, 26, 26]
        B, N, C, H, W = subtiles.shape
        subtiles_reshaped = subtiles.contiguous().reshape(B * N, C, H, W)  # ✅ 強制 contiguous
        f_sub = self.encoder_subtiles(subtiles_reshaped).reshape(B, N, -1).mean(dim=1)  # ✅ reshape

        # neighbors: [B, 8, 3, 78, 78]
        B, N, C, H, W = neighbors.shape
        neighbors_reshaped = neighbors.contiguous().reshape(B * N, C, H, W)  # ✅ 強制 contiguous
        f_neigh = self.encoder_neighbors(neighbors_reshaped).reshape(B, N, -1).mean(dim=1)  # ✅ reshape

        x = torch.cat([f_tile, f_sub, f_neigh, coords], dim=1)
        return self.decoder(x)



