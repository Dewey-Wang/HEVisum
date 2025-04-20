import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, negative_slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu2 = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu2(out)
        return out


class ResNetLikeEncoder(nn.Module):
    def __init__(self, out_dim, in_channels=3, negative_slope=0.01):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 32, stride=1, negative_slope=negative_slope),
            ResidualBlock(32, 32, stride=1, negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=1, negative_slope=negative_slope),
            ResidualBlock(64, 64, stride=1, negative_slope=negative_slope),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=35, negative_slope=0.01):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p=0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.shared(x)


class VisionMLP_MultiTask(nn.Module):
    def __init__(self, cnn_out_dim=64, output_dim=35, num_freqs=4, negative_slope=0.01):
        super().__init__()
        self.encoder_tile      = ResNetLikeEncoder(cnn_out_dim, in_channels=3, negative_slope=negative_slope)
        self.encoder_subtiles  = ResNetLikeEncoder(cnn_out_dim, in_channels=3, negative_slope=negative_slope)
        self.encoder_neighbors = ResNetLikeEncoder(cnn_out_dim, in_channels=3, negative_slope=negative_slope)

        self.num_freqs = num_freqs
        self.pe_dim    = 4 * num_freqs
        decoder_in_dim = cnn_out_dim * 3 + self.pe_dim
        self.decoder = MLPDecoder(input_dim=decoder_in_dim,
                                  output_dim=output_dim,
                                  negative_slope=negative_slope)

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        pe_list = []
        for i in range(self.num_freqs):
            freq = (2.0 ** i) * math.pi
            pe_list.append(torch.sin(coords * freq))
            pe_list.append(torch.cos(coords * freq))
        return torch.cat(pe_list, dim=1)

    def forward(self, tile, subtiles, neighbors, norm_coord):
        B = tile.size(0)
        f_tile = self.encoder_tile(tile)

        # subtiles
        _, N, C, H, W = subtiles.shape
        sub = subtiles.contiguous().reshape(B * N, C, H, W)
        f_sub = self.encoder_subtiles(sub).contiguous().reshape(B, N, -1).mean(dim=1)

        # neighbors
        _, M, C, H, W = neighbors.shape
        neigh = neighbors.contiguous().reshape(B * M, C, H, W)
        f_neigh = self.encoder_neighbors(neigh).contiguous().reshape(B, M, -1).mean(dim=1)

        pe = self.positional_encoding(norm_coord)
        x = torch.cat([f_tile, f_sub, f_neigh, pe], dim=1)
        return self.decoder(x)

model = VisionMLP_MultiTask()
