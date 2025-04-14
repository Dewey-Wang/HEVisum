import torch
import torch.nn as nn
class MultiTaskDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=35, dropout_rate=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])

    def forward(self, x):
        shared_feat = self.shared(x)
        outs = [head(shared_feat) for head in self.heads]
        return torch.cat(outs, dim=1)


class SimpleCNNEncoder(nn.Module):
    def __init__(self, out_dim, in_channels=4, dropout_rate=0.3):
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
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dropout(x)    # ✅ Apply dropout before final linear
        x = self.linear(x)
        return x





class VisionWithCoord_MTL(nn.Module):
    def __init__(self, cnn_out_dim=64, output_dim=35, dropout_rate=0.3):
        super().__init__()
        self.encoder = SimpleCNNEncoder(out_dim=cnn_out_dim, in_channels=4, dropout_rate=dropout_rate)
        self.decoder = MultiTaskDecoder(cnn_out_dim + 2, output_dim, dropout_rate=dropout_rate)

    def forward(self, tile, coords):
        feat = self.encoder(tile)
        x = torch.cat([feat, coords], dim=1)
        out = self.decoder(x)
        return out



model = VisionWithCoord_MTL()