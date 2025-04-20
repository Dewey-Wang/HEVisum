import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet50_backbone(output_dim=128):
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]  # remove last fc layer, keep conv & pool
    resnet_backbone = nn.Sequential(*modules)
    for param in resnet_backbone.parameters():
        param.requires_grad = False  # freeze if desired
    # projection maps flattened features to desired dim
    projection = nn.Sequential(
        nn.Flatten(),
        nn.Linear(resnet.fc.in_features, output_dim),
        nn.LeakyReLU(0.01)
    )
    return nn.Sequential(resnet_backbone, projection)


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


class VisionMLP_MultiTask_NoGraph(nn.Module):
    def __init__(self,
                 cnn_out_dim=128,
                 output_dim=35,
                 negative_slope=0.01):
        super().__init__()
        # three vision branches
        self.encoder_tile      = get_resnet50_backbone(cnn_out_dim)
        self.encoder_subtiles  = get_resnet50_backbone(cnn_out_dim)
        self.encoder_neighbors= get_resnet50_backbone(cnn_out_dim)

        # decoder input: 3 vision outputs + norm_coord (2 dims)
        decoder_in_dim = cnn_out_dim * 3 + 2
        self.decoder = MLPDecoder(input_dim=decoder_in_dim,
                                  output_dim=output_dim,
                                  negative_slope=negative_slope)

    def forward(self,
                tile: torch.Tensor,           # [B, 3, H, W]
                subtiles: torch.Tensor,       # [B, 9, 3, H, W]
                neighbors: torch.Tensor,      # [B, 8, 3, H, W]
                norm_coord: torch.Tensor      # [B, 2]
               ) -> torch.Tensor:
        B = tile.size(0)
        # tile branch
        f_tile = self.encoder_tile(tile)  # [B, cnn_out_dim]
        # subtiles: average over 9
        _, N, C, H, W = subtiles.shape
        f_sub = self.encoder_subtiles(
            subtiles.view(B * N, C, H, W)
        ).view(B, N, -1).mean(dim=1)      # [B, cnn_out_dim]
        # neighbors: average over 8
        _, M, C, H, W = neighbors.shape
        f_neigh = self.encoder_neighbors(
            neighbors.view(B * M, C, H, W)
        ).view(B, M, -1).mean(dim=1)     # [B, cnn_out_dim]

        # concat vision + coords
        # ensure norm_coord is float
        coords_flat = norm_coord.float()
        x = torch.cat([f_tile, f_sub, f_neigh, coords_flat], dim=1)
        return self.decoder(x)

# Example instantiation
model = VisionMLP_MultiTask_NoGraph()
