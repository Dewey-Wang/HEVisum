import torch
import torch.nn as nn
import torchvision.models as models

class GraphFeatureEncoder(nn.Module):
    """
    Encode (norm_coord, node_feat, edge_feat) into a fixed-size vector.
    Assumes edge_feat[..., -1] is the weight for each neighbor.
    """
    def __init__(self,
                 node_feat_dim: int,
                 edge_feat_dim: int,
                 coord_dim: int = 2,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 negative_slope: float = 0.01):
        super().__init__()
        # embed the per-node features
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope),
        )
        # embed the aggregated edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim - 1, hidden_dim),  # drop the weight dim
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope),
        )
        # final fusion MLP
        fusion_in = coord_dim + hidden_dim + hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self,
                norm_coord: torch.Tensor,
                node_feat: torch.Tensor,
                adj_list,                  # unused, since edge_feat has weights
                edge_feat: torch.Tensor) -> torch.Tensor:
        """
        norm_coord: [B,2]
        node_feat:  [B, F_node]
        edge_feat:  [B, K, F_edge] where edge_feat[..., -1] = weight
        adj_list:   list-of-lists (ignored here)
        """
        B, K, F = edge_feat.shape

        # 1) node embedding
        h_node = self.node_mlp(node_feat)           # [B, hidden_dim]

        # 2) weighted aggregate of edge features (drop last dim = weight)
        feats= edge_feat[..., :-1]                  # [B, K, F_edge-1]
        weights = edge_feat[..., -1].unsqueeze(-1)   # [B, K, 1]
        weighted_sum = (feats * weights).sum(dim=1)  # [B, F_edge-1]
        norm = weights.sum(dim=1).clamp(min=1e-6)    # [B,1]
        agg = weighted_sum / norm                    # [B, F_edge-1]

        h_edge = self.edge_mlp(agg)                  # [B, hidden_dim]

        # 3) fuse with norm_coord
        x = torch.cat([norm_coord, h_node, h_edge], dim=1)  # [B, coord+hid+hid]
        return self.fusion_mlp(x)     


def get_resnet50_backbone(output_dim=128):
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]  # remove last fc layer
    resnet_backbone = nn.Sequential(*modules)
    for param in resnet_backbone.parameters():
        param.requires_grad = False  # freeze if retraining is not desired
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


class VisionMLP_MultiTask(nn.Module):
    def __init__(self,
                 cnn_out_dim=128,
                 graph_out_dim=32,   # 最終 graph 特徵維度
                 output_dim=35,
                 negative_slope=0.01):
        super().__init__()
        # 三條 vision 分支不動
        self.encoder_tile       = get_resnet50_backbone(cnn_out_dim)
        self.encoder_subtiles   = get_resnet50_backbone(cnn_out_dim)
        self.encoder_neighbors = get_resnet50_backbone(cnn_out_dim)

        # 重新設定 graph encoder 的輸入維度
        NODE_FEAT_DIM = 12     # 你的 node_feat 長度
        EDGE_FEAT_DIM = 5      # 每條邊 5 維 (distance, dx, dy, angle, weight)
        self.encoder_graph = GraphFeatureEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            coord_dim=2,       # norm_coord = 2 維
            hidden_dim=64,
            output_dim=graph_out_dim,
            negative_slope=negative_slope
        )

        # decoder 現在接 3 條 CNN output + 一條 graph embedding
        decoder_in_dim = cnn_out_dim * 3 + graph_out_dim
        self.decoder = MLPDecoder(input_dim=decoder_in_dim,
                                  output_dim=output_dim,
                                  negative_slope=negative_slope)

    def forward(self,
                tile: torch.Tensor,           # [B,  3, H, W]
                subtiles: torch.Tensor,       # [B,  9, 3, H, W]
                neighbors: torch.Tensor,      # [B,  8, 3, H, W]  ← 這是影像鄰居
                norm_coord: torch.Tensor,     # [B,  2]
                node_feat: torch.Tensor,      # [B, 12]
                adj_list,                     # list-of-lists (每筆長度 15)，GraphEncoder 接口保留
                edge_feat: torch.Tensor       # [B, 15, 5]
               ) -> torch.Tensor:

        B = tile.size(0)
        # —— CNN 分支 —— 
        f_tile = self.encoder_tile(tile)  # [B, cnn_out_dim]
        
        # subtiles 平均池化
        _, N, C, H, W = subtiles.shape   # N=9
        f_sub = self.encoder_subtiles(
            subtiles.view(B*N, C, H, W)
        ).view(B, N, -1).mean(1)         # [B, cnn_out_dim]

        # image neighbors 平均池化
        _, M, C, H, W = neighbors.shape  # M=8
        f_neigh = self.encoder_neighbors(
            neighbors.view(B*M, C, H, W)
        ).view(B, M, -1).mean(1)         # [B, cnn_out_dim]

        # —— graph features 分支 ——
        # norm_coord [B,2], node_feat [B,12], edge_feat [B,15,5]
        f_graph = self.encoder_graph(norm_coord, node_feat, adj_list, edge_feat)  # [B, graph_out_dim]

        # 串起所有特徵
        x = torch.cat([f_tile, f_sub, f_neigh, f_graph], dim=1)  # [B, cnn_out_dim*3 + graph_out_dim]
        return self.decoder(x)



model = VisionMLP_MultiTask()
