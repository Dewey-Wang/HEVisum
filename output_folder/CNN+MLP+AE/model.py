import torch
import torch.nn as nn

# ---------------------------
# 原始 CNNEncoder 部分（和你原本的相同）
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, out_dim)

    def forward(self, x):  # x: (B, 3, H, W)
        x = self.cnn(x)     # → (B, 64, 1, 1)
        x = self.flatten(x) # → (B, 64)
        x = self.linear(x)  # → (B, out_dim)
        return x

# ---------------------------
# Intermediate MLP 將 concat 後的特徵映射到 AE latent 空間（15 維）
# ---------------------------
class MLPIntermediate(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # 映射到 latent 空間，例如 15 維
        )

    def forward(self, x):
        return self.mlp(x)

# ---------------------------
# 定義和你訓練時一致的 AE decoder 架構
# ---------------------------

class AE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, 24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, output_dim)
        )
        
    def forward(self, x):
        out = self.decoder(x)
        return out, x  # 此處返回 out 與原始 latent（或你也只返回 out）



# ---------------------------
# 新模型：結合 CNNEncoder、Intermediate MLP 與預先訓練好的 AE Decoder
# ---------------------------
class VisionMLPModelWithCoordAndPretrainedAEDecoder(nn.Module):
    def __init__(self, cnn_out_dim=64, ae_latent_dim=15, output_dim=35, ae_decoder_path=None):
        """
        cnn_out_dim: CNN encoder 輸出維度  
        ae_latent_dim: AE latent 空間維度（例如 15）  
        output_dim: 最終預測輸出維度（例如 35）  
        ae_decoder_path: 預訓練 AE decoder 權重的檔案路徑，若 None 則不載入
        """
        super().__init__()
        self.encoder_spot = CNNEncoder(cnn_out_dim)
        self.encoder_subtiles = CNNEncoder(cnn_out_dim)
        self.encoder_neighbors = CNNEncoder(cnn_out_dim)

        # 輸入到 Intermediate MLP 的維度為 3 * cnn_out_dim + 2 (包含 coords)
        self.intermediate_mlp = MLPIntermediate(input_dim=cnn_out_dim*3 + 2, latent_dim=ae_latent_dim)
        
        # 建立 AE decoder
        self.ae_decoder = AE_Decoder(latent_dim=ae_latent_dim, output_dim=output_dim)
        if ae_decoder_path is not None:
            # 載入完整模型的 state_dict
            full_state = torch.load(ae_decoder_path)
            # 過濾出所有以 "decoder." 開頭的鍵
            decoder_state = {}
            for key, value in full_state.items():
                if key.startswith("decoder."):
                    # 去除 "decoder." 前綴，因為新模型的 self.ae_decoder.decoder 的鍵不帶前綴
                    new_key = key[len("decoder."):]
                    decoder_state[new_key] = value
            # 載入過濾後的權重到 self.ae_decoder.decoder
            self.ae_decoder.decoder.load_state_dict(decoder_state)
            # 如果不微調 decoder，則 freeze 它
            for param in self.ae_decoder.parameters():
                param.requires_grad = False

    def forward(self, center_tile, subtiles, neighbor_tiles, coords):
        # center_tile: (B, 3, H, W)
        # subtiles: (B, 9, 3, h, w)
        # neighbor_tiles: (B, 8, 3, H, W)
        # coords: (B, 2)
        B = center_tile.size(0)

        # 取得 center 的特徵
        f_center = self.encoder_spot(center_tile)  # (B, cnn_out_dim)

        # Subtiles 平均特徵
        B, N, C, h, w = subtiles.shape
        subtiles = subtiles.view(B * N, C, h, w)
        f_sub = self.encoder_subtiles(subtiles).view(B, N, -1).mean(dim=1)  # (B, cnn_out_dim)

        # Neighbors 平均特徵
        B, N, C, H, W = neighbor_tiles.shape
        neighbor_tiles = neighbor_tiles.view(B * N, C, H, W)
        f_neigh = self.encoder_neighbors(neighbor_tiles).view(B, N, -1).mean(dim=1)  # (B, cnn_out_dim)
        
        # Concatenate 各部分特徵與座標
        concat_features = torch.cat([f_center, f_sub, f_neigh, coords], dim=1)  # (B, 3*cnn_out_dim+2)

        # Intermediate MLP 映射到 AE latent 空間 (B, ae_latent_dim)
        latent = self.intermediate_mlp(concat_features)
        # 經由預先訓練好的 AE decoder 還原到最終輸出 (B, output_dim)
        out, _ = self.ae_decoder(latent)
        return out

# 測試新模型
if __name__ == '__main__':
    # 假設 batch size 為 4
    B = 4
    center_tile = torch.randn(B, 3, 64, 64)
    subtiles = torch.randn(B, 9, 3, 32, 32)
    neighbor_tiles = torch.randn(B, 8, 3, 64, 64)
    coords = torch.randn(B, 2)

    # 指定已訓練好 AE decoder 的路徑，假設存放在 "best_ae_decoder.pt"
    ae_decoder_path = "./spot data cleaning/AE_15_2layer/best_autoencoder.pt"
    model = VisionMLPModelWithCoordAndPretrainedAEDecoder(cnn_out_dim=64, ae_latent_dim=15, output_dim=35,
                                                         ae_decoder_path=ae_decoder_path)
    output = model(center_tile, subtiles, neighbor_tiles, coords)
    print("Output shape:", output.shape)  # 預期 (B, 35)
