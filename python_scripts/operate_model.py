# operate_model.py
import inspect
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import matplotlib.gridspec as gridspec
from captum.attr import IntegratedGradients
import math

def get_model_inputs(model, print_sig=True):
    sig = inspect.signature(model.forward)
    if print_sig:
        print(f"Model forward signature: {sig}")
    return sig

def automate_ig_analysis(model, batch, target_index=0, n_steps=50, image_dict=None):
    """
    自動計算並視覺化模型對給定 batch 輸入的 Integrated Gradients。

    參數:
        model: 已初始化且設為 eval 模式的模型
        batch: 從 DataLoader 中取得的一個 batch，
               格式需符合模型 forward 輸入順序，例如 tuple/list，
               順序應與 get_model_inputs(model) 所顯示的參數一致。
        target_index: 欲計算歸因的目標輸出 index (例如某個 cell type 的 index)
        n_steps: IG 計算時的步數 (步數越多，近似越精確，但計算成本增加)
        image_dict: dict，指定每個參數名稱是否為 image 類型，
                    例如 {'tile': True, 'subtiles': True, 'neighbors': True, 'coords': False}，
                    若沒提供，則預設全部視為非 image。
                    
    返回:
        一個 dict，鍵為輸入參數名稱，值為對應的 attribution tensor（仍保留 batch 維度）。
    """
    model.eval()
    
    # 取得模型 forward 的參數名稱
    sig = get_model_inputs(model, print_sig=True)
    param_names = list(sig.parameters.keys())
    if 'self' in param_names:
        param_names.remove('self')
    print("Model input parameters:", param_names)
    
    # 根據 param_names 從 batch 中依序取得輸入
    if isinstance(batch, dict):
        inputs_list = [batch[name] for name in param_names]
    else:
        # 假設是 tuple/list，順序與 param_names 一致
        inputs_list = list(batch)
    
    # 取 batch 中第一筆樣本作為範例進行 IG 分析，保留 batch 維度 = 1
    inputs_sample = [inp[0:1] for inp in inputs_list]
    # 建立 baseline：採用全零張量
    baselines = [torch.zeros_like(inp) for inp in inputs_sample]

    # 使用 Captum 的 IntegratedGradients 計算歸因
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs=tuple(inputs_sample),
        baselines=tuple(baselines),
        target=target_index,
        n_steps=n_steps
    )
    
    # 組成輸出 dict，參數名稱對應其 attribution (保持 batch 維度)
    attr_dict = dict(zip(param_names, attributions))
    
    # 開始整合視覺化: 針對每個參數各自建立一個區塊
    num_params = len(param_names)
    fig = plt.figure(figsize=(10, num_params * 5))
    outer_gs = gridspec.GridSpec(num_params, 1, hspace=0.5)
    
    for i, name in enumerate(param_names):
        # 移除 batch 維度
        attr = attr_dict[name].squeeze(0)
        attr_np = attr.detach().cpu().numpy()
        
        ax = None  # 初始化
        # 判斷是否為 image 資料；若 image_dict 未提供則預設 False
        is_image = image_dict.get(name, False) if image_dict is not None else False
        
        if is_image:
            # 處理 image 輸入
            if attr_np.ndim == 3:
                # 形狀 (C, H, W) --> 單張圖片
                ax = fig.add_subplot(outer_gs[i])
                avg_attr = np.mean(attr_np, axis=0)
                im = ax.imshow(avg_attr, cmap='viridis')
                ax.set_title(f"Attribution for {name} (avg over channels)")
                fig.colorbar(im, ax=ax)
            elif attr_np.ndim == 4:
                # 形狀 (N, C, H, W) --> 多張圖片，N 為圖片數量
                N = attr_np.shape[0]
                grid_cols = math.ceil(math.sqrt(N))
                grid_rows = math.ceil(N / grid_cols)
                # 建立內層 GridSpec 於外層位置
                sub_gs = gridspec.GridSpecFromSubplotSpec(grid_rows, grid_cols, subplot_spec=outer_gs[i], 
                                                          wspace=0.3, hspace=0.3)
                # 迭代顯示每一張圖片
                for j in range(N):
                    ax_sub = fig.add_subplot(sub_gs[j])
                    avg_attr = np.mean(attr_np[j], axis=0)
                    im = ax_sub.imshow(avg_attr, cmap='viridis')
                    ax_sub.set_title(f"{name} Image {j}")
                    ax_sub.axis('off')
                    fig.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)
            else:
                ax = fig.add_subplot(outer_gs[i])
                ax.text(0.5, 0.5, f"無法視覺化 shape: {attr_np.shape}", 
                        horizontalalignment='center', fontsize=12)
                ax.set_title(f"Attribution for {name}")
        else:
            # 非 image 資料：以長條圖呈現(flatten後)
            ax = fig.add_subplot(outer_gs[i])
            flat_attr = attr_np.flatten()
            ax.bar(range(len(flat_attr)), flat_attr, color='skyblue')
            ax.set_title(f"Attribution for {name} (flattened)")
            
    plt.tight_layout()
    plt.show()
    
    return attr_dict

  
def make_input_to_device(model, batch, device, label_key="label", need_label=True):
    """
    根據 model.forward 的參數列表，自動從 batch 中提取輸入，
    並搬移到指定 device。如果參數 label_key 有設定，則也會嘗試提取該 key 的資料。
    
    若在 batch 中找不到 label_key 的資料，則拋出 KeyError，
    提示「你的 dataset 裡面沒有 label，或是你要自己更改成你預測的東西」。
    """
    # 取得 model.forward 的參數簽名（不包含 self）
    sig = get_model_inputs(model, print_sig=False)
    # 儲存要傳給 forward 的參數
    inputs = {}
    for name, _ in sig.parameters.items():
        if name == 'self':
            continue

        if name in batch:
            inputs[name] = batch[name].to(device)
        else:
            raise KeyError(f"Model 需要 {sig}. Batch 中找不到對應 '{name}' 的資料，請確認 dataloader 的輸出鍵與 model.forward 參數名稱一致。")
    
    # 對 label_key 作特別處理：若 label_key 已定義但不在 inputs 中
    if need_label:
        if label_key in batch:
            label = batch[label_key].to(device)
        else:
            raise KeyError(f"你的 dataset 裡面沒有 '{label_key}' 資料，或是你要自己更改成你預測的東西。")
    else:
        label = None
    return inputs, label



def predict(model, dataloader, device, **kwargs):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 同樣利用 get_model_inputs 提取輸入
            inputs, _ = make_input_to_device(model, batch, device, need_label=False)
            out = model(**inputs)
            all_preds.append(out.cpu())

    
    return torch.cat(all_preds).numpy()

class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True



def plot_losses(train_losses, val_losses, ax=None, title="Training vs Validation Loss"):
    """
    繪製 train/validation loss 曲線
    :param train_losses: list 或 array，訓練 loss 數值
    :param val_losses: list 或 array，驗證 loss 數值
    :param ax: (optional) 傳入 matplotlib Axes 物件，如果沒有提供，則自行創建一個圖表
    :param title: (string) 圖表標題
    """
    # 如果沒有 ax，則新建立一個 figure 與 ax，並用 clear_output 以避免重疊
    if ax is None:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(train_losses, label="Train Loss", marker='o')
    ax.plot(val_losses, label="Val Loss", marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_title(title)
# 收集資料
def plot_per_cell_metrics(mse_vals, spearman_vals, cell_names=None, top_k=5, ax_mse=None, ax_spearman=None):
    """
    繪製每個 cell type 的 MSE 和 Spearman 柱狀圖。
    可指定兩個 axes，用於自訂排版（和 plot_losses 對齊）。

    Params:
        mse_vals: array-like, 每個 cell type 的 MSE
        spearman_vals: array-like, 每個 cell type 的 Spearman
        cell_names: list of str, cell type 名稱（預設為 C1 ~ C35）
        top_k: int, 要標記的最佳/最差項目數量
        ax_mse: matplotlib Axes，用來畫 MSE 圖
        ax_spearman: matplotlib Axes，用來畫 Spearman 圖
    """
    if cell_names is None:
        cell_names = [f"C{i+1}" for i in range(len(mse_vals))]

    sorted_idx_mse = np.argsort(mse_vals)
    sorted_idx_spearman = np.argsort(spearman_vals)

    if ax_mse is None or ax_spearman is None:
        fig, (ax_mse, ax_spearman) = plt.subplots(1, 2, figsize=(14, 4))

    # Left: MSE per gene
    ax_mse.clear()
    ax_mse.bar(cell_names, mse_vals, color='skyblue')
    ax_mse.set_title("Per-cell MSE")
    ax_mse.tick_params(axis='x', rotation=45)
    for i in sorted_idx_mse[:top_k]:
        ax_mse.text(i, mse_vals[i] + 0.01, "↓", ha='center', color='red')
    for i in sorted_idx_mse[-top_k:]:
        ax_mse.text(i, mse_vals[i] + 0.01, "↑", ha='center', color='green')

    # Right: Spearman per gene
    ax_spearman.clear()
    ax_spearman.bar(cell_names, spearman_vals, color='orange')
    ax_spearman.set_title("Per-cell Spearman")
    ax_spearman.tick_params(axis='x', rotation=45)
    for i in sorted_idx_spearman[:top_k]:
        ax_spearman.text(i, spearman_vals[i] + 0.01, "↓", ha='center', color='red')
    for i in sorted_idx_spearman[-top_k:]:
        ax_spearman.text(i, spearman_vals[i] + 0.01, "↑", ha='center', color='green')

    if ax_mse is None or ax_spearman is None:
        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

# =======================
# 提供的函數定義
# =======================

def get_alpha(epoch, initial_alpha=0.3, final_alpha=0.8, target_epoch=50, method="linear"):
    """
    根據目前 epoch 和指定方法計算 alpha 值。
    當 epoch >= target_epoch 時，直接返回 final_alpha。

    :param epoch: 當前 epoch（整數）
    :param initial_alpha: 初始 alpha 值
    :param final_alpha: 最終 alpha 值（target_epoch 時達到）
    :param target_epoch: 希望在此 epoch 時 alpha 達到 final_alpha
    :param method: 調度方法，可選 "linear", "exponential", "cosine", "log"
    :return: 當前 epoch 的 alpha 值
    """
    if not isinstance(epoch, (int, float)):
        raise TypeError(f"`epoch` must be int or float, but got {type(epoch)}")

    if epoch >= target_epoch:
        return final_alpha

    progress = epoch / target_epoch  # 進度比例 (0 ~ 1)
    if method == "linear":
        # 線性上升：epoch==target_epoch 時 alpha = final_alpha
        return initial_alpha + (final_alpha - initial_alpha) * progress
    elif method == "exponential":
        # 指數型上升：初期變化較慢，後期較快
        return initial_alpha * ((final_alpha / initial_alpha) ** progress)
    elif method == "cosine":
        # Cosine 衰減：用 cosine 曲線平滑過渡
        # 當 epoch==0 時：cos(0)=1, alpha = final_alpha + (initial_alpha-final_alpha)*1 = initial_alpha
        # 當 epoch==target_epoch 時：cos(pi)= -1, alpha = final_alpha + (initial_alpha-final_alpha)*0 = final_alpha
        return final_alpha + (initial_alpha - final_alpha) * (np.cos(np.pi * progress) + 1) / 2
    elif method == "log":
        # 對數型：使用 log₂ 過渡，當 epoch==target_epoch 時正好達到 final_alpha
        return initial_alpha + (final_alpha - initial_alpha) * np.log2(1 + epoch) / np.log2(1 + target_epoch)
    else:
        raise ValueError(f"Unknown method: {method}")

def spearman_corr_loss(pred, target, eps=1e-8):
    """
    計算 batch-wise Spearman correlation 損失。
    Returns: 1 - mean_corr, corr 越高 → loss 越小
    pred, target shape: (B, num_cells)
    """
    # 取得各自的 rank，argsort 兩次可把數值變為 rank
    pred_rank = pred.argsort(dim=1).argsort(dim=1).float()
    target_rank = target.argsort(dim=1).argsort(dim=1).float()

    # 每一筆資料都去中心化
    pred_rank = pred_rank - pred_rank.mean(dim=1, keepdim=True)
    target_rank = target_rank - target_rank.mean(dim=1, keepdim=True)

    # 計算 batch 中每筆的 spearman correlation
    corr = (pred_rank * target_rank).sum(dim=1) / (
        pred_rank.norm(dim=1) * target_rank.norm(dim=1) + eps
    )
    # 取 1 - 平均 corr 當作 loss
    return 1.0 - corr.mean()

def hybrid_weighted_spearman_loss(pred, target, alpha=0.5, beta=1.0):
    """
    將 z-score weighted MSE 與 Spearman Loss 結合的 Hybrid Loss。
    
    pred, target shape: (B, num_cells)
    alpha: Spearman loss 的權重比例 (0~1 之間)
    beta: 加權倍數，越大表示對高 z-score 區塊懲罰越強
    """
    # 權重：讓 z-score 絕對值越大者 → 權重越大
    weighting = 1.0 + beta * target.abs()
    weighted_mse = weighting * (pred - target)**2
    w_mse = weighted_mse.mean()

    # Spearman correlation loss
    s_loss = spearman_corr_loss(pred, target)

    # 最終 Loss = (1 - alpha)*MSE + alpha*Spearman
    loss = (1 - alpha) * w_mse + alpha * s_loss
    return loss

# =======================
# 改進版 train_one_epoch 與 evaluate
# =======================

def train_one_epoch(model, dataloader, optimizer, device, current_epoch,
                    initial_alpha=0.3, final_alpha=0.9, target_epoch=15, beta=1.0, method="linear"):
    """
    訓練一個 epoch，使用動態 alpha 計算 hybrid loss。
    僅保留必要參數：
      - model, dataloader, optimizer, device, current_epoch, total_epochs
    另外，使用預設: initial_alpha=0.3, final_alpha=0.8, 當 epoch >= target_epoch (預設50) 後 alpha 固定為 final_alpha，
      且 beta 固定為 1.0，調度方法由 method 決定。
    
    :return: 平均 loss 與平均 Spearman 相關性
    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    # 根據當前 epoch 計算動態 alpha
    alpha = get_alpha(current_epoch, initial_alpha, final_alpha, target_epoch, method)
    
    for batch in pbar:
        inputs, label = make_input_to_device(model, batch, device)
        optimizer.zero_grad()
        out = model(**inputs)
        loss = hybrid_weighted_spearman_loss(out, label, alpha=alpha, beta=beta)
        loss.backward()
        optimizer.step()
        
        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        
        all_preds.append(out.cpu())
        all_targets.append(label.cpu())
        
        avg_loss = total_loss / ((pbar.n + 1) * dataloader.batch_size)
        pbar.set_postfix(loss=loss.item(), avg=avg_loss)
    
    all_preds = torch.cat(all_preds).detach().numpy()
    all_targets = torch.cat(all_targets).detach().numpy()
    
    # 計算每個 cell type 的 Spearman 相關，取平均
    scores = [spearmanr(all_preds[:, i], all_targets[:, i])[0] for i in range(all_preds.shape[1])]
    spearman_avg = np.nanmean(scores)
    
    avg_epoch_loss = total_loss / len(dataloader.dataset)
    return avg_epoch_loss, spearman_avg


def evaluate(model, dataloader, device, current_epoch,
             initial_alpha=0.3, final_alpha=0.8, target_epoch=15, beta=1.0, method="linear"):
    """
    評估函數：使用與 train 一致的動態 alpha 計算 hybrid loss。
    僅保留必要參數：
    
    :return: 平均 loss, 平均 Spearman, 每個 cell type 的 MSE 與 Spearman 值
    """
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    total_mse = torch.zeros(35).to(device)  # 假設有 35 個 cell types
    n_samples = 0
    
    # 根據當前 epoch 計算動態 alpha
    alpha = get_alpha(current_epoch, initial_alpha, final_alpha, target_epoch, method)
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            inputs, label = make_input_to_device(model, batch, device)
            out = model(**inputs)
            loss = hybrid_weighted_spearman_loss(out, label, alpha=alpha, beta=beta)
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            preds.append(out.cpu())
            targets.append(label.cpu())
            loss_per_cell = ((out - label) ** 2).sum(dim=0)  # (35,)
            total_mse += loss_per_cell
            n_samples += batch_size
            pbar.set_postfix(loss=loss.item())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse_per_cell = (total_mse / n_samples).cpu().numpy()
    spearman_per_cell = [spearmanr(preds[:, i], targets[:, i])[0] for i in range(preds.shape[1])]
    spearman_avg = np.nanmean(spearman_per_cell)
    
    avg_epoch_loss = total_loss / n_samples
    return avg_epoch_loss, spearman_avg, mse_per_cell, spearman_per_cell


__all__ = [
    "get_model_inputs",
    "train_one_epoch",
    "evaluate",
    "predict",
    "EarlyStopping",
    "plot_losses"
]
