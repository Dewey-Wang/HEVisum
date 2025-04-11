# operate_model.py
import inspect
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

def get_model_inputs(model, print_sig=True):
    sig = inspect.signature(model.forward)
    if print_sig:
        print(f"Model forward signature: {sig}")
    return sig

    
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

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, **kwargs):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in pbar:
        inputs, label = make_input_to_device(model, batch, device)
        optimizer.zero_grad()
        out = model(**inputs)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        
        batch_size = label.size(0)
        total_loss += loss.item() * batch_size
        
        # 儲存每個 batch 的預測與真實值，用於最後計算 Spearman 相關
        all_preds.append(out.cpu())
        all_targets.append(label.cpu())
        
        avg_loss = total_loss / ((pbar.n + 1) * dataloader.batch_size)
        pbar.set_postfix(loss=loss.item(), avg=avg_loss)
    
    # 合併所有 batch 的預測與標籤
    all_preds = torch.cat(all_preds).detach().numpy()
    all_targets = torch.cat(all_targets).detach().numpy()
    
    # 計算每個 cell type 的 Spearman 相關，並求平均
    scores = [spearmanr(all_preds[:, i], all_targets[:, i])[0] for i in range(all_preds.shape[1])]
    spearman_avg = np.nanmean(scores)
    
    avg_epoch_loss = total_loss / len(dataloader.dataset)
    return avg_epoch_loss, spearman_avg

def evaluate(model, dataloader, loss_fn, device, **kwargs):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            inputs, label = make_input_to_device(model, batch, device)
            out = model(**inputs)
            loss = loss_fn(out, label)
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            preds.append(out.cpu())
            targets.append(label.cpu())
            pbar.set_postfix(loss=loss.item())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    
    # 計算每個 cell type (gene) 的 Spearman 相關
    scores = [spearmanr(preds[:, i], targets[:, i])[0] for i in range(preds.shape[1])]
    spearman_avg = np.nanmean(scores)
    
    return total_loss / len(dataloader.dataset), spearman_avg

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


__all__ = [
    "get_model_inputs",
    "train_one_epoch",
    "evaluate",
    "predict",
    "EarlyStopping",
    "plot_losses"
]
