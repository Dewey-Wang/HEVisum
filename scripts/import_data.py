import torch
from torch.utils.data import Dataset
from .operate_model import get_model_inputs
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

def preprocess_data(data, image_keys, transform):
    """
    對指定的 data（字典）中 image_keys 的欄位使用 transform 做預處理，
    若欄位中的每筆資料為單一圖片，則逐筆轉換；
    若每筆資料本身是 list（例如 subtiles 或 neighbor_tiles），則對其中每個圖片做轉換。

    參數:
      data: 原始資料字典，格式為 {key: list([...])}，
            其中 key 可能包含圖片（例如 'center_tile', 'subtiles', ...）。
      image_keys: list 或 set，指定哪些欄位是圖片資料，需要預處理。
      transform: torchvision 轉換函數，例如 transforms.Compose([...])
      
    回傳:
      processed_data: 一個新的資料字典，其中 image_keys 中的欄位已經經過轉換，
                      其他欄位保持不變。
    """
    processed_data = {}
    for key, value in data.items():
        if key in image_keys:
            # 檢查該欄位的第一筆資料是否為 list
            if isinstance(value[0], list):
                # 對每一筆資料中的每張圖片做處理
                processed_data[key] = [
                    [transform(img) for img in sublist] for sublist in value
                ]
            else:
                # 單一圖片逐筆處理
                processed_data[key] = [transform(img) for img in value]
        else:
            # 非圖片欄位保持原樣
            processed_data[key] = value
    return processed_data

# ===========================
# 範例使用
# ===========================
if __name__ == "__main__":
    # 定義轉換流程，只針對圖片使用
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 假設原始圖片使用 PIL.Image
    dummy_img = Image.new("RGB", (64, 64))  # 建立一個 64x64 的空白 RGB 圖片

    num_samples = 100
    original_data = {
        'center_tile': [dummy_img for _ in range(num_samples)],
        'subtiles': [[dummy_img for _ in range(9)] for _ in range(num_samples)],
        'neighbor_tiles': [[dummy_img for _ in range(8)] for _ in range(num_samples)],
        'coords': [[0.5, 0.5] for _ in range(num_samples)],
        'label': [torch.randn(35, dtype=torch.float32) for _ in range(num_samples)]
    }

    image_keys = ['center_tile', 'subtiles', 'neighbor_tiles']

    # 對指定的圖片欄位做預處理
    processed_data = preprocess_data(original_data, image_keys, my_transform)

    # 接下來你可以把 processed_data 傳入 ValidatedDataset，transform 選 identity
    # 例如：
    # dataset = ValidatedDataset(
    #    data_dict=processed_data,
    #    model=DummyModel(),      # 你的模型
    #    image_keys=image_keys,
    #    transform=lambda x: x,   # identity transform，因為圖片已預處理
    #    print_sig=True
    # )



import torch
from torch.utils.data import Dataset
import inspect
import numpy as np

def convert_item(item, is_image=False):
    """
    將任意 list / numpy.ndarray / Python scalar 轉成 torch.Tensor。
    如果 is_image=True，並且 item 是 ndarray，就在 numpy 端做 H×W×3 → 3×H×W 的轉置，
    然後直接轉成 Tensor；其餘情況都走到遞歸把純 Python 結構變成 Tensor。
    """
    # 1) 如果是影像的 numpy array，直接在 numpy 端做 channel-last → channel-first
    if is_image and isinstance(item, np.ndarray):
        arr = item.astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # [H, W, 3] → [3, H, W]
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 4 and arr.shape[-1] == 3:
            # [B, H, W, 3] → [B, 3, H, W]
            arr = arr.transpose(0, 3, 1, 2)
        return torch.from_numpy(arr)

    # 2) 所有其他 numpy array，先降成純 Python list
    if isinstance(item, np.ndarray):
        try:
            item = item.tolist()
        except Exception:
            # 若 tolist() 失敗，先強制成 float32 再降 list
            item = np.asarray(item, dtype=np.float32).tolist()

    # 3) 如果是 list，遞歸 convert 然後 stack
    if isinstance(item, list):
        converted = [convert_item(elem, is_image=is_image) for elem in item]
        try:
            return torch.stack(converted)
        except Exception:
            raise ValueError(f"轉換列表中的元素失敗，列表內容: {item}")

    # 4) 如果已經是 Tensor，直接返回
    if isinstance(item, torch.Tensor):
        return item

    # 5) 剩下的都是 Python 標量（int/float/...），直接用 torch.tensor()
    try:
        return torch.tensor(item)
    except Exception:
        raise ValueError(f"無法轉換資料為 tensor，輸入資料: {item}")


class importDataset(Dataset):
    def __init__(self, data_dict, model, image_keys=None, transform=None, print_sig=False):
        self.data = data_dict
        self.image_keys = set(image_keys) if image_keys is not None else set()
        self.transform = transform if transform is not None else lambda x: x
        self.forward_keys = list(get_model_inputs(model, print_sig=print_sig).parameters.keys())

        expected_length = None
        for key, value in self.data.items():
            if expected_length is None:
                expected_length = len(value)
            if len(value) != expected_length:
                raise ValueError(f"資料欄位 '{key}' 的長度 ({len(value)}) 與預期 ({expected_length}) 不一致。")

        for key in self.forward_keys:
            if key not in self.data:
                raise ValueError(f"data_dict 缺少模型 forward 所需欄位: '{key}'。目前可用的欄位: {list(self.data.keys())}")
        if "label" not in self.data:
            raise ValueError(f"data_dict 必須包含 'label' 欄位。可用的欄位: {list(self.data.keys())}")
        if "source_idx" not in self.data:
            raise ValueError("data_dict 必須包含 'source_idx' 欄位，用於 trace 原始順序對應。")
        if "position" not in self.data:
            raise ValueError("data_dict 必須包含 'position' 欄位，用於 trace 原始順序對應。")
    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        sample = {}
        for key in self.forward_keys:
            value = self.data[key][idx]
            value = self.transform(value)
            value = convert_item(value, is_image=(key in self.image_keys))
            if isinstance(value, torch.Tensor):
                value = value.float()
            sample[key] = value

        label = self.transform(self.data["label"][idx])
        label = convert_item(label, is_image=False)
        if isinstance(label, torch.Tensor):
            label = label.float()
        sample["label"] = label

        # 加入 source_idx
        source_idx = self.data["source_idx"][idx]
        sample["source_idx"] = torch.tensor(source_idx, dtype=torch.long)
        # 加入 position （假设 data_dict 中 'position' 是 (x, y) 或 [x, y]）
        pos = self.data["position"][idx]
        sample["position"] = torch.tensor(pos, dtype=torch.float)
        return sample
    def check_item(self, idx=0, num_lines=5):
        expected_keys = self.forward_keys + ['label', 'source_idx', 'position']
        sample = self[idx]
        print(f"🔍 Checking dataset sample: {idx}")
        for key in expected_keys:
            if key not in sample:
                print(f"❌ 資料中缺少 key: {key}")
                continue
            tensor = sample[key]
            if isinstance(tensor, torch.Tensor):
                try:
                    shape = tensor.shape
                except Exception:
                    shape = "N/A"
                dtype = tensor.dtype if hasattr(tensor, "dtype") else "N/A"
                output_str = f"📏 {key} shape: {shape} | dtype: {dtype}"
                if tensor.numel() > 0:
                    try:
                        tensor_float = tensor.float()
                        mn = tensor_float.min().item()
                        mx = tensor_float.max().item()
                        mean = tensor_float.mean().item()
                        std = tensor_float.std().item()
                        output_str += f" | min: {mn:.3f}, max: {mx:.3f}, mean: {mean:.3f}, std: {std:.3f}"
                    except Exception:
                        output_str += " | 無法計算統計數據"
                print(output_str)
                if key not in self.image_keys:
                    if tensor.ndim == 0:
                        print(f"--- {key} 資料為純量:", tensor)
                    elif tensor.ndim == 1:
                        print(f"--- {key} head (前 {num_lines} 個元素):")
                        print(tensor[:num_lines])
                    else:
                        print(f"--- {key} head (前 {num_lines} 列):")
                        print(tensor[:num_lines])
            else:
                # 如果 position 存的是 list/tuple/etc，也会走这里
                print(f"📏 {key} (非 tensor 資料):", tensor)
        print("✅ All checks passed!")



import os
import torch
import random

def load_all_tile_data(folder_path,
                       model,
                       fraction: float = 1.0,
                       shuffle : bool = False):
    """
    回傳 dict，其中包含：
        - Model forward() 需要的欄位
        - 'label'
        - 'slide_idx'    ← for GroupKFold
        - 'source_idx'   ← 從 .pt 檔案內部讀取
    """
    sig            = get_model_inputs(model, print_sig=False)
    fwd_keys       = list(sig.parameters.keys())
    required_keys  = set(fwd_keys + ['label', 'slide_idx', 'position'])   # include slide_idx
    keep_meta_keys = required_keys.union({'source_idx'})

    pt_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.pt'))
    N        = len(pt_files)
    keep_n   = max(1, int(N * fraction))
    pt_files = random.sample(pt_files, keep_n) if shuffle else pt_files[-keep_n:]

    data_dict = {k: [] for k in keep_meta_keys}

    for fname in pt_files:
        fpath = os.path.join(folder_path, fname)
        d = torch.load(fpath, map_location='cpu')

        # ✅ 優先從檔案內部讀取 source_idx
        if 'source_idx' in d:
            data_dict['source_idx'].append(d['source_idx'])
        else:
            data_dict['source_idx'].append(fname)  # optional fallback

        # ➋ 補入其他欄位
        for k in required_keys:
            data_dict[k].append(d.get(k, None))

    return data_dict




def load_node_feature_data(pt_path: str, model, num_cells: int = 35) -> dict:
    """
    根據 model.forward 的參數，自動載入 .pt 檔案中所需欄位，
    並自動補 'label'（若不存在）為 0 tensor。
    支援自動讀取 'position' 和 'source_idx' 欄位（若 forward 有用到）。

    返回：
      dict: key 對應 forward 的參數名 + label, position, source_idx（如需）
    """
    import torch
    import inspect

    raw = torch.load(pt_path, map_location="cpu")

    # 模型需要哪些參數？
    sig = inspect.signature(model.forward)
    param_names = [p for p in sig.parameters if p != "self"]
    param_names.append('source_idx')
    param_names.append('position')

    out = {}
    for name in param_names:
        # a) 直接同名
        if name in raw:
            out[name] = raw[name]
            continue
        # b) name + 's'（plural）
        if name + "s" in raw:
            out[name] = raw[name + "s"]
            continue
        # c) 模糊匹配
        cands = [k for k in raw if name in k or k in name]
        if len(cands) == 1:
            out[name] = raw[cands[0]]
            continue
        raise KeyError(f"❌ 無法找到 '{name}'，raw keys: {list(raw.keys())}")

    # 推斷 batch 大小
    dataset_size = None
    for v in out.values():
        if hasattr(v, "__len__"):
            dataset_size = len(v)
            print(f"⚠️ 從 '{type(v)}' 推斷樣本數量: {dataset_size}")
            break
    if dataset_size is None:
        raise RuntimeError("❌ 無法推斷樣本數量。")

    # 預設補上 label
    out["label"] = raw.get("label", torch.zeros((dataset_size, num_cells), dtype=torch.float32))

    # ✅ 額外補上 position 和 source_idx（如有）
    for meta_key in ["position", "source_idx"]:
        if meta_key in raw:
            out[meta_key] = raw[meta_key]

    return out

# ==============================================
# 範例使用
# ==============================================
if __name__ == "__main__":
    # 定義一個模型，假設 forward 所需參數為 center_tile, subtiles, neighbor_tiles, coords
    class DummyModel:
        def forward(self, center_tile, subtiles, neighbor_tiles, coords):
            pass

    model = DummyModel()

    # 模擬 100 筆資料，每筆資料的圖片原始格式為 channel-first (3, H, W)
    num_samples = 100
    dummy_center = [torch.randn(3, 64, 64) for _ in range(num_samples)]
    dummy_subtiles = [[torch.randn(3, 32, 32) for _ in range(9)] for _ in range(num_samples)]
    dummy_neighbor = [[torch.randn(3, 64, 64) for _ in range(8)] for _ in range(num_samples)]
    dummy_coords = [[0.5, 0.5] for _ in range(num_samples)]
    dummy_label = [torch.randn(35, dtype=torch.float32) for _ in range(num_samples)]  # 假設 label 長度為 35

    # 建立資料字典，key 命名必須與 DummyModel.forward 相符，再加上 label
    data = {
        'center_tile': dummy_center,
        'subtiles': dummy_subtiles,
        'neighbor_tiles': dummy_neighbor,
        'coords': dummy_coords,
        'label': dummy_label
    }

    # 指定哪些欄位為圖片資料
    image_keys = ['center_tile', 'subtiles', 'neighbor_tiles']

    # 建立 ValidatedDataset
    dataset = ValidatedDataset(
        data_dict=data,
        model=model,
        image_keys=image_keys,
        transform=lambda x: x,  # identity transform
        print_sig=True
    )

    # 取得第一筆資料
    sample = dataset[0]
    # 印出組合資料的順序，其順序為 (center_tile, subtiles, neighbor_tiles, coords, label)
    print("取得的 sample 資料順序：")
    print("center_tile shape:", sample[0].shape)
    print("subtiles shape:", sample[1].shape)
    print("neighbor_tiles shape:", sample[2].shape)
    print("coords:", sample[3])
    print("label shape:", sample[4].shape)

    # 檢查第一筆資料
    dataset.check_item(idx=0)
    
    
    
