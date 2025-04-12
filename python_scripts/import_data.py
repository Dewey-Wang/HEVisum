import torch
from torch.utils.data import Dataset
from .operate_model import get_model_inputs
from torchvision import transforms
from PIL import Image
import torch

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

def convert_item(item, is_image=False):
    """
    將 item 轉換成 torch.tensor，並確保圖片資料是 [C, H, W] 格式。
    """
    if isinstance(item, list):
        try:
            converted_list = [convert_item(elem, is_image=is_image) for elem in item]
            return torch.stack(converted_list)
        except Exception as e:
            raise ValueError(f"轉換列表中的元素失敗，列表內容: {item}") from e

    if not isinstance(item, torch.Tensor):
        try:
            item = torch.tensor(item)
        except Exception as e:
            raise ValueError(f"無法轉換資料為 tensor，輸入資料: {item}") from e

    if is_image:
        # 強制圖片轉為 [C, H, W] 格式
        if item.ndim == 3:
            if item.shape[-1] == 3:  # 若為 [H, W, 3] → permute to [3, H, W]
                item = item.permute(2, 0, 1)
    return item


class importDataset(Dataset):
    def __init__(self, data_dict, model, image_keys=None, transform=None, print_sig=False):
        """
        參數:
          data_dict: dict，每個 key 對應一個 list，其中 list[i] 表示第 i 筆資料的該欄位內容。
                     資料欄位必須包含模型 forward 所需的參數名稱，另外還必須有 "label" 欄位。
          model: 模型物件，將根據 model.forward 的參數順序決定輸入組合順序。
          image_keys: list 或 set，標記哪些欄位需要視為圖片資料，處理時會轉換為 channel-last 格式。
          transform: 每筆資料的轉換函數，若未指定則為 identity function。
          print_sig: 是否印出模型 forward 函式的簽名。
        """
        self.data = data_dict
        self.image_keys = set(image_keys) if image_keys is not None else set()
        self.transform = transform if transform is not None else lambda x: x

        self.forward_keys = list(get_model_inputs(model, print_sig=print_sig).parameters.keys())

        # 資料長度檢查：所有欄位的 list 長度必須一致
        expected_length = None
        for key, value in self.data.items():
            if expected_length is None:
                expected_length = len(value)
            if len(value) != expected_length:
                raise ValueError(f"資料欄位 '{key}' 的長度 ({len(value)}) 與預期 ({expected_length}) 不一致。")
        
        # 檢查必要欄位：必須包含模型 forward 所需欄位與 'label'
        for key in self.forward_keys:
            if key not in self.data:
                raise ValueError(f"data_dict 缺少模型 forward 所需欄位: '{key}'。目前可用的欄位: {list(self.data.keys())}")
        if "label" not in self.data:
            raise ValueError(f"data_dict 必須包含 'label' 欄位。可用的欄位: {list(self.data.keys())}")

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        sample = {}
        # 依照模型 forward 的順序依次取出資料並處理（保持 key 名稱不變）
        for key in self.forward_keys:
            try:
                value = self.data[key][idx]
            except IndexError as e:
                raise IndexError(f"索引 {idx} 超出欄位 '{key}' 的資料範圍，共有 {len(self.data[key])} 筆資料。") from e
            try:
                value = self.transform(value)
            except Exception as e:
                raise ValueError(f"轉換欄位 '{key}' 的資料時出錯，資料: {value}") from e
            try:
                if key in self.image_keys:
                    value = convert_item(value, is_image=True)
                else:
                    value = convert_item(value, is_image=False)
            except Exception as e:
                raise ValueError(f"轉換欄位 '{key}' 的資料為 tensor 時出錯，資料內容: {value}") from e
            # 轉換成 float32
            if isinstance(value, torch.Tensor):
                value = value.float()
            sample[key] = value
        
        # 處理 label
        try:
            label = self.data["label"][idx]
        except IndexError as e:
            raise IndexError(f"索引 {idx} 超出 'label' 欄位的資料範圍，共有 {len(self.data['label'])} 筆資料。") from e
        try:
            label = self.transform(label)
        except Exception as e:
            raise ValueError(f"轉換 'label' 資料時出錯，資料內容: {label}") from e
        try:
            label = convert_item(label, is_image=False)
        except Exception as e:
            raise ValueError(f"轉換 'label' 為 tensor 時出錯，資料內容: {label}") from e
        if isinstance(label, torch.Tensor):
            label = label.float()
        sample["label"] = label

        return sample

    def check_item(self, idx=0, num_lines=5):
        """
        檢查第 idx 筆資料中每個欄位的詳細資訊。
        對每個欄位（依據 forward_keys 加上 'label'）印出：
          - shape 與 dtype，
          - 如果是 tensor，印出 min, max, mean, std（計算時強制轉為 float32），
          - 對於非圖片資料，印出該 tensor 前 num_lines 列/元素的內容。
        """
        expected_keys = self.forward_keys + ['label']
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
                        # 將 tensor 轉成 float32 計算統計數據
                        tensor_float = tensor.float()
                        mn = tensor_float.min().item()
                        mx = tensor_float.max().item()
                        mean = tensor_float.mean().item()
                        std = tensor_float.std().item()
                        output_str += f" | min: {mn:.3f}, max: {mx:.3f}, mean: {mean:.3f}, std: {std:.3f}"
                    except Exception:
                        output_str += " | 無法計算統計數據"
                print(output_str)
                # 若非圖片資料，印出前 num_lines 列/元素
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
                print(f"📏 {key} (非 tensor 資料):", tensor)
        print("✅ All checks passed!")



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
    
    
    
