import os
import inspect
import importlib.util
def load_model_classes(folder_path, module_filename="model.py"):
    """
    從指定的資料夾讀取 module_filename（預設為 model.py），
    並動態匯入該檔案中所有定義的 class。

    參數:
      folder_path (str): 指定的資料夾路徑。
      module_filename (str): 要讀取的 Python 檔名，預設為 'model.py'

    回傳:
      dict: 一個字典，鍵為 class 的名稱，值為對應的 class 物件。

    範例:
      classes = load_model_classes("/path/to/folder")
      print(classes.keys())  # 印出 model.py 中所有 class 的名稱
    """
    # 組合完整的檔案路徑
    module_path = os.path.join(folder_path, module_filename)
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"找不到檔案: {module_path}")
    
    # 使用 importlib 載入該 module
    spec = importlib.util.spec_from_file_location("model", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 取得模組中所有定義的 class（過濾掉從其他模組 import 進來的）
    classes = {
        name: cls for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    }
    
    return classes