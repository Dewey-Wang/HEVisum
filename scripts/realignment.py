from __future__ import annotations
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import NamedTuple

class SpotArray(NamedTuple):
    x: np.ndarray
    y: np.ndarray

    def shifted(self, dx: float, dy: float) -> "SpotArray":
        """回傳平移後的新座標（不改變原物件）。"""
        return SpotArray(self.x + dx, self.y + dy)

# ---------- 工具函式 ---------- #
def diameter_px_to_s(diameter_px: float, ax: plt.Axes) -> float:
    """
    把「像素直徑」轉成 plt.scatter 的 s 面積值 (pt²)。
    """
    if diameter_px <= 0:
        raise ValueError("diameter_px 必須為正")
    dpi = ax.figure.dpi
    diameter_pt = diameter_px * 72.0 / dpi   # px -> pt
    return diameter_pt ** 2                  # 面積 (pt²)

def align_and_plot(
    image: np.ndarray,
    spots: SpotArray,
    dx: float,
    dy: float,
    spot_diameter_px: float = 26.0,
    title: str | None = None,
    cmap: str = "gray"
) -> SpotArray:
    """
    手動平移 spots 並繪圖。
    回傳平移後的 SpotArray 供後續寫回或分析。
    """
    aligned = spots.shifted(dx, dy)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap=cmap, aspect="auto")
    s_val = diameter_px_to_s(spot_diameter_px, ax)
    ax.scatter(aligned.x, aligned.y, s=s_val, c="red", alpha=0.5)

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return aligned