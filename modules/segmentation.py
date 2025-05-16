# modules/segmentation.py

import numpy as np
import torch
import cv2
from typing import List, Dict
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class ObjectSegmentor:
    """
    使用 SAM (Segment Anything Model) 對影像進行自動物件分割。
    適用於 CPU-only 環境（如 Mac M3）。
    """

    def __init__(self, model_type: str = "vit_b", sam_checkpoint: str = "sam_vit_b.pth", device: str = "cpu"):
        """
        初始化 Segment Anything 模型

        Args:
            model_type: SAM 模型類型（vit_b、vit_h、vit_l）
            sam_checkpoint: 模型權重路徑
            device: 使用的裝置（"cpu" 或 "mps"）
        """
        self.device = torch.device(device)

        # 載入 SAM 模型架構與權重
        print(f"Loading SAM model: {model_type} on {self.device} ...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)

        # 初始化自動遮罩生成器
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        print("SAM 模型載入完成")

    def segment_objects(self, image: np.ndarray) -> List[Dict]:
        """
        對單張影像執行自動物件遮罩生成。

        Args:
            image: 輸入影像（BGR numpy array）

        Returns:
            mask_list: 每個遮罩是 dict，包含 segmentation mask、score 等欄位
        """
        # SAM 使用 RGB 影像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 執行遮罩生成
        mask_list = self.mask_generator.generate(rgb_image)

        # 可選：依據面積排序（大面積優先）
        mask_list = sorted(mask_list, key=lambda x: x['area'], reverse=True)

        return mask_list

    def generate_occlusion_mask(self, mask_list: List[Dict], image_shape: tuple) -> np.ndarray:
        """
        根據物件遮罩，產生總遮擋遮罩（用於阻擋 3D 物件穿越前景）

        Args:
            mask_list: SAM 傳回的 mask 資料
            image_shape: 原始圖像形狀 (H, W, 3)

        Returns:
            occlusion_mask: 單通道遮罩，前景為 255，背景為 0
        """
        height, width = image_shape[:2]
        occlusion_mask = np.zeros((height, width), dtype=np.uint8)

        for i, mask_dict in enumerate(mask_list):
            mask = mask_dict['segmentation']
            occlusion_mask[mask] = 255  # 將所有前景合併

        return occlusion_mask
