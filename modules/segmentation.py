# modules/segmentation.py

import numpy as np
import torch
import cv2
import os
from typing import List, Dict, Tuple
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class ObjectSegmentor:
    """
    使用 SAM 進行自動物件遮罩分割，支援 CPU / MPS。
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        sam_checkpoint: str = "models/sam_vit_b.pth",
        device: str = "cpu"
    ):
        """
        初始化 SAM 模型與遮罩生成器

        Args:
            model_type: SAM 模型類型，如 vit_b、vit_h
            sam_checkpoint: 權重檔案路徑
            device: "cpu" 或 "mps"
        """
        self.device = torch.device(device)
        print(f"🔧 載入 SAM 模型 ({model_type}) 至 {device}...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        print("✅ SAM 初始化完成")

    def segment_objects(self, image: np.ndarray) -> List[Dict]:
        """
        使用 SAM 產生所有遮罩

        Args:
            image: BGR numpy 圖像

        Returns:
            mask_list: List of mask dicts
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_list = self.mask_generator.generate(rgb_image)
        mask_list = sorted(mask_list, key=lambda x: x['area'], reverse=True)
        return mask_list

    def generate_occlusion_mask(self, mask_list: List[Dict], image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        將所有物件遮罩合併成一張「前景遮擋遮罩」

        Args:
            mask_list: SAM 遮罩清單
            image_shape: 原圖尺寸

        Returns:
            單通道遮罩 (255=前景，0=背景)
        """
        h, w = image_shape[:2]
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_dict in mask_list:
            mask = mask_dict['segmentation']
            occlusion_mask[mask] = 255
        return occlusion_mask

    def visualize_masks(self, image: np.ndarray, mask_list: List[Dict], alpha: float = 0.6) -> np.ndarray:
        """
        為每個遮罩指定顏色，疊加在原圖上

        Args:
            image: 原始 BGR 圖像
            mask_list: 遮罩資料
            alpha: 疊加透明度

        Returns:
            彩色視覺化圖像
        """
        overlay = np.zeros_like(image, dtype=np.uint8)
        num_masks = len(mask_list)
        for i, mask_dict in enumerate(mask_list):
            mask = mask_dict['segmentation']
            color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)
            overlay[mask] = color

        vis = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return vis

    def save_masks_to_disk(self, mask_list: List[Dict], base_path: str) -> None:
        """
        儲存每個遮罩為二值圖（debug 用）

        Args:
            mask_list: 遮罩清單
            base_path: 儲存目錄（建議用 frame 名）
        """
        os.makedirs(base_path, exist_ok=True)
        for i, mask_dict in enumerate(mask_list):
            mask = mask_dict['segmentation'].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(base_path, f"mask_{i:03d}.png"), mask)
        print(f"✅ 儲存 {len(mask_list)} 個遮罩至 {base_path}/")

    def overlay_occlusion_mask(self, image: np.ndarray, occlusion_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        將遮擋遮罩（灰階）疊加在原圖上，用於展示遮擋區域

        Args:
            image: 原始圖像
            occlusion_mask: 單通道遮罩
            alpha: 疊加程度

        Returns:
            疊加後圖像
        """
        mask_colored = cv2.cvtColor(occlusion_mask, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
