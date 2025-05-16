import os
import numpy as np
import torch
import cv2
from typing import Dict, Tuple, List, Union, Optional

# MiDaS深度估計
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

# 用於平面檢測的工具
from skimage import measure
import yaml


class DepthEstimator:
    """使用MiDaS進行單張圖像深度估計"""
    
    def __init__(self, config: Dict):
        """
        初始化深度估計器
        
        Args:
            config: 配置字典，包含模型參數
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # 載入模型
        print(f"Loading MiDaS {config['model_type']} model...")
        
        # 使用torch hub載入模型
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", config['model_type'])
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            raise
            
        # 設定模型前處理參數
        if config['model_type'] == "DPT_Large" or config['model_type'] == "DPT_Hybrid":
            self.transform = self._get_transform(384, 384)
        else:
            self.transform = self._get_transform(256, 256)
            
        print("MiDaS model loaded successfully")
        
    def _get_transform(self, width: int, height: int):
        """獲取標準化和調整大小的轉換函數"""
        
        def transform(img: np.ndarray) -> torch.Tensor:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = torch.from_numpy(img).float().permute(2, 0, 1)  # HWC -> CHW
            
            # 標準化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            
            # 調整大小
            img = F.interpolate(
                img.unsqueeze(0), 
                size=(height, width), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)
            
            return img
            
        return transform
        
    def estimate_depth(self, img: np.ndarray) -> np.ndarray:
        """
        從單一圖像估計深度圖
        
        Args:
            img: 輸入BGR圖像 (H, W, 3)
            
        Returns:
            深度圖 (H, W)，值越大表示越近
        """
        # 轉換並準備批次
        input_tensor = self.transform(img).to(self.device).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
            # 處理不同模型的輸出格式
            if self.config['model_type'] == "DPT_Large" or self.config['model_type'] == "DPT_Hybrid":
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)
            else:
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
        
        depth = prediction.cpu().numpy().squeeze()
        
        # 標準化深度，使值在0-1範圍內
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)
        
        return depth


class PlaneDetector:
    """平面檢測和分析"""
    
    def __init__(self, config: Dict):
        """
        初始化平面檢測器
        
        Args:
            config: 配置字典，包含模型參數
        """
        self.config = config
        self.device = torch.device(config['device'])
        self.confidence_threshold = config['confidence_threshold']
        
        # 注意：完整的PlaneRCNN實現需要完整模型
        # 這裡使用一個基於深度圖的簡化實現
        print("Initializing plane detection module...")
        
    def detect_planes(self, img: np.ndarray, depth_map: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        基於深度圖檢測平面
        
        Args:
            img: 原始BGR圖像 (H, W, 3)
            depth_map: 深度圖 (H, W)
            
        Returns:
            - 平面列表，每個平面是一個字典，包含mask、法線向量等信息
            - 平面遮罩，每個像素值表示該點屬於哪個平面ID
        """
        # 對深度圖進行邊緣檢測，找出深度不連續的區域
        depth_edges = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
        
        # 平滑深度圖以減少噪音
        smoothed_depth = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        # 使用分水嶺算法或連通區域進行平面分割
        # 為了簡化，這裡使用基於深度的閾值分割
        labels = measure.label(smoothed_depth > 0.5, connectivity=2)
        
        # 過濾小區域
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if count < 1000:  # 過濾小於1000像素的區域
                labels[labels == label] = 0
                
        # 重新標記標籤以確保連續
        labels = measure.label(labels > 0, connectivity=2)
        
        # 為每個檢測到的平面創建數據結構
        planes = []
        for plane_id in range(1, labels.max() + 1):
            mask = (labels == plane_id).astype(np.uint8)
            
            # 計算平面區域
            area = np.sum(mask)
            if area < 5000:  # 忽略太小的平面
                continue
                
            # 找到平面的輪廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 計算平面的法線向量（這裡是簡化實現）
            # 在實際的 PlaneRCNN 中，法線向量來自深度學習模型
            y_indices, x_indices = np.where(mask > 0)
            
            # 通過隨機採樣點來估計平面方程（簡化）
            if len(y_indices) > 100:
                # 隨機選擇100個點
                idx = np.random.choice(len(y_indices), 100, replace=False)
                samples_y, samples_x = y_indices[idx], x_indices[idx]
                depths = depth_map[samples_y, samples_x]
                
                # 使用最小二乘法擬合平面方程：ax + by + cz + d = 0
                # 其中 (a, b, c) 是法線向量
                A = np.column_stack([samples_x, samples_y, np.ones(len(samples_x))])
                b = depths
                
                try:
                    # 求解方程 Ax = b
                    normal, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    # 法線向量標準化
                    norm = np.sqrt(normal[0]**2 + normal[1]**2 + 1)
                    normal = np.array([normal[0]/norm, normal[1]/norm, 1/norm])
                except:
                    normal = np.array([0, 0, 1])  # 默認向上的法線
            else:
                normal = np.array([0, 0, 1])  # 默認向上的法線
            
            # 計算平面的中心點
            center_y, center_x = np.mean(y_indices), np.mean(x_indices)
            center_depth = np.median(depth_map[y_indices, x_indices])
            
            # 添加平面數據
            planes.append({
                'id': plane_id,
                'mask': mask,
                'normal': normal,
                'center': (center_x, center_y, center_depth),
                'area': area,
                'contours': contours,
                # 根據平面法線方向估計平面類型 (地面、牆、桌面等)
                'type': 'ground' if normal[2] > 0.8 else ('wall' if normal[2] < 0.3 else 'other')
            })
        
        # 創建彩色標籤圖像用於可視化
        plane_mask = np.zeros_like(labels)
        for plane in planes:
            plane_mask[plane['mask'] > 0] = plane['id']
        
        return planes, plane_mask


class SceneUnderstanding:
    """場景理解總模組，結合深度估計和平面檢測"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化場景理解模組
        
        Args:
            config_path: 配置文件路徑
        """
        # 載入配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化子模組
        self.depth_estimator = DepthEstimator(self.config['models']['midas'])
        self.plane_detector = PlaneDetector(self.config['models']['planercnn'])
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        處理單一影片幀，進行場景理解
        
        Args:
            frame: BGR格式的影片幀
            
        Returns:
            包含深度圖、平面信息等的字典
        """
        # 1. 估計深度
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        # 2. 檢測平面
        planes, plane_mask = self.plane_detector.detect_planes(frame, depth_map)
        
        # 3. 返回結果
        return {
            'depth_map': depth_map,
            'planes': planes,
            'plane_mask': plane_mask,
            'frame_shape': frame.shape
        }
    
    def visualize_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        可視化場景理解結果
        
        Args:
            frame: 原始影片幀
            results: process_frame返回的結果
            
        Returns:
            可視化圖像
        """
        # 複製原始幀
        vis_img = frame.copy()
        
        # 1. 顯示深度圖
        depth_colored = cv2.applyColorMap((results['depth_map'] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_vis = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)
        
        # 2. 顯示平面
        plane_vis = frame.copy()
        # 為每個平面使用不同的顏色
        colors = [
            (0, 0, 255),    # 紅色
            (0, 255, 0),    # 綠色
            (255, 0, 0),    # 藍色
            (0, 255, 255),  # 黃色
            (255, 0, 255),  # 紫色
            (255, 255, 0),  # 青色
            (128, 0, 0),    # 暗藍色
            (0, 128, 0),    # 暗綠色
            (0, 0, 128),    # 暗紅色
        ]
        
        for i, plane in enumerate(results['planes']):
            color = colors[i % len(colors)]
            # 繪製平面輪廓
            cv2.drawContours(plane_vis, plane['contours'], -1, color, 2)
            
            # 在平面中心顯示ID和類型
            center = (int(plane['center'][0]), int(plane['center'][1]))
            cv2.putText(
                plane_vis, 
                f"#{plane['id']} {plane['type']}", 
                center,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # 繪製法線向量（簡單可視化）
            normal = plane['normal']
            end_point = (
                int(center[0] + normal[0] * 50),
                int(center[1] + normal[1] * 50)
            )
            cv2.arrowedLine(plane_vis, center, end_point, color, 2)
        
        # 3. 合併可視化結果
        h, w = frame.shape[:2]
        vis_combined = np.zeros((h * 2, w, 3), dtype=np.uint8)
        vis_combined[:h, :] = depth_vis
        vis_combined[h:, :] = plane_vis
        
        return vis_combined


# 單元測試代碼
if __name__ == "__main__":
    # 載入配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化場景理解模組
    scene_module = SceneUnderstanding("config.yaml")
    
    # 測試單張圖像
    img_path = "test_image.jpg"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            # 處理圖像
            results = scene_module.process_frame(img)
            
            # 可視化結果
            vis_img = scene_module.visualize_results(img, results)
            
            # 顯示結果
            cv2.imshow("Scene Understanding Results", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"檢測到 {len(results['planes'])} 個平面")
            
            # 保存結果
            cv2.imwrite("scene_understanding_results.jpg", vis_img)
        else:
            print(f"無法讀取圖像: {img_path}")
    else:
        print(f"圖像檔案不存在: {img_path}")
