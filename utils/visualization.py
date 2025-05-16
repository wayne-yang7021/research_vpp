import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.cm as cm


def visualize_depth_map(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    將深度圖可視化為顏色圖
    
    Args:
        depth_map: 深度圖 (H, W)，值範圍在0-1之間
        colormap: OpenCV色彩映射方案
        
    Returns:
        彩色化的深度圖
    """
    # 確保深度圖在0-1之間
    depth_map = np.clip(depth_map, 0, 1)
    
    # 轉換為8位整數
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    
    # 應用色彩映射
    colored_depth = cv2.applyColorMap(depth_uint8, colormap)
    
    return colored_depth


def get_plane_color(index: int, total: int = 20) -> Tuple[int, int, int]:
    """
    根據 index 使用 colormap 產生 RGB 顏色

    Args:
        index: 平面索引
        total: 顏色總類（會循環）

    Returns:
        RGB 顏色 (int, int, int)
    """
    color = cm.get_cmap('tab20')(index % total)  # RGBA 值 [0,1]
    return tuple(int(c * 255) for c in color[:3])  # 轉成 RGB [0,255]

# def visualize_planes(image: np.ndarray, planes: List[Dict], alpha: float = 0.5) -> np.ndarray:
#     """
#     在原始圖像上可視化檢測到的平面
    
#     Args:
#         image: 原始BGR圖像
#         planes: 平面列表，每個平面是一個包含'mask'、'id'等鍵的字典
#         alpha: 平面疊加透明度
        
#     Returns:
#         疊加平面可視化的圖像
#     """
#     # 複製原始圖像
#     result = image.copy()
    
#     # 不同平面的顏色
#     colors = [
#         (0, 0, 255),    # 紅色
#         (0, 255, 0),    # 綠色
#         (255, 0, 0),    # 藍色
#         (0, 255, 255),  # 黃色
#         (255, 0, 255),  # 紫色
#         (255, 255, 0),  # 青色
#         (128, 0, 0),    # 暗藍色
#         (0, 128, 0),    # 暗綠色
#         (0, 0, 128),    # 暗紅色
#     ]
    
#     # 創建一個覆蓋層
#     overlay = np.zeros_like(image)
    
#     # 為每個平面填充不同顏色
#     for i, plane in enumerate(planes):
#         color = colors[i % len(colors)]
#         mask = plane['mask']
        
#         # 填充平面區域
#         overlay[mask > 0] = color
        
#         # 繪製平面邊界
#         contours = plane['contours']
#         cv2.drawContours(result, contours, -1, color, 2)
        
#         # 添加標籤
#         if 'center' in plane:
#             cx, cy = int(plane['center'][0]), int(plane['center'][1])
#             text = f"#{plane['id']}"
#             if 'type' in plane:
#                 text += f" {plane['type']}"
#             cv2.putText(result, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
#     # 將平面疊加到原始圖像上
#     cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0, result)
    
#     return result
def visualize_planes(image: np.ndarray, planes: List[Dict], alpha: float = 0.5) -> np.ndarray:
    """
    在原始圖像上可視化檢測到的平面，使用動態 colormap 配色

    Args:
        image: 原始BGR圖像
        planes: 平面列表，每個平面是一個包含'mask'、'id'等鍵的字典
        alpha: 平面疊加透明度

    Returns:
        疊加平面可視化的圖像
    """
    # 複製原始圖像
    result = image.copy()
    
    # 創建覆蓋層
    overlay = np.zeros_like(image)

    for i, plane in enumerate(planes):
        color = get_plane_color(i)  # 動態取得顏色
        mask = plane['mask']
        
        # 填充平面區域
        overlay[mask > 0] = color
        
        # 繪製平面邊界
        contours = plane['contours']
        cv2.drawContours(result, contours, -1, color, 2)
        
        # 添加標籤
        if 'center' in plane:
            cx, cy = int(plane['center'][0]), int(plane['center'][1])
            text = f"#{plane['id']}"
            if 'type' in plane:
                text += f" {plane['type']}"
            cv2.putText(result, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 疊加平面顏色圖層
    cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0, result)
    
    return result


def visualize_normal_vectors(image: np.ndarray, planes: List[Dict], scale: float = 50.0) -> np.ndarray:
    """
    可視化平面的法線向量
    
    Args:
        image: 原始BGR圖像
        planes: 平面列表
        scale: 法線向量的顯示比例
        
    Returns:
        帶有法線向量可視化的圖像
    """
    # 複製原始圖像
    result = image.copy()
    
    # 為每個平面繪製法線向量
    for plane in planes:
        if 'normal' in plane and 'center' in plane:
            normal = plane['normal']
            center = (int(plane['center'][0]), int(plane['center'][1]))
            
            # 計算終點
            end_point = (
                int(center[0] + normal[0] * scale),
                int(center[1] + normal[1] * scale)
            )
            
            # 繪製箭頭
            cv2.arrowedLine(result, center, end_point, (0, 255, 255), 2)
    
    return result


def save_visualization(output_dir: str, frame_id: int, results: Dict, original_frame: np.ndarray) -> None:
    """
    保存可視化結果
    
    Args:
        output_dir: 輸出目錄
        frame_id: 幀ID
        results: 場景理解結果
        original_frame: 原始圖像
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存深度圖
    depth_map = results['depth_map']
    depth_vis = visualize_depth_map(depth_map)
    cv2.imwrite(os.path.join(output_dir, f"depth_{frame_id:06d}.jpg"), depth_vis)
    
    # 2. 保存平面分割結果
    if 'planes' in results:
        planes_vis = visualize_planes(original_frame, results['planes'])
        cv2.imwrite(os.path.join(output_dir, f"planes_{frame_id:06d}.jpg"), planes_vis)
        
        # 保存法線向量可視化
        normals_vis = visualize_normal_vectors(original_frame, results['planes'])
        cv2.imwrite(os.path.join(output_dir, f"normals_{frame_id:06d}.jpg"), normals_vis)
    
    # 3. 保存組合可視化
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Frame")
    plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Depth Map")
    plt.imshow(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    if 'planes' in results:
        plt.subplot(2, 2, 3)
        plt.title("Plane Detection")
        plt.imshow(cv2.cvtColor(planes_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.title("Normal Vectors")
        plt.imshow(cv2.cvtColor(normals_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_{frame_id:06d}.jpg"))
    plt.close()


def create_progress_video(image_dir: str, output_path: str, fps: int = 30) -> None:
    """
    從一系列圖像創建影片
    
    Args:
        image_dir: 包含圖像的目錄
        output_path: 輸出影片路徑
        fps: 影片幀率
    """
    # 獲取所有JPG圖像並排序
    images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
    images.sort()
    
    if not images:
        print(f"在 {image_dir} 中沒有找到圖像")
        return
    
    # 讀取第一張圖像獲取尺寸
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, _ = frame.shape
    
    # 創建視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 寫入每一幀
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
    
    # 釋放資源
    video_writer.release()
    print(f"已創建影片: {output_path}")
