import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional
import shutil
import json
import yaml


def create_directory(directory: str) -> None:
    """
    創建目錄，如果已存在則不執行任何操作
    
    Args:
        directory: 要創建的目錄路徑
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"創建目錄: {directory}")


def clean_directory(directory: str) -> None:
    """
    清空目錄中的所有文件和子目錄
    
    Args:
        directory: 要清空的目錄路徑
    """
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"清空目錄: {directory}")


def extract_frames(video_path: str, output_dir: str, fps: Optional[float] = None) -> int:
    """
    從影片中提取幀
    
    Args:
        video_path: 輸入影片路徑
        output_dir: 輸出幀的目錄
        fps: 提取的幀率，None表示提取所有幀
        
    Returns:
        提取的幀數
    """
    # 創建輸出目錄
    create_directory(output_dir)
    
    # 打開影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法打開影片: {video_path}")
    
    # 獲取影片信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"影片信息: {frame_count} 幀, {video_fps:.2f} FPS")
    
    # 計算幀採樣間隔
    if fps is not None and fps < video_fps:
        frame_interval = int(video_fps / fps)
    else:
        frame_interval = 1
    
    # 提取幀
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_idx += 1
    
    # 釋放資源
    cap.release()
    
    print(f"提取了 {saved_count} 幀到 {output_dir}")
    return saved_count


def save_json(data: Union[Dict, List], filepath: str) -> None:
    """
    將數據保存為JSON文件
    
    Args:
        data: 要保存的數據
        filepath: 輸出文件路徑
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"保存JSON到: {filepath}")


def load_json(filepath: str) -> Union[Dict, List]:
    """
    從JSON文件載入數據
    
    Args:
        filepath: JSON文件路徑
        
    Returns:
        載入的數據
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_yaml(data: Dict, filepath: str) -> None:
    """
    將數據保存為YAML文件
    
    Args:
        data: 要保存的數據
        filepath: 輸出文件路徑
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
    print(f"保存YAML到: {filepath}")


def load_yaml(filepath: str) -> Dict:
    """
    從YAML文件載入數據
    
    Args:
        filepath: YAML文件路徑
        
    Returns:
        載入的數據
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data


def load_config(config_path: str) -> Dict:
    """
    載入配置文件
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        配置字典
    """
    # 檢查文件擴展名
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        return load_yaml(config_path)
    elif ext.lower() == '.json':
        return load_json(config_path)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    調整圖像大小
    
    Args:
        image: 輸入圖像
        target_size: 目標大小 (寬, 高)
        
    Returns:
        調整大小後的圖像
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def save_numpy_data(data: np.ndarray, filepath: str) -> None:
    """
    保存NumPy數組
    
    Args:
        data: NumPy數組
        filepath: 輸出文件路徑
    """
    np.save(filepath, data)
    print(f"保存NumPy數據到: {filepath}")


def load_numpy_data(filepath: str) -> np.ndarray:
    """
    載入NumPy數組
    
    Args:
        filepath: 輸入文件路徑
        
    Returns:
        NumPy數組
    """
    return np.load(filepath)


def create_video_from_frames(
    frames_dir: str, 
    output_path: str, 
    fps: int = 30, 
    frame_pattern: str = "frame_%06d.jpg",
    start_number: int = 0
) -> None:
    """
    從一系列圖像幀創建影片
    
    Args:
        frames_dir: 包含圖像幀的目錄
        output_path: 輸出影片路徑
        fps: 影片幀率
        frame_pattern: 幀文件名模式
        start_number: 起始幀編號
    """
    # 獲取第一幀以確定尺寸
    first_frame_path = os.path.join(frames_dir, frame_pattern % start_number)
    if not os.path.exists(first_frame_path):
        raise ValueError(f"找不到第一幀: {first_frame_path}")
    
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # 創建視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 獲取所有幀文件並排序
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    
    # 寫入每一幀
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)
    
    # 釋放資源
    video_writer.release()
    print(f"創建影片: {output_path}")


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    混合兩張圖像
    
    Args:
        img1: 第一張圖像
        img2: 第二張圖像
        alpha: 混合係數，0.0表示完全使用img1，1.0表示完全使用img2
        
    Returns:
        混合後的圖像
    """
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
