import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


def compute_plane_equation(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    計算平面方程 ax + by + cz + d = 0
    其中 (a, b, c) 是單位法線向量
    
    Args:
        points: 平面上的點，形狀為 (N, 3)
        
    Returns:
        (normal, d) - 法線向量和平面方程的 d 參數
    """
    # 至少需要3個點確定一個平面
    if points.shape[0] < 3:
        raise ValueError("至少需要3個點來確定平面")
    
    # 形成矩陣方程
    centroid = np.mean(points, axis=0)
    
    # 將點相對於中心平移以提高數值穩定性
    points_centered = points - centroid
    
    # 使用SVD求解最佳擬合平面
    u, s, vh = np.linalg.svd(points_centered)
    
    # 最小奇異值對應的向量是法線方向
    normal = vh[2, :]
    
    # 標準化法線向量
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-10:
        raise ValueError("法線向量計算錯誤")
    
    normal = normal / normal_length
    
    # 計算平面方程的 d 參數
    d = -np.dot(normal, centroid)
    
    return normal, d


def plane_intersection_line(plane1: Tuple[np.ndarray, float], plane2: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算兩個平面的交線
    
    Args:
        plane1: (normal1, d1) - 第一個平面
        plane2: (normal2, d2) - 第二個平面
        
    Returns:
        (point, direction) - 交線上的一點和方向向量
    """
    normal1, d1 = plane1
    normal2, d2 = plane2
    
    # 交線的方向是兩個法線的叉積
    direction = np.cross(normal1, normal2)
    direction_length = np.linalg.norm(direction)
    
    if direction_length < 1e-10:
        raise ValueError("平面平行或重合")
    
    # 標準化方向向量
    direction = direction / direction_length
    
    # 找到交線上的一點
    # 使用線性代數求解
    A = np.array([normal1, normal2])
    b = np.array([-d1, -d2])
    
    # 由於方程是超定的，我們選擇使用未知數最少的維度
    # 找到最大法線分量的索引
    idx = np.argmax(np.abs(direction))
    
    # 創建一個零向量，並設置選擇的坐標為0
    point = np.zeros(3)
    
    # 組裝3x2矩陣，刪除選定的行
    indices = [i for i in range(3) if i != idx]
    A_reduced = A[:, indices]
    
    # 求解2x2線性方程組
    solution = np.linalg.solve(A_reduced, b)
    
    # 將解填入到點的坐標中
    for i, val in zip(indices, solution):
        point[i] = val
    
    return point, direction


def point_to_plane_distance(point: np.ndarray, plane: Tuple[np.ndarray, float]) -> float:
    """
    計算點到平面的距離
    
    Args:
        point: 點坐標 (x, y, z)
        plane: (normal, d) - 平面方程
        
    Returns:
        點到平面的距離
    """
    normal, d = plane
    # 點到平面的距離公式: |ax0 + by0 + cz0 + d| / √(a² + b² + c²)
    # 由於法線已標準化，分母為1
    return abs(np.dot(normal, point) + d)


def project_point_to_plane(point: np.ndarray, plane: Tuple[np.ndarray, float]) -> np.ndarray:
    """
    將點投影到平面上
    
    Args:
        point: 點坐標 (x, y, z)
        plane: (normal, d) - 平面方程
        
    Returns:
        投影點的坐標
    """
    normal, d = plane
    # 計算點到平面的有符號距離
    dist = np.dot(normal, point) + d
    
    # 沿法線方向投影
    projected_point = point - dist * normal
    
    return projected_point


def is_point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    判斷點是否在多邊形內（2D）
    使用射線法
    
    Args:
        point: 點坐標 [x, y]
        polygon: 多邊形頂點 shape=(N, 2)
        
    Returns:
        點是否在多邊形內
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersect:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def fit_plane_ransac(points: np.ndarray, max_iterations: int = 100, threshold: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    使用RANSAC算法擬合平面
    
    Args:
        points: 3D點雲，shape=(N, 3)
        max_iterations: 最大迭代次數
        threshold: 內點閾值
        
    Returns:
        (normal, d) - 最佳擬合平面
    """
    best_normal = None
    best_d = None
    max_inliers = 0
    
    num_points = points.shape[0]
    
    for _ in range(max_iterations):
        # 隨機選擇3個點
        idx = np.random.choice(num_points, 3, replace=False)
        sample_points = points[idx]
        
        try:
            # 計算平面
            normal, d = compute_plane_equation(sample_points)
            
            # 計算每個點到平面的距離
            distances = np.abs(np.dot(points, normal) + d)
            
            # 計算內點數量
            inliers = np.sum(distances < threshold)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_normal = normal
                best_d = d
        
        except ValueError:
            continue
    
    # 如果沒有找到有效平面
    if best_normal is None:
        raise ValueError("無法找到有效平面")
    
    # 使用所有內點重新擬合平面以提高精度
    distances = np.abs(np.dot(points, best_normal) + best_d)
    inlier_indices = np.where(distances < threshold)[0]
    
    if len(inlier_indices) >= 3:
        inlier_points = points[inlier_indices]
        try:
            best_normal, best_d = compute_plane_equation(inlier_points)
        except ValueError:
            pass
    
    return best_normal, best_d


def transform_point_to_plane_coordinates(point: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """
    將3D點轉換到平面2D坐標系
    
    Args:
        point: 3D點 [x, y, z]
        origin: 平面坐標系原點
        x_axis: 平面x軸單位向量
        y_axis: 平面y軸單位向量
        
    Returns:
        平面坐標系中的2D點 [u, v]
    """
    # 計算點相對於原點的位移
    relative_point = point - origin
    
    # 投影到平面坐標軸上
    u = np.dot(relative_point, x_axis)
    v = np.dot(relative_point, y_axis)
    
    return np.array([u, v])


def transform_plane_to_world_coordinates(plane_point: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    將平面坐標系中的2D點轉換回3D世界坐標
    
    Args:
        plane_point: 平面坐標系中的點 [u, v]
        origin: 平面坐標系原點
        x_axis: 平面x軸單位向量
        y_axis: 平面y軸單位向量
        z_axis: 平面法線向量（標準化）
        
    Returns:
        3D世界坐標點 [x, y, z]
    """
    u, v = plane_point
    # 轉換回世界坐標
    world_point = origin + u * x_axis + v * y_axis
    
    return world_point


def create_plane_coordinate_system(normal: np.ndarray, point_on_plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    為平面創建坐標系
    
    Args:
        normal: 平面法線向量（標準化）
        point_on_plane: 平面上的一點
        
    Returns:
        (origin, x_axis, y_axis, z_axis) - 平面坐標系
    """
    # 以提供的點作為原點
    origin = point_on_plane
    
    # z軸為法線方向
    z_axis = normal / np.linalg.norm(normal)
    
    # 選擇一個不與z軸平行的軸作為參考
    if abs(z_axis[0]) < abs(z_axis[1]):
        ref_axis = np.array([1.0, 0.0, 0.0])
    else:
        ref_axis = np.array([0.0, 1.0, 0.0])
    
    # 計算x軸（垂直於z軸）
    x_axis = np.cross(ref_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 計算y軸（垂直於x軸和z軸）
    y_axis = np.cross(z_axis, x_axis)
    
    return origin, x_axis, y_axis, z_axis
