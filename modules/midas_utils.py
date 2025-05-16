# modules/midas_utils.py

import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# from midas.midas_net_custom import MidasNet_small
# from midas.dpt_depth import DPTDepthModel
# from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torch.hub

def load_model(model_type: str, device: torch.device):
    """
    使用 torch.hub 載入 MiDaS 模型，支援 midas_v21_small。
    """

    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "MiDaS_small":
        transform = transforms.small_transform
    elif "dpt" in model_type:
        transform = transforms.dpt_transform
    else:
        raise ValueError(f"未知模型類型: {model_type}")

    return model, transform
