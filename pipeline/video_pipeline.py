# pipeline/video_processing.py

import os
import cv2
from tqdm import tqdm
from modules.scene_understanding import SceneUnderstanding
from modules.segmentation import ObjectSegmentor
from utils import (
    extract_frames, create_directory, create_video_from_frames,
    blend_images
)
import yaml


class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.input_video = self.config["input_video"]
        self.frame_dir = self.config["temp_frame_dir"]
        self.output_dir = self.config["output_dir"]
        self.fps = self.config.get("fps", 5)

        create_directory(self.frame_dir)
        create_directory(self.output_dir)

        print("🔧 初始化模組中...")
        self.scene_model = SceneUnderstanding(config_path)
        self.segmentor = ObjectSegmentor(**self.config["sam"])
        print("✅ 模組初始化完成")

    def run(self):
        print("🎞️ 拆解影片為幀...")
        num_frames = extract_frames(self.input_video, self.frame_dir, fps=self.fps)

        print(f"📦 共擷取 {num_frames} 幀，開始逐幀處理...")
        for i in tqdm(range(num_frames)):
            frame_path = os.path.join(self.frame_dir, f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # 深度 + 平面
            scene = self.scene_model.process_frame(frame)

            # 遮擋遮罩
            masks = self.segmentor.segment_objects(frame)
            occ_mask = self.segmentor.generate_occlusion_mask(masks, frame.shape)

            # 疊加可視化
            blended = blend_images(frame, cv2.cvtColor(occ_mask, cv2.COLOR_GRAY2BGR), alpha=0.5)

            # 儲存視覺化圖
            out_path = os.path.join(self.output_dir, f"processed_{i:06d}.jpg")
            cv2.imwrite(out_path, blended)

        print("🎬 正在合併為影片...")
        output_video = os.path.join(self.output_dir, "processed_video.mp4")
        create_video_from_frames(
            frames_dir=self.output_dir,
            output_path=output_video,
            fps=self.fps,
            frame_pattern="processed_%06d.jpg"
        )

        print(f"✅ 處理完成，輸出影片：{output_video}")
