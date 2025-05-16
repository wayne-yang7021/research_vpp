#!/bin/bash

# 創建模型儲存資料夾
MODEL_DIR="models"
mkdir -p $MODEL_DIR

echo "🔽 開始下載模型檔案..."

# 下載 SAM vit_b 模型
if [ ! -f "$MODEL_DIR/sam_vit_b.pth" ]; then
    echo "🔽 下載 SAM vit_b 模型..."
    wget -O "$MODEL_DIR/sam_vit_b.pth" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "✅ 已存在: sam_vit_b.pth"
fi

echo "✅ 所有模型下載完成"
