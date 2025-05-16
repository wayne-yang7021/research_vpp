# research_vpp

## 目前的檔案夾目錄
```markdown
research_vpp/
├── main.py                 # 主程式執行入口
├── requirements.txt        # 所需依賴包
├── config.yaml             # 系統配置檔
│
└── modules/
    ├── __init__.py
    ├── scene_understanding.py  # Step 1: 場景理解與平面檢測
    ├── segmentation.py         # Step 2: 物件分割與遮擋處理
    ├── camera_tracking.py      # Step 3: 相機追蹤與姿態估計
    ├── object_insertion.py     # Step 4: 3D物件插入與初步渲染
    ├── lighting.py             # Step 5: 光照與風格調和
    └── quality_control.py      # Step 6: 自動品質控制
│
└── utils/
    ├── __init__.py
    ├── visualization.py        # 視覺化工具
    ├── io_utils.py             # 輸入輸出處理
    └── geometry.py             # 幾何計算工具
│
└── web/                      # 可選的Web界面
    ├── app.py
    ├── static/
    └── templates/
```

