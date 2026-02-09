"""
可视化模块。

提供实验结果的可视化功能：
- 时序图：LLM vs Human 预测值随时间变化
- 更多图表类型待扩展...

目录结构：
    visualization/
    ├── config.py       # 配置（路径、颜色、模型映射）
    ├── data/           # 数据处理
    │   └── prepare_cache.py
    └── plots/          # 绘图
        └── timeseries.py

用法：
    cd src/visualization

    # 1. 预处理数据（生成 pkl 缓存）
    python data/prepare_cache.py

    # 2. 绘制时序图
    python plots/timeseries.py
"""
