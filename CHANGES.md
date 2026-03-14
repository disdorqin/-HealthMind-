# CHANGES
## 版本：多模型融合生产化版本
### 1. 整体结构变更
- 新增 `src/data`、`src/models`、`src/utils` 三层结构。
- 根入口统一为 `app.py`。
- `main.py` 支持模型存在即跳过训练。
### 2. 核心技术升级
- 接入 LSTM/GRU/XGBoost/Moirai 四模型。
- 增加 `StackingManager` 融合输出。
- 训练监控加入进度与 ETA。
### 3. 展示与产品价值提升
- 支持多模型曲线同框对比。
- 支持交易收益和风险展示。
- Streamlit Cloud 一键部署能力增强。
