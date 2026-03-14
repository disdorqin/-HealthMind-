# QUICKSTART_REFACTORED
## 1. 整体结构
- 入口：`app.py`（生产）
- 训练/检查：`main.py`
- 模型服务：`src/models/model_service.py`
## 2. 核心技术
- LSTM/GRU（PyTorch）
- XGBoost
- Moirai(Zero-shot)
- Stacking 融合
## 3. 展示与价值
- 训练进度 + ETA
- 多模型同框对比
- 交易收益与风险指标
## 快速命令
```bash
pip install -r requirements.txt
python main.py --train-only
streamlit run app.py
```
