# Architecture
## 1. 整体结构
### 1.1 目录分层
- UI: `app.py`
- Service: `src/models/model_service.py`
- Models: `src/models/*.py`
- Data: `src/data/*.py`
- Utils: `src/utils/*.py`
- Business: `src/logic/*.py`
### 1.2 数据流与执行顺序
1. 数据读入与特征工程。
2. lookback 序列化与时间切分。
3. 基模型训练/推理。
4. stacking 融合。
5. 结果可视化与交易评估。
## 2. 核心技术
- PyTorch: LSTM/GRU
- XGBoost: 结构化非线性建模
- Uni2TS/Moirai: 零样本时序推理
- Scikit-learn: Stacking 元学习
- Streamlit: 一体化交互展示
### 创新点（对比传统电力预测）
- 从单模型到多模型融合，稳定性更高。
- 引入零样本大模型能力，增强新场景泛化。
- 预测直接服务交易优化，形成业务闭环。
## 3. 展示与产品价值
- 模型对比看板支持快速分析误差来源。
- 训练可视化降低运维排查成本。
- 单进程部署降低上线复杂度与资源消耗。
