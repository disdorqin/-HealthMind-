# QUICK_REFERENCE
## 1. 整体结构
- 数据层：`src/data/*`
- 模型层：`src/models/*`
- 工具层：`src/utils/*`
- 展示层：`app.py`
## 2. 核心技术
- 多模型统一接口 `train/predict`
- Moirai 零样本封装 + lightweight
- stacking 融合输出最终预测
## 3. 展示与业务价值
- 预测曲线对比
- 训练进度可观测
- 交易优化辅助决策
## 常用命令
```bash
python test_training_progress.py
python main.py --train-only
streamlit run app.py
```
