# 风芒可测 - 多模型电力预测与交易优化平台
## 1. 整体结构
### 1.1 文件目录
```text
app.py                              # Streamlit 根入口（生产入口）
main.py                             # 本地检查/自动训练/启动脚本
requirements.txt                    # Python 依赖
packages.txt                        # Streamlit Cloud 系统依赖
.streamlit/
  config.toml                       # Streamlit 配置
  secrets.toml.example              # Secrets 模板
src/
  data/
    data_service.py                 # 数据准备统一入口
    dataset_builder.py              # 时序窗口构建与切分
    feature_engineering.py          # 特征工程
    mysql_client.py                 # Secrets 数据库配置读取
  models/
    base_model.py                   # 统一接口(train/predict/save/load)
    lstm_model.py                   # LSTM
    gru_model.py                    # GRU
    xgboost_model.py                # XGBoost
    moirai_model.py                 # Moirai(Zero-shot + lightweight)
    stacking_manager.py             # 融合元学习器
    model_registry.py               # 模型注册与工厂
    model_service.py                # 前端直调服务层
  utils/
    env.py                          # Cloud环境检测与轻量模式
    eta.py                          # 训练剩余时长估计
    paths.py                        # 路径管理
    progress.py                     # 进度事件结构
  logic/
    __init__.py                     # 业务兼容门面层
    trade.py                        # 交易优化逻辑
docs/
  architecture.md                   # 架构文档
```
### 1.2 数据流
1. CSV 数据进入 `src/data/feature_engineering.py` 进行数值化与时间特征提取。  
2. `src/data/dataset_builder.py` 完成 lookback 窗口构造与 train/val/test 切分。  
3. `src/models/model_service.py` 调用四类模型训练与推理。  
4. `src/models/stacking_manager.py` 对基模型输出做线性融合得到最终预测。  
5. `app.py` 将结果可视化并输出交易建议相关指标。  
### 1.3 执行顺序
- 本地总览：`python main.py`  
- 仅训练：`python main.py --train-only`  
- 仅前端：`streamlit run app.py`  
- 云端部署：Streamlit Cloud 指向 `app.py`  
### 1.4 业务逻辑
- 训练阶段：支持多模型并行配置 + 融合学习。  
- 预测阶段：支持单模型曲线与 stacking 融合曲线同屏对比。  
- 决策阶段：基于预测功率 + 分时电价生成交易优化建议、收益与风险评估。  
## 2. 核心技术
### 2.1 技术栈
- 前端：Streamlit  
- 深度学习：PyTorch（LSTM/GRU）  
- 机器学习：XGBoost + Scikit-learn（Stacking 元学习器）  
- 大模型时序：Uni2TS/Moirai（零样本推理封装）  
- 工程化：Pathlib、Secrets、Cloud lightweight 策略  
### 2.2 技术体现
- 统一模型接口，降低新增模型成本。  
- 训练进度/ETA 实时反馈，提升可观测性。  
- Cloud 下轻量模式自动启用，降低内存溢出风险。  
- 旧 Flask 路由降级为 legacy，主链路保持单进程稳定。  
### 2.3 相比传统电力预测的创新点
- 从单模型升级为“深度模型 + 树模型 + 大模型”的融合预测框架。  
- Moirai 引入零样本能力，在少标注/新场景下增强泛化能力。  
- Stacking 元学习器将模型互补性转化为稳定收益提升。  
- 将预测直接闭环到交易策略，形成“预测-决策一体化”。  
### 2.4 前后端分布
- 当前生产形态：单体 Streamlit（前后端逻辑在同一进程）。  
- 模型服务层：`src/models/model_service.py`，供 UI 直接调用。  
- 兼容形态：`src/backend/api.py` 保留 legacy 入口（默认禁用）。  
## 3. 展示与价值
### 3.1 结果展示
- 数据页：原始数据规模与质量可视化。  
- 训练页：模型进度、损失、ETA、训练摘要。  
- 预测页：LSTM/GRU/XGBoost/Moirai/Stacking 同框曲线对比。  
- 交易页：收益、成本、风险分层指标。  
### 3.2 对企业产品的优点
- 降低调度与交易决策延迟，提升运营效率。  
- 提高预测鲁棒性，减少单模型失效带来的业务波动。  
- 降低云端部署与运维复杂度，支持快速上线迭代。  
- 形成可解释的业务闭环，便于对外展示与内部汇报。  
