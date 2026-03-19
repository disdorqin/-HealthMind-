# ❤️ HealthMind 健康风险管理平台

> 基于 AI 的心血管疾病风险预测与健康管理平台

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心功能](#-核心功能)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [API 文档](#-api-文档)
- [模型说明](#-模型说明)
- [配置说明](#-配置说明)
- [常见问题](#-常见问题)
- [开发团队](#-开发团队)

---

## 🌟 项目简介

HealthMind 是一个集成先进机器学习技术的健康管理平台，专注于心血管疾病风险预测与干预建议。平台采用 Stacking 集成学习架构，融合 LSTM/GRU 时序模型、XGBoost 特征模型和 Moirai 时序基础模型，通过 SHAP 可解释性分析为用户提供透明、可信的健康风险评估。

### 核心优势

- **精准预测**：Stacking 集成学习，准确率 > 70%
- **可解释性**：SHAP 归因分析，明确风险因素贡献度
- **个性化干预**：基于风险因素的定制化健康建议
- **激励机制**：健康积分系统，促进用户持续参与

---

## ✨ 核心功能

### 1. 多维度风险趋势 📈

- **24 小时风险趋势图**：实时监测心血管、血压、血糖风险变化
- **周风险预测**：未来 7 天健康风险趋势预测
- **风险仪表盘**：三维度实时风险等级指示（低/中/高）

### 2. 目标管理 🎯

| 目标类型 | 默认目标 | 进度追踪 |
|---------|---------|---------|
| 步数目标 | 10,000 步/日 | ✅ 周趋势图 |
| 睡眠目标 | 8 小时/日 | ✅ 周趋势图 |
| 蔬菜摄入 | 7 份/日 | ✅ 进度条 |
| 饮水量 | 2500ml/日 | ✅ 进度条 |
| 冥想时间 | 60 分钟/日 | ✅ 进度条 |

### 3. SHAP 风险归因 🔍

- **瀑布图可视化**：红色柱表示增加风险因素，绿色柱表示保护因素
- **风险因素详情表**：当前值 vs 正常范围对比
- **风险贡献分布饼图**：各因素占比一目了然

### 4. 健康建议 💡

- **优先级排序**：按风险贡献度自动排序
- **预期效果量化**：每条建议标注预期风险降低比例
- **交互式清单**：✅ 标记完成 / 📝 记录进展

### 5. 激励系统 🏆

- **积分规则**：
  - 每日预测：+10 分
  - 完成建议：+20 分
  - 连续 7 天打卡：+50 分
  - 风险改善：+30 分
- **等级体系**：青铜 → 白银 → 黄金 → 铂金
- **排行榜**：社区互动，健康达人 PK

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                     HealthMind 架构                          │
├─────────────────────────────────────────────────────────────┤
│  前端层 (Streamlit)                                          │
│  ├─ 风险趋势可视化 (Pyecharts)                               │
│  ├─ 目标管理进度条                                           │
│  ├─ SHAP 归因瀑布图                                          │
│  └─ 健康积分排行榜                                           │
├─────────────────────────────────────────────────────────────┤
│  服务层 (src/services/service_layer.py)                      │
│  ├─ SHAPExplainer: 风险归因分析                              │
│  ├─ DecisionEngine: 干预建议生成                             │
│  └─ HealthPointsSystem: 积分激励系统                         │
├─────────────────────────────────────────────────────────────┤
│  模型层 (src/models.py)                                      │
│  ├─ StackingEnsemble: 集成学习框架                           │
│  │  ├─ LSTMBaseLearner: 双层双向 LSTM                        │
│  │  ├─ GRUBaseLearner: 双层双向 GRU                          │
│  │  ├─ XGBoostBaseLearner: 非线性特征交互                    │
│  │  └─ MoiraiMockLearner: 冷启动预测 (预留 API)               │
│  └─ MetaLearner: 场景感知融合 (逻辑回归)                      │
├─────────────────────────────────────────────────────────────┤
│  数据处理层 (src/data_processing.py)                         │
│  ├─ 数据加载：cardio_train.csv (分号分隔)                     │
│  ├─ 特征转换：年龄→岁数、BMI 计算、血压差提取                  │
│  └─ 时序模拟：滑动窗口生成 7 天序列                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9 - 3.11
- 内存 >= 4GB
- 磁盘空间 >= 2GB

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/disdorqin/EcoLife.git
cd 计算机设计大赛
```

#### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
# 基础安装（推荐）
pip install -r requirements.txt

# 国内用户加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 4. 启动应用

```bash
# 启动 HealthMind 前端
streamlit run app.py

# 浏览器访问
# http://localhost:8501
```

---

## 📁 项目结构

```
计算机设计大赛/
├── app.py                          # Streamlit 前端应用
├── requirements.txt                # Python 依赖
├── README.md                       # 项目文档
├── data/
│   └── cardio_train.csv           # 心血管训练数据 (70,000 样本)
├── src/
│   ├── data_processing.py         # 数据处理模块
│   ├── models.py                  # Stacking 集成模型
│   └── services/
│       └── service_layer.py       # 服务层 (SHAP/决策/积分)
├── models/
│   └── checkpoints/               # 模型检查点
├── logs/                          # 日志文件
└── docs/                          # 详细文档
    ├── 模型训练与预测流程详解.md
    └── 调参优化指南.md
```

---

## 📊 API 文档

### HealthMindService

```python
from src.services.service_layer import HealthMindService

# 初始化服务
service = HealthMindService()

# 预测并解释
prediction = service.predict_and_explain(
    user_id='user_001',
    X=user_features,  # shape: (1, 17)
    feature_names=feature_names
)

# 生成干预计划
plan = service.generate_intervention_plan(prediction)

# 记录用户行为
result = service.record_user_action(
    user_id='user_001',
    recommendation_idx=0,
    completed=True
)

# 获取用户仪表板
dashboard = service.get_user_dashboard('user_001')
```

### StackingEnsemble

```python
from src.models import create_healthmind_ensemble

# 创建模型
ensemble = create_healthmind_ensemble(
    input_dim=17,
    target_accuracy=0.70
)

# 训练
result = ensemble.train(
    X_train, y_train,
    X_val, y_val,
    is_sequential=False
)

# 预测
predictions = ensemble.predict(X_test)

# 评估
metrics = ensemble.evaluate(X_test, y_test)
```

---

## 🧠 模型说明

### Stacking 集成架构

| 组件 | 类型 | 作用 | 超参数 |
|------|------|------|--------|
| LSTM | 双层双向 RNN | 捕捉健康指标周期性 | hidden=64, layers=2 |
| GRU | 双层双向 RNN | 捕捉健康指标周期性 | hidden=64, layers=2 |
| XGBoost | 梯度提升树 | 挖掘非线性交互 | depth=5, lr=0.1 |
| Moirai | Mock 模型 | 冷启动预测 | - |
| MetaLearner | 逻辑回归 | 场景感知融合 | - |

### 特征工程

| 特征 | 说明 | 转换方式 |
|------|------|---------|
| age_years | 年龄（岁） | age / 365.25 |
| bmi | BMI 指数 | weight / (height/100)² |
| bmi_encoded | BMI 分类编码 | 0-3 |
| bp_diff | 血压差 | ap_hi - ap_lo |
| map | 平均动脉压 | (ap_hi + 2*ap_lo) / 3 |
| bp_ratio | 血压比值 | ap_hi / ap_lo |
| bp_encoded | 血压分类编码 | 0-4 |

---

## ⚙️ 配置说明

### 环境变量 (.env)

```bash
# API 配置
API_HOST=localhost
API_PORT=5000

# 模型配置
MODEL_PATH=models/checkpoints
RANDOM_SEED=42

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### 模型配置 (EnsembleConfig)

```python
@dataclass
class EnsembleConfig:
    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # LSTM/GRU 超参数
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3
    rnn_epochs: int = 50
    rnn_batch_size: int = 32
    rnn_lr: float = 1e-3
    
    # XGBoost 超参数
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_lr: float = 0.1
    xgb_subsample: float = 0.8
    
    # 元学习器
    meta_learner_type: str = 'logistic'
    
    # 性能目标
    target_accuracy: float = 0.70
```

---

## ❓ 常见问题

### Q1: streamlit-echarts 安装失败

**解决方案**：
```bash
# 方法 1：使用特定版本
pip install streamlit-echarts==0.4.0

# 方法 2：国内源
pip install streamlit-echarts==0.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: pyecharts 图表报错

**常见错误及修复**：
- `is_fill` 参数不存在 → 改用 `linestyle_opts`
- `PointerOpts` 不存在 → 移除该参数
- `TitleOpts` 的 `pos_center` 不存在 → 移除该参数

### Q3: 模型训练准确率低

**优化建议**：
1. 增加训练轮次：`rnn_epochs=100`
2. 调整学习率：`rnn_lr=5e-4`
3. 增加 Dropout：`rnn_dropout=0.4`
4. 数据增强：增加时序模拟样本数

### Q4: SHAP 归因不可用

**说明**：SHAP 库为可选依赖，未安装时自动切换到基于规则的归因方法。

```bash
# 安装完整 SHAP 支持
pip install shap
```

---

## 👥 开发团队

| 角色 | 职责 |
|------|------|
| 算法工程师 | 模型开发与优化 |
| 后端工程师 | API 与服务层开发 |
| 前端工程师 | Streamlit 界面开发 |
| 数据工程师 | 数据处理与管道 |

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 心血管疾病数据集：[Kaggle Cardiovascular Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- 可视化库：[Pyecharts](https://pyecharts.org/)
- Web 框架：[Streamlit](https://streamlit.io/)
- 深度学习：[PyTorch](https://pytorch.org/)

---

## 📞 联系方式

- 项目仓库：https://github.com/disdorqin/EcoLife
- 问题反馈：请在 GitHub 提交 Issue

---

*最后更新：2026 年 3 月*