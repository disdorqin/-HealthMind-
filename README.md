# EcoLife：个人碳足迹预测与管理平台

<div align="center">

🌿 **EcoLife - 您的智能碳足迹管理助手**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/pytorch-latest-orange.svg)](https://pytorch.org/)

**计算机设计大赛参赛作品 | 多模型融合预测 | 智能碳管理**

</div>

---

## 📖 目录

- [项目简介](#-项目简介)
- [核心功能](#-核心功能)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [模型说明](#-模型说明)
- [使用指南](#-使用指南)
- [性能指标](#-性能指标)
- [开发团队](#-开发团队)

---

## 🌟 项目简介

**EcoLife** 是一个面向个人碳足迹管理的智能预测与决策支持平台。项目结合深度学习、机器学习和时序基础模型，实现了对个人碳排放行为的高精度预测，并提供个性化的减碳建议和交易决策支持。

### 项目背景

在全球碳中和的大背景下，个人碳足迹管理成为实现减排目标的重要环节。本项目通过构建多模型融合的预测系统，帮助用户：
- 📊 **了解** 自身碳排放模式
- 🔮 **预测** 未来碳排放趋势
- 💡 **获取** 个性化减碳建议
- 🏆 **追踪** 减碳成果与荣誉

---

## ✨ 核心功能

### 1. 多模型融合预测
集成四种先进模型进行碳排放预测：
| 模型 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| **LSTM** | 深度学习 | 双向 LSTM，捕捉时序依赖 | 短期波动预测 |
| **GRU** | 深度学习 | 门控循环单元，训练高效 | 短期波动预测 |
| **XGBoost** | 机器学习 | 梯度提升树，特征重要性 | 特征驱动预测 |
| **Moirai** | 时序基础模型 | 零样本学习，通用性强 | 长期趋势预测 |

### 2. Stacking 融合机制
- **简单平均融合**：等权重集成各模型预测
- **元学习融合**：使用 XGBoost 作为元模型，学习最优权重
- **自适应选择**：根据预测时间尺度自动调整模型权重

### 3. 智能碳管理
- 📈 **碳预算设置**：设定月度碳排放目标
- 🎯 **实时追踪**：监控当前排放与预算进度
- 💰 **碳积分系统**：基于预测准确度奖励减碳行为
- 🏅 **荣誉体系**：等级晋升与成就徽章

### 4. 个性化减碳建议
- 🥗 **饮食建议**：低碳食谱推荐，无肉日提醒
- 🚌 **出行建议**：公共交通 vs 私家车排放对比
- 🏠 **用能建议**：家庭能源消耗优化方案

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit 前端                          │
│  ┌───────────┬───────────┬───────────┬───────────┐         │
│  │ 智能预测  │ 减碳计划  │ 碳积分    │ 全局指标  │         │
│  └───────────┴───────────┴───────────┴───────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      服务层 (Services)                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │PredictionSvc │ │ TrainingSvc  │ │  TradeSvc    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐                         │
│  │ForecasterMgr │ │ CarbonEngine │                         │
│  └──────────────┘ └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      模型层 (Models)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  LSTM    │ │   GRU    │ │XGBoost   │ │ Moirai   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                    ┌──────────────┐                         │
│                    │Stacking Meta │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据层 (Data)                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │LSTM Processing│ │XGB Processing│ │Moirai Proc  │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 类别 | 技术 |
|------|------|
| **前端框架** | Streamlit, Pyecharts, Streamlit-Echarts |
| **深度学习** | PyTorch (LSTM, GRU) |
| **机器学习** | XGBoost, Scikit-learn |
| **时序模型** | Moirai (Uni2TS) |
| **数据处理** | NumPy, Pandas, SciPy |
| **可视化** | Matplotlib, Pyecharts |
| **数据存储** | SQLite, MySQL (可选) |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 操作系统：Windows / Linux / macOS
- 推荐显存：4GB+ (用于 GPU 加速)

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/disdorqin/EcoLife.git
cd EcoLife
```

#### 2. 创建虚拟环境

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.template .env

# 编辑 .env 文件，配置数据库等参数（如使用 MySQL）
```

### 运行方式

#### 方式一：启动前端界面（推荐）

```bash
streamlit run app.py
```

然后在浏览器中访问 `http://localhost:8501`

#### 方式二：命令行交互模式

```bash
python main.py
```

#### 方式三：命令行参数模式

```bash
# 训练模式（默认 XGBoost）
python main.py --train

# 启用 Stacking 融合
python main.py --train --stack

# 指定模型组合
python main.py --train --models 1,3     # LSTM + XGBoost
python main.py --train --models 1,2,3,4 --stack  # 全模型融合

# 查看帮助
python main.py --help
```

**参数说明：**
- `--train`: 训练模式
- `--stack`: 启用 Stacking 融合
- `--models`: 模型选择 (1=LSTM, 2=GRU, 3=XGBoost, 4=Moirai)
- `--interactive`: 强制交互模式
- `--data`: 数据文件路径

---

## 📁 项目结构

```
EcoLife/
├── app.py                          # Streamlit 主入口
├── main.py                         # 命令行交互入口
├── requirements.txt                # Python 依赖
├── packages.txt                    # 系统依赖
├── README.md                       # 项目说明
├── .env.template                   # 环境变量模板
│
├── config/                         # 配置文件
│   ├── __init__.py
│   ├── settings.py                 # 设置加载器
│   └── settings.yaml               # YAML 配置
│
├── src/                            # 源代码
│   ├── core/                       # 核心工具
│   │   ├── config/                 # 配置管理
│   │   ├── exceptions/             # 异常定义
│   │   └── utils/                  # 工具函数
│   │       ├── logger.py           # 日志系统
│   │       └── training_progress.py # 训练进度
│   │
│   ├── data/                       # 数据处理
│   │   ├── lstm_processing.py      # LSTM 数据预处理
│   │   ├── xgboost_processing.py   # XGBoost 数据预处理
│   │   ├── moirai_processing.py    # Moirai 数据预处理
│   │   └── mysql_client.py         # MySQL 客户端
│   │
│   ├── model_layer/                # 模型封装层
│   │   ├── base_model.py           # 基类定义
│   │   ├── lstm_model.py           # LSTM 封装
│   │   ├── xgboost_model.py        # XGBoost 封装
│   │   ├── model_trainer.py        # 多模型训练器
│   │   ├── model_evaluator.py      # 模型评估器
│   │   └── metrics_manager.py      # 指标管理
│   │
│   ├── models/                     # 模型定义
│   │   ├── lstm_model.py           # LSTM 模型
│   │   ├── gru_model.py            # GRU 模型
│   │   ├── xgboost_model.py        # XGBoost 模型
│   │   ├── moirai_model.py         # Moirai 模型
│   │   ├── stacking_manager.py     # Stacking 管理器
│   │   └── model_registry.py       # 模型注册表
│   │
│   ├── services/                   # 业务服务
│   │   ├── prediction_service.py   # 预测服务
│   │   ├── training_service.py     # 训练服务
│   │   ├── trade_service.py        # 交易服务
│   │   ├── forecaster_manager.py   # 预测调度器
│   │   └── carbon_engine.py        # 碳引擎
│   │
│   ├── runner/                     # 运行器
│   │   ├── pipeline_router.py      # 管道路由
│   │   ├── auto_diagnosis.py       # 自动诊断
│   │   └── lstm_runner.py          # LSTM 运行器
│   │
│   ├── pipeline/                   # 处理管道
│   │   └── lstm_pipeline.py        # LSTM 管道
│   │
│   ├── trade_logic/                # 交易逻辑
│   │   └── optimizer.py            # 优化器
│   │
│   ├── visualization/              # 可视化
│   │   └── echarts_options.py      # ECharts 配置
│   │
│   └── utils/                      # 通用工具
│       ├── data_processor.py       # 数据处理器
│       ├── env.py                  # 环境工具
│       ├── eta.py                  # ETA 计算
│       └── paths.py                # 路径工具
│
├── data/                           # 数据目录
│   └── personal_carbon_footprint_behavior.csv
│
├── models/                         # 模型存储
│   └── checkpoints/                # 模型检查点
│
├── logs/                           # 日志目录
│   ├── plots/                      # 生成的图表
│   └── metrics/                    # 评估指标
│
└── docs/                           # 文档目录
    ├── 项目执行流程详解.md
    ├── 模型训练与预测流程详解.md
    └── 调参优化指南.md
```

---

## 🧠 模型说明

### LSTM (长短期记忆网络)

```python
# 配置参数
input_dim = 10        # 输入特征维度
hidden_dim = 128      # 隐藏层维度
num_layers = 2        # LSTM 层数
bidirectional = True  # 双向 LSTM
```

**特点：**
- 双向 LSTM 捕捉前后文依赖
- 自适应学习率调度 (ReduceLROnPlateau)
- 早停机制防止过拟合

### XGBoost (极端梯度提升)

```python
# 配置参数
n_estimators = 100    # 树的数量
learning_rate = 0.1   # 学习率
max_depth = 6         # 树的最大深度
```

**特点：**
- 特征重要性分析
- 内置正则化防止过拟合
- 训练速度快，解释性强

### Stacking 融合

```
第一层（基学习器）: LSTM, GRU, XGBoost, Moirai
                    ↓
第二层（元学习器）: XGBoost Regressor
                    ↓
              最终预测结果
```

**融合策略：**
- 简单平均：各模型预测的算术平均
- 加权融合：元模型学习最优权重
- 动态选择：根据时间尺度调整权重

---

## 📖 使用指南

### 前端界面使用

#### 1. 智能预测页面
- 选择时间维度（天/周/月）
- 查看各模型预测曲线
- 对比融合预测结果
- 查看模型贡献度雷达图

#### 2. 减碳计划页面
- 获取个性化饮食建议
- 查看无肉日提醒
- 浏览低碳食谱推荐

#### 3. 碳积分荣誉页面
- 查看本周获得积分
- 追踪总积分和等级
- 查看积分明细

#### 4. 全局指标页面
- 查看模型性能指标
- 对比各模型误差
- 分析历史表现

### 命令行使用

#### 训练单个模型

```bash
# 训练 XGBoost
python main.py --train --models 3

# 训练 LSTM
python main.py --train --models 1
```

#### 启用 Stacking 融合

```bash
# LSTM + XGBoost 融合
python main.py --train --stack --models 1,3

# 全模型融合
python main.py --train --stack --models 1,2,3,4
```

---

## 📊 性能指标

### 模型性能对比（验证集）

| 模型 | MAE | RMSE | R² | 准确率 |
|------|-----|------|----|--------|
| LSTM | 2.01 | 2.54 | -0.02 | 50.6% |
| GRU | 2.01 | 2.54 | -0.02 | 50.5% |
| XGBoost | 0.25 | 0.33 | 0.98 | 95.7% |
| Moirai | 1.96 | 2.44 | -0.02 | 46.8% |
| **Stacking 融合** | **1.21** | **1.51** | **0.64** | **83.8%** |

> 注：XGBoost 在本数据集上表现最佳，Stacking 融合有效整合了各模型优势。

### 分类指标（碳排放等级预测）

| 模型 | 精确率 | 召回率 | F1 分数 |
|------|--------|--------|---------|
| XGBoost | 0.25 | 0.59 | 0.35 |
| LSTM | 0.24 | 0.55 | 0.34 |
| Stacking | 0.84 | 0.84 | 0.84 |

---

## 👥 开发团队

**EcoLife 团队**

- 项目地址：https://github.com/disdorqin/EcoLife
- 参赛组别：计算机设计大赛

---

## 📄 许可证

本项目仅供学习和竞赛使用。

---

## 🙏 致谢

感谢计算机设计大赛组委会提供的平台！

---

<div align="center">

**🌿 EcoLife - 让每一次呼吸都更清新**

Made with ❤️ by EcoLife Team

</div>