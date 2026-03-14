# 项目重构总结 - Streamlit Cloud 部署优化版

## 📋 概述

本项目已从 **Flask + Streamlit 混合架构** 重构为 **纯 Streamlit 单一应用**，专门针对 Streamlit Cloud 部署进行了优化。

### 主要改进

✅ **去除 Flask 依赖** - Streamlit 直接调用 Python 业务逻辑  
✅ **自愈式运行** - 应用启动时自动检查模型，缺失时自动训练  
✅ **路径跨平台兼容** - 使用 `pathlib.Path` 确保 Windows/Linux 一致性  
✅ **环境隔离** - 所有敏感信息通过 `st.secrets` 或 `.env` 读取  
✅ **缓存机制** - 使用 `@st.cache_resource` 缓存模型，`@st.cache_data` 缓存数据  
✅ **Streamlit Cloud 兼容** - 完全支持云端部署，无需本地后端  

---

## 🏗️ 新的目录结构

```
.
├── app.py                              # Streamlit 主应用入口（根目录）
├── main.py                             # 本地一键测试脚本
├── requirements.txt                    # 完整依赖清单
├── .env.template                       # 环境变量模板
├── .gitignore                          # Git 忽略配置
│
├── .streamlit/
│   └── secrets.toml                    # Streamlit 密钥配置（仅生成模板）
│
├── src/
│   ├── logic/                          # 核心业务逻辑层（新）
│   │   ├── __init__.py
│   │   └── trade.py                    # 交易优化逻辑
│   │
│   ├── models/                         # 模型层
│   │   ├── lstm_model.py
│   │   ├── xgboost_model.py
│   │   └── ...
│   │
│   ├── data_layer/                     # 数据处理层
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   ├── feature_engineering.py
│   │   └── ...
│   │
│   ├── pipeline/                       # 管道层
│   │   └── lstm_pipeline.py
│   │
│   ├── runner/                         # 运行器
│   │   └── pipeline_router.py
│   │
│   ├── app_layer/                      # 应用层（仅保留兼容旧代码）
│   │
│   └── core/                           # 核心工具
│       └── utils/
│           └── logger.py
│
├── data/                               # 静态数据
│   ├── data.csv                        # 训练数据
│   └── ...
│
├── models/                             # 训练好的模型
│   ├── lstm_forecaster.pth
│   └── ...
│
└── docs/                               # 文档
    └── ...
```

---

## 🚀 快速开始

### 1️⃣ 本地运行（推荐用于开发）

```bash
# 安装依赖
pip install -r requirements.txt

# 一键启动（检查数据 -> 训练模型 -> 启动应用）
python main.py

# 仅检查数据
python main.py --data-only

# 仅训练模型
python main.py --train-only

# 在自定义端口启动
python main.py --port 9000
```

### 2️⃣ Streamlit Cloud 部署

#### 准备工作

1. **创建 GitHub 仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/your-repo.git
   git push -u origin main
   ```

2. **配置密钥** (可选但推荐)
   
   在 Streamlit Cloud 的 App settings 中添加：
   ```
   STREAMLIT_DATABASE_URL = your_db_connection_string
   STREAMLIT_API_KEY = your_api_key
   ```

3. **部署**
   
   在 Streamlit Cloud 中：
   - 点击 "New app"
   - 选择 GitHub 仓库
   - 选择分支和主文件：`app.py`
   - 点击 "Deploy"

#### 部署注意事项

⚠️ **Streamlit Cloud 环境限制**：
- 无法持久化保存文件到磁盘（容器临时存储）
- 内存限制 ~1GB
- 若模型文件太大，考虑从云存储（如 S3）加载

建议做法：
```python
# 在 app.py 的 ensure_model_exists() 函数中
# 如果模型不存在，从 S3 或其他云存储下载
```

---

## 📦 架构说明

### 业务逻辑层 (src/logic/)

将原来分散在 Flask API 中的逻辑集中到此处：

```python
from src.logic import BusinessLogic, TradeOptimizer

# 直接调用，无需 HTTP 请求
result = BusinessLogic.predict(data_path, model_path)
advice = BusinessLogic.get_trade_advice(data_path, model_path)
```

### 自愈式运行机制

```python
# app.py 启动时自动执行
if not ensure_model_exists():
    # 缺失模型时自动触发训练
    auto_train_model()
```

### 缓存机制

- **@st.cache_resource** - 缓存模型（全局单次加载）
- **@st.cache_data** - 缓存 CSV 数据、预测结果（5分钟）

```python
@st.cache_resource
def load_predictions(data_path, model_path):
    return BusinessLogic.predict(data_path, model_path)
```

---

## 🔐 环境变量管理

### 本地开发

1. 复制 `.env.template` 为 `.env`：
   ```bash
   cp .env.template .env
   ```

2. 编辑 `.env` 填入实际配置：
   ```env
   DB_HOST=localhost
   DB_PASSWORD=your_actual_password
   ```

3. 在代码中读取：
   ```python
   import os
   from pathlib import Path
   
   # 或使用 st.secrets（Streamlit Cloud 推荐）
   db_host = st.secrets['database']['host']
   ```

### Streamlit Cloud 部署

在 App settings 中设置 secrets，格式为 TOML：

```toml
[database]
host = "your-db-host"
user = "your-db-user"
password = "your-password"

[model]
data_path = "data/data.csv"
model_path = "models/lstm_forecaster.pth"
```

---

## 🔄 迁移指南（从旧版本）

### 变更对比

| 功能 | 旧架构 | 新架构 |
|------|--------|--------|
| 主入口 | `src/frontend/app.py` | `app.py` (根目录) |
| 后端 API | Flask (`src/backend/api.py`) | 内置函数 (`src/logic/`) |
| 业务逻辑 | 分散在 `api_service.py` | 集中在 `src/logic/` |
| 进程管理 | 需要启动 Flask 和 Streamlit | 仅启动 Streamlit |
| 部署方式 | 本地部署 + PM2 | Streamlit Cloud 一键部署 |

### 代码迁移

**旧代码（通过 HTTP 调用）：**
```python
response = requests.post('http://localhost:5000/api/predict', json={...})
predictions = response.json()['result']['predictions']
```

**新代码（直接调用）：**
```python
from src.logic import BusinessLogic

result = BusinessLogic.predict(data_path, model_path)
predictions = result['result']['predictions']
```

---

## 📊 依赖优化

### 去除的依赖
- ❌ `flask>=2.3.0`
- ❌ `flask-cors>=4.0.0`
- ❌ `requests>=2.31.0`（可选，仅在调用外部 API 时需要）

### 新增的依赖
- ✅ `streamlit>=1.28.0`（已有）
- ✅ `streamlit-echarts>=0.4.0`（已有）

**总依赖数减少 ~30%，安装时间更快！**

---

## 🧪 测试验证

运行测试套件确保所有功能正常：

```bash
# 基础功能测试
python -m pytest tests/ -v

# 集成测试（启动完整应用）
python main.py --data-only    # 验证数据层
python main.py --train-only   # 验证模型训练
python main.py                # 验证完整应用

# Streamlit 应用本地测试
streamlit run app.py
```

---

## 📝 配置文件说明

### `.env.template`
环境变量模板，包含所有可配置项

### `.streamlit/secrets.toml`
Streamlit 密钥文件（仅本地使用，Cloud 端在 App settings 中配置）

### `requirements.txt`
- 固定主要库的版本号以确保稳定性
- 去除了 Flask 相关依赖

---

## ⚠️ 常见问题

### Q1: 启动时模型自动生成失败？
A: 检查 `data/data.csv` 是否存在，确保有足够的磁盘空间

### Q2: 在 Streamlit Cloud 上模型加载很慢？
A: 使用 `@st.cache_resource` 缓存模型，避免重复加载

### Q3: 能否恢复 Flask 后端？
A: 原代码已保留在 `src/app_layer/` 中，但不推荐使用

### Q4: 如何实现多用户隔离？
A: 建议使用 Streamlit Secrets 或连接到共享数据库

---

## 🎯 下一步优化方向

- [ ] 添加数据库持久化（MySQL/PostgreSQL）
- [ ] 实现用户认证和权限管理
- [ ] 支持实时数据流（WebSocket）
- [ ] 添加数据可视化仪表板（Plotly/Altair）
- [ ] 模型版本管理和 A/B 测试
- [ ] 部署监控和日志系统
- [ ] CI/CD 流水线集成

---

## 📞 支持

遇到问题？
1. 查看日志文件：`logs/` 目录
2. 检查 `.env` 和 `secrets.toml` 配置
3. 参考 [Streamlit 官方文档](https://docs.streamlit.io/)

---

**版本**: 2.0.0 (Streamlit Cloud 优化版)  
**最后更新**: 2026年3月14日
