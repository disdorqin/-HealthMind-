# ⚡ 风芒可测 - 快速开始指南（重构版）

## 🎯 目标

这个指南将帮助你在本地或云端快速部署 **风芒可测 - 电力预测与交易优化系统**。

## 📋 重构亮点

✨ **一切皆 Python** - 无需 Flask 后端，Streamlit 直接调用业务逻辑  
✨ **自愈系统** - 应用启动时自动检查和生成模型  
✨ **跨平台兼容** - Windows、Linux、macOS 无差别运行  
✨ **Streamlit Cloud 就绪** - 一键部署到云端  
✨ **环境隔离** - 敏感信息使用 secrets 管理  

---

## 🖥️ 本地运行（5分钟快速开始）

### 前提条件

```bash
# Python 3.8+ 和 pip
python --version  # 应该是 3.8 或更高
pip --version
```

### 步骤 1: 克隆并进入项目

```bash
cd d:\作业\竞赛\计算机设计大赛
```

### 步骤 2: 安装依赖

```bash
pip install -r requirements.txt
```

第一次安装可能需要 2-5 分钟，主要是 PyTorch 比较大。

### 步骤 3: 一键启动 ✨

```bash
python main.py
```

程序将自动执行以下步骤：
1. ✓ [1/4] 检查数据文件 (`data/data.csv`)
2. ✓ [2/4] 检查模型文件（如果不存在自动训练）
3. ✓ [3/4] 启动模型训练（仅在模型缺失时）
4. ✓ [4/4] 启动 Streamlit 应用

等待看到输出：
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

### 步骤 4: 打开浏览器

访问 `http://localhost:8501` 即可看到应用界面！

---

## 🎛️ CommandLine 选项

如果不想执行全部步骤，可以使用以下命令：

```bash
# 仅检查数据文件
python main.py --data-only

# 仅训练模型（需要数据文件存在）
python main.py --train-only

# 在自定义端口启动应用（例如 9000）
python main.py --port 9000

# 跳过模型训练，直接启动应用（模型必须已存在）
python main.py --skip-train

# 显示帮助
python main.py --help
```

---

## 📚 应用功能说明

启动后，Streamlit 应用提供以下页面：

### 📊 数据预览
- 查看原始数据的统计信息
- 检查数据格式和列数据类型
- 识别缺失值

### 🤖 模型训练
- 配置训练参数（轮数、批大小等）
- 一键启动模型训练
- 实时查看训练进度

### 🔮 预测结果
- 显示 24 小时功率预测曲线
- 展示预测统计（最小值、最大值、平均值）
- 可视化预测结果

### 💰 交易建议
- 基于分时电价和预测结果生成买卖建议
- 展示预期收益和成本节约
- 了解削峰填谷机会

### 📈 交易指标
- 日发电量计算
- 收益潜力评估
- 削峰和填谷效益分析

### ⚠️ 风险分析
- 预测波动性评估
- 风险等级判定（极低/低/中/高/极高）
- 操作建议

---

## ☁️ 部署到 Streamlit Cloud（云端）

### 前置条件

1. GitHub 账号（免费）
2. Streamlit 账号（免费，关联 GitHub）

### 部署步骤

#### 1️⃣ 推送代码到 GitHub

```bash
# 初始化 Git（如果还没有）
git init
git add .
git commit -m "Initial commit: Streamlit Cloud 优化版"

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

#### 2️⃣ 在 Streamlit Cloud 中部署

1. 访问 https://share.streamlit.io/
2. 点击 "New app"
3. 选择以下配置：
   - **Repository**: `your-username/your-repo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. 点击 "Deploy"

#### 3️⃣ 等待部署完成

第一次部署需要 1-2 分钟，后续更新更快。

部署完成后，你将获得一个公开链接，类似：
```
https://your-username-your-repo-xxxxxxxxxx.streamlit.app
```

**就这样！你的应用已经上线了！** 🎉

---

## 🔐 如何安全地处理敏感信息

### 本地开发

1. 复制 `.env.template` 为 `.env`：
   ```bash
   cp .env.template .env
   ```

2. 编辑 `.env`，填入敏感信息：
   ```
   DB_PASSWORD=your_actual_password
   DB_HOST=192.168.1.100
   ```

3. 在代码中读取（Python）：
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   db_password = os.getenv('DB_PASSWORD')
   ```

**⚠️ 重要**: `.env` 已在 `.gitignore` 中，不会被 Git 提交。

### Streamlit Cloud 部署

1. 在你的应用页面，点击 "Settings"
2. 点击 "Secrets" 标签
3. 添加密钥，格式为 TOML：
   ```toml
   [database]
   host = "your-db-host"
   user = "your-db-user"
   password = "your-password"
   
   [model]
   data_path = "data/data.csv"
   ```

4. 在代码中读取（Python）：
   ```python
   import streamlit as st
   
   db_host = st.secrets["database"]["host"]
   ```

---

## 🔨 故障排除

### 问题 1: `ModuleNotFoundError: No module named 'torch'`

**解决方案**：重新安装依赖

```bash
pip install -r requirements.txt --no-cache-dir
```

### 问题 2: 模型训练失败

**检查清单**：
- ✓ `data/data.csv` 文件存在？
- ✓ 磁盘空间充足（至少 500 MB）？
- ✓ 内存充足（RAM 至少 4 GB）？

**快速修复**：
```bash
python main.py --train-only
```

### 问题 3: Streamlit 应用加载很慢

**优化方案**：
- 使用 `@st.cache_resource` 缓存模型（已实现）
- 使用 `@st.cache_data` 缓存预测结果（已实现）
- 考虑减少预测数据的大小

### 问题 4: Streamlit Cloud 上模型加载失败

**原因**：容器环境不同，模型可能无法加载  
**解决方案**：
- 从云存储（S3）动态下载模型
- 使用更小的模型文件
- 确保模型在同一个 Python 版本和 PyTorch 版本下训练

---

## 📊 项目结构速览

```
.
├── app.py                    # Streamlit 主应用 ⭐
├── main.py                   # 本地启动脚本 ⭐
├── requirements.txt          # 依赖清单 ⭐
├── .env.template            # 环境变量模板
├── .gitignore               # Git 忽略文件
│
├── src/logic/               # 业务逻辑（新！）
│   ├── __init__.py
│   └── trade.py             # 交易优化逻辑
│
├── src/data_layer/          # 数据处理
├── src/model_layer/         # 模型层
├── src/pipeline/            # 管道
│
├── data/                    # 数据文件
│   └── data.csv
├── models/                  # 训练好的模型
│   └── lstm_forecaster.pth
└── .streamlit/
    └── secrets.toml         # Streamlit 密钥
```

---

## 🚀 性能优化建议

### 本地运行

```python
# 为了加快首次加载，可以预先加载模型
@st.cache_resource
def load_model():
    # 这个函数只在应用启动时执行一次
    return load_lstm_model('models/lstm_forecaster.pth')

# 缓存数据读取
@st.cache_data(ttl=300)  # 5分钟缓存
def load_data():
    return pd.read_csv('data/data.csv')
```

### Streamlit Cloud

- 压缩模型文件大小（< 50MB 为佳）
- 使用 CDN 或云存储加载大文件
- 避免在应用启动时进行重计算

---

## 📈 下一步（高级功能）

一旦基础版本运行稳定，可以考虑：

1. **数据库集成** - 从 MySQL/PostgreSQL 读取实时数据
2. **用户认证** - 添加登录功能和细粒度权限检查
3. **实时数据接入** - WebSocket 支持流式数据
4. **更多可视化** - Plotly、Altair、PyDeck 等
5. **模型版本管理** - MLflow 或 Weights & Biases 等
6. **自动化部署** - GitHub Actions CI/CD 流水线

---

## 💬 需要帮助？

1. **检查日志** - 查看 `logs/` 目录中的日志文件
2. **阅读完整文档** - 见 `PROJECT_REFACTORING.md`
3. **查看示例代码** - `src/logic/` 中有完整的业务逻辑示例

---

## ✅ 验证清单

部署前确保完成以下步骤：

- [ ] Python 3.8+ 已安装
- [ ] 依赖已通过 `pip install -r requirements.txt` 安装
- [ ] `data/data.csv` 数据文件存在
- [ ] 本地测试通过：`python main.py`
- [ ] 应用能在 `http://localhost:8501` 正常访问
- [ ] (可选) 代码已推送到 GitHub
- [ ] (可选) 已在 Streamlit Cloud 中部署

---

**祝你部署成功！如有问题，查看 `PROJECT_REFACTORING.md` 获取更多详情。** 🎉

**版本**: 2.0.0 (Streamlit 单一应用优化版)  
**最后更新**: 2026年3月14日
