# EcoLife 项目改进方案（务实版）

> 基于团队现状定制的可行方案
> - 数据集：Kaggle Personal Carbon Footprint
> - 现状：框架已完成，时间紧张，多人协作
> - 目标：在现有基础上最大化比赛竞争力

---

## 📋 目录

1. [团队现状与约束](#1-团队现状与约束)
2. [国奖作品核心特点分析](#2-国奖作品核心特点分析)
3. [改进方向一：用户交互与数据系统](#3-改进方向一用户交互与数据系统)
4. [改进方向二：模型优化策略](#4-改进方向二模型优化策略)
5. [改进方向三：前端与部署](#5-改进方向三前端与部署)
6. [协作开发指南](#6-协作开发指南)
7. [时间规划与任务分配](#7-时间规划与任务分配)

---

## 1. 团队现状与约束

### 1.1 当前状态

| 维度 | 状态 | 说明 |
|------|------|------|
| **数据集** | Kaggle Personal Carbon Footprint | 已固定，不宜大改 |
| **框架** | 完整训练 + 预测流程 | main.py 可运行 |
| **团队** | 多人协作 | 工程经验有限 |
| **时间** | 紧张 | 需快速产出 |
| **代码管理** | GitHub 仓库 | 存在冲突风险 |

### 1.2 约束条件

⚠️ **不推荐的大改动**：
- ❌ 重新设计特征工程（会破坏现有流程）
- ❌ 更换数据集（需要重新验证）
- ❌ 重构核心架构（时间不够）

✅ **推荐的改进方向**：
- ✅ 在现有框架上添加功能
- ✅ 前端和展示层优化
- ✅ 模型参数调优和对比实验
- ✅ 部署和演示准备

---

## 2. 国奖作品核心特点分析

### 2.1 国奖作品必备要素

根据近几年计算机设计大赛国奖作品分析，以下要素是获奖关键：

| 要素 | 重要性 | EcoLife 当前状态 | 改进建议 |
|------|--------|-----------------|----------|
| **完整的功能演示** | ⭐⭐⭐⭐⭐ | ✅ 已有 | 保持并优化 |
| **可公开访问的 Demo** | ⭐⭐⭐⭐⭐ | ❌ 本地运行 | **优先部署** |
| **清晰的创新点** | ⭐⭐⭐⭐⭐ | ⚠️ 需提炼 | 提炼多模型融合亮点 |
| **量化效果对比** | ⭐⭐⭐⭐ | ⚠️ 部分有 | 增加对比实验 |
| **专业的展示材料** | ⭐⭐⭐⭐⭐ | ⚠️ 需完善 | 制作视频 + 文档 |
| **实际应用场景** | ⭐⭐⭐⭐ | ⚠️ 需加强 | 添加用户系统 |
| **技术深度** | ⭐⭐⭐⭐ | ✅ 多模型融合 | 深化 Stacking 机制 |

### 2.2 与 EcoLife 最相关的国奖特点

**1. 多模型融合**（我们的核心优势）
- 国奖作品普遍采用集成学习方法
- EcoLife 已有 4 模型 + Stacking，技术路线正确
- **建议**：深化 Stacking 机制说明，突出创新点

**2. 交互式展示**
- 国奖作品都有良好的交互体验
- EcoLife 当前 Streamlit 界面较简单
- **建议**：优化 UI，增加交互组件

**3. 可访问的 Demo**
- 国奖作品基本都有在线演示
- EcoLife 仅本地运行
- **建议**：优先部署到 Streamlit Cloud

**4. 完整的文档**
- 国奖作品文档齐全
- EcoLife 已有 README 和技术文档
- **建议**：补充对比实验文档

---

## 3. 改进方向一：用户交互与数据系统

### 3.1 现状分析

**当前架构**：
```
用户 → Streamlit 界面 → 读取 CSV 文件 → 预测 → 展示
```

**问题**：
- 数据是静态的（Kaggle 数据集）
- 无法保存用户历史数据
- 无法实现个性化预测

### 3.2 改进方案（轻量级）

考虑到时间和团队能力，推荐**轻量级用户系统**方案：

#### 方案 A：Streamlit Session State + JSON 存储（推荐 ⭐⭐⭐⭐⭐）

**优点**：
- 无需数据库，代码改动最小
- 可实现基本的用户交互
- 适合演示

**实施步骤**：

**1. 创建用户数据管理模块**（新建 `src/utils/user_data.py`）：
```python
"""
用户数据管理模块
使用 JSON 文件存储用户数据，无需数据库
"""
import json
import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/user_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_user_file(user_id: str) -> Path:
    """获取用户数据文件路径"""
    return DATA_DIR / f"{user_id}.json"

def create_user(user_id: str, username: str):
    """创建用户"""
    user_data = {
        "user_id": user_id,
        "username": username,
        "created_at": datetime.now().isoformat(),
        "carbon_records": [],
        "total_credits": 0,
        "settings": {
            "weekly_budget": 350,  # 周碳预算 (kg)
            "diet_type": "omnivore"  # 饮食类型
        }
    }
    
    with open(get_user_file(user_id), 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    return user_data

def get_user(user_id: str) -> dict:
    """获取用户数据"""
    user_file = get_user_file(user_id)
    if not user_file.exists():
        return None
    
    with open(user_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def add_carbon_record(user_id: str, date: str, carbon_value: float, 
                      predicted_value: float = None):
    """添加碳足迹记录"""
    user_data = get_user(user_id)
    if not user_data:
        return False
    
    record = {
        "date": date,
        "carbon_value": carbon_value,
        "predicted_value": predicted_value or carbon_value,
        "timestamp": datetime.now().isoformat()
    }
    
    user_data["carbon_records"].append(record)
    
    # 保存
    with open(get_user_file(user_id), 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    return True

def get_user_history(user_id: str, days: int = 30) -> list:
    """获取用户历史记录"""
    user_data = get_user(user_id)
    if not user_data:
        return []
    
    records = user_data.get("carbon_records", [])[-days:]
    return records

def add_credits(user_id: str, credits: int, reason: str = ""):
    """添加积分"""
    user_data = get_user(user_id)
    if not user_data:
        return False
    
    user_data["total_credits"] += credits
    
    # 记录积分变更
    if "credit_history" not in user_data:
        user_data["credit_history"] = []
    
    user_data["credit_history"].append({
        "credits": credits,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    })
    
    with open(get_user_file(user_id), 'w', encoding='utf-8') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    return True
```

**2. 修改 app.py 添加用户登录功能**：
```python
# 在 app.py 开头添加
import streamlit as st
from src.utils.user_data import create_user, get_user, add_carbon_record, get_user_history

# 添加侧边栏用户登录
with st.sidebar:
    st.title("👤 用户登录")
    
    if "user_id" not in st.session_state:
        # 登录界面
        username = st.text_input("用户名", key="login_username")
        if st.button("登录/注册"):
            if username:
                st.session_state.user_id = username
                st.session_state.username = username
                
                # 如果是新用户，创建账户
                user = get_user(username)
                if not user:
                    create_user(username, username)
                st.rerun()
    else:
        # 已登录状态
        st.success(f"欢迎，{st.session_state.username}!")
        
        if st.button("退出登录"):
            st.session_state.clear()
            st.rerun()
        
        # 显示用户统计
        user = get_user(st.session_state.user_id)
        if user:
            st.metric("总积分", user.get("total_credits", 0))
            st.metric("记录数", len(user.get("carbon_records", [])))
```

**3. 在预测后保存用户数据**：
```python
# 在预测功能后添加
if "user_id" in st.session_state:
    # 保存预测记录
    add_carbon_record(
        st.session_state.user_id,
        datetime.now().strftime("%Y-%m-%d"),
        actual_value,  # 实际值（用户输入或模拟）
        predicted_value  # 预测值
    )
```

---

#### 方案 B：SQLite 数据库（备选 ⭐⭐⭐）

如果团队有数据库经验，可使用 SQLite：

**优点**：
- 数据管理更规范
- 支持复杂查询

**缺点**：
- 代码改动较大
- 需要处理数据库连接

**实施代码**（已在 `database/user_db.py` 有基础实现）：
```python
# 简化版实现
import sqlite3

class SimpleUserDB:
    def __init__(self):
        self.conn = sqlite3.connect("database/ecolife.db", check_same_thread=False)
        self._init_tables()
    
    def _init_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS carbon_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                date DATE,
                carbon_value REAL,
                predicted_value REAL,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        self.conn.commit()
    
    def add_record(self, username, date, carbon_value, predicted_value):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO carbon_records (username, date, carbon_value, predicted_value) VALUES (?, ?, ?, ?)",
            (username, date, carbon_value, predicted_value)
        )
        self.conn.commit()
    
    def get_history(self, username, days=30):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT date, carbon_value, predicted_value FROM carbon_records WHERE username = ? ORDER BY date DESC LIMIT ?",
            (username, days)
        )
        return cursor.fetchall()

# 在 app.py 中使用
db = SimpleUserDB()
```

---

## 4. 改进方向二：模型优化策略

### 4.1 现状分析

| 模型 | 当前 R² | 问题 | 改进空间 |
|------|---------|------|----------|
| LSTM | ≈ -0.02 | 欠拟合 | 中等 |
| GRU | ≈ -0.02 | 欠拟合 | 中等 |
| XGBoost | ≈ 0.98 | 表现好 | 保持 |
| Moirai | ≈ -0.02 | 零样本模式 | 有限 |
| Stacking | ≈ 0.64 | 依赖基模型 | 中等 |

### 4.2 改进方案

#### 方案 A：参数调优（推荐 ⭐⭐⭐⭐⭐）

**原因**：
- 不改变代码逻辑
- 团队成员都能参与
- 快速见效

**需要调整的参数**：

**LSTM 参数**（`src/models/lstm_model.py`）：
```python
# 当前配置
hidden_dim: int = 128,      # 可尝试：64, 128, 256
num_layers: int = 2,        # 可尝试：1, 2, 3
learning_rate: float = 1e-3, # 可尝试：1e-4, 5e-4, 1e-3
epochs: int = 100,          # 可尝试：50, 100, 200
batch_size: int = 32,       # 可尝试：16, 32, 64
patience: int = 15,         # 可尝试：10, 15, 20
```

**GRU 参数**（`src/models/gru_model.py`）：
```python
# 当前配置
hidden_dim: int = 64,       # 可尝试：64, 128, 256
num_layers: int = 2,        # 可尝试：1, 2, 3
learning_rate: float = 1e-3, # 可尝试：1e-4, 5e-4, 1e-3
```

**实验记录表**（建议使用在线文档共享）：

| 日期 | 模型 | hidden_dim | num_layers | learning_rate | R² | MAE | 备注 |
|------|------|------------|------------|---------------|----|-----|------|
| 3.17 | LSTM | 128 | 2 | 0.001 | -0.02 | 2.01 | 基线 |
| 3.17 | LSTM | 256 | 2 | 0.001 | ? | ? | 实验 1 |
| 3.17 | LSTM | 128 | 3 | 0.001 | ? | ? | 实验 2 |

**执行命令**：
```bash
# 每次修改参数后运行
python main.py --train --models 1  # LSTM
python main.py --train --models 2  # GRU
```

---

#### 方案 B：对比实验（推荐 ⭐⭐⭐⭐⭐）

**目的**：说明为什么选择这些模型

**对比维度**：

**1. 模型特性对比**
| 模型 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| LSTM | 长序列依赖 | 捕捉长期模式 | 训练慢 |
| GRU | 中等序列 | 训练快 | 表达能力稍弱 |
| XGBoost | 表格数据 | 精度高，解释性强 | 无法处理序列 |
| Moirai | 零样本预测 | 无需训练 | 依赖预训练质量 |

**2. 性能对比实验**

修改 `main.py` 添加对比实验模式：
```bash
# 运行所有模型对比
python main.py --train --models 1,2,3,4 --stack
```

**3. 消融实验**（说明 Stacking 的必要性）
- 实验 1：仅 LSTM
- 实验 2：仅 XGBoost
- 实验 3：LSTM + XGBoost
- 实验 4：LSTM + XGBoost + GRU + Moirai (Stacking)

---

#### 方案 C：模型逻辑优化（谨慎 ⭐⭐⭐）

**如果时间允许，可考虑以下小改动**：

**1. 增加 LSTM 序列长度**
```python
# 修改 src/data/lstm_processing.py
# 将 window_size 从 3 改为 7
X, y, scaler = process_data_for_lstm(data_path, window_size=7)
```

**2. 添加 Dropout**
```python
# 在 src/models/lstm_model.py 的 _LSTMRegressor 类中
self.dropout = nn.Dropout(0.3)  # 添加 dropout
```

**⚠️ 注意**：模型逻辑改动需要重新验证所有实验，时间紧张时不建议。

---

## 5. 改进方向三：前端与部署

### 5.1 Streamlit Cloud 部署（强烈推荐 ⭐⭐⭐⭐⭐）

**为什么必须部署**：
1. 国奖作品都有可公开访问的 Demo
2. 方便评委体验
3. 展示项目完整性

**部署步骤**（1 小时完成）：

**步骤 1：确保 GitHub 仓库已更新**
```bash
git add .
git commit -m "准备 Streamlit Cloud 部署"
git push origin main
```

**步骤 2：在 Streamlit Cloud 部署**
1. 访问 https://streamlit.io/cloud
2. 点击 "Deploy an app"
3. 选择 "Connect your GitHub"
4. 授权 GitHub
5. 选择 `EcoLife` 仓库
6. 设置入口文件为 `app.py`
7. 点击 "Deploy!"

**步骤 3：获取公开链接**
- 部署成功后获得链接，如：`https://ecolife-demo.streamlit.app`
- 将此链接放入 README 和比赛材料

**⚠️ 注意事项**：
- 确保 `requirements.txt` 包含所有依赖
- 数据文件不能太大（<200MB）
- 模型文件如果太大，考虑使用 Git LFS

---

### 5.2 UI 优化（推荐 ⭐⭐⭐⭐）

**当前问题**：界面较简单，缺少视觉吸引力

**优化方案**：

**1. 添加专业配色**（修改 `app.py`）：
```python
st.markdown("""
<style>
    /* 主色调 */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #66BB6A;
        --accent-color: #FFA726;
    }
    
    /* 卡片样式 */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 按钮渐变 */
    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)
```

**2. 添加欢迎界面**：
```python
# 在 app.py 开头添加
st.set_page_config(
    page_title="EcoLife - 个人碳足迹管理",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加首页横幅
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #2E7D32;'>🌿 EcoLife</h1>
    <p style='font-size: 18px; color: #666;'>个人碳足迹预测与管理平台</p>
    <p style='color: #888;'>计算机设计大赛参赛作品</p>
</div>
""", unsafe_allow_html=True)
```

**3. 添加数据可视化**：
```python
# 在预测页面添加
import plotly.graph_objects as go

# 创建对比图
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['LSTM', 'GRU', 'XGBoost', 'Moirai', 'Stacking'],
    y=[-0.02, -0.02, 0.98, -0.02, 0.64],
    name='R² 分数',
    marker_color=['#ef5350', '#ef5350', '#66bb6a', '#ef5350', '#ffa726']
))
st.plotly_chart(fig, use_container_width=True)
```

---

### 5.3 新增网站功能（推荐 ⭐⭐⭐⭐）

**功能 1：实时预测输入**

在侧边栏添加用户输入：
```python
with st.sidebar:
    st.subheader("📝 今日数据输入")
    
    # 模拟用户输入
    transport = st.selectbox("交通方式", ["公交", "地铁", "私家车", "自行车", "步行"])
    diet = st.selectbox("饮食类型", ["素食", "杂食", "高蛋白"])
    energy = st.slider("用电量 (kWh)", 0, 50, 10)
    
    if st.button("计算今日碳排放"):
        # 简单计算公式（示例）
        carbon_map = {
            "公交": 0.5, "地铁": 0.3, "私家车": 3.0,
            "自行车": 0, "步行": 0
        }
        diet_map = {"素食": 2.0, "杂食": 4.0, "高蛋白": 5.0}
        
        today_carbon = (
            carbon_map.get(transport, 1.0) +
            diet_map.get(diet, 3.0) +
            energy * 0.5
        )
        
        st.success(f"今日碳排放：{today_carbon:.2f} kg")
        
        # 保存到用户记录
        if "user_id" in st.session_state:
            add_carbon_record(
                st.session_state.user_id,
                datetime.now().strftime("%Y-%m-%d"),
                today_carbon
            )
```

**功能 2：个人碳足迹趋势图**

```python
# 在首页添加
if "user_id" in st.session_state:
    history = get_user_history(st.session_state.user_id, days=30)
    if history:
        dates = [record[0] for record in history]
        values = [record[1] for record in history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='lines+markers',
            name='碳排放趋势'
        ))
        st.plotly_chart(fig, use_container_width=True)
```

---

## 6. 协作开发指南

### 6.1 Git 协作规范

**问题**：多人同时修改容易冲突

**解决方案**：

**1. 分支管理**
```bash
# 主分支（稳定版本）
main

# 功能分支（每人一个）
feature/user-auth      # 用户认证
feature/model-tuning   # 模型调优
feature/ui-improvement # UI 优化
```

**2. 每人工作流程**
```bash
# 1. 从 main 创建自己的分支
git checkout main
git pull origin main
git checkout -b feature/your-feature

# 2. 在自己的分支开发
# ... 修改代码 ...

# 3. 提交前检查冲突
git fetch origin
git rebase origin/main

# 4. 提交并推送
git add .
git commit -m "feat: 添加用户登录功能"
git push origin feature/your-feature

# 5. 在 GitHub 创建 Pull Request
```

**3. 避免冲突的技巧**
- 每人负责不同的文件
- 修改同一文件前先沟通
- 小步提交，频繁合并

---

### 6.2 任务分配建议

| 成员 | 任务 | 负责文件 | 截止时间 |
|------|------|----------|----------|
| **成员 A** | 用户系统 | `src/utils/user_data.py`, `app.py` | 3 月 20 日 |
| **成员 B** | 模型调优 | `src/models/lstm_model.py`, `src/models/gru_model.py` | 3 月 20 日 |
| **成员 C** | UI 优化 | `app.py` (CSS 部分) | 3 月 21 日 |
| **成员 D** | 部署 + 文档 | GitHub, Streamlit Cloud, README | 3 月 21 日 |
| **全体** | 对比实验 | 运行 main.py 记录结果 | 3 月 22 日 |

---

### 6.3 沟通机制

**每日站会**（15 分钟）：
- 昨天做了什么
- 今天计划做什么
- 遇到什么问题

**代码审查**：
- 每人完成功能后，在群里@其他人 review
- 确认没问题后再合并到 main 分支

**文档共享**：
- 使用在线文档记录实验结果
- 会议记录共享

---

## 7. 时间规划与任务分配

### 7.1 第一周（3.17-3.23）- 基础功能完善

| 日期 | 任务 | 负责人 | 交付物 |
|------|------|--------|--------|
| 3.17 | 团队分工确认 | 全体 | 任务分配表 |
| 3.18-3.19 | 用户系统实现 | 成员 A | `user_data.py` |
| 3.18-3.19 | LSTM 参数调优 | 成员 B | 实验记录 |
| 3.20-3.21 | UI 优化 | 成员 C | 新版界面 |
| 3.20-3.21 | Streamlit Cloud 部署 | 成员 D | 公开链接 |
| 3.22 | 对比实验运行 | 全体 | 实验数据 |
| 3.23 | 周会 + 进度检查 | 全体 | 周会记录 |

### 7.2 第二周（3.24-3.30）- 展示材料准备

| 日期 | 任务 | 负责人 | 交付物 |
|------|------|--------|--------|
| 3.24-3.25 | 演示视频脚本 | 全体 | 脚本 |
| 3.26-3.27 | 演示视频录制 | 成员 D | 视频文件 |
| 3.26-3.27 | 比赛文档撰写 | 全体 | 文档 |
| 3.28-3.29 | 最终测试 | 全体 | 测试报告 |
| 3.30 | 提交材料准备 | 成员 D | 提交包 |

### 7.3 里程碑检查点

| 日期 | 检查点 | 完成标准 |
|------|--------|----------|
| 3.20 | 用户系统完成 | 可登录、可保存数据 |
| 3.21 | 部署完成 | 公开链接可访问 |
| 3.22 | 实验完成 | 至少 5 组对比数据 |
| 3.27 | 视频完成 | 3-5 分钟演示视频 |
| 3.30 | 全部完成 | 所有材料就绪 |

---

## 总结

### 核心改进清单

| 改进方向 | 具体内容 | 优先级 | 预计时间 |
|----------|----------|--------|----------|
| **用户系统** | JSON 存储用户数据 | P0 | 4 小时 |
| **模型调优** | LSTM/GRU 参数调整 | P0 | 8 小时 |
| **对比实验** | 4 模型 +Stacking 对比 | P0 | 4 小时 |
| **Streamlit 部署** | Cloud 公开访问 | P0 | 1 小时 |
| **UI 优化** | CSS+ 交互组件 | P1 | 4 小时 |
| **演示视频** | 3-5 分钟 | P0 | 4 小时 |

### 国奖竞争力分析

| 评选维度 | 当前状态 | 改进后目标 |
|----------|----------|------------|
| 技术深度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 功能完整性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 展示效果 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 用户体验 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 文档材料 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 关键建议

1. **不要大改框架**：时间紧张，在现有基础上优化
2. **优先部署**：Streamlit Cloud 部署是 P0 任务
3. **做好对比实验**：这是评委最看重的
4. **演示视频要精美**：第一印象很重要
5. **团队协作要顺畅**：使用 Git 规范，避免冲突

---

<div align="center">

**🌿 EcoLife Team - 务实高效，冲击国奖！**

</div>