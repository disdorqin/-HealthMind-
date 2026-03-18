# EcoLife 用户系统 API 开发需求文档

> 前后端完全分离架构 | 本地 JSON 存储 | APP 形式输出

---

## 一、项目概述

### 架构说明
```
┌─────────────┐      HTTP/JSON      ┌─────────────┐
│   前端 APP   │ ←─────────────────→ │  后端 API   │
│  (app.py)   │                     │ (api.py)    │
│             │                     │             │
│ - 用户界面  │                     │ - 数据管理  │
│ - 数据展示  │                     │ - JSON 存储  │
│ - 用户输入  │                     │ - 业务逻辑  │
└─────────────┘                     └─────────────┘
         ↓                                   ↓
    本地测试运行                      本地 JSON 文件存储
```

### 技术栈
| 模块 | 技术 | 说明 |
|------|------|------|
| **后端 API** | Flask | 提供 HTTP 接口 |
| **前端** | Streamlit | 本地运行测试，后续可打包为 APP |
| **数据存储** | JSON 文件 | 本地存储用户数据 |
| **通信协议** | HTTP/JSON | 前后端通过 REST API 通信 |

---

## 二、后端 API 开发需求

### 1. 数据管理模块

**文件位置**：`backend/user_manager.py`

**功能需求**：
1. 创建用户
2. 获取用户数据
3. 保存碳排放记录
4. 获取历史记录
5. 添加积分

**数据存储格式**（JSON）：
```json
{
  "user_id": "zhangsan",
  "username": "张三",
  "created_at": "2026-03-18T09:00:00",
  "total_credits": 150,
  "carbon_records": [
    {
      "date": "2026-03-17",
      "carbon_value": 12.5,
      "predicted_value": 13.2,
      "timestamp": "2026-03-17T20:00:00"
    },
    {
      "date": "2026-03-18",
      "carbon_value": 10.8,
      "predicted_value": 11.5,
      "timestamp": "2026-03-18T09:00:00"
    }
  ],
  "credit_history": [
    {
      "credits": 10,
      "reason": "低碳一天",
      "timestamp": "2026-03-17T20:00:00"
    }
  ]
}
```

**函数签名**：
```python
def create_user(user_id: str, username: str) -> dict
def get_user(user_id: str) -> dict | None
def add_carbon_record(user_id: str, date: str, carbon_value: float, 
                      predicted_value: float = None) -> bool
def get_user_history(user_id: str, days: int = 30) -> list
def add_credits(user_id: str, credits: int, reason: str = "") -> bool
```

---

### 2. API 接口设计

**文件位置**：`backend/api.py`

#### 2.1 用户登录/注册

**接口**：`POST /api/login`

**请求**：
```json
{
  "username": "张三"
}
```

**响应（成功）**：
```json
{
  "success": true,
  "data": {
    "user_id": "zhangsan",
    "username": "张三",
    "total_credits": 0,
    "record_count": 0
  },
  "message": "登录成功"
}
```

**响应（失败）**：
```json
{
  "success": false,
  "message": "用户名不能为空"
}
```

---

#### 2.2 获取用户信息

**接口**：`GET /api/user/<user_id>`

**响应**：
```json
{
  "success": true,
  "data": {
    "user_id": "zhangsan",
    "username": "张三",
    "created_at": "2026-03-18T09:00:00",
    "total_credits": 150,
    "record_count": 5
  },
  "message": "获取成功"
}
```

---

#### 2.3 添加碳排放记录

**接口**：`POST /api/record`

**请求**：
```json
{
  "user_id": "zhangsan",
  "date": "2026-03-18",
  "carbon_value": 10.5,
  "predicted_value": 11.0
}
```

**响应**：
```json
{
  "success": true,
  "data": {
    "record_id": "rec_001"
  },
  "message": "记录保存成功"
}
```

---

#### 2.4 获取历史记录

**接口**：`GET /api/history/<user_id>?days=30`

**参数**：
- `days`：可选，默认 30 天

**响应**：
```json
{
  "success": true,
  "data": {
    "user_id": "zhangsan",
    "records": [
      {
        "date": "2026-03-17",
        "carbon_value": 12.5,
        "predicted_value": 13.2
      },
      {
        "date": "2026-03-18",
        "carbon_value": 10.8,
        "predicted_value": 11.5
      }
    ]
  },
  "message": "获取成功"
}
```

---

#### 2.5 添加积分

**接口**：`POST /api/credits`

**请求**：
```json
{
  "user_id": "zhangsan",
  "credits": 10,
  "reason": "低碳一天"
}
```

**响应**：
```json
{
  "success": true,
  "data": {
    "new_total": 160
  },
  "message": "积分添加成功"
}
```

---

### 3. 技术细节

#### 3.1 目录结构
```
backend/
├── api.py              # Flask API 主程序
├── user_manager.py     # 用户数据管理模块
├── test_api.py         # 测试脚本
└── user_data/          # JSON 数据存储目录
    ├── zhangsan.json
    ├── lisi.json
    └── ...
```

#### 3.2 CORS 配置（允许前端调用）
```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有来源访问
```

#### 3.3 错误处理
```python
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "接口不存在"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "服务器内部错误"
    }), 500
```

#### 3.4 APP 形式输出的适配
- **本地测试**：后端 API 运行在 `http://localhost:5000`
- **打包为 APP 后**：
  - 方案 A：API 部署到服务器，APP 通过网络调用
  - 方案 B：APP 内置轻量级 API 服务（使用 Flask 或 FastAPI）
  - 方案 C：直接调用数据管理模块（跳过 HTTP 层）

---

## 三、前端开发需求

### 1. 前端调用方式

**文件位置**：`app.py`

**API 客户端封装**：
```python
import requests

API_BASE_URL = "http://localhost:5000/api"

class APIClient:
    def login(self, username):
        resp = requests.post(f"{API_BASE_URL}/login", 
                            json={"username": username})
        return resp.json()
    
    def get_user(self, user_id):
        resp = requests.get(f"{API_BASE_URL}/user/{user_id}")
        return resp.json()
    
    def add_record(self, user_id, date, carbon_value, predicted_value):
        resp = requests.post(f"{API_BASE_URL}/record",
                            json={
                                "user_id": user_id,
                                "date": date,
                                "carbon_value": carbon_value,
                                "predicted_value": predicted_value
                            })
        return resp.json()
    
    def get_history(self, user_id, days=30):
        resp = requests.get(f"{API_BASE_URL}/history/{user_id}",
                           params={"days": days})
        return resp.json()
    
    def add_credits(self, user_id, credits, reason):
        resp = requests.post(f"{API_BASE_URL}/credits",
                            json={
                                "user_id": user_id,
                                "credits": credits,
                                "reason": reason
                            })
        return resp.json()
```

---

### 2. 前端界面流程

```
┌─────────────────────────────────────────┐
│           启动 app.py                    │
│                  ↓                       │
│  ┌─────────────────────────────────┐    │
│  │      检查是否已登录？            │    │
│  │         ↓          ↓            │    │
│  │       是          否            │    │
│  │       ↓           ↓             │    │
│  │   显示主界面   显示登录框        │    │
│  │                  ↓              │    │
│  │            输入用户名           │    │
│  │                  ↓              │    │
│  │            调用/login API       │    │
│  │                  ↓              │    │
│  │            登录成功→主界面       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

### 3. 主界面功能

#### 3.1 侧边栏
- 用户信息（用户名、总积分、记录数）
- 数据输入表单（交通、饮食、用电）
- 退出登录按钮

#### 3.2 主界面
- 欢迎横幅
- 今日碳排放计算结果
- 历史记录趋势图（Plotly 折线图）
- 模型预测功能（调用原有预测服务）

---

## 四、测试开发流程

### 步骤 1：启动后端 API
```bash
cd backend
python api.py
# 输出：* Running on http://localhost:5000
```

### 步骤 2：测试 API（可选）
```bash
python test_api.py
# 输出：所有测试通过！
```

### 步骤 3：启动前端
```bash
streamlit run app.py
# 输出：* Running on http://localhost:8501
```

### 步骤 4：浏览器访问
打开 `http://localhost:8501` 测试前端界面

---

## 五、开发清单

### 后端任务
| 文件 | 功能 | 代码量 | 预计时间 |
|------|------|--------|----------|
| `backend/user_manager.py` | 数据管理模块 | ~100 行 | 2 小时 |
| `backend/api.py` | Flask API | ~150 行 | 3 小时 |
| `backend/test_api.py` | 测试脚本 | ~50 行 | 1 小时 |

### 前端任务
| 文件 | 功能 | 代码量 | 预计时间 |
|------|------|--------|----------|
| `app.py` | 修改登录逻辑 | ~100 行 | 2 小时 |
| `app.py` | API 客户端封装 | ~50 行 | 1 小时 |
| `app.py` | UI 美化 | ~50 行 | 1 小时 |

---

## 六、常见问题

### Q1: 本地测试时后端和前端都要运行吗？
**A**: 是的，需要同时运行：
- 终端 1：`python backend/api.py`
- 终端 2：`streamlit run app.py`

### Q2: 打包为 APP 后怎么处理？
**A**: 三种方案：
1. **云端 API**：后端部署到服务器，APP 通过网络调用
2. **内置 API**：APP 内置 Flask 服务（启动时自动运行）
3. **直接调用**：APP 直接导入 `user_manager` 模块（跳过 HTTP）

### Q3: JSON 文件存在哪里？
**A**: `backend/user_data/` 目录下，每个用户一个 JSON 文件

### Q4: 多人同时写入会冲突吗？
**A**: 添加文件锁机制：
```python
import filelock

lock = filelock.FileLock(f"data/user_data/{user_id}.lock")
with lock:
    # 写入操作
```

---

## 七、总结

### 核心架构
- **前后端分离**：通过 HTTP/JSON 通信
- **本地 JSON 存储**：简单快速，无需数据库
- **APP 形式输出**：前端可打包为独立应用

### 开发顺序
1. 后端先写 `user_manager.py`
2. 后端再写 `api.py`
3. 后端运行 `test_api.py` 测试
4. 前端修改 `app.py` 调用 API
5. 前后端联调测试

### 交付物
- 后端 API 服务（可运行）
- 前端 APP（可本地测试）
- 测试脚本（验证功能）

---

<div align="center">

**🌿 EcoLife Team**

</div>