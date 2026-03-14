# 风芒可测——电力预测与交易优化系统

⚡ 面向新能源消纳的多模型融合功率预测与交易优化系统

## 项目简介

本系统基于深度学习 LSTM 模型，结合分时电价机制，为新能源发电企业提供功率预测和交易优化建议。通过"谷时充电、峰时放电"的峰谷套利模式，帮助企业实现电费节约和收益最大化。

## 功能特性

### 📊 实时监控
- 实时功率监测与预测对比
- 一键同步最新数据并重新预测
- 模型评估指标展示（MAE、RMSE、R²、MAPE）

### 📈 多维预测
- 天预测（24 小时）
- 周预测（168 小时）
- 月预测（720 小时）

### 💰 交易助手
- 买卖电时段建议
- 预期收益与成本节约分析
- 削峰填谷效果展示
- 分时电价信息

## 项目结构

```
项目根目录/
├── main.py                          # 主入口（命令行训练/预测）
├── src/
│   ├── core/                        # 核心配置和工具
│   │   ├── config/config_manager.py
│   │   ├── exceptions/base.py
│   │   └── utils/logger.py
│   ├── pipeline/                    # 管道层
│   │   └── lstm_pipeline.py         # LSTM 训练/预测管道
│   ├── runner/                      # 执行器层
│   │   ├── lstm_runner.py           # LSTM 模型核心实现
│   │   └── pipeline_router.py       # 管道路由器
│   ├── backend/                     # 后端 API
│   │   └── api.py                   # Flask RESTful 服务
│   ├── frontend/                    # 前端界面
│   │   └── app.py                   # Streamlit 数据看板
│   └── trade_logic/                 # 交易逻辑
│       └── optimizer.py             # 交易优化算法
├── models/                          # 模型保存目录
├── data/                            # 数据目录
└── requirements.txt                 # 依赖安装
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型（可选，命令行方式）

```bash
python main.py --mode train
```

### 3. 启动后端 API 服务

```bash
python src/backend/api.py
```

后端将在 http://localhost:5000 启动，提供以下 API 接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/run_all` | POST | 全链路执行（训练 + 预测） |
| `/api/forecast/<scale>` | GET | 多尺度预测（day/week/month） |
| `/api/trade/advice` | POST | 交易优化建议 |
| `/api/data/plot` | GET | 绘图数据接口 |
| `/api/metrics` | GET | 模型评估指标 |

### 4. 启动前端界面

打开新终端，运行：

```bash
streamlit run src/frontend/app.py
```

前端将在 http://localhost:8501 启动（自动打开浏览器）

## API 接口详解

### 全链路执行接口 `/api/run_all`

```bash
curl -X POST http://localhost:5000/api/run_all
```

响应示例：
```json
{
  "status": "success",
  "message": "全链路执行完成",
  "training": {
    "mae": 27364.18,
    "rmse": 40705.78,
    "r2": -0.82,
    "mape": 15.5
  },
  "prediction": {
    "count": 114677,
    "min": 20.96,
    "max": 34.29,
    "mean": 25.5
  }
}
```

### 多尺度预测接口 `/api/forecast/<scale>`

```bash
# 天预测
curl http://localhost:5000/api/forecast/day

# 周预测
curl http://localhost:5000/api/forecast/week

# 月预测
curl http://localhost:5000/api/forecast/month
```

### 交易优化接口 `/api/trade/advice`

```bash
curl -X POST http://localhost:5000/api/trade/advice
```

响应示例：
```json
{
  "status": "success",
  "data": {
    "buy_advice": [
      {
        "hour": 2,
        "power": 18.5,
        "price": 0.504,
        "reason": "低谷电价 (0.504 元/kWh)，适合充电储能"
      }
    ],
    "sell_advice": [
      {
        "hour": 12,
        "power": 32.1,
        "price": 0.704,
        "reason": "高峰电价 (0.704 元/kWh)，适合放电自用"
      }
    ],
    "expected_revenue": 1234.56,
    "cost_saving": 567.89
  }
}
```

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 模型层 | PyTorch | LSTM 深度学习模型 |
| 后端层 | Flask | RESTful API 服务 |
| 前端层 | Streamlit + Pyecharts | 数据可视化看板 |
| 数据库 | MySQL（可选） | 数据存储 |

## 分时电价说明

| 时段 | 时间 | 电价（元/kWh） | 建议操作 |
|------|------|---------------|----------|
| 高峰 | 11-14 点，18-23 点 | 0.704 | 卖电（放电/自用） |
| 平段 | 其他时段 | 0.604 | 正常调度 |
| 低谷 | 23-7 点 | 0.504 | 买电（充电/储能） |

## 常见问题

### Q: 后端 API 无法启动？
A: 确保已安装 `flask` 和 `flask-cors`，检查端口 5000 是否被占用。

### Q: 前端无法连接后端？
A: 确保后端 API 已启动，检查 `API_BASE_URL` 配置是否正确。

### Q: 模型训练失败？
A: 检查 `data/data.csv` 文件是否存在，确保数据格式正确。

## 参考文档

- [项目计划书](计算机设计大赛准备.docx)
- [架构文档](docs/architecture.md)
- [项目数据流详解](项目数据流详解.md)

## License

MIT License