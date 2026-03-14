# Streamlit Cloud 发布检查清单

## 1. 仓库与入口

- [ ] 主入口文件为 app.py
- [ ] README 中启动命令为 streamlit run app.py
- [ ] 不再要求先启动 Flask

## 2. 依赖与系统包

- [ ] requirements.txt 包含 torch, xgboost, uni2ts
- [ ] packages.txt 已包含 libgomp1
- [ ] 删除/避免无关开发依赖进入生产环境

## 3. Secrets 与路径

- [ ] Streamlit Cloud Secrets 已配置 database/model/runtime 区块
- [ ] 本地存在 .streamlit/secrets.toml.example 用于模板说明
- [ ] 代码路径统一使用 pathlib

## 4. 运行模式

- [ ] 默认 lightweight_mode 在 Cloud 可自动启用
- [ ] 如需强制，设置环境变量 LIGHTWEIGHT_MODE=true
- [ ] legacy Flask 默认禁用，避免误启动

## 5. 功能验证

- [ ] 训练页可看到实时进度与 ETA
- [ ] 预测页可同框对比多模型曲线
- [ ] stacking 输出可用
- [ ] 交易建议页可正常生成结果

## 6. 回归验证

- [ ] 运行 test_training_progress.py 冒烟测试
- [ ] 检查 models 目录生成 *_model.bin 与 stacking_meta.bin
