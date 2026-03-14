"""
Model Layer 重构示例 - 完整使用说明
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# 导入重构后的模型组件
from src.model_layer import (
    LSTMModelWrapper,
    XGBoostModel,
    StackingTrainer,
    ModelTrainingPipeline,
    ModelMetricsManager,
)

from src.core.utils.logger import logger


def example_1_single_model_training():
    """
    示例1：单模型训练（LSTM）
    
    演示如何使用直接训练单个模型
    """
    logger.info("\n" + "="*70)
    logger.info("示例1：单一模型训练（LSTM）")
    logger.info("="*70)
    
    # 生成示例数据
    X_train = np.random.randn(1000, 8)  # 8个特征，1000个样本
    y_train = np.random.randn(1000) * 100 + 500  # 目标：功率（MW）
    X_val = np.random.randn(200, 8)
    y_val = np.random.randn(200) * 100 + 500
    X_test = np.random.randn(200, 8)
    y_test = np.random.randn(200) * 100 + 500
    
    # 创建模型
    lstm_model = LSTMModelWrapper(
        input_dim=8,
        hidden_dim=64,
        num_layers=2,
        lookback=24
    )
    
    # 训练模型
    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    logger.info(f"✓ 训练完成。历史: {history}")
    
    # 预测
    y_pred = lstm_model.predict(X_test)
    logger.info(f"预测形状: {y_pred.shape}")
    
    # 评估
    metrics = lstm_model.evaluate(y_test, y_pred)
    logger.info(f"✓ 评估指标: {metrics}")
    
    # 保存模型
    model_path = lstm_model.save('models/lstm_example.pth')
    logger.info(f"✓ 模型已保存到: {model_path}")
    
    return lstm_model, metrics


def example_2_xgboost_model():
    """
    示例2：XGBoost模型训练
    
    演示XGBoost模型的训练和预测
    """
    logger.info("\n" + "="*70)
    logger.info("示例2：XGBoost模型训练")
    logger.info("="*70)
    
    # 生成示例数据
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randn(1000) * 100 + 500
    X_val = np.random.randn(200, 10)
    y_val = np.random.randn(200) * 100 + 500
    X_test = np.random.randn(200, 10)
    y_test = np.random.randn(200) * 100 + 500
    
    # 创建XGBoost模型
    xgb_model = XGBoostModel(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # 训练
    logger.info("开始训练XGBoost模型...")
    history = xgb_model.train(X_train, y_train, X_val, y_val)
    logger.info(f"✓ 训练完成")
    
    # 预测
    y_pred = xgb_model.predict(X_test)
    
    # 评估
    metrics = xgb_model.evaluate(y_test, y_pred)
    logger.info(f"✓ XGBoost评估指标: {metrics}")
    
    # 获取特征重要性
    feature_importance = xgb_model.get_feature_importance(top_n=5)
    logger.info(f"✓ 特征重要性（Top 5）: {feature_importance}")
    
    # 保存模型
    model_path = xgb_model.save('models/xgboost_example.pkl')
    logger.info(f"✓ 模型已保存到: {model_path}")
    
    return xgb_model, metrics


def example_3_stacking_ensemble():
    """
    示例3：Stacking集成学习
    
    演示如何使用Stacking组合LSTM和XGBoost
    完整流程：
    1. 基模型1（LSTM）在训练集训练
    2. 基模型2（XGBoost）在训练集训练
    3. 生成验证集元特征（基模型预测）
    4. 元学习器在元特征上训练
    5. 最终预测
    """
    logger.info("\n" + "="*70)
    logger.info("示例3：Stacking集成学习（LSTM + XGBoost）")
    logger.info("="*70)
    
    # 生成示例数据
    X_train = np.random.randn(1000, 8)
    y_train = np.random.randn(1000) * 100 + 500
    X_val = np.random.randn(200, 8)
    y_val = np.random.randn(200) * 100 + 500
    X_test = np.random.randn(200, 8)
    y_test = np.random.randn(200) * 100 + 500
    
    # 创建基模型
    logger.info("创建基模型...")
    lstm_model = LSTMModelWrapper(
        input_dim=8,
        hidden_dim=64,
        num_layers=2,
        lookback=24
    )
    
    xgb_model = XGBoostModel(
        n_estimators=100,
        max_depth=6
    )
    
    # 创建Stacking训练器
    stacking = StackingTrainer(
        base_models=[lstm_model, xgb_model],
        meta_learner='linear'  # 或 'xgboost', 'ridge'
    )
    
    # 训练Stacking
    logger.info("开始Stacking训练...")
    fit_result = stacking.fit(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    logger.info(f"✓ Stacking训练完成")
    logger.info(f"验证集指标: {fit_result['val_metrics']}")
    
    if fit_result['test_metrics']:
        logger.info(f"测试集指标: {fit_result['test_metrics']}")
    
    # 获取摘要
    summary = stacking.get_summary()
    logger.info(f"✓ Stacking摘要: {summary}")
    
    # 预测
    y_pred_final, base_predictions = stacking.predict(X_test, return_base_predictions=True)
    logger.info(f"集成预测形状: {y_pred_final.shape}")
    logger.info(f"基模型预测: {base_predictions.keys()}")
    
    # 保存Stacking模型
    model_path = stacking.save('models/stacking_example.pkl')
    logger.info(f"✓ Stacking模型已保存到: {model_path}")
    
    return stacking, fit_result


def example_4_training_pipeline():
    """
    示例4：使用ModelTrainingPipeline进行完整训练
    
    这是推荐的方式！一键完成：
    1. 模型训练
    2. 模型保存
    3. 指标保存到数据库
    """
    logger.info("\n" + "="*70)
    logger.info("示例4：ModelTrainingPipeline（推荐用法）")
    logger.info("="*70)
    
    # 生成示例数据
    X_train = np.random.randn(1000, 8)
    y_train = np.random.randn(1000) * 100 + 500
    X_val = np.random.randn(200, 8)
    y_val = np.random.randn(200) * 100 + 500
    X_test = np.random.randn(200, 8)
    y_test = np.random.randn(200) * 100 + 500
    
    # 方式A：训练单个LSTM模型
    logger.info("\n--- 方式A：单个LSTM模型 ---")
    pipeline_lstm = ModelTrainingPipeline(
        model_type='LSTM',
        auto_save=True,
        auto_db_save=True
    )
    
    result = pipeline_lstm.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_kwargs={
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        input_dim=8,
        hidden_dim=64
    )
    
    logger.info(f"✓ LSTM训练完成")
    logger.info(f"  - 指标: {result['val_metrics']}")
    logger.info(f"  - 模型路径: {result.get('model_path', 'N/A')}")
    logger.info(f"  - 耗时: {result['training_time']:.2f}s")
    
    # 方式B：Stacking集成学习
    logger.info("\n--- 方式B：Stacking集成学习 ---")
    
    # 创建基模型
    lstm_base = LSTMModelWrapper(input_dim=8, hidden_dim=64)
    xgb_base = XGBoostModel(n_estimators=100)
    
    pipeline_stacking = ModelTrainingPipeline(
        model_type='Stacking',
        models=[lstm_base, xgb_base],
        auto_save=True,
        auto_db_save=True
    )
    
    result = pipeline_stacking.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        meta_learner='linear'
    )
    
    logger.info(f"✓ Stacking训练完成")
    logger.info(f"  - 指标: {result['val_metrics']}")
    logger.info(f"  - 耗时: {result['training_time']:.2f}s")
    
    return pipeline_lstm, pipeline_stacking


def example_5_metrics_manager():
    """
    示例5：使用ModelMetricsManager查询和比较指标
    
    演示如何查询历史指标和比较不同模型的性能
    """
    logger.info("\n" + "="*70)
    logger.info("示例5：指标管理与查询")
    logger.info("="*70)
    
    # 创建指标管理器
    metrics_mgr = ModelMetricsManager(db_available=True)
    
    # 保存示例指标
    logger.info("保存训练指标...")
    
    result1 = metrics_mgr.save_metrics(
        model_name='LSTM',
        mae=50.5,
        rmse=75.3,
        r2=0.92,
        mape=3.5,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        training_time=120.5,
        dataset_size=1000,
        version='20260313',
        notes='第一次训练'
    )
    logger.info(f"✓ LSTM指标已保存: {result1}")
    
    result2 = metrics_mgr.save_metrics(
        model_name='XGBoost',
        mae=48.2,
        rmse=72.1,
        r2=0.93,
        mape=3.2,
        training_time=45.3,
        dataset_size=1000,
        version='20260313',
        notes='100棵树'
    )
    logger.info(f"✓ XGBoost指标已保存: {result2}")
    
    result3 = metrics_mgr.save_metrics(
        model_name='Stacking',
        mae=45.1,
        rmse=68.5,
        r2=0.95,
        mape=2.8,
        training_time=200.0,
        dataset_size=1000,
        version='20260313',
        notes='LSTM + XGBoost集成'
    )
    logger.info(f"✓ Stacking指标已保存: {result3}")
    
    # 查询单个模型的历史指标
    logger.info("\n查询LSTM的历史指标...")
    lstm_history = metrics_mgr.query_model_metrics('LSTM', limit=5)
    for record in lstm_history:
        logger.info(f"  - {record['run_time']}: MAE={record['mae']:.2f}, RMSE={record['rmse']:.2f}")
    
    # 比较所有模型
    logger.info("\n按MAE比较所有模型:")
    comparison = metrics_mgr.compare_models(metric='mae')
    for i, model in enumerate(comparison, 1):
        logger.info(f"{i}. {model['model_name']}: MAE={model['mae']:.2f}, RMSE={model['rmse']:.2f}")
    
    # 获取最优模型
    best_model = metrics_mgr.get_best_model(metric='mae')
    if best_model:
        logger.info(f"\n✓ 最优模型（按MAE）: {best_model['model_name']}")
    
    # 生成报告
    logger.info("\n生成性能报告:")
    report = metrics_mgr.generate_report()
    logger.info(report)


def example_6_complete_workflow():
    """
    示例6：完整工作流程
    
    从数据准备到模型部署的完整流程
    """
    logger.info("\n" + "="*70)
    logger.info("示例6：完整工作流程（Data → Model → Evaluate → Save）")
    logger.info("="*70)
    
    # 1. 准备数据
    logger.info("\n[步骤1] 数据准备")
    X_train = np.random.randn(5000, 15)  # 更大的数据集
    y_train = np.sin(X_train[:, 0]) * 100 + np.random.randn(5000) * 10 + 500
    X_val = np.random.randn(1000, 15)
    y_val = np.sin(X_val[:, 0]) * 100 + np.random.randn(1000) * 10 + 500
    X_test = np.random.randn(1000, 15)
    y_test = np.sin(X_test[:, 0]) * 100 + np.random.randn(1000) * 10 + 500
    logger.info(f"✓ 数据准备完成 - 训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    # 2. 训练多个模型
    logger.info("\n[步骤2] 模型训练")
    
    # 2a. LSTM
    lstm_pipeline = ModelTrainingPipeline(
        model_type='LSTM',
        auto_save=True,
        auto_db_save=True
    )
    lstm_result = lstm_pipeline.train(
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_kwargs={'epochs': 10, 'batch_size': 32},
        input_dim=15, hidden_dim=128
    )
    logger.info(f"✓ LSTM完成 - MAE: {lstm_result['val_metrics']['mae']:.2f}")
    
    # 2b. XGBoost
    xgb_model = XGBoostModel(n_estimators=200, max_depth=8)
    xgb_model.train(X_train, y_train, X_val, y_val)
    y_val_pred_xgb = xgb_model.predict(X_val)
    xgb_metrics = xgb_model.evaluate(y_val, y_val_pred_xgb)
    logger.info(f"✓ XGBoost完成 - MAE: {xgb_metrics['mae']:.2f}")
    
    # 3. Stacking集成
    logger.info("\n[步骤3] Stacking集成学习")
    lstm_base = LSTMModelWrapper(input_dim=15, hidden_dim=128)
    xgb_base = XGBoostModel(n_estimators=200)
    
    stacking_pipeline = ModelTrainingPipeline(
        model_type='Stacking',
        models=[lstm_base, xgb_base],
        auto_save=True,
        auto_db_save=True
    )
    
    stacking_result = stacking_pipeline.train(
        X_train, y_train, X_val, y_val, X_test, y_test,
        meta_learner='linear'
    )
    logger.info(f"✓ Stacking完成 - MAE: {stacking_result['val_metrics']['mae']:.2f}")
    
    # 4. 性能评估
    logger.info("\n[步骤4] 性能评估与比较")
    metrics_mgr = ModelMetricsManager(db_available=True)
    
    comparison = metrics_mgr.compare_models(metric='mae')
    logger.info("\n模型性能排名：")
    for i, model in enumerate(comparison, 1):
        logger.info(f"{i}. {model['model_name']}: MAE={model['mae']:.2f}, "
                   f"RMSE={model['rmse']:.2f}, R²={model.get('r2', 'N/A')}")
    
    # 5. 保存最优模型
    logger.info("\n[步骤5] 保存最优模型")
    best_model = comparison[0] if comparison else None
    if best_model:
        logger.info(f"✓ 最优模型: {best_model['model_name']}")
        logger.info(f"  MAE: {best_model['mae']:.2f}")
        logger.info(f"  RMSE: {best_model['rmse']:.2f}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ 完整工作流程完成！")
    logger.info("="*70)
    
    return lstm_result, xgb_metrics, stacking_result


def run_all_examples():
    """运行所有示例"""
    logger.info("\n\n")
    logger.info("#" * 70)
    logger.info("# Model Layer 重构 - 完整使用示例")
    logger.info("#" * 70)
    
    try:
        # 示例1：单个LSTM
        lstm_model, lstm_metrics = example_1_single_model_training()
        
        # 示例2：XGBoost
        xgb_model, xgb_metrics = example_2_xgboost_model()
        
        # 示例3：Stacking
        stacking_model, stacking_result = example_3_stacking_ensemble()
        
        # 示例4：使用Pipeline
        pipeline_lstm, pipeline_stacking = example_4_training_pipeline()
        
        # 示例5：指标管理
        example_5_metrics_manager()
        
        # 示例6：完整工作流程
        example_6_complete_workflow()
        
        logger.info("\n\n" + "="*70)
        logger.info("✅ 所有示例运行完成！")
        logger.info("="*70)
    
    except Exception as e:
        logger.error(f"示例执行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_examples()
