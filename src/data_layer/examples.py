"""数据层快速开始 - 展示新功能的使用示例"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.utils.logger import logger
from src.data_layer import (
    ETLPipeline,
    TrainingDataPipeline,
    DataSchema,
    DataIntegrityChecker,
    DataExporter,
)


def example_etl_pipeline():
    """示例1：自动化ETL流程"""
    logger.info("\n" + "="*60)
    logger.info("示例1：自动化ETL流程（CSV → MySQL）")
    logger.info("="*60)
    
    # 创建ETL管道
    etl = ETLPipeline(
        csv_path='data/data.csv',
        table_name='power_weather'
    )
    
    # 运行ETL流程
    result = etl.run_etl(nrows=1000)
    
    logger.info(f"\nETL执行结果：")
    logger.info(f"  - 成功: {result['success']}")
    logger.info(f"  - CSV行数: {result['csv_rows']}")
    logger.info(f"  - 验证行数: {result['validated_rows']}")
    logger.info(f"  - 清洗行数: {result['cleaned_rows']}")
    logger.info(f"  - 新记录数: {result['new_records']}")
    logger.info(f"  - 插入行数: {result['inserted_rows']}")
    
    return result


def example_training_pipeline():
    """示例2：完整的训练数据生成管道"""
    logger.info("\n" + "="*60)
    logger.info("示例2：完整的训练数据生成管道")
    logger.info("="*60)
    
    # 创建训练进度管道
    pipeline = TrainingDataPipeline(
        csv_path='data/data.csv',
        lookback=24,
        test_size=0.2
    )
    
    # 运行管道（自动处理所有步骤）
    result = pipeline.run(
        nrows=2000,
        target_col='actual_power',
        feature_cols=['wind_speed', 'temperature', 'irradiance'],
        output_dir='data/training',
        batch_size=32
    )
    
    logger.info(f"\n训练数据生成完成：")
    logger.info(f"  - 输出目录: {result['metadata']['scaler_path']}")
    logger.info(f"  - X_train形状: {result['metadata']['X_train_shape']}")
    logger.info(f"  - y_train形状: {result['metadata']['y_train_shape']}")
    logger.info(f"  - X_test形状: {result['metadata']['X_test_shape']}")
    logger.info(f"  - y_test形状: {result['metadata']['y_test_shape']}")
    logger.info(f"  - 批大小: {result['metadata']['batch_size']}")
    logger.info(f"  - Scaler路径: {result['metadata']['scaler_path']}")
    
    return result


def example_data_validation():
    """示例3：数据完整性检查"""
    logger.info("\n" + "="*60)
    logger.info("示例3：数据完整性检查")
    logger.info("="*60)
    
    import pandas as pd
    
    # 加载数据
    data = pd.read_csv('data/data.csv', nrows=100)
    
    # 验证数据
    valid, errors = DataSchema.validate(data)
    
    logger.info(f"\n数据验证结果：")
    logger.info(f"  - 有效: {valid}")
    
    if not valid:
        logger.info(f"  - 错误:")
        for error in errors:
            logger.info(f"    * {error}")
    
    # 生成完整报告
    report = DataIntegrityChecker.generate_report(data)
    logger.info(f"\n{report}")
    
    return valid, errors


def example_feature_export():
    """示例4：特征导出和Scaler保存"""
    logger.info("\n" + "="*60)
    logger.info("示例4：特征导出和Scaler保存")
    logger.info("="*60)
    
    from src.data_layer import FeatureEngineer
    import numpy as np
    
    # 创建示例数据
    X_train = np.random.randn(100, 24)
    y_train = np.random.randn(100)
    X_test = np.random.randn(20, 24)
    y_test = np.random.randn(20)
    
    # 创建特征工程师
    engineer = FeatureEngineer(lookback=24)
    
    # 归一化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    engineer.scaler = scaler
    engineer.feature_names = [f'feat_{i}' for i in range(24)]
    
    # 导出训练数据
    result = engineer.export_training_data(
        X_train, y_train, X_test, y_test,
        output_dir='data/exported',
        batch_size=32
    )
    
    logger.info(f"\n特征导出完成：")
    logger.info(f"  - 输出目录: data/exported")
    logger.info(f"  - 数据文件:")
    logger.info(f"    * X_train.npy")
    logger.info(f"    * y_train.npy")
    logger.info(f"    * X_test.npy")
    logger.info(f"    * y_test.npy")
    logger.info(f"  - Scaler: scaler.pkl")
    logger.info(f"  - 元数据: metadata.json")
    logger.info(f"  - DataLoader: train_loader, test_loader")
    
    return result


def main():
    """运行所有示例"""
    logger.info("\n")
    logger.info("╔════════════════════════════════════════════════════╗")
    logger.info("║     数据层优化功能演示                             ║")
    logger.info("║     - 自动化ETL                                   ║")
    logger.info("║     - 特征解耦和导出                              ║")
    logger.info("║     - 数据完整性校验                              ║")
    logger.info("╚════════════════════════════════════════════════════╝")
    logger.info("")
    
    try:
        # 运行示例
        examples = [
            ("自动化ETL流程", example_etl_pipeline, False),
            ("训练数据生成", example_training_pipeline, False),
            ("数据完整性检查", example_data_validation, True),
            ("特征导出", example_feature_export, False),
        ]
        
        for name, func, always_run in examples:
            try:
                logger.info(f"\n运行：{name}")
                
                # 跳过需要真实数据的示例
                if not always_run:
                    logger.info(f"⏭️  跳过（需要真实数据）")
                    continue
                
                func()
            
            except FileNotFoundError as e:
                logger.warning(f"⚠ 跳过示例（文件未找到）：{e}")
            
            except Exception as e:
                logger.error(f"❌ 示例执行失败：{e}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ 演示完成")
        logger.info("="*60)
        logger.info("\n使用指南：")
        logger.info("  1. 自动化ETL: etl = ETLPipeline(); etl.run_etl()")
        logger.info("  2. 训练数据: pipeline = TrainingDataPipeline(); pipeline.run()")
        logger.info("  3. 数据验证: valid, errors = DataSchema.validate(df)")
        logger.info("  4. 特征导出: engineer.export_training_data(...)")
    
    except Exception as e:
        logger.error(f"错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
