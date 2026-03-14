"""模型评估器"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.core.utils.logger import logger


class ModelEvaluator:
    """模型评估器 - 计算性能指标"""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            字典包含各项指标
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        logger.info(f"模型评估 - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        return metrics
    
    @staticmethod
    def evaluate_models(models: Dict[str, nn.Module], 
                       X_test: torch.Tensor, y_test: torch.Tensor,
                       device: str = 'cpu') -> Dict[str, Dict[str, float]]:
        """
        评估多个模型
        
        Args:
            models: 模型字典
            X_test: 测试特征
            y_test: 测试标签
            device: 设备
        
        Returns:
            各模型的评估结果
        """
        results = {}
        device = torch.device(device)
        
        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                X_test = X_test.to(device)
                y_test_tensor = y_test.to(device)
                
                y_pred = model(X_test)
                
                # 转换为numpy
                y_true_np = y_test.cpu().numpy().flatten()
                y_pred_np = y_pred.cpu().numpy().flatten()
                
                metrics = ModelEvaluator.evaluate(y_true_np, y_pred_np)
                results[model_name] = metrics
                
                logger.info(f"模型 {model_name} 评估完成")
        
        return results
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> str:
        """
        比较模型性能
        
        Args:
            results: 各模型的评估结果
        
        Returns:
            比较报告字符串
        """
        report = "\n" + "="*60 + "\n"
        report += "模型性能对比\n"
        report += "="*60 + "\n"
        
        for model_name, metrics in results.items():
            report += f"\n{model_name}:\n"
            for metric_name, value in metrics.items():
                report += f"  {metric_name.upper()}: {value:.6f}\n"
        
        # 找最佳模型
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        report += f"\n最佳模型：{best_model[0]} (RMSE: {best_model[1]['rmse']:.6f})\n"
        report += "="*60 + "\n"
        
        logger.info(report)
        
        return report
