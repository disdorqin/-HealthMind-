#!/usr/bin/env python3
"""
EcoLife Interactive Internal Testing Platform
Refactored for Model Tuning & Stacking Comparison.

支持两种运行模式：
1. 命令行参数模式：python main.py --train --stack --models 1,3
2. 交互模式：python main.py (直接回车进入交互式)
"""

import os
import sys
import time
import argparse
import logging
import yaml
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

# scientific computing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor

# Local Imports
# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

from src.core.utils.logger import logger
from src.models.lstm_model import LSTMForecastModel
from src.models.gru_model import GRUForecastModel
from src.models.xgboost_model import XGBoostForecastModel
from src.models.moirai_model import MoiraiZeroShotModel
from src.data.lstm_processing import process_data_for_lstm
from src.data.xgboost_processing import process_data_for_xgboost
from src.data.moirai_processing import process_data_for_moirai

# --- Configuration Constants ---
LOG_DIR = Path("logs")
PLOT_DIR = LOG_DIR / "plots"
RESULTS_FILE = LOG_DIR / "test_results.csv"
CHECKPOINT_DIR = Path("models/checkpoints")

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Carbon Footprint Thresholds (kg/day)
# Baseline approx 12.5. 
# Low < 10, High > 15, Middle [10, 15]
THRESHOLD_LOW = 10.0
THRESHOLD_HIGH = 15.0

# Model Map
MODEL_MAP = {'1': 'LSTM', '2': 'GRU', '3': 'XGBoost', '4': 'Moirai'}


class UnifiedEvaluator:
    """Handles Metric Calculation, Visualization, and Logging."""
    
    @staticmethod
    def _get_classification_label(value: float) -> str:
        if value < THRESHOLD_LOW:
            return "Low"
        elif value > THRESHOLD_HIGH:
            return "High"
        else:
            return "Medium"

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Convert continuous regression output to classification metrics."""
        y_true_labels = [UnifiedEvaluator._get_classification_label(y) for y in y_true]
        y_pred_labels = [UnifiedEvaluator._get_classification_label(y) for y in y_pred]
        
        labels = ["Low", "Medium", "High"]
        
        return {
            "Accuracy": accuracy_score(y_true_labels, y_pred_labels),
            "Precision": precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            "Recall": recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            "F1": f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        }

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, mode: str = "Test") -> Dict[str, float]:
        """Comprehensive evaluation."""
        # Regression Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Classification Metrics
        cls_metrics = UnifiedEvaluator.calculate_classification_metrics(y_true, y_pred)
        
        metrics = {
            "Model": model_name,
            "Mode": mode,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            **{k: round(v, 4) for k, v in cls_metrics.items()},
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"\n[{model_name}] Performance Report ({mode}):")
        print(f"Regression  | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        print(f"Classific.  | Acc: {metrics['Accuracy']:.2%}, F1: {metrics['F1']:.4f}")
        return metrics

    @staticmethod
    def log_result(metrics: Dict[str, Any]):
        """Append results to CSV."""
        df = pd.DataFrame([metrics])
        header = not RESULTS_FILE.exists()
        df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
        logger.info(f"Results logged to {RESULTS_FILE}")

    @staticmethod
    def plot_loss(history: Dict[str, List[float]], model_name: str):
        """Plot training loss curve."""
        if not history or 'train_loss' not in history or len(history.get('train_loss', [])) == 0:
            logger.warning(f"No training history available for {model_name}")
            return
            
        plt.figure(figsize=(10, 6))
        
        if 'train_loss' in history and history['train_loss']:
            plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
            
        plt.title(f"{model_name} Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        save_path = PLOT_DIR / f"loss_curve_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Loss curve saved to {save_path}")

    @staticmethod
    def plot_validation(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot Predicted vs True on Validation Set."""
        plt.figure(figsize=(12, 6))
        
        # Limit to first 100 points for clarity if too large
        limit = 100
        indices = range(min(len(y_true), limit))
        
        plt.plot(indices, y_true[:limit], label='True Value', color='black', alpha=0.7)
        plt.plot(indices, y_pred[:limit], label='Predicted', color='green', linestyle='--')
        
        # Draw Threshold Zones
        plt.axhline(THRESHOLD_LOW, color='blue', linestyle=':', alpha=0.5, label='Low Carbon < 10')
        plt.axhline(THRESHOLD_HIGH, color='red', linestyle=':', alpha=0.5, label='High Carbon > 15')
        
        plt.title(f"{model_name} Validation Preview (Next Day Prediction)")
        plt.xlabel("Sample Index")
        plt.ylabel("Carbon Footprint (kg)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = PLOT_DIR / f"fit_compare_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Validation comparison saved to {save_path}")


class ModelTrainer:
    """Encapsulates training logic for all supported models."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.val_ratio = 0.2
        self.seed = 42

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Hard 8:2 Split."""
        split_idx = int(len(X) * (1 - self.val_ratio))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def train_lstm(self) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
        logger.info("Initializing LSTM Training...")
        # 1. Processing
        X, y, scaler = process_data_for_lstm(self.data_path, window_size=3) # Assume 3-day window for next day
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # 2. Config & Init (Input dim from data)
        input_dim = X.shape[2]
        model = LSTMForecastModel(input_dim=input_dim, hidden_dim=128, num_layers=2)
        
        # 3. Train
        history = model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
        
        # 4. Predict
        preds_scaled = model.predict(X_val)
        # Inverse Transform
        if scaler:
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            y_val_orig = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        else:
            preds = preds_scaled.flatten()
            y_val_orig = y_val.flatten()
            
        return model, history, y_val_orig, preds

    def train_gru(self) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
        logger.info("Initializing GRU Training...")
        # Reuse LSTM processing as GRU accepts same shape
        X, y, scaler = process_data_for_lstm(self.data_path, window_size=3)
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        input_dim = X.shape[2]
        model = GRUForecastModel(input_dim=input_dim, hidden_dim=64, num_layers=2)
        
        # GRU train wrapper usually returns dict history, accepts standard X, y if we assume fix to GRU wrapper or direct call
        # Assuming GRU class method train accepts standard numpy arrays
        history = model.train(X_train, y_train, X_val, y_val, epochs=30)
        
        preds_scaled = model.predict(X_val)
        if scaler:
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            y_val_orig = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        else:
            preds = preds_scaled.flatten()
            y_val_orig = y_val.flatten()
            
        return model, history, y_val_orig, preds

    def train_xgboost(self) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
        logger.info("Initializing XGBoost Training...")
        X, y = process_data_for_xgboost(self.data_path)
        # Handle potential dataframe output
        if hasattr(X, 'values'): X = X.values
        if hasattr(y, 'values'): y = y.values
        
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        model = XGBoostForecastModel(n_estimators=100, learning_rate=0.1)
        history_metrics = model.train(X_train, y_train, X_val, y_val) 
        
        preds = model.predict(X_val)
        
        return model, {}, y_val, preds

    def train_moirai(self) -> Tuple[Any, Dict, np.ndarray, np.ndarray]:
        logger.info("Initializing Moirai Training...")
        # Moirai uses dataframe directly usually
        df = process_data_for_moirai(self.data_path)
        
        # Split Dataframe
        split_idx = int(len(df) * (1 - self.val_ratio))
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:]
        
        model = MoiraiZeroShotModel()
        # Train creates internal history or loaded model
        model.train(df_train, None, df_val, df_val['carbon_footprint_kg'].values)
        
        preds = model.predict(df_val)
        y_val = df_val['carbon_footprint_kg'].values
        
        return model, {}, y_val, preds


def get_interactive_config() -> Tuple[bool, bool, List[str]]:
    """Get configuration via interactive prompts."""
    print("="*60)
    print("      EcoLife Interactive Internal Testing Platform      ")
    print("="*60)
    
    # 1. Interactive Selection
    print("\n[Q1] 选择任务类型:")
    print("   [1] 模型训练与调优 (Train)")
    print("   [2] 推理模式 (跳过训练，加载现有模型)")
    try:
        task_choice = input(">> 请选择 (1/2): ").strip()
    except EOFError:
        print("\n检测到非交互式环境，使用默认配置：训练模式")
        task_choice = '1'
    
    is_training = (task_choice == '1')
    
    # 2. Stacking Choice
    print("\n[Q2] 是否启用 Stacking 融合？")
    try:
        stacking_choice = input(">> 启用 (Y/N): ").strip().upper()
    except EOFError:
        print("\n检测到非交互式环境，使用默认配置：不启用 Stacking")
        stacking_choice = 'N'
    is_stacking = (stacking_choice == 'Y')
    
    # 3. Model Selection
    selected_models = []
    
    if not is_stacking:
        print("\n[Q3] 选择要测试的单个模型:")
        print("   [1] LSTM")
        print("   [2] GRU")
        print("   [3] XGBoost")
        print("   [4] Moirai")
        try:
            choice = input(">> 选择 (1-4): ").strip()
        except EOFError:
            print("\n检测到非交互式环境，使用默认模型：XGBoost")
            choice = '3'
        if choice in MODEL_MAP:
            selected_models = [MODEL_MAP[choice]]
    else:
        print("\n[Q3] 输入 Stacking 融合的模型组合 (逗号分隔):")
        print("   示例：1,3 或 1,2,3,4")
        try:
            choices = input(">> 模型: ").strip().split(',')
        except EOFError:
            print("\n检测到非交互式环境，使用默认组合：LSTM+XGBoost")
            choices = ['1', '3']
        for c in choices:
            c = c.strip()
            if c in MODEL_MAP:
                selected_models.append(MODEL_MAP[c])
    
    return is_training, is_stacking, selected_models


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EcoLife Internal Testing Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 交互模式
  python main.py --train                  # 训练模式，默认 XGBoost
  python main.py --train --stack          # 启用 Stacking 融合
  python main.py --train --models 1,3     # 训练 LSTM 和 XGBoost
  python main.py --train --models 1,2,3,4 --stack  # 全模型 Stacking
        """
    )
    
    parser.add_argument('--train', action='store_true', 
                        help='训练模式 (默认：推理模式)')
    parser.add_argument('--stack', action='store_true',
                        help='启用 Stacking 融合')
    parser.add_argument('--models', type=str, default='3',
                        help='模型选择 (1=LSTM, 2=GRU, 3=XGBoost, 4=Moirai)，逗号分隔，例如：1,3')
    parser.add_argument('--interactive', action='store_true',
                        help='强制使用交互模式')
    parser.add_argument('--data', type=str, 
                        default='data/personal_carbon_footprint_behavior.csv',
                        help='数据文件路径')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Determine if interactive mode
    # Interactive if: --interactive flag is set, or no arguments provided, or stdin is a TTY
    is_interactive = args.interactive or (len(sys.argv) == 1) or (hasattr(sys.stdin, 'isatty') and sys.stdin.isatty())
    
    if is_interactive and len(sys.argv) == 1:
        # Pure interactive mode
        is_training, is_stacking, selected_models = get_interactive_config()
    else:
        # Command line mode
        is_training = args.train
        is_stacking = args.stack
        
        # Parse model selection
        model_choices = args.models.split(',')
        selected_models = []
        for c in model_choices:
            c = c.strip()
            if c in MODEL_MAP:
                selected_models.append(MODEL_MAP[c])
        
        if not selected_models:
            logger.error("No valid models specified. Using default: XGBoost")
            selected_models = ['XGBoost']
        
        print("="*60)
        print("      EcoLife Internal Testing Platform      ")
        print("="*60)
        print(f"\n配置：任务={'训练' if is_training else '推理'}, 模型={selected_models}, Stacking={is_stacking}")
    
    # Validate data path
    data_path = args.data
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    if not selected_models:
        logger.error("No valid models selected.")
        return

    print(f"\n开始执行：任务={'训练' if is_training else '推理'}, 模型={selected_models}, Stacking={is_stacking}")
    
    trainer = ModelTrainer(data_path)
    
    # Storage for Stacking
    val_predictions = {} 
    val_targets = None
    
    # Process Execution
    for model_name in selected_models:
        print(f"\n... 处理 {model_name} ...")
        
        y_val_true = None
        y_val_pred = None
        history = {}
        
        try:
            if model_name == 'LSTM':
                _, history, y_val_true, y_val_pred = trainer.train_lstm()
            elif model_name == 'GRU':
                _, history, y_val_true, y_val_pred = trainer.train_gru()
            elif model_name == 'XGBoost':
                _, _, y_val_true, y_val_pred = trainer.train_xgboost()
            elif model_name == 'Moirai':
                _, _, y_val_true, y_val_pred = trainer.train_moirai()
                
            # Store for stacking
            if len(y_val_pred) > 0:
                val_predictions[model_name] = y_val_pred
                if val_targets is None:
                    val_targets = y_val_true # Assume aligned
                else:
                    # Align lengths if mismatched (simple truncation for safety)
                    min_len = min(len(val_targets), len(y_val_true))
                    val_targets = val_targets[:min_len]
                    val_predictions[model_name] = val_predictions[model_name][:min_len]
                    
            # Evaluation & Visualization
            metrics = UnifiedEvaluator.evaluate(y_val_true, y_val_pred, model_name)
            UnifiedEvaluator.log_result(metrics)
            UnifiedEvaluator.plot_loss(history, model_name)
            UnifiedEvaluator.plot_validation(y_val_true, y_val_pred, model_name)
            
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Stacking Logic
    if is_stacking and len(val_predictions) > 1:
        print("\n... 执行 Stacking 融合 ...")
        
        # Align all predictions to shortest length
        if val_targets is None:
             logger.error("No valid predictions generated for stacking.")
             return

        min_len = min([len(v) for v in val_predictions.values()] + [len(val_targets)])
        
        X_stack = []
        feature_names = []
        for name, preds in val_predictions.items():
            X_stack.append(preds[:min_len])
            feature_names.append(name)
            
        X_stack = np.column_stack(X_stack)
        y_stack = val_targets[:min_len]
        
        # Simple Blending (Average)
        avg_preds = np.mean(X_stack, axis=1)
        UnifiedEvaluator.evaluate(y_stack, avg_preds, "Stacking_Mean_Blend")
        
        # Meta Model (XGBoost)
        logger.info("Training Meta-Model on Validation Predictions (Fusion Layer)...")
        meta_sub_model = XGBRegressor(n_estimators=50, max_depth=3)
        meta_sub_model.fit(X_stack, y_stack)
        meta_preds = meta_sub_model.predict(X_stack)
        
        # Evaluate
        metrics = UnifiedEvaluator.evaluate(y_stack, meta_preds, "Stacking_Meta_XGB")
        UnifiedEvaluator.log_result(metrics)
        UnifiedEvaluator.plot_validation(y_stack, meta_preds, "Stacking_Fusion")
        
        # Save Meta Model
        joblib.dump(meta_sub_model, CHECKPOINT_DIR / "stacking_meta_model.joblib")
        print(f"Meta-Model 已保存到 {CHECKPOINT_DIR / 'stacking_meta_model.joblib'}")
    
    print("\n" + "="*60)
    print("执行完成!")
    print("="*60)


if __name__ == "__main__":
    main()