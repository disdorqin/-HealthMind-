from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.core.utils.logger import logger
from src.models.lstm_model import LSTMForecastModel
from src.models.xgboost_model import XGBoostForecastModel
from src.models.moirai_model import MoiraiZeroShotModel

# Reuse processing logic (ideally refactored to be reusable for inference, but calling directly for now)
# Note: In a real production system, we'd have a separate 'InferencePipeline' class.
from src.data.lstm_processing import process_data_for_lstm
from src.data.xgboost_processing import process_data_for_xgboost
from src.data.moirai_processing import process_data_for_moirai

import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class PredictionService:
    """
    Unified Service for EcoLife Predictions.
    Orchestrates the Two-Step Model Strategy (Base Models -> Meta Stacking).
    """

    def __init__(self, model_dir: str = "models/checkpoints"):
        self.model_dir = Path(model_dir)
        self.lstm_model = None
        self.xgb_model = None
        self.moirai_model = None
        self.meta_model = None
        self.lstm_target_scaler = None
        self.lstm_config = {}
        self.is_loaded = False

    def load_models(self):
        """Loads all trained models from disk."""
        logger.info("Loading EcoLife models...")
        try:
            # 0. Load Config & Scaler
            try:
                with open(self.model_dir / "lstm_config.json", "r") as f:
                    self.lstm_config = json.load(f)
                self.lstm_target_scaler = joblib.load(self.model_dir / "lstm_target_scaler.joblib")
            except Exception as e:
                logger.warning(f"Failed to load LSTM config/scaler: {e}. Using defaults.")

            # 1. Load LSTM
            # Re-init model with config if available
            input_dim = self.lstm_config.get("input_dim", 1) # Fallback 1 if unknown, usually 11 is standard
            hidden_dim = self.lstm_config.get("hidden_dim", 128)
            
            self.lstm_model = LSTMForecastModel(input_dim=input_dim, hidden_dim=hidden_dim) 
            
            try:
                self.lstm_model.load(self.model_dir / "lstm_best.pth")
            except Exception as e:
                logger.warning(f"Failed to load LSTM: {e}")

            # ... (rest is same)

            # 2. Load XGBoost
            self.xgb_model = XGBoostForecastModel()
            try:
                self.xgb_model.load(self.model_dir / "xgboost_best.joblib")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost: {e}")

            # 3. Load Moirai
            self.moirai_model = MoiraiZeroShotModel()
            try:
                self.moirai_model.load(self.model_dir / "moirai_best.joblib")
            except Exception as e:
                logger.warning(f"Failed to load Moirai: {e}")
                
            # 4. Load Meta-Model
            try:
                self.meta_model = joblib.load(self.model_dir / "stacking_meta_model.joblib")
            except Exception as e:
                logger.warning(f"Failed to load Meta-Model: {e}")

            self.is_loaded = True
            logger.info("Models loaded successfully (partial failures logged).")
            
        except Exception as e:
            logger.error(f"Critical error loading models: {e}")

    def predict_next_cycle(self, data_path: str) -> Dict[str, Any]:
        """
        Generate predictions for the next cycle using all models and stacking.
        
        Args:
            data_path: Path to the latest data file.
            
        Returns:
            Dict containing individual and stacked predictions.
        """
        if not self.is_loaded:
            self.load_models()
            
        results = {}
        
        # 1. LSTM Prediction
        try:
            scaler_type_str = self.lstm_config.get("scaler_type", "MinMax")
            scaler_cls = StandardScaler if scaler_type_str == "Standard" else MinMaxScaler
            
            # Re-process data with correct scaler class for features
            X_lstm, _, _ = process_data_for_lstm(data_path, window_size=3, scaler_cls=scaler_cls) 
            
            last_seq = X_lstm[-1:] 
            pred_scaled = self.lstm_model.predict(last_seq)
            
            if self.lstm_target_scaler:
                pred_lstm = self.lstm_target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            else:
                # Fallback if no target scaler (assume unscaled or generic MinMax)
                pred_lstm = pred_scaled.flatten()[0]
                
            results['lstm'] = float(pred_lstm)
        except Exception as e:
            logger.error(f"LSTM Prediction failed: {e}")
            results['lstm'] = 0.0

        # 2. XGBoost Prediction
        try:
            X_xgb, _ = process_data_for_xgboost(data_path)
            if isinstance(X_xgb, pd.DataFrame): X_xgb = X_xgb.values
            last_row = X_xgb[-1:]
            pred_xgb = self.xgb_model.predict(last_row)[0]
            results['xgboost'] = float(pred_xgb)
        except Exception as e:
            logger.error(f"XGBoost Prediction failed: {e}")
            results['xgboost'] = 0.0

        # 3. Moirai Prediction
        try:
            df_moirai = process_data_for_moirai(data_path)
            # Moirai history is loaded, so we pass the context.
            # Predict for the user in the last row
            last_user_df = df_moirai.iloc[-1:]
            pred_moirai = self.moirai_model.predict(last_user_df)[0]
            results['moirai'] = float(pred_moirai)
        except Exception as e:
            logger.error(f"Moirai Prediction failed: {e}")
            results['moirai'] = 0.0

        # 4. Meta-Stacking
        if self.meta_model:
            try:
                X_meta = np.array([[results['lstm'], results['xgboost'], results['moirai']]])
                final_pred = self.meta_model.predict(X_meta)[0]
                results['ensemble_meta'] = float(final_pred)
            except Exception as e:
                logger.error(f"Meta-Prediction failed: {e}")
                # Fallback to average
                results['ensemble_meta'] = np.mean([results['lstm'], results['xgboost'], results['moirai']])
        else:
             results['ensemble_meta'] = np.mean([results['lstm'], results['xgboost'], results['moirai']])

        return results
