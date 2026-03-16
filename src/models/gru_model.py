from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.utils.logger import logger
from src.utils.data_processor import DataProcessor
from src.utils.metrics import calculate_direction_accuracy
from .base_model import BaseForecastModel


class _GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class GRUForecastModel(BaseForecastModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(name="gru")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = float(learning_rate)
        self.model = _GRURegressor(input_dim, hidden_dim, num_layers)
        # Force CPU to avoid potential CUDA initialization hangs in diagnosis mode
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.scaler = MinMaxScaler()

    def load_and_preprocess(
        self, 
        data_path: Union[str, Path], 
        lookback: int = 24, 
        target_col: str = 'YD15'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load raw data and perform GRU-specific preprocessing (scaling + windowing).
        """
        logger.info(f"GRU: Loading data from {data_path}")
        df = DataProcessor.load_data(data_path)
        
        # Ensure target column exists
        if target_col not in df.columns:
            # Fallback to last numeric column if target missing
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
                logger.warning(f"Target column not found, using {target_col}")
            else:
                 raise ValueError("No numeric columns found in data")

        # Select numeric features only
        numeric_df = df.select_dtypes(include=[np.number])
        feature_data = numeric_df.values
        target_idx = numeric_df.columns.get_loc(target_col)
        
        # Scale
        self.scaler.fit(feature_data)
        scaled_data = self.scaler.transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i : i + lookback])
            y.append(scaled_data[i + lookback, target_idx])
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Update input_dim if needed
        if X.shape[-1] != self.input_dim:
            logger.info(f"Updating input_dim from {self.input_dim} to {X.shape[-1]}")
            self.input_dim = X.shape[-1]
            self.model = _GRURegressor(self.input_dim, self.hidden_dim, self.num_layers)
            self.model.to(self.device)
            
        # Time-based split (70/20/10)
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Data prepared: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        progress_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        epochs = int(kwargs.get("epochs", 20))
        batch_size = int(kwargs.get("batch_size", 64))
        lr = float(kwargs.get("lr", self.learning_rate))

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        history = []
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            val_loss = None
            val_acc = None
            if X_val is not None and y_val is not None and len(X_val) > 0:
                metrics = self._eval_metrics(X_val, y_val)
                val_loss = metrics["val_loss"]
                val_acc = metrics["val_acc"]

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
            if progress_callback is not None:
                progress_callback(
                    {
                        "model": self.name,
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

        logger.info("GRU training completed")
        return {"history": history}

    def _eval_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            yb = torch.tensor(y, dtype=torch.float32).to(self.device)
            preds = self.model(xb)
            loss = nn.MSELoss()(preds, yb)
            
            preds_np = preds.cpu().numpy()
            y_np = y
            acc = calculate_direction_accuracy(y_np, preds_np)
            
        self.model.train()
        return {"val_loss": float(loss.item()), "val_acc": float(acc)}

    def _eval_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        # Deprecated logic kept for safety
        metrics = self._eval_metrics(X, y)
        return metrics["val_loss"]

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(xb).detach().cpu().numpy()
        return pred.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "learning_rate": self.learning_rate,
            },
            path,
        )
        # Save scaler
        scaler_path = path.with_suffix('.scaler.joblib')
        joblib.dump(self.scaler, scaler_path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = int(checkpoint["input_dim"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.num_layers = int(checkpoint["num_layers"])
        self.learning_rate = float(checkpoint.get("learning_rate", self.learning_rate))
        self.model = _GRURegressor(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        
        # Load scaler if exists
        scaler_path = path.with_suffix('.scaler.joblib')
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            logger.warning(f"Scaler not found at {scaler_path}, creating new one")
            self.scaler = MinMaxScaler()
