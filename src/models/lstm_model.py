from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from src.core.utils.logger import logger
from .base_model import BaseForecastModel


class _LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        # Bidirectional for better context
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        # Hidden dim * 2 for bidirectional
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        output, _ = self.lstm(x)
        # output: (batch, seq_len, hidden * 2)
        # Take last time step
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)
        return self.head(last_step).squeeze(-1)


class LSTMForecastModel(BaseForecastModel):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128, # Increased capacity
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        name: str = "lstm",
    ):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = _LSTMRegressor(input_dim, hidden_dim, num_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100, # Increased epochs for convergence
        batch_size: int = 32,
        patience: int = 15, # Increased patience
        save_path: str = "models/checkpoints/lstm_best.pth",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train with Enhanced Logic:
        1. Bidirectional LSTM
        2. LR Scheduler (ReduceLROnPlateau)
        3. Strict specific checkpointing (Early Stopping based on best Val Loss)
        4. R2 Sanity Check
        """
        if not isinstance(X_train, torch.Tensor):
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
        else:
            X_train_t, y_train_t = X_train, y_train
            
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None:
             if not isinstance(X_val, torch.Tensor):
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32)
             else:
                X_val_t, y_val_t = X_val, y_val
             val_ds = TensorDataset(X_val_t, y_val_t)
             val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
             
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # --- 1. Adaptive Learning Rate ---
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            epoch_loss /= len(train_ds)
            train_losses.append(epoch_loss)
            
            # Validation
            val_loss = epoch_loss
            r2 = -float('inf')
            
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                preds = []
                targets = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                        preds.extend(outputs.cpu().numpy())
                        targets.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_ds)
                val_losses.append(val_loss)
                r2 = r2_score(targets, preds)
                
                # --- Scheduler Step ---
                scheduler.step(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model logic directly here or outside
                    if save_path:
                        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                if patience_counter >= patience:
                    logger.info("Early stopping triggered by patience.")
                    break
            else:
                 logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        # Return history for higher-level auditing
        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_val_loss": best_val_loss
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor)
            return preds.cpu().numpy()

    def save(self, path: Union[str, Path]) -> None:
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, str(path))

    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.model = _LSTMRegressor(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

    def _plot_metrics(self, history: List[float], save_dir: Path, prefix: str):
        plt.figure(figsize=(10, 5))
        plt.plot(history, label='Train Loss')
        plt.title(f'{prefix} Training Loss (Enhanced)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plot_path = save_dir.parent / 'metrics' / f'{prefix}_loss.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
