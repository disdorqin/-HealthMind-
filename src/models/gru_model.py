from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.utils.logger import logger
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__(name="gru")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = _GRURegressor(input_dim, hidden_dim, num_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
        lr = float(kwargs.get("lr", 1e-3))

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
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loss = self._eval_loss(X_val, y_val)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if progress_callback is not None:
                progress_callback(
                    {
                        "model": self.name,
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )

        logger.info("GRU training completed")
        return {"history": history}

    def _eval_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            yb = torch.tensor(y, dtype=torch.float32).to(self.device)
            preds = self.model(xb)
            loss = nn.MSELoss()(preds, yb)
        self.model.train()
        return float(loss.item())

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
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = int(checkpoint["input_dim"])
        self.hidden_dim = int(checkpoint["hidden_dim"])
        self.num_layers = int(checkpoint["num_layers"])
        self.model = _GRURegressor(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
