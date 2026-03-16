import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

class PlotManager:
    @staticmethod
    def save_loss_curve(history: List[Dict[str, Any]], save_path: Path) -> None:
        """
        Save training loss curve to file.
        
        Args:
            history: List of dictionaries containing 'epoch', 'train_loss', 'val_loss'.
            save_path: Path to save the plot.
        """
        if not history:
            return

        epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
        train_loss = [h.get('train_loss') for h in history]
        val_loss = [h.get('val_loss') for h in history]
        
        plt.figure(figsize=(10, 6))
        
        # Filter None values for plotting
        valid_train = [(e, l) for e, l in zip(epochs, train_loss) if l is not None]
        if valid_train:
            plt.plot(*zip(*valid_train), label='Train Loss', marker='o')
            
        valid_val = [(e, l) for e, l in zip(epochs, val_loss) if l is not None]
        if valid_val:
            plt.plot(*zip(*valid_val), label='Validation Loss', marker='x')

        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved to {save_path}")
