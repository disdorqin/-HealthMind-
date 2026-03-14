"""训练进度跟踪工具"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

class TrainingProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self, progress_file: str = 'logs/training_progress.json'):
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.current_task_id = None

    def start_training(self, task_id: str, total_epochs: int, model_name: str = "LSTM"):
        """开始训练任务"""
        with self.lock:
            self.current_task_id = task_id
            progress_data = {
                'task_id': task_id,
                'model_name': model_name,
                'total_epochs': total_epochs,
                'current_epoch': 0,
                'progress': 0.0,
                'status': 'running',
                'start_time': time.time(),
                'loss_history': [],
                'metrics': {}
            }
            self._save_progress(progress_data)

    def update_progress(self, task_id: str, current_epoch: int, loss: float = None, 
                       metrics: Dict[str, float] = None, additional_info: Dict[str, Any] = None):
        """更新训练进度"""
        if task_id != self.current_task_id:
            return

        with self.lock:
            progress_data = self._load_progress()
            if progress_data and progress_data.get('task_id') == task_id:
                total_epochs = progress_data.get('total_epochs', 1)
                progress_data['current_epoch'] = current_epoch
                progress_data['progress'] = min(current_epoch / total_epochs * 100, 100.0)
                
                # 构建损失历史记录（支持 train_loss 和 val_loss）
                loss_entry = {
                    'epoch': current_epoch,
                    'timestamp': time.time()
                }
                
                # 兼容旧的 loss 参数
                if loss is not None:
                    loss_entry['loss'] = loss
                
                # 支持新的 metrics 参数（train_loss, val_loss 等）
                if metrics:
                    if 'train_loss' in metrics:
                        loss_entry['train_loss'] = metrics['train_loss']
                    if 'val_loss' in metrics:
                        loss_entry['val_loss'] = metrics['val_loss']
                    # 如果没有单独的 loss 字段，使用 train_loss 作为默认 loss
                    if 'loss' not in loss_entry and 'train_loss' in metrics:
                        loss_entry['loss'] = metrics['train_loss']
                
                progress_data['loss_history'].append(loss_entry)
                
                if metrics:
                    progress_data['metrics'].update(metrics)
                
                if additional_info:
                    progress_data.update(additional_info)
                
                self._save_progress(progress_data)

    def finish_training(self, task_id: str, success: bool = True, final_metrics: Dict[str, float] = None):
        """完成训练任务"""
        with self.lock:
            progress_data = self._load_progress()
            if progress_data and progress_data.get('task_id') == task_id:
                progress_data['status'] = 'completed' if success else 'failed'
                progress_data['end_time'] = time.time()
                if final_metrics:
                    progress_data['final_metrics'] = final_metrics
                self._save_progress(progress_data)
            self.current_task_id = None

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取训练进度"""
        progress_data = self._load_progress()
        if progress_data and progress_data.get('task_id') == task_id:
            return progress_data
        return None

    def _save_progress(self, data: Dict[str, Any]):
        """保存进度数据到文件"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_progress(self) -> Optional[Dict[str, Any]]:
        """从文件加载进度数据"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def cleanup(self):
        """清理进度文件"""
        with self.lock:
            if self.progress_file.exists():
                self.progress_file.unlink()
            self.current_task_id = None


# 全局训练进度跟踪器实例
global_tracker = TrainingProgressTracker()


def get_training_tracker() -> TrainingProgressTracker:
    """获取全局训练进度跟踪器"""
    return global_tracker