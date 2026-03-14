from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_file: Path
    model_dir: Path
    logs_dir: Path

    @classmethod
    def detect(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        return cls(
            root=root,
            data_file=root / "data" / "data.csv",
            model_dir=root / "models",
            logs_dir=root / "logs",
        )
