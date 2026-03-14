from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ProgressEvent:
    stage: str
    progress: float
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
