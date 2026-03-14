from __future__ import annotations

from typing import Iterable, List


def to_float_list(values: Iterable[float], digits: int = 4) -> List[float]:
    return [round(float(v), digits) for v in values]
