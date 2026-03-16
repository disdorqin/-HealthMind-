from __future__ import annotations

import json
import importlib
from typing import Dict, List

try:
    _pyecharts_opts = importlib.import_module("pyecharts.options")
    _pyecharts_charts = importlib.import_module("pyecharts.charts")
    opts = _pyecharts_opts
    Line = getattr(_pyecharts_charts, "Line")
    HAS_PYECHARTS = True
except Exception:
    opts = None
    Line = None
    HAS_PYECHARTS = False


def build_forecast_option(
    x_axis: List[str],
    curves: Dict[str, List[float]],
    selected: List[str],
    title: str,
) -> Dict:
    palette = {
        "ground_truth": "#1f2937",
        "lstm": "#ef4444",
        "gru": "#f59e0b",
        "xgboost": "#0ea5e9",
        "moirai": "#10b981",
        "ensemble": "#8b5cf6",
    }

    if HAS_PYECHARTS:
        line = Line()
        line.add_xaxis(x_axis)

        for name, values in curves.items():
            is_visible = name in selected or name == "ground_truth"
            line.add_yaxis(
                series_name=name,
                y_axis=[float(v) for v in values],
                is_smooth=True,
                is_symbol_show=False,
                is_selected=is_visible,
                linestyle_opts=opts.LineStyleOpts(width=2.6 if name == "ensemble" else 1.8),
                itemstyle_opts=opts.ItemStyleOpts(color=palette.get(name, "#6b7280")),
                label_opts=opts.LabelOpts(is_show=False),
            )

        line.set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                pos_left="left",
                title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold", color="#0f172a"),
            ),
            legend_opts=opts.LegendOpts(type_="scroll", pos_top="8%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
            ],
        )

        return json.loads(line.dump_options())

    series = []
    for name, values in curves.items():
        series.append(
            {
                "name": name,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"width": 2.6 if name == "ensemble" else 1.8},
                "itemStyle": {"color": palette.get(name, "#6b7280")},
                "data": [float(v) for v in values],
            }
        )

    return {
        "title": {"text": title, "left": "left"},
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll"},
        "xAxis": {"type": "category", "data": x_axis},
        "yAxis": {"type": "value"},
        "series": series,
    }
