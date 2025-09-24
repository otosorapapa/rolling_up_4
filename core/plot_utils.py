from typing import Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.design_tokens import get_color, get_font_stack, rgba


PRIMARY = get_color("primary")
PRIMARY_TEXT = get_color("text")
ACCENT_SOFT = get_color("accent", "soft")

LIGHT_TEXT = PRIMARY_TEXT
LIGHT_GRID = rgba(PRIMARY, 0.10)
LIGHT_AXIS = rgba(PRIMARY, 0.28)
DARK_TEXT = "#E6EFF8"
DARK_GRID = rgba(ACCENT_SOFT, 0.20)
DARK_AXIS = rgba(ACCENT_SOFT, 0.35)


def apply_elegant_theme(fig: go.Figure, theme: str = "light") -> go.Figure:
    """Apply subdued, elegant styling to Plotly figures when enabled."""
    if not st.session_state.get("elegant_on", True):
        return fig
    if theme == "dark":
        dark_bg = "#0F1A2C"
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=dark_bg,
            plot_bgcolor=dark_bg,
            font=dict(
                family=get_font_stack("body"),
                size=12,
                color=DARK_TEXT,
            ),
            legend=dict(
                bgcolor=rgba(PRIMARY, 0.65),
                bordercolor=rgba(ACCENT_SOFT, 0.32),
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor=rgba(PRIMARY, 0.85),
                bordercolor=rgba(ACCENT_SOFT, 0.35),
                font=dict(color=DARK_TEXT),
            ),
        )
        grid = DARK_GRID
        axisline = DARK_AXIS
        marker_border = rgba(ACCENT_SOFT, 0.45)
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=get_color("surface"),
            plot_bgcolor=get_color("surface"),
            font=dict(
                family=get_font_stack("body"),
                size=12,
                color=LIGHT_TEXT,
            ),
            legend=dict(
                bgcolor=rgba(get_color("surface"), 0.88),
                bordercolor=rgba(PRIMARY, 0.16),
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor=rgba(get_color("surface"), 0.98),
                bordercolor=rgba(PRIMARY, 0.16),
                font=dict(color=LIGHT_TEXT),
            ),
        )
        grid = LIGHT_GRID
        axisline = LIGHT_AXIS
        marker_border = rgba(PRIMARY, 0.24)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        linecolor=axisline,
        ticks="outside",
        ticklen=4,
        tickcolor=axisline,
        showline=True,
        linewidth=1,
        title_standoff=14,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        linecolor=axisline,
        ticks="outside",
        ticklen=4,
        tickcolor=axisline,
        showline=True,
        linewidth=1,
        title_standoff=16,
    )
    fig.update_traces(
        selector=lambda t: "markers" in getattr(t, "mode", ""),
        marker=dict(size=6, line=dict(width=1.2, color=marker_border)),
    )
    return fig


def _plot_area_height(fig: go.Figure) -> int:
    h = fig.layout.height or 520
    m = fig.layout.margin or {}
    t = getattr(m, "t", 40) or 40
    b = getattr(m, "b", 60) or 60
    return max(120, int(h - t - b))


def _y_to_px(y, y0, y1, plot_h):
    if y1 == y0:
        y1 = y0 + 1.0
    return float((1 - (y - y0) / (y1 - y0)) * plot_h)


def add_latest_labels_no_overlap(
    fig: go.Figure,
    df_long: pd.DataFrame,
    label_col: str = "display_name",
    x_col: str = "month",
    y_col: str = "year_sum",
    max_labels: int = 12,
    min_gap_px: int = 12,
    alternate_side: bool = True,
    xpad_px: int = 8,
    font_size: int = 11,
):
    last = df_long.sort_values(x_col).groupby(label_col, as_index=False).tail(1)
    if len(last) == 0:
        return
    cand = last.sort_values(y_col, ascending=False).head(max_labels).copy()
    yaxis = fig.layout.yaxis
    if getattr(yaxis, "range", None):
        y0, y1 = yaxis.range
    else:
        y0, y1 = float(df_long[y_col].min()), float(df_long[y_col].max())
    plot_h = _plot_area_height(fig)
    cand["y_px"] = cand[y_col].apply(lambda v: _y_to_px(v, y0, y1, plot_h))
    cand = cand.sort_values("y_px")
    placed = []
    for _, r in cand.iterrows():
        base = r["y_px"]
        if placed and base <= placed[-1] + min_gap_px:
            base = placed[-1] + min_gap_px
        base = float(np.clip(base, 0 + 6, plot_h - 6))
        placed.append(base)
        yshift = -(base - r["y_px"])
        xshift = xpad_px if (not alternate_side or (len(placed) % 2 == 1)) else -xpad_px
        fig.add_annotation(
            x=r[x_col],
            y=r[y_col],
            text=f"{r[label_col]}：{r[y_col]:,.0f}（{pd.to_datetime(r[x_col]).strftime('%Y-%m')}）",
            showarrow=False,
            xanchor="left" if xshift >= 0 else "right",
            yanchor="middle",
            xshift=xshift,
            yshift=yshift,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=font_size),
        )


def render_plotly_with_spinner(
    fig: go.Figure,
    *,
    spinner_text: str = "グラフを描画中…",
    use_container_width: bool = True,
    config: dict | None = None,
    **kwargs: Any,
) -> None:
    """Render a Plotly figure with a spinner to highlight processing."""

    with st.spinner(spinner_text):
        height = kwargs.pop("height", None)
        if height is not None:
            fig.update_layout(height=height)
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
            config=config,
            **kwargs,
        )
