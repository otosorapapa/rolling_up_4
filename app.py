import html
import io
import json
import math
import textwrap
from string import Template
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Iterable, Callable

import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ai_features import (
    summarize_dataframe,
    generate_comment,
    explain_analysis,
    generate_actions,
    answer_question,
    generate_anomaly_brief,
)
from core.i18n import (
    get_available_languages,
    get_current_language,
    init_language,
    language_name,
    t,
)
from core.design_tokens import (
    darken,
    get_color,
    get_color_rgb,
    get_font_stack,
    get_layout_token,
    get_plotly_palette,
    get_typography,
    hex_to_rgb_tuple,
    lighten,
    mix,
    rgba,
)


PRIMARY_COLOR = get_color("primary")
PRIMARY_RGB = get_color_rgb("primary")
PRIMARY_DARK = darken(PRIMARY_COLOR, 0.25)
PRIMARY_DEEP = darken(PRIMARY_COLOR, 0.4)
PRIMARY_LIGHT = lighten(PRIMARY_COLOR, 0.25)
ACCENT_COLOR = get_color("accent")
ACCENT_RGB = get_color_rgb("accent")
ACCENT_SOFT = get_color("accent", "soft")
ACCENT_SOFT_RGB = get_color_rgb("accent", "soft")
ACCENT_EMPHASIS = get_color("accent", "emphasis")
BACKGROUND_COLOR = get_color("background")
BACKGROUND_MUTED = mix(BACKGROUND_COLOR, get_color("surface_alt"), 0.45)
SURFACE_COLOR = get_color("surface")
SURFACE_ALT_COLOR = get_color("surface_alt")
TEXT_COLOR = get_color("text")
MUTED_COLOR = get_color("muted")
BORDER_COLOR = get_color("border")
BORDER_STRONG = get_color("border", "strong")
SUCCESS_COLOR = get_color("success")
SUCCESS_RGB = get_color_rgb("success")
WARNING_COLOR = get_color("warning")
WARNING_RGB = get_color_rgb("warning")
ERROR_COLOR = get_color("error")

body_typography = get_typography("body")
heading_typography = get_typography("heading")
numeric_typography = get_typography("numeric")
BODY_FONT_SIZE = body_typography.get("size_px", {}).get("base", 15)
BODY_LINE_HEIGHT = body_typography.get("line_height", 1.5)
HEADING_LINE_HEIGHT = heading_typography.get("line_height", 1.35)
FONT_BODY = get_font_stack("body")
FONT_HEADING = get_font_stack("heading")
FONT_NUMERIC = get_font_stack("numeric")
PRIMARY_DEEP_RGB = ",".join(str(c) for c in hex_to_rgb_tuple(PRIMARY_DEEP))

card_tokens = get_layout_token("card")
card_radius_tokens = card_tokens.get("radius_px", {}) if isinstance(card_tokens, dict) else {}
CARD_RADIUS = card_radius_tokens.get("base", card_radius_tokens.get("min", 10))
CARD_SHADOW = card_tokens.get("shadow", "0 12px 24px rgba(11,31,59,0.08)") if isinstance(card_tokens, dict) else "0 12px 24px rgba(11,31,59,0.08)"
if "rgba(" in CARD_SHADOW:
    CARD_SHADOW = CARD_SHADOW.replace("rgba(11,31,59", f"rgba({PRIMARY_RGB}")
SPACING_UNIT = get_layout_token("spacing", "unit_px")

px.defaults.color_discrete_sequence = get_plotly_palette()

init_language()
current_language = get_current_language()

PLOTLY_CONFIG = {
    "locale": "ja",
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "autoScale2d",
        "resetViewMapbox",
        "toggleSpikelines",
        "select2d",
        "lasso2d",
        "zoom3d",
        "orbitRotation",
        "tableRotation",
    ],
    "toImageButtonOptions": {"format": "png", "filename": "年計比較"},
}
PLOTLY_CONFIG["locale"] = "ja" if current_language == "ja" else "en"

ICON_SVGS: Dict[str, str] = {
    "template": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <rect x=\"5\" y=\"3\" width=\"14\" height=\"18\" rx=\"2.2\" ry=\"2.2\"/>
      <path d=\"M8 9h8M8 13h5\"/>
    </svg>
    """,
    "upload": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M4 16.5v3.3A1.2 1.2 0 0 0 5.2 21h13.6A1.2 1.2 0 0 0 20 19.8v-3.3\"/>
      <path d=\"M12 4v11.5\"/>
      <path d=\"m7.5 8.5 4.5-4.5 4.5 4.5\"/>
    </svg>
    """,
    "download": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M4 16v3.2A1.2 1.2 0 0 0 5.2 20.4h13.6A1.2 1.2 0 0 0 20 19.2V16\"/>
      <path d=\"M12 3.5v11.5\"/>
      <path d=\"m7.5 10.5 4.5 4.5 4.5-4.5\"/>
    </svg>
    """,
    "metrics": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M5 19V9m7 10V5m7 14v-7\"/>
      <path d=\"M3 19h18\"/>
    </svg>
    """,
    "quality": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M12 3 4.8 6v6c0 4.3 3 7.9 7.2 9 4.2-1.1 7.2-4.7 7.2-9V6Z\"/>
      <path d=\"m9.5 12 1.9 1.9 3.1-3.4\"/>
    </svg>
    """,
    "info": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <circle cx=\"12\" cy=\"12\" r=\"9\"/>
      <path d=\"M12 8.5h.01M10.8 11.3h1.2v4.2\"/>
    </svg>
    """,
    "alert": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M12 4 3 19h18Z\"/>
      <path d=\"M12 10.2v3.6M12 16.8h.01\"/>
    </svg>
    """,
    "check": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"m5.5 12.8 3.6 3.6L18.4 7\"/>
    </svg>
    """,
    "policy": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M4 7h16M6 12h12M8 17h8\"/>
      <path d=\"M9 5v4M15 10v4M12 15v4\"/>
    </svg>
    """,
    "dataset": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M4 6.5C4 4.6 7.6 3 12 3s8 1.6 8 3.5S16.4 10 12 10 4 8.4 4 6.5Z\"/>
      <path d=\"M4 11.5C4 13.4 7.6 15 12 15s8-1.6 8-3.5\"/>
      <path d=\"M4 16.5C4 18.4 7.6 20 12 20s8-1.6 8-3.5\"/>
    </svg>
    """,
    "trend": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"m5 15 4-4 3 3 6-7\"/>
      <path d=\"M16 7h4v4\"/>
    </svg>
    """,
    "delta": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <path d=\"M5 7h14l-7 10Z\"/>
    </svg>
    """,
    "sku": """
    <svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\">
      <rect x=\"3.5\" y=\"5\" width=\"6\" height=\"14\" rx=\"1.6\"/>
      <rect x=\"14.5\" y=\"5\" width=\"6\" height=\"14\" rx=\"1.6\"/>
      <path d=\"M6.5 9h0.01M6.5 12h0.01M6.5 15h0.01M17.5 9h0.01M17.5 12h0.01M17.5 15h0.01\"/>
    </svg>
    """,
}

METRIC_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "年計総額": {
        "footnote": "直近12ヶ月の売上を合計した値で、年間の規模感を把握します。",
        "tooltip": "計算式: 直近12ヶ月の売上高を単純合計\n意味: 年間の累計売上を把握し、規模の推移を確認します。",
    },
    "年計YoY": {
        "footnote": "前年同期比の成長率。直近12ヶ月と一年前の12ヶ月を比較します。",
        "tooltip": "計算式: (直近12ヶ月売上 ÷ 前年同期12ヶ月売上) - 1\n意味: 年計ベースで前年からどれだけ伸びたかを示します。",
    },
    "前月差(Δ)": {
        "footnote": "直近の年計モメンタム。前月とのギャップを確認します。",
        "tooltip": "計算式: 今月の年計売上 - 先月の年計売上\n意味: 直近で売上が加速しているか減速しているかを把握します。",
    },
    "HHI(集中度)": {
        "footnote": "Herfindahl-Hirschman Index でSKUの集中度を測定します。",
        "tooltip": "計算式: 各SKUの売上シェアの二乗和\n意味: 売上が一部SKUに集中しているか分散しているかを示す指標です。",
    },
    "SKU数": {
        "footnote": "年計期間中に売上が発生したSKUの件数です。",
        "tooltip": "計算式: 期間内に売上が0より大きいSKUのユニーク件数\n意味: アクティブな商品点数を把握し、ポートフォリオの広さを確認します。",
    },
}


MESSAGE_PRESETS: Dict[str, Dict[str, object]] = {
    "empty": {
        "component": "warning",
        "text": "データがありません。条件を変更して再試行してください。",
        "action": {"kind": "modify", "label": "条件を変更"},
    },
    "loading": {
        "component": "spinner",
        "text": "データを取得しています…",
    },
    "error": {
        "component": "error",
        "text": "データ取得に失敗しました。ネットワーク接続を確認してください。",
        "action": {"kind": "retry", "label": "再試行"},
    },
    "completed": {
        "component": "success",
        "text": "CSVをダウンロードしました。",
    },
}

SPINNER_MESSAGE = str(MESSAGE_PRESETS["loading"]["text"])


@contextmanager
def loading_message(detail: Optional[str] = None):
    """Display the standard loading spinner message with optional detail."""

    base = SPINNER_MESSAGE
    text = base if not detail else f"{base}\n{detail}".strip()
    with st.spinner(text):
        yield


def render_status_message(
    state: str,
    *,
    key: Optional[str] = None,
    on_retry: Optional[Callable[[], None]] = None,
    on_modify: Optional[Callable[[], None]] = None,
    guide: Optional[str] = None,
    disable_actions: bool = False,
) -> None:
    """Render status feedback based on the shared message dictionary."""

    config = MESSAGE_PRESETS.get(state)
    if not config:
        return
    component = str(config.get("component", "info"))
    if component == "spinner":
        # Use the loading_message context manager for spinners.
        return
    message = str(config.get("text", ""))
    container = st.container()
    if component == "warning":
        container.warning(message)
    elif component == "error":
        container.error(message)
    elif component == "success":
        container.success(message)
    else:
        container.info(message)

    action_cfg = config.get("action") or {}
    label = action_cfg.get("label")
    kind = action_cfg.get("kind")
    if label and kind in {"retry", "modify"}:
        btn_key = f"{key or state}_action"
        if kind == "retry":
            container.button(
                str(label),
                key=btn_key,
                on_click=on_retry,
                disabled=(on_retry is None) or disable_actions,
            )
        elif kind == "modify":
            container.button(
                str(label),
                key=btn_key,
                on_click=on_modify,
                disabled=(on_modify is None) or disable_actions,
            )

    final_guide = guide or config.get("guide")
    if final_guide:
        container.caption(str(final_guide))


def icon_svg(name: str) -> str:
    return ICON_SVGS.get(name, ICON_SVGS["info"])


@contextmanager
def compat_modal(title: str, *, key: Optional[str] = None, **kwargs):
    """Provide a modal-like context manager on Streamlit versions without st.modal."""

    modal_fn = getattr(st, "modal", None)
    if callable(modal_fn):
        with modal_fn(title, key=key, **kwargs) as modal:
            yield modal
        return

    fallback_container = st.container()
    with fallback_container:
        if title:
            st.markdown(f"### {title}")
        yield fallback_container


def render_icon_label(
    icon_key: str,
    primary: str,
    secondary: Optional[str] = None,
    *,
    help_text: Optional[str] = None,
) -> None:
    icon_html = icon_svg(icon_key)
    primary_html = html.escape(primary)
    secondary_html = (
        f"<span class='mck-inline-label__secondary'>{html.escape(secondary)}</span>"
        if secondary
        else ""
    )
    help_html = ""
    if help_text:
        tooltip = html.escape(help_text).replace("\n", "&#10;")
        help_html = (
            "<span class='mck-inline-label__help' tabindex='0' aria-label='ヘルプ' "
            f"data-tooltip='{tooltip}'>?</span>"
        )
    st.markdown(
        """
        <div class="mck-inline-label mck-animated">
          <span class="mck-inline-label__icon" aria-hidden="true">{icon}</span>
          <div class="mck-inline-label__texts">
            <span class="mck-inline-label__primary">{primary}</span>
            {secondary}
          </div>
          {help}
        </div>
        """.format(icon=icon_html, primary=primary_html, secondary=secondary_html, help=help_html),
        unsafe_allow_html=True,
    )


def _chunk_list(items: List[Dict[str, object]], size: int) -> List[List[Dict[str, object]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def detect_metric_icon(name: str) -> str:
    lowered = (name or "").lower()
    if "在庫" in name:
        return "dataset"
    if "客" in name or "単価" in name:
        return "metrics"
    if "粗利" in name or "利益" in name:
        return "delta"
    if "率" in name or "YoY" in name or "成長" in name:
        return "trend"
    return "metrics"


def render_metric_cards(
    cards: List[Dict[str, object]],
    *,
    columns: int = 3,
) -> None:
    if not cards:
        return
    rows = _chunk_list(cards, columns)
    for row in rows:
        cols = st.columns(len(row))
        for col, card in zip(cols, row):
            icon_html = icon_svg(str(card.get("icon", "metrics")))
            title_text = str(card.get("title", "指標"))
            title = html.escape(title_text)
            subtitle = card.get("subtitle")
            subtitle_html = (
                f"<span class='mck-metric-card__subtitle'>{html.escape(str(subtitle))}</span>"
                if subtitle
                else ""
            )
            value = html.escape(str(card.get("value", "—")))
            footnote = card.get("footnote")
            footnote_html = (
                f"<div class='mck-metric-card__footnote'>{html.escape(str(footnote))}</div>"
                if footnote
                else ""
            )
            tooltip_raw = card.get("tooltip") or footnote
            tooltip_text = str(tooltip_raw) if tooltip_raw is not None else ""
            tooltip_attr = ""
            aria_attr = ""
            tab_attr = ""
            info_html = ""
            classes = ["mck-metric-card", "mck-animated"]
            if tooltip_text.strip():
                tooltip_attr = (
                    " data-tooltip=\""
                    + html.escape(tooltip_text, quote=True).replace("\n", "&#10;")
                    + "\""
                )
                aria_label = f"{title_text}: {tooltip_text.replace('\n', ' ')}".strip()
                aria_attr = f" aria-label=\"{html.escape(aria_label, quote=True)}\""
                tab_attr = " tabindex=\"0\""
                classes.append("has-tooltip")
                info_html = (
                    f"<span class='mck-metric-card__info' aria-hidden='true'>{icon_svg('info')}</span>"
                )
            class_attr = " ".join(classes)
            col.markdown(
                """
                <div class="{classes}" role="group"{tooltip}{tab}{aria}>
                  <div class="mck-metric-card__header">
                    <span class="mck-metric-card__icon" aria-hidden="true">{icon}</span>
                    <div class="mck-metric-card__title-group">
                      <div class="mck-metric-card__title">{title}</div>
                      {subtitle}
                    </div>
                    {info}
                  </div>
                  <div class="mck-metric-card__value">{value}</div>
                  {footnote}
                </div>
                """.format(
                    classes=class_attr,
                    tooltip=tooltip_attr,
                    tab=tab_attr,
                    aria=aria_attr,
                    icon=icon_html,
                    title=title,
                    subtitle=subtitle_html,
                    info=info_html,
                    value=value,
                    footnote=footnote_html,
                ),
                unsafe_allow_html=True,
            )


def render_metric_bar_chart(metrics_list: List[Dict[str, object]]) -> None:
    if not metrics_list:
        return
    chart_records: List[Dict[str, object]] = []
    for metric in metrics_list:
        value = metric.get("value")
        if not isinstance(value, (int, float)):
            continue
        unit = metric.get("unit", "")
        numeric = float(value)
        if unit == "%":
            numeric *= 100
        chart_records.append(
            {
                "Metric": metric.get("name", "指標"),
                "Normalized": numeric,
                "Display": format_template_metric(metric),
            }
        )
    if not chart_records:
        return
    chart_df = pd.DataFrame(chart_records)
    max_abs = chart_df["Normalized"].abs().max()
    if not math.isfinite(max_abs) or max_abs <= 0:
        return
    chart_df["Normalized"] = chart_df["Normalized"] / max_abs
    fig = px.bar(
        chart_df,
        y="Metric",
        x="Normalized",
        orientation="h",
        text="Display",
        title="推奨KPIプレビュー（単位差を正規化）",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(showticklabels=False),
        height=280,
        margin=dict(l=10, r=10, t=46, b=30),
    )
    fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "light"))
    render_plotly_with_spinner(
        fig, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
    )
    st.caption("棒グラフは単位差を補正するために正規化値を使用しています。カードの数値で実数を確認してください。")


def render_quality_summary_panel(summary: Dict[str, object]) -> None:
    missing = int(summary.get("missing", 0) or 0)
    total = int(summary.get("total", 0) or 0)
    sku_count = int(summary.get("sku_count", 0) or 0)
    period_start = summary.get("period_start", "—")
    period_end = summary.get("period_end", "—")
    completeness = 1.0
    if total > 0:
        completeness = max(0.0, min(1.0, 1.0 - (missing / total)))
    level = "success" if completeness >= 0.95 else ("warning" if completeness >= 0.8 else "danger")
    icon_key = "check" if level == "success" else "alert"
    completeness_pct = completeness * 100
    st.markdown(
        """
        <div class="mck-alert mck-alert--{level} mck-animated">
          <div class="mck-alert__icon" aria-hidden="true">{icon}</div>
          <div class="mck-alert__content">
            <strong>データ品質サマリー</strong>
            <p>欠測セルと期間を自動チェックしました。下記を確認して次のステップへ進んでください。</p>
            <div class="mck-progress">
              <div class="mck-progress__bar" style="width:{width:.1f}%;"></div>
            </div>
            <div class="mck-progress__meta">完全性 {width:.1f}% ｜ 欠測 {missing:,} / {total:,}</div>
            <ul class="mck-alert__meta">
              <li>SKU数 {sku:,}</li>
              <li>期間 {start} 〜 {end}</li>
            </ul>
          </div>
        </div>
        """.format(
            level=level,
            icon=icon_svg(icon_key),
            width=completeness_pct,
            missing=missing,
            total=total,
            sku=sku_count,
            start=html.escape(str(period_start)),
            end=html.escape(str(period_end)),
        ),
        unsafe_allow_html=True,
    )

APP_TITLE = t("header.title", language=current_language)
st.set_page_config(
    page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded"
)


@st.cache_data(ttl=600)
def _ai_sum_df(df: pd.DataFrame) -> str:
    return summarize_dataframe(df)


@st.cache_data(ttl=600)
def _ai_explain(d: dict) -> str:
    return explain_analysis(d)


@st.cache_data(ttl=600)
def _ai_comment(t: str) -> str:
    return generate_comment(t)


@st.cache_data(ttl=600)
def _ai_actions(metrics: Dict[str, float], focus: str) -> str:
    return generate_actions(metrics, focus)


@st.cache_data(ttl=600)
def _ai_answer(question: str, context: str) -> str:
    return answer_question(question, context)


@st.cache_data(ttl=600)
def _ai_anomaly_report(df: pd.DataFrame) -> str:
    return generate_anomaly_brief(df)


from services import (
    parse_uploaded_table,
    fill_missing_months,
    compute_year_rolling,
    compute_slopes,
    abc_classification,
    compute_hhi,
    build_alerts,
    aggregate_overview,
    build_indexed_series,
    latest_yearsum_snapshot,
    resolve_band,
    filter_products_by_band,
    get_yearly_series,
    top_growth_codes,
    trend_last6,
    slopes_snapshot,
    shape_flags,
    detect_linear_anomalies,
)
from sample_data import (
    SampleCSVMeta,
    get_sample_csv_bytes,
    list_sample_csv_meta,
    load_sample_csv_dataframe,
    load_sample_dataset,
)
from core.chart_card import toolbar_sku_detail, build_chart_card
from core.plot_utils import apply_elegant_theme, render_plotly_with_spinner
from core.correlation import (
    corr_table,
    fisher_ci,
    fit_line,
    maybe_log1p,
    narrate_top_insights,
    winsorize_frame,
)
from core.product_clusters import render_correlation_category_module

# Brand-aligned light theme baseline
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
:root{
  --font-heading:'Georgia','Times New Roman','Hiragino Mincho ProN','Yu Mincho',serif;
  --font-base:'Arial','Noto Sans JP','Hiragino Kaku Gothic ProN','Meiryo',sans-serif;
  --bg:#ffffff;
  --bg-muted:#f4f7fb;
  --panel:#ffffff;
  --panel-alt:#f6f8fc;
  --ink:var(--primary,#0B1F3B);
  --ink-subtle:#40526d;
  --accent:var(--accent,#1E88E5);
  --accent-strong:#0b2f4c;
  --accent-soft:var(--accent-soft,#56A5EB);
  --muted:#5a6880;
  --border:#d4deee;
  --border-strong:#b7c5da;
  --metric-positive:var(--accent,#1E88E5);
  --metric-negative:#b24646;
}
body, .stApp, [data-testid="stAppViewContainer"]{
  background:var(--bg) !important;
  color:var(--ink) !important;
  font-family:var(--font-base);
  font-size:1rem;
  line-height:1.65;
}
[data-testid="stAppViewContainer"] > .main{
  padding-top:0;
}
[data-testid="stHeader"]{
  background:linear-gradient(90deg,var(--primary,#0B1F3B) 0%,var(--primary-light,#153C72) 100%);
  border-bottom:1px solid rgba(var(--primary-rgb,11,31,59),0.45);
}
[data-testid="stHeader"] *{
  color:var(--surface,#FFFFFF) !important;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--primary-deep,#08172C) 0%,var(--primary,#0B1F3B) 100%);
  color:var(--surface-alt,#EEF1F5);
  padding:1.6rem 1.2rem;
}
[data-testid="stSidebar"] *{
  color:var(--surface-alt,#EEF1F5) !important;
  font-family:var(--font-base);
}
[data-testid="stSidebar"] .stButton>button{
  background:rgba(255,255,255,0.16);
  border:1px solid rgba(255,255,255,0.38);
  color:#ffffff;
  box-shadow:none;
  font-weight:600;
}
[data-testid="stSidebar"] .stButton>button:hover{
  background:rgba(255,255,255,0.24);
}
.mck-inline-label{
  display:flex;
  align-items:center;
  gap:0.75rem;
  margin:0.2rem 0 0.6rem;
}
.mck-inline-label__icon{
  width:24px;
  height:24px;
  border-radius:8px;
  background:rgba(var(--accent-rgb,30,136,229),0.12);
  color:var(--accent-strong);
  display:inline-flex;
  align-items:center;
  justify-content:center;
}
.mck-inline-label__icon svg{
  width:14px;
  height:14px;
}
.mck-inline-label__texts{
  display:flex;
  flex-direction:column;
  gap:0.1rem;
}
.mck-inline-label__primary{
  font-weight:600;
  font-size:0.98rem;
}
.mck-inline-label__secondary{
  font-size:0.82rem;
  color:var(--muted);
}
.mck-inline-label__help{
  margin-left:auto;
  width:20px;
  height:20px;
  border-radius:50%;
  background:rgba(var(--accent-rgb,30,136,229),0.18);
  color:var(--accent-strong);
  font-size:0.75rem;
  font-weight:700;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  cursor:help;
  position:relative;
  transition:background 0.2s ease;
}
.mck-inline-label__help:focus-visible,
.mck-inline-label__help:hover{
  background:rgba(var(--accent-rgb,30,136,229),0.28);
}
.mck-inline-label__help::after{
  content:attr(data-tooltip);
  position:absolute;
  bottom:calc(100% + 10px);
  left:50%;
  transform:translateX(-50%);
  background:var(--ink);
  color:#ffffff;
  padding:0.45rem 0.65rem;
  border-radius:8px;
  font-size:0.78rem;
  line-height:1.4;
  max-width:280px;
  box-shadow:0 12px 24px rgba(var(--primary-rgb,11,31,59),0.18);
  opacity:0;
  visibility:hidden;
  pointer-events:none;
  transition:opacity 0.2s ease;
  white-space:pre-line;
  z-index:2000;
}
.mck-inline-label__help:hover::after,
.mck-inline-label__help:focus-visible::after{
  opacity:1;
  visibility:visible;
}
h1,h2,h3,h4{
  color:var(--ink);
  font-family:var(--font-heading);
  font-weight:600;
  letter-spacing:0.2px;
  margin-bottom:0.45rem;
}
h1{ font-size:1.4rem; }
h2{ font-size:1.25rem; }
h3{ font-size:1.12rem; }
h4{ font-size:1.0rem; }
p,li,span,div{
  color:var(--ink);
  font-family:var(--font-base);
  font-size:1rem;
  line-height:1.68;
}
small, .text-small{
  font-size:0.82rem;
  color:var(--muted);
}
.block-container{
  padding:1.8rem 2.6rem 2.6rem;
  max-width:1380px;
}
.element-container{
  margin-bottom:1.4rem;
}
[data-testid="stHorizontalBlock"]{
  gap:1.4rem !important;
}
[data-testid="stMetric"]{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:14px;
  padding:0.85rem 1rem;
  box-shadow:0 12px 28px rgba(var(--primary-rgb,11,31,59),0.08);
}
[data-testid="stMetricValue"]{
  color:var(--accent-strong);
  font-family:var(--font-heading);
  font-weight:600;
  font-variant-numeric:tabular-nums;
}
[data-testid="stMetricDelta"]{
  font-weight:600;
}
[data-testid="stMetricLabel"]{
  color:var(--muted);
  font-weight:600;
  letter-spacing:0.08em;
  text-transform:uppercase;
}
.mck-sidebar-summary{
  background:rgba(255,255,255,0.12);
  border-radius:12px;
  padding:0.85rem;
  margin-bottom:1.2rem;
  font-size:0.9rem;
  line-height:1.6;
  color:#f6fbff;
}
.mck-sidebar-summary strong{ color:#ffffff; }
.mck-hero{
  background:linear-gradient(135deg, rgba(var(--primary-rgb,11,31,59),0.96) 0%, rgba(var(--accent-rgb,30,136,229),0.88) 100%);
  color:#ffffff;
  padding:1.8rem 2rem;
  border-radius:20px;
  margin-bottom:1.2rem;
  box-shadow:0 24px 40px rgba(var(--primary-rgb,11,31,59),0.32);
  position:relative;
  overflow:hidden;
  font-family:var(--font-base);
}
.mck-hero::after{
  content:"";
  position:absolute;
  inset:auto -18% -32% auto;
  width:220px;
  height:220px;
  background:rgba(255,255,255,0.18);
  border-radius:50%;
}
.mck-hero h1{
  color:#ffffff;
  margin-bottom:0.5rem;
  font-size:1.65rem;
  font-family:var(--font-heading);
}
.mck-hero p{
  color:rgba(235,242,250,0.88);
  font-size:1.02rem;
  margin-bottom:0;
}
.mck-hero__eyebrow{
  text-transform:uppercase;
  letter-spacing:.16em;
  font-size:0.78rem;
  font-weight:600;
  color:rgba(235,242,250,0.88);
  margin-bottom:0.6rem;
  display:inline-flex;
  align-items:center;
  gap:0.5rem;
}
.mck-hero__eyebrow:before{ content:"◦"; font-size:0.9rem; }
.mck-section-header{
  display:flex;
  align-items:flex-start;
  gap:0.85rem;
  margin:0.8rem 0 0.6rem;
}
.mck-section-header h2{
  margin:0;
  font-size:1.35rem;
  line-height:1.2;
  color:var(--accent-strong);
  font-family:var(--font-heading);
}
.mck-section-subtitle{
  margin:0.25rem 0 0;
  font-size:0.96rem;
  color:var(--ink-subtle);
}
.mck-section-icon{
  width:34px;
  height:34px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border-radius:50%;
  background:rgba(var(--accent-rgb,30,136,229),0.12);
  color:var(--accent-strong);
  font-size:1rem;
  flex-shrink:0;
  margin-top:0.1rem;
}
.mck-section-icon svg{
  width:20px;
  height:20px;
}
.mck-ai-answer{
  background:var(--panel);
  border-radius:12px;
  border:1px solid var(--border);
  padding:0.75rem 0.9rem;
  box-shadow:0 12px 26px rgba(var(--primary-rgb,11,31,59),0.12);
  margin-top:0.75rem;
}
.mck-ai-answer strong{ color:var(--accent-strong); }
.stTabs [data-baseweb="tab-list"]{ gap:0.6rem; }
.stTabs [data-baseweb="tab"]{
  background:var(--panel);
  padding:0.6rem 1rem;
  border-radius:999px;
  border:1px solid var(--border);
  color:var(--muted);
  font-weight:600;
}
.stTabs [data-baseweb="tab"]:hover{ border-color:var(--accent); color:var(--accent-strong); }
.stTabs [data-baseweb="tab"]:focus{ outline:none; box-shadow:0 0 0 3px rgba(var(--accent-rgb,30,136,229),0.18); }
.stTabs [aria-selected="true"]{ background:var(--accent); color:#ffffff; border-color:var(--accent); }
.stDataFrame{ border-radius:14px !important; }
.stButton>button{
  border-radius:999px;
  padding:0.45rem 1.2rem;
  font-weight:700;
  border:1px solid var(--accent-strong);
  color:#ffffff;
  background:var(--accent);
  box-shadow:0 12px 26px rgba(15,76,129,0.28);
  transition:background .2s ease, box-shadow .2s ease;
}
.stButton>button:hover{
  background:var(--accent-strong);
  border-color:var(--accent-strong);
  color:#ffffff;
  box-shadow:0 14px 30px rgba(11,46,92,0.32);
}
[data-testid="stFileUploaderDropzoneInstructions"] > :first-child{
  width:40px;
  height:40px;
  min-width:40px;
  min-height:40px;
  display:flex;
  align-items:center;
  justify-content:center;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg{
  width:22px !important;
  height:22px !important;
}
.mck-metric-card{
  border-radius:18px;
  border:1px solid var(--border);
  background:var(--panel);
  padding:1.1rem 1.25rem;
  box-shadow:0 18px 40px rgba(var(--primary-rgb,11,31,59),0.14);
  display:flex;
  flex-direction:column;
  gap:0.4rem;
  min-height:150px;
}
.mck-metric-card__icon{
  width:26px;
  height:26px;
  border-radius:9px;
  background:rgba(var(--accent-rgb,30,136,229),0.12);
  color:var(--accent-strong);
  display:inline-flex;
  align-items:center;
  justify-content:center;
}
.mck-metric-card__icon svg{
  width:14px;
  height:14px;
}
.mck-metric-card__header{
  display:flex;
  align-items:flex-start;
  gap:0.75rem;
}
.mck-metric-card__title-group{
  display:flex;
  flex-direction:column;
  gap:0.2rem;
  flex:1;
}
.mck-metric-card__title{
  font-size:0.95rem;
  font-weight:600;
  color:var(--ink);
}
.mck-metric-card__subtitle{
  font-size:0.82rem;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:0.08em;
}
.mck-metric-card__value{
  font-size:1.45rem;
  font-family:var(--font-heading);
  color:var(--accent-strong);
  line-height:1.2;
}
.mck-metric-card__info{
  width:22px;
  height:22px;
  border-radius:50%;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  background:rgba(var(--accent-rgb,30,136,229),0.12);
  color:var(--accent-strong);
  flex-shrink:0;
}
.mck-metric-card.has-tooltip .mck-metric-card__info{
  cursor:help;
}
.mck-metric-card__info svg{
  width:14px;
  height:14px;
}
.mck-metric-card__footnote{
  font-size:0.78rem;
  color:var(--muted);
}
.mck-alert{
  border-radius:18px;
  border:1px solid transparent;
  padding:1rem 1.2rem;
  display:flex;
  gap:0.9rem;
  align-items:flex-start;
  margin:0.6rem 0 1rem;
}
.mck-alert__icon{
  width:24px;
  height:24px;
  border-radius:10px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
}
.mck-alert__icon svg{
  width:14px;
  height:14px;
}
.mck-alert__content{
  flex:1;
  font-size:0.95rem;
}
.mck-alert__content p{
  margin:0.2rem 0 0.6rem;
  color:var(--ink-subtle);
  font-size:0.9rem;
}
.mck-alert__meta{
  list-style:none;
  padding:0;
  margin:0.8rem 0 0;
  display:flex;
  flex-wrap:wrap;
  gap:0.75rem;
  font-size:0.82rem;
  color:var(--muted);
}
.mck-alert__meta li::before{
  content:"•";
  margin-right:0.35rem;
  color:currentColor;
}
.mck-alert--success{
  background:rgba(31,142,94,0.12);
  border-color:rgba(31,142,94,0.25);
  color:#135a3d;
}
.mck-alert--success .mck-alert__icon{
  background:rgba(31,142,94,0.18);
  color:#135a3d;
}
.mck-alert--warning{
  background:rgba(255,193,37,0.16);
  border-color:rgba(255,193,37,0.32);
  color:#6b4c00;
}
.mck-alert--warning .mck-alert__icon{
  background:rgba(255,193,37,0.24);
  color:#6b4c00;
}
.mck-alert--danger{
  background:rgba(178,70,70,0.16);
  border-color:rgba(178,70,70,0.32);
  color:#6b1f1f;
}
.mck-alert--danger .mck-alert__icon{
  background:rgba(178,70,70,0.24);
  color:#6b1f1f;
}
.mck-progress{
  width:100%;
  height:10px;
  border-radius:999px;
  background:rgba(var(--primary-rgb,11,31,59),0.08);
  overflow:hidden;
  position:relative;
}
.mck-progress__bar{
  height:100%;
  background:var(--accent);
  border-radius:999px;
  transition:width 0.3s ease;
}
.mck-progress__meta{
  margin-top:0.35rem;
  font-size:0.82rem;
  color:var(--muted);
}
.mck-animated{
  animation:mck-fade-in 0.3s ease;
}
@keyframes mck-fade-in{
  from{ opacity:0; transform:translateY(8px); }
  to{ opacity:1; transform:translateY(0); }
}
.tour-banner{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:20px;
  padding:1.2rem 1.5rem;
  margin:0 0 1.2rem;
  position:relative;
  overflow:hidden;
  box-shadow:0 20px 44px rgba(var(--primary-rgb,11,31,59),0.18);
}
.tour-banner::before{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(135deg, rgba(var(--accent-rgb,30,136,229),0.12), rgba(var(--accent-soft-rgb,86,165,235),0.12));
  opacity:0.25;
  pointer-events:none;
}
.tour-banner > div{ position:relative; z-index:1; }
.tour-banner--muted{
  background:linear-gradient(135deg, rgba(var(--primary-rgb,11,31,59),0.05), rgba(var(--primary-rgb,11,31,59),0.02));
  border-style:dashed;
  box-shadow:none;
}
.tour-banner__progress{
  text-transform:uppercase;
  letter-spacing:.12em;
  font-size:.78rem;
  color:var(--muted);
  margin-bottom:.35rem;
  font-weight:600;
}
.tour-banner__title{
  font-size:1.42rem;
  font-weight:700;
  color:var(--accent-strong);
  margin-bottom:.35rem;
  font-family:var(--font-heading);
}
.tour-banner__desc{
  margin-bottom:.25rem;
  color:var(--ink-subtle);
  font-size:.96rem;
}
.tour-banner__details{
  margin:0;
  font-size:.94rem;
  color:var(--ink);
}
.tour-banner__section{
  display:inline-flex;
  align-items:center;
  gap:0.45rem;
  padding:0.25rem 0.8rem;
  border-radius:999px;
  background:rgba(var(--accent-rgb,30,136,229),0.16);
  color:var(--accent-strong);
  font-weight:700;
  font-size:.82rem;
  letter-spacing:.02em;
  text-transform:none;
  margin-bottom:.55rem;
}
.tour-banner__section span{
  font-size:.72rem;
  color:var(--muted);
  font-weight:700;
  letter-spacing:.08em;
}
.tour-progress{
  margin:0.85rem 0 0;
}
.tour-progress__meta{
  display:flex;
  justify-content:space-between;
  font-size:.78rem;
  color:var(--muted);
  font-weight:600;
  letter-spacing:.05em;
  text-transform:uppercase;
  margin-bottom:.35rem;
}
.tour-progress__track{
  position:relative;
  height:8px;
  border-radius:999px;
  background:rgba(var(--accent-rgb,30,136,229),0.18);
  overflow:hidden;
}
.tour-progress__bar{
  position:absolute;
  inset:0;
  border-radius:inherit;
  background:linear-gradient(90deg, var(--accent), var(--accent-soft));
  transition:width .3s ease;
}
.tour-banner__nav{
  margin-top:1.05rem;
  display:flex;
  gap:.65rem;
}
.tour-banner__nav [data-testid="column"]{
  flex:1;
}
.tour-banner__nav [data-testid="column"] [data-testid="stButton"]>button{
  width:100%;
  border-radius:12px;
  font-weight:700;
  box-shadow:0 12px 24px rgba(var(--accent-rgb,30,136,229),0.22);
}
.tour-banner__nav [data-testid="column"] [data-testid="stButton"]>button:disabled{
  opacity:.55;
  cursor:not-allowed;
  box-shadow:none;
}
.tour-banner__nav [data-testid="column"]:first-child [data-testid="stButton"]>button,
.tour-banner__nav [data-testid="column"]:last-child [data-testid="stButton"]>button{
  background:#ffffff;
  color:var(--accent-strong);
  border:1px solid var(--accent-strong);
  box-shadow:none;
}
.tour-banner__nav [data-testid="column"]:first-child [data-testid="stButton"]>button:hover{
  background:rgba(var(--accent-rgb,30,136,229),0.08);
}
.tour-banner__nav [data-testid="column"]:last-child [data-testid="stButton"]>button{
  border-color:#b24646;
  color:#b24646;
}
.tour-banner__nav [data-testid="column"]:last-child [data-testid="stButton"]>button:hover{
  background:rgba(178,70,70,0.1);
  border-color:#962d2d;
  color:#962d2d;
}
.tour-banner__nav--resume{
  justify-content:flex-start;
}
.tour-banner__nav--resume [data-testid="column"]:first-child [data-testid="stButton"]>button{
  background:var(--accent);
  color:#ffffff;
  border-color:var(--accent);
  box-shadow:0 12px 24px rgba(var(--accent-rgb,30,136,229),0.22);
}
.tour-banner__nav--resume [data-testid="column"]:first-child [data-testid="stButton"]>button:hover{
  background:var(--accent-strong);
  border-color:var(--accent-strong);
  color:#ffffff;
}
.tour-banner__nav--resume [data-testid="column"]:last-child [data-testid="stButton"]>button{
  background:#ffffff;
  color:var(--accent-strong);
  border:1px solid var(--accent-strong);
  box-shadow:none;
}
.tour-banner__nav--resume [data-testid="column"]:last-child [data-testid="stButton"]>button:hover{
  background:rgba(var(--accent-rgb,30,136,229),0.08);
  border-color:var(--accent-strong);
  color:var(--accent-strong);
}
.tour-highlight-heading{
  position:relative;
  border-radius:18px;
  outline:3px solid rgba(var(--accent-rgb,30,136,229),0.45);
  box-shadow:0 18px 36px rgba(var(--accent-rgb,30,136,229),0.22);
  background:linear-gradient(135deg, rgba(var(--primary-rgb,11,31,59),0.08), rgba(var(--accent-soft-rgb,86,165,235),0.18));
  transition:box-shadow .3s ease;
}
.tour-highlight-heading h2{ color:var(--accent-strong) !important; font-family:var(--font-heading); }
.tour-highlight-heading::after{
  content:"";
  position:absolute;
  inset:8px;
  border-radius:14px;
  border:1px solid rgba(var(--accent-rgb,30,136,229),0.28);
  pointer-events:none;
}
section[data-testid="stSidebar"] label.tour-highlight-nav{
  border:1.6px solid rgba(255,255,255,0.72);
  border-radius:12px;
  background:rgba(255,255,255,0.18);
  box-shadow:0 0 0 3px rgba(255,255,255,0.24);
}
section[data-testid="stSidebar"] label.tour-highlight-nav *{
  color:#ffffff !important;
}
.tour-banner--muted .tour-banner__progress{ color:var(--muted); }
.tour-banner--muted .tour-banner__section{
  background:rgba(var(--accent-rgb,30,136,229),0.08);
  color:var(--muted);
}
.tour-banner--muted .tour-banner__section span{ color:var(--muted); }
.tour-banner--muted .tour-progress__meta{ color:var(--muted); }
.tour-banner--muted .tour-progress__track{
  background:rgba(var(--accent-rgb,30,136,229),0.1);
}
.tour-banner--muted .tour-progress__bar{
  background:rgba(var(--accent-rgb,30,136,229),0.22);
}
.tour-banner--muted .tour-banner__desc{ color:var(--muted); }
.chart-card{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:16px;
  box-shadow:0 16px 32px rgba(var(--primary-rgb,11,31,59),0.08);
}
.chart-toolbar{
  background:linear-gradient(180deg, rgba(var(--primary-rgb,11,31,59),0.05), rgba(var(--primary-rgb,11,31,59),0.02));
  border-bottom:1px solid rgba(var(--primary-rgb,11,31,59),0.18);
}
</style>
    """,
    unsafe_allow_html=True,
)

brand_override_template = Template(
    textwrap.dedent(
        """
        <style>
        :root{
          --font-heading:${font_heading};
          --font-base:${font_body};
          --font-body:${font_body};
          --font-mono:${font_numeric};
          --primary:${primary};
          --primary-rgb:${primary_rgb};
          --primary-dark:${primary_dark};
          --primary-deep:${primary_deep};
          --primary-deep-rgb:${primary_deep_rgb};
          --primary-light:${primary_light};
          --bg:${background};
          --bg-muted:${background_muted};
          --panel:${surface};
          --panel-alt:${surface_alt};
          --ink:${text};
          --ink-subtle:${muted};
          --muted:${muted};
          --accent:${accent};
          --accent-rgb:${accent_rgb};
          --accent-soft:${accent_soft};
          --accent-soft-rgb:${accent_soft_rgb};
          --accent-strong:${accent_emphasis};
          --border:${border};
          --border-strong:${border_strong};
          --metric-positive:${success};
          --metric-negative:${error};
          --success:${success};
          --success-rgb:${success_rgb};
          --warning:${warning};
          --warning-rgb:${warning_rgb};
          --error:${error};
          --spacing-unit:${spacing_unit}px;
          --radius-card:${card_radius}px;
          --card-shadow:${card_shadow};
          --line-height-body:${body_line_height};
          --line-height-heading:${heading_line_height};
        }

        body, .stApp, [data-testid="stAppViewContainer"]{
          font-family:${font_body};
          font-size:${body_font_size}px;
          line-height:var(--line-height-body);
          background:${background} !important;
          color:${text} !important;
        }

        [data-testid="stHeader"]{
          background:linear-gradient(90deg, ${primary} 0%, ${primary_light} 100%);
          border-bottom:1px solid rgba(${primary_rgb},0.45);
        }

        [data-testid="stHeader"] *{
          color:${surface} !important;
          font-family:${font_body};
        }

        [data-testid="stSidebar"]{
          background:linear-gradient(180deg, ${primary_deep} 0%, ${primary} 100%);
          color:${surface_alt};
        }

        [data-testid="stSidebar"] *{
          color:${surface_alt} !important;
          font-family:${font_body};
        }

        .mck-inline-label__icon,
        .mck-inline-label__help{
          background:rgba(${accent_rgb},0.12);
          color:${accent_emphasis};
          border-color:rgba(${accent_rgb},0.22);
        }

        .stTabs [aria-selected="true"]{
          background:${accent};
          color:${surface};
          border-color:${accent};
        }

        .stTabs [data-baseweb="tab"]:focus{
          box-shadow:0 0 0 3px rgba(${accent_rgb},0.18);
        }

        .stButton>button{
          font-family:${font_body};
        }

        .stButton>button:focus-visible,
        .stButton>button:hover{
          border-color:${accent_emphasis};
        }

        .stMetric-value{
          font-family:${font_numeric};
        }

        .mck-ai-answer strong{
          color:${accent_emphasis};
          font-family:${font_heading};
        }
        </style>
        """
    )
)

st.markdown(
    brand_override_template.substitute(
        font_heading=FONT_HEADING,
        font_body=FONT_BODY,
        font_numeric=FONT_NUMERIC,
        primary=PRIMARY_COLOR,
        primary_rgb=PRIMARY_RGB,
        primary_dark=PRIMARY_DARK,
        primary_deep=PRIMARY_DEEP,
        primary_deep_rgb=PRIMARY_DEEP_RGB,
        primary_light=PRIMARY_LIGHT,
        background=BACKGROUND_COLOR,
        background_muted=BACKGROUND_MUTED,
        surface=SURFACE_COLOR,
        surface_alt=SURFACE_ALT_COLOR,
        text=TEXT_COLOR,
        muted=MUTED_COLOR,
        accent=ACCENT_COLOR,
        accent_rgb=ACCENT_RGB,
        accent_soft=ACCENT_SOFT,
        accent_soft_rgb=ACCENT_SOFT_RGB,
        accent_emphasis=ACCENT_EMPHASIS,
        border=BORDER_COLOR,
        border_strong=BORDER_STRONG,
        success=SUCCESS_COLOR,
        success_rgb=SUCCESS_RGB,
        warning=WARNING_COLOR,
        warning_rgb=WARNING_RGB,
        error=ERROR_COLOR,
        spacing_unit=SPACING_UNIT,
        card_radius=CARD_RADIUS,
        card_shadow=CARD_SHADOW,
        body_line_height=BODY_LINE_HEIGHT,
        heading_line_height=HEADING_LINE_HEIGHT,
        body_font_size=BODY_FONT_SIZE,
    ),
    unsafe_allow_html=True,
)

# ===== Elegant（品格）UI ON/OFF & Language Selector =====
if "elegant_on" not in st.session_state:
    st.session_state["elegant_on"] = True
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "light"

with st.container():
    control_left, control_right = st.columns([3, 1])
    with control_left:
        toggle_cols = st.columns(2)
        with toggle_cols[0]:
            elegant_on = st.toggle(
                t("header.elegant_toggle.label"),
                value=st.session_state.get("elegant_on", True),
                help=t("header.elegant_toggle.help"),
                key="elegant_ui_toggle",
            )
            st.session_state["elegant_on"] = elegant_on
        with toggle_cols[1]:
            dark_mode = st.toggle(
                t("header.dark_mode.label", default="ダークモード"),
                value=st.session_state.get("dark_mode", False),
                help=t(
                    "header.dark_mode.help",
                    default="長時間の閲覧向けに暗色テーマへ切り替えます",
                ),
                key="mck_dark_mode_toggle",
            )
            st.session_state["dark_mode"] = dark_mode
            st.session_state["ui_theme"] = "dark" if dark_mode else "light"
    with control_right:
        language_codes = get_available_languages()
        if language_codes:
            current_value = st.session_state.get("language")
            if current_value not in language_codes:
                st.session_state["language"] = language_codes[0]
        else:
            language_codes = [get_current_language()]
        st.selectbox(
            t("header.language_selector.label"),
            options=language_codes,
            key="language",
            format_func=lambda code: language_name(code),
        )

elegant_on = st.session_state.get("elegant_on", True)
dark_mode = st.session_state.get("dark_mode", False)

# ===== 品格UI CSS（配色/余白/フォント/境界の見直し） =====
if elegant_on:
    if dark_mode:
        st.markdown(
            """
            <style>
              :root{
                --ink:#e6efff;
                --ink-subtle:#a9bddc;
                --bg:#08121f;
                --bg-muted:#0d1c30;
                --panel:#0f2138;
                --panel-alt:#162d4a;
                --border:#1f3a5d;
                --border-strong:#2f4c74;
                --accent:var(--accent,#1E88E5);
                --accent-strong:var(--accent-soft,#56A5EB);
                --accent-soft:#8fc2ff;
                --muted:#9cb1d1;
                --metric-positive:var(--accent-soft,#56A5EB);
                --metric-negative:#f18c8c;
              }
              body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--ink) !important; }
              [data-testid="stHeader"]{
                background:linear-gradient(90deg,#050a14 0%,#0d2136 100%);
                border-bottom:1px solid rgba(var(--accent-soft-rgb,86,165,235),0.28);
              }
              [data-testid="stHeader"] *{ color:#e6efff !important; }
              [data-testid="stSidebar"]{
                background:linear-gradient(180deg,#050b16 0%,#0d2239 100%);
                color:#e6efff;
              }
              [data-testid="stSidebar"] *{ color:#e6efff !important; }
              .chart-card, .stDataFrame{
                box-shadow:0 18px 38px rgba(var(--primary-deep-rgb,8,23,44),0.55) !important;
                border:1px solid var(--border) !important;
                background:var(--panel) !important;
              }
              [data-testid="stMetric"]{
                background:var(--panel-alt);
                box-shadow:0 18px 40px rgba(var(--primary-deep-rgb,8,23,44),0.5);
                border:1px solid var(--border);
              }
              .stTabs [data-baseweb="tab"]{
                background:var(--panel-alt);
                border-color:var(--border);
                color:var(--ink-subtle);
              }
              .stTabs [aria-selected="true"]{
                background:var(--accent);
                color:#041020;
                border-color:var(--accent);
              }
              .stButton>button{
                border:1px solid var(--accent-strong);
                background:var(--accent);
                color:#041020;
                box-shadow:0 16px 32px rgba(8,25,46,0.46);
              }
              .stButton>button:hover{
                background:var(--accent-strong);
                border-color:var(--accent-strong);
                color:#041020;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
              :root{
                --ink:var(--primary,#0B1F3B);
                --ink-subtle:#40526d;
                --bg:#ffffff;
                --bg-muted:#f4f7fb;
                --panel:#ffffff;
                --panel-alt:#f6f8fc;
                --border:#d4deee;
                --border-strong:#b7c5da;
                --accent:var(--accent,#1E88E5);
                --accent-strong:#0b2f4c;
                --accent-soft:var(--accent-soft,#56A5EB);
                --muted:#5a6880;
                --metric-positive:var(--accent,#1E88E5);
                --metric-negative:#b24646;
              }
              body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--ink) !important; }
              [data-testid="stHeader"]{
                background:linear-gradient(90deg,var(--primary,#0B1F3B) 0%,var(--primary-light,#153C72) 100%);
                border-bottom:1px solid rgba(var(--primary-rgb,11,31,59),0.45);
              }
              [data-testid="stSidebar"]{
                background:linear-gradient(180deg,var(--primary-deep,#08172C) 0%,var(--primary,#0B1F3B) 100%);
              }
              .chart-card, .stDataFrame{
                border:1px solid var(--border) !important;
                box-shadow:0 16px 32px rgba(var(--primary-rgb,11,31,59),0.12) !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

INDUSTRY_TEMPLATES: Dict[str, Dict[str, object]] = {
    "restaurant": {
        "label": "飲食業",
        "description": "メニュー別売上や客数、FLコストを前提とした飲食業向けテンプレートです。",
        "goal": "テンプレート適用で入力工数を約50％削減（30分→15分以内）することを目標とします。",
        "fields": [
            "メニュー/コース名",
            "店舗・業態",
            "日次または月次の客数",
            "食材費・人件費などのFLコスト",
        ],
        "recommended_metrics": [
            {
                "name": "客単価",
                "value": 2800,
                "unit": "円",
                "description": "売上高 ÷ 来店客数",
            },
            {
                "name": "席稼働率",
                "value": 0.85,
                "unit": "%",
                "description": "ピーク帯の稼働席数 ÷ 総席数",
            },
            {
                "name": "FLコスト比率",
                "value": 0.60,
                "unit": "%",
                "description": "食材費＋人件費 ÷ 売上高",
            },
        ],
        "settings": {
            "yoy_threshold": -0.08,
            "delta_threshold": -200000.0,
            "slope_threshold": -0.7,
        },
        "template_columns": ["店舗名", "メニュー名", "カテゴリ", "SKUコード"],
        "template_sample_rows": [
            {
                "店舗名": "丸の内本店",
                "メニュー名": "看板ランチセット",
                "カテゴリ": "ランチ",
                "SKUコード": "MENU001",
            },
            {
                "店舗名": "丸の内本店",
                "メニュー名": "季節ディナーコース",
                "カテゴリ": "ディナー",
                "SKUコード": "MENU002",
            },
        ],
        "financial_profile": {
            "cogs_ratio": 0.35,
            "opex_ratio": 0.32,
            "other_income_ratio": 0.02,
            "interest_ratio": 0.01,
            "tax_ratio": 0.23,
            "asset_turnover": 3.0,
            "balance_assets": [
                {"item": "現金及び預金", "ratio": 0.28},
                {"item": "売掛金", "ratio": 0.18},
                {"item": "棚卸資産", "ratio": 0.22},
                {"item": "固定資産", "ratio": 0.32},
            ],
            "balance_liabilities": [
                {"item": "流動負債", "ratio": 0.40},
                {"item": "固定負債", "ratio": 0.18},
                {"item": "純資産", "ratio": 0.42},
            ],
            "cash_flow": [
                {"item": "営業キャッシュフロー", "ratio": 0.12},
                {"item": "投資キャッシュフロー", "ratio": -0.05},
                {"item": "財務キャッシュフロー", "ratio": -0.03},
            ],
        },
    },
    "retail": {
        "label": "小売業",
        "description": "カテゴリ別の販売計画や在庫を管理する小売業向けテンプレートです。",
        "goal": "テンプレート適用で入力工数を約50％削減（30分→15分以内）することを目標とします。",
        "fields": [
            "商品カテゴリ",
            "SKU/商品コード",
            "在庫数量",
            "仕入原価",
        ],
        "recommended_metrics": [
            {
                "name": "在庫回転率",
                "value": 14,
                "unit": "回/年",
                "description": "売上原価 ÷ 平均在庫",
            },
            {
                "name": "客単価",
                "value": 6200,
                "unit": "円",
                "description": "売上高 ÷ 購買客数",
            },
            {
                "name": "粗利率",
                "value": 0.38,
                "unit": "%",
                "description": "（売上高 − 売上原価）÷ 売上高",
            },
        ],
        "settings": {
            "yoy_threshold": -0.06,
            "delta_threshold": -300000.0,
            "slope_threshold": -0.5,
        },
        "template_columns": ["店舗名", "商品カテゴリ", "商品名", "SKUコード"],
        "template_sample_rows": [
            {
                "店舗名": "旗艦店",
                "商品カテゴリ": "食品",
                "商品名": "定番スイーツA",
                "SKUコード": "SKU001",
            },
            {
                "店舗名": "旗艦店",
                "商品カテゴリ": "日用品",
                "商品名": "人気雑貨B",
                "SKUコード": "SKU002",
            },
        ],
        "financial_profile": {
            "cogs_ratio": 0.62,
            "opex_ratio": 0.22,
            "other_income_ratio": 0.01,
            "interest_ratio": 0.005,
            "tax_ratio": 0.23,
            "asset_turnover": 2.4,
            "balance_assets": [
                {"item": "現金及び預金", "ratio": 0.18},
                {"item": "売掛金", "ratio": 0.22},
                {"item": "棚卸資産", "ratio": 0.36},
                {"item": "固定資産", "ratio": 0.24},
            ],
            "balance_liabilities": [
                {"item": "仕入債務", "ratio": 0.35},
                {"item": "短期借入金", "ratio": 0.18},
                {"item": "純資産", "ratio": 0.47},
            ],
            "cash_flow": [
                {"item": "営業キャッシュフロー", "ratio": 0.09},
                {"item": "投資キャッシュフロー", "ratio": -0.03},
                {"item": "財務キャッシュフロー", "ratio": -0.01},
            ],
        },
    },
    "service": {
        "label": "サービス業",
        "description": "契約継続率や稼働率を重視するサービス業向けテンプレートです。",
        "goal": "テンプレート適用で入力工数を約50％削減（30分→15分以内）することを目標とします。",
        "fields": [
            "サービス名",
            "担当チーム",
            "契約ID/顧客ID",
            "稼働時間・提供工数",
        ],
        "recommended_metrics": [
            {
                "name": "稼働率",
                "value": 0.78,
                "unit": "%",
                "description": "提供工数 ÷ 提供可能工数",
            },
            {
                "name": "契約継続率",
                "value": 0.92,
                "unit": "%",
                "description": "継続契約数 ÷ 全契約数",
            },
            {
                "name": "人時売上高",
                "value": 9500,
                "unit": "円",
                "description": "売上高 ÷ 提供工数",
            },
        ],
        "settings": {
            "yoy_threshold": -0.04,
            "delta_threshold": -150000.0,
            "slope_threshold": -0.4,
        },
        "template_columns": ["サービス名", "担当チーム", "契約ID", "SKUコード"],
        "template_sample_rows": [
            {
                "サービス名": "サポートプランA",
                "担当チーム": "CS本部",
                "契約ID": "PLAN-A",
                "SKUコード": "SRV001",
            },
            {
                "サービス名": "プロフェッショナルサービスB",
                "担当チーム": "導入G",
                "契約ID": "PLAN-B",
                "SKUコード": "SRV002",
            },
        ],
        "financial_profile": {
            "cogs_ratio": 0.28,
            "opex_ratio": 0.45,
            "other_income_ratio": 0.03,
            "interest_ratio": 0.01,
            "tax_ratio": 0.24,
            "asset_turnover": 1.8,
            "balance_assets": [
                {"item": "現金及び預金", "ratio": 0.32},
                {"item": "売掛金", "ratio": 0.28},
                {"item": "無形資産", "ratio": 0.25},
                {"item": "投資その他", "ratio": 0.15},
            ],
            "balance_liabilities": [
                {"item": "流動負債", "ratio": 0.34},
                {"item": "固定負債", "ratio": 0.16},
                {"item": "純資産", "ratio": 0.50},
            ],
            "cash_flow": [
                {"item": "営業キャッシュフロー", "ratio": 0.18},
                {"item": "投資キャッシュフロー", "ratio": -0.07},
                {"item": "財務キャッシュフロー", "ratio": -0.04},
            ],
        },
    },
}

INDUSTRY_TEMPLATE_ORDER = ["restaurant", "retail", "service"]
DEFAULT_TEMPLATE_KEY = "retail"

st.markdown(
    """
    <style>
    .mobile-sticky-actions{
      position:sticky;
      bottom:0;
      padding:0.85rem 1rem;
      background:linear-gradient(180deg, rgba(243,246,251,0), rgba(243,246,251,0.92) 60%, rgba(243,246,251,1) 100%);
      border-top:1px solid var(--border);
      box-shadow:0 -8px 24px rgba(11,44,74,0.12);
      z-index:90;
    }
    .mobile-sticky-actions .mobile-action-caption{
      margin:0.35rem 0 0;
      font-size:0.82rem;
      color:var(--muted);
      text-align:center;
    }
    @media (max-width: 880px){
      body, .stApp, [data-testid="stAppViewContainer"]{ font-size:15px; }
      .mck-hero{ padding:1.25rem; }
      .mobile-sticky-actions{ padding:0.75rem 0.85rem; }
      .mobile-sticky-actions .stButton>button{ padding:0.85rem 1.1rem; font-size:1rem; }
      .stTabs [data-baseweb="tab"]{ font-size:0.9rem; padding:0.45rem 0.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Session State ----------------
if "data_monthly" not in st.session_state:
    st.session_state.data_monthly = None  # long-form DF
if "data_year" not in st.session_state:
    st.session_state.data_year = None
if "settings" not in st.session_state:
    default_template = INDUSTRY_TEMPLATES.get(DEFAULT_TEMPLATE_KEY, {})
    template_defaults = default_template.get("settings", {})
    st.session_state.settings = {
        "window": 12,
        "last_n": 12,
        "missing_policy": "zero_fill",
        "yoy_threshold": template_defaults.get("yoy_threshold", -0.10),
        "delta_threshold": template_defaults.get("delta_threshold", -300000.0),
        "slope_threshold": template_defaults.get("slope_threshold", -1.0),
        "currency_unit": "円",
        "industry_template": DEFAULT_TEMPLATE_KEY,
        "template_kpi_targets": [
            dict(metric) for metric in default_template.get("recommended_metrics", [])
        ],
    }
else:
    if "industry_template" not in st.session_state.settings:
        st.session_state.settings["industry_template"] = DEFAULT_TEMPLATE_KEY
    if "template_kpi_targets" not in st.session_state.settings:
        tpl = INDUSTRY_TEMPLATES.get(
            st.session_state.settings.get("industry_template", DEFAULT_TEMPLATE_KEY),
            INDUSTRY_TEMPLATES[DEFAULT_TEMPLATE_KEY],
        )
        st.session_state.settings["template_kpi_targets"] = [
            dict(metric) for metric in tpl.get("recommended_metrics", [])
        ]
if "notes" not in st.session_state:
    st.session_state.notes = {}  # product_code -> str
if "tags" not in st.session_state:
    st.session_state.tags = {}  # product_code -> List[str]
if "saved_views" not in st.session_state:
    st.session_state.saved_views = {}  # name -> dict
if "compare_params" not in st.session_state:
    st.session_state.compare_params = {}
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None
if "copilot_answer" not in st.session_state:
    st.session_state.copilot_answer = ""
if "copilot_context" not in st.session_state:
    st.session_state.copilot_context = ""
if "copilot_focus" not in st.session_state:
    st.session_state.copilot_focus = "全体サマリー"
if "tour_active" not in st.session_state:
    st.session_state.tour_active = True
if "tour_step_index" not in st.session_state:
    st.session_state.tour_step_index = 0
if "tour_completed" not in st.session_state:
    st.session_state.tour_completed = False
if "onboarding_seen" not in st.session_state:
    st.session_state.onboarding_seen = False
if "show_onboarding_modal" not in st.session_state:
    st.session_state.show_onboarding_modal = True
if "sample_data_notice" not in st.session_state:
    st.session_state.sample_data_notice = False
if "sample_data_message" not in st.session_state:
    st.session_state.sample_data_message = ""

# track user interactions and global filters
if "click_log" not in st.session_state:
    st.session_state.click_log = {}
if "filters" not in st.session_state:
    st.session_state.filters = {}

# currency unit scaling factors
UNIT_MAP = {"円": 1, "千円": 1_000, "百万円": 1_000_000}


def log_click(name: str):
    """Increment click count for command bar actions."""
    st.session_state.click_log[name] = st.session_state.click_log.get(name, 0) + 1


def get_active_template_key() -> str:
    settings = st.session_state.get("settings", {})
    key = settings.get("industry_template", DEFAULT_TEMPLATE_KEY)
    return key if key in INDUSTRY_TEMPLATES else DEFAULT_TEMPLATE_KEY


def get_template_config(template_key: Optional[str] = None) -> Dict[str, object]:
    key = template_key or get_active_template_key()
    return INDUSTRY_TEMPLATES.get(key, INDUSTRY_TEMPLATES[DEFAULT_TEMPLATE_KEY])


def apply_industry_template(template_key: str) -> None:
    if "settings" not in st.session_state:
        return
    template = INDUSTRY_TEMPLATES.get(template_key)
    if not template:
        return
    settings = st.session_state.settings
    template_defaults = template.get("settings", {})
    for field, value in template_defaults.items():
        settings[field] = value
    settings["industry_template"] = template_key
    settings["template_kpi_targets"] = [
        dict(metric) for metric in template.get("recommended_metrics", [])
    ]


def build_industry_template_csv(template_key: str, months: int = 12) -> bytes:
    template = INDUSTRY_TEMPLATES.get(template_key)
    if not template:
        return b""
    base_columns = template.get("template_columns", ["品目名"])
    sample_rows = template.get("template_sample_rows") or [{}]
    end_period = pd.Timestamp.today().to_period("M")
    periods = pd.period_range(end_period - (months - 1), end_period, freq="M")
    month_columns = [period.strftime("%Y-%m") for period in periods]
    rows: List[List[object]] = []
    for row in sample_rows:
        base_values = [row.get(col, "") for col in base_columns]
        month_values = [0 for _ in month_columns]
        rows.append(base_values + month_values)
    if not rows:
        rows.append(["" for _ in base_columns + month_columns])
    df = pd.DataFrame(rows, columns=base_columns + month_columns)
    return df.to_csv(index=False).encode("utf-8-sig")


def _normalize_statement_items(items: List[Dict[str, float]]) -> List[Dict[str, float]]:
    normalized: List[Dict[str, float]] = []
    total = 0.0
    for item in items:
        label = item.get("item") or item.get("label") or "項目"
        ratio_raw = item.get("ratio", 0.0)
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ratio) or ratio == 0:
            continue
        normalized.append({"item": label, "ratio": ratio})
        total += ratio
    if not normalized or total == 0:
        return [{"item": "合計", "ratio": 1.0}]
    return [
        {"item": entry["item"], "ratio": entry["ratio"] / total}
        for entry in normalized
    ]


def build_financial_statements(
    year_df: Optional[pd.DataFrame],
    month: Optional[str],
    template_key: str,
) -> Dict[str, object]:
    template = get_template_config(template_key)
    profile = template.get("financial_profile", {})
    meta_template = {
        "template_key": template_key,
        "template_label": template.get("label", template_key),
        "revenue": 0.0,
        "assets_total": 0.0,
        "net_income": 0.0,
    }
    if year_df is None or month is None:
        return {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cash": pd.DataFrame(),
            "meta": meta_template,
        }
    snapshot = year_df[year_df["month"] == month].dropna(subset=["year_sum"])
    if snapshot.empty:
        return {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cash": pd.DataFrame(),
            "meta": meta_template,
        }

    revenue = float(snapshot["year_sum"].sum())
    if not math.isfinite(revenue) or revenue <= 0:
        return {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cash": pd.DataFrame(),
            "meta": meta_template,
        }

    cogs_ratio = profile.get("cogs_ratio", 0.6)
    opex_ratio = profile.get("opex_ratio", 0.25)
    other_income_ratio = profile.get("other_income_ratio", 0.01)
    interest_ratio = profile.get("interest_ratio", 0.0)
    tax_ratio = profile.get("tax_ratio", 0.23)

    cogs_value = -revenue * cogs_ratio
    gross_profit = revenue + cogs_value
    opex_value = -revenue * opex_ratio
    operating_income = gross_profit + opex_value
    other_income = revenue * other_income_ratio
    interest_expense = -revenue * interest_ratio
    ordinary_income = operating_income + other_income + interest_expense
    tax_base = max(ordinary_income, 0.0)
    taxes_value = -tax_base * tax_ratio
    net_income = ordinary_income + taxes_value

    income_records: List[Dict[str, object]] = [
        {"項目": "売上高", "金額": revenue, "構成比": 1.0},
        {
            "項目": profile.get("cogs_label", "売上原価"),
            "金額": cogs_value,
            "構成比": cogs_value / revenue,
        },
        {"項目": "売上総利益", "金額": gross_profit, "構成比": gross_profit / revenue},
        {"項目": "販管費", "金額": opex_value, "構成比": opex_value / revenue},
        {"項目": "営業利益", "金額": operating_income, "構成比": operating_income / revenue},
    ]
    if other_income:
        income_records.append(
            {
                "項目": "営業外収益",
                "金額": other_income,
                "構成比": other_income / revenue,
            }
        )
    if interest_expense:
        income_records.append(
            {
                "項目": "支払利息等",
                "金額": interest_expense,
                "構成比": interest_expense / revenue,
            }
        )
    income_records.append(
        {
            "項目": "経常利益",
            "金額": ordinary_income,
            "構成比": ordinary_income / revenue,
        }
    )
    if taxes_value:
        income_records.append(
            {
                "項目": "法人税等",
                "金額": taxes_value,
                "構成比": taxes_value / revenue,
            }
        )
    income_records.append(
        {"項目": "当期純利益", "金額": net_income, "構成比": net_income / revenue}
    )
    income_df = pd.DataFrame(income_records)

    asset_turnover = profile.get("asset_turnover", 2.5)
    if not asset_turnover or not math.isfinite(asset_turnover):
        asset_turnover = 2.5
    assets_total = revenue / asset_turnover if asset_turnover else revenue

    assets_items = _normalize_statement_items(profile.get("balance_assets", []))
    liabilities_items = _normalize_statement_items(
        profile.get("balance_liabilities", [])
    )

    balance_records: List[Dict[str, object]] = [
        {
            "区分": "資産",
            "項目": item["item"],
            "金額": assets_total * item["ratio"],
            "構成比": item["ratio"],
        }
        for item in assets_items
    ]
    balance_records.extend(
        {
            "区分": "負債・純資産",
            "項目": item["item"],
            "金額": assets_total * item["ratio"],
            "構成比": item["ratio"],
        }
        for item in liabilities_items
    )
    balance_df = pd.DataFrame(balance_records)

    cash_records: List[Dict[str, object]] = []
    net_cash = 0.0
    for item in profile.get("cash_flow", []):
        label = item.get("item") or "キャッシュフロー"
        ratio_raw = item.get("ratio", 0.0)
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ratio) or ratio == 0:
            continue
        amount = revenue * ratio
        net_cash += amount
        cash_records.append({"項目": label, "金額": amount, "構成比": ratio})
    if cash_records:
        cash_records.append(
            {
                "項目": "フリーキャッシュフロー",
                "金額": net_cash,
                "構成比": net_cash / revenue,
            }
        )
    cash_df = pd.DataFrame(cash_records)

    return {
        "income": income_df,
        "balance": balance_df,
        "cash": cash_df,
        "meta": {
            "template_key": template_key,
            "template_label": template.get("label", template_key),
            "revenue": revenue,
            "assets_total": assets_total,
            "net_income": net_income,
        },
    }


def format_template_metric(metric: Dict[str, object]) -> str:
    unit = metric.get("unit", "")
    value = metric.get("value", "—")
    if isinstance(value, (int, float)):
        if unit == "%":
            return f"{value * 100:.1f}%"
        if unit == "円":
            return f"{format_int(value)} 円"
        if unit:
            return f"{value}{unit}"
        return format_int(value)
    value_str = value if isinstance(value, str) else str(value)
    if unit and not value_str.endswith(unit):
        return f"{value_str}{unit}"
    return value_str


def render_app_hero():
    st.markdown(
        f"""
        <div class=\"mck-hero\">
            <div class=\"mck-hero__eyebrow\">{t("header.eyebrow")}</div>
            <h1>{t("header.title")}</h1>
            <p>{t("header.description")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_onboarding_modal() -> None:
    if st.session_state.get("onboarding_seen"):
        return
    if not st.session_state.get("show_onboarding_modal", True):
        return
    with compat_modal("クイックスタートガイド", key="onboarding_modal"):
        st.write("数分で主要なワークフローを体験できます。下記の流れで操作を進めましょう。")
        st.markdown(
            "- **データ取込** — サンプルや自社データをアップロードして分析を有効化\n"
            "- **ダッシュボード** — KPIカードとAIサマリーで全体像を確認\n"
            "- **分析ツール** — ランキングや比較ビューで気になるSKUを深掘り"
        )
        st.caption("ヒント: メニューのアイコンやボタンにカーソルを合わせると詳細のツールチップが表示されます。")
        start_col, skip_col = st.columns(2)
        if start_col.button(
            "ツアーを開始",
            key="onboarding_start",
            help="基礎編から順に案内する操作ツアーを開始します。",
        ):
            st.session_state.onboarding_seen = True
            st.session_state.show_onboarding_modal = False
            st.session_state.tour_active = True
            st.session_state.tour_completed = False
            st.session_state.tour_step_index = 0
            if TOUR_STEPS:
                st.session_state.tour_pending_nav = TOUR_STEPS[0]["nav_key"]
            st.rerun()
        if skip_col.button(
            "あとで見る",
            key="onboarding_skip",
            help="オンボーディングを閉じて通常画面に進みます。",
        ):
            st.session_state.onboarding_seen = True
            st.session_state.show_onboarding_modal = False
            st.rerun()


def get_current_tour_step() -> Optional[Dict[str, str]]:
    if not st.session_state.get("tour_active", True):
        return None
    if not TOUR_STEPS:
        return None
    idx = max(0, min(st.session_state.get("tour_step_index", 0), len(TOUR_STEPS) - 1))
    return TOUR_STEPS[idx]


def render_tour_banner() -> None:
    if not TOUR_STEPS:
        return

    total = len(TOUR_STEPS)
    idx = max(0, min(st.session_state.get("tour_step_index", 0), total - 1))
    st.session_state.tour_step_index = idx
    active = st.session_state.get("tour_active", True)

    banner = st.container()
    with banner:
        banner_class = "tour-banner" if active else "tour-banner tour-banner--muted"
        st.markdown(f"<div class='{banner_class}'>", unsafe_allow_html=True)
        if active:
            step = TOUR_STEPS[idx]
            section_label = step.get("section", "")
            section_index = step.get("section_index", idx + 1)
            section_total = step.get("section_total", total)
            title_text = step.get("title") or step.get("heading") or step.get("label") or ""
            description_text = step.get("description", "")
            details_text = step.get("details", "")

            if section_label:
                st.markdown(
                    f"<div class='tour-banner__section'>{html.escape(section_label)}<span>{section_index} / {section_total}</span></div>",
                    unsafe_allow_html=True,
                )

            if title_text:
                st.markdown(
                    f"<div class='tour-banner__title'>{html.escape(title_text)}</div>",
                    unsafe_allow_html=True,
                )
            if description_text:
                st.markdown(
                    f"<p class='tour-banner__desc'>{html.escape(description_text)}</p>",
                    unsafe_allow_html=True,
                )
            if details_text:
                st.markdown(
                    f"<p class='tour-banner__details'>{html.escape(details_text)}</p>",
                    unsafe_allow_html=True,
                )

            section_progress_label = (
                f"{section_label} {section_index} / {section_total}"
                if section_label
                else f"STEP {idx + 1} / {total}"
            )
            progress_percent = ((idx + 1) / total) * 100 if total else 0
            progress_html = f"""
<div class='tour-progress'>
  <div class='tour-progress__meta'>
    <span>{html.escape(section_progress_label)}</span>
    <span>STEP {idx + 1} / {total}</span>
  </div>
  <div class='tour-progress__track' role='progressbar' aria-valuemin='1' aria-valuemax='{total}' aria-valuenow='{idx + 1}'>
    <div class='tour-progress__bar' style='width: {progress_percent:.2f}%;'></div>
  </div>
</div>
"""
            st.markdown(progress_html, unsafe_allow_html=True)

            st.markdown("<div class='tour-banner__nav'>", unsafe_allow_html=True)
            prev_col, next_col, finish_col = st.columns(3)
            prev_clicked = prev_col.button(
                "前へ",
                key="tour_prev",
                use_container_width=True,
                disabled=idx == 0,
            )
            next_clicked = next_col.button(
                "次へ",
                key="tour_next",
                use_container_width=True,
                disabled=idx >= total - 1,
            )
            finish_clicked = finish_col.button(
                "終了",
                key="tour_finish",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if prev_clicked and idx > 0:
                new_idx = idx - 1
                st.session_state.tour_step_index = new_idx
                st.session_state.tour_pending_nav = TOUR_STEPS[new_idx]["nav_key"]
                st.session_state.tour_completed = False
                st.rerun()

            if next_clicked and idx < total - 1:
                new_idx = idx + 1
                st.session_state.tour_step_index = new_idx
                st.session_state.tour_pending_nav = TOUR_STEPS[new_idx]["nav_key"]
                st.session_state.tour_completed = False
                st.rerun()

            if finish_clicked:
                st.session_state.tour_active = False
                st.session_state.tour_completed = True
                st.session_state.pop("tour_pending_nav", None)
                st.rerun()
        else:
            completed = st.session_state.get("tour_completed", False)
            last_step = TOUR_STEPS[idx] if 0 <= idx < total else None
            section_label = last_step.get("section", "") if last_step else ""
            section_index = last_step.get("section_index", 0) if last_step else 0
            section_total = last_step.get("section_total", 0) if last_step else 0
            title_text = (
                last_step.get("title")
                or last_step.get("heading")
                or last_step.get("label")
                or ""
                if last_step
                else ""
            )

            if section_label:
                st.markdown(
                    f"<div class='tour-banner__section'>{html.escape(section_label)}<span>{section_index} / {section_total}</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<p class='tour-banner__progress'>チュートリアルツアー</p>",
                unsafe_allow_html=True,
            )

            if completed and idx == total - 1:
                desc_text = "基礎編から応用編までのツアーを完了しました。必要なときにいつでも振り返りできます。"
            elif last_step:
                desc_text = (
                    f"前回は{section_label}の「{title_text}」まで進みました。途中から続きが再開できます。"
                )
            else:
                desc_text = "再開ボタンでいつでもハイライトを確認できます。"

            st.markdown(
                f"<p class='tour-banner__desc'>{html.escape(desc_text)}</p>",
                unsafe_allow_html=True,
            )

            if last_step:
                section_progress_label = (
                    f"{section_label} {section_index} / {section_total}"
                    if section_label
                    else f"STEP {idx + 1} / {total}"
                )
                progress_percent = ((idx + 1) / total) * 100 if total else 0
                progress_html = f"""
<div class='tour-progress'>
  <div class='tour-progress__meta'>
    <span>{html.escape(section_progress_label)}</span>
    <span>STEP {idx + 1} / {total}</span>
  </div>
  <div class='tour-progress__track' role='progressbar' aria-valuemin='1' aria-valuemax='{total}' aria-valuenow='{idx + 1}'>
    <div class='tour-progress__bar' style='width: {progress_percent:.2f}%;'></div>
  </div>
</div>
"""
                st.markdown(progress_html, unsafe_allow_html=True)

            st.markdown(
                "<div class='tour-banner__nav tour-banner__nav--resume'>",
                unsafe_allow_html=True,
            )
            resume_col, restart_col = st.columns(2)
            resume_clicked = resume_col.button(
                "再開",
                key="tour_resume",
                use_container_width=True,
            )
            restart_clicked = restart_col.button(
                "最初から",
                key="tour_restart",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if resume_clicked:
                st.session_state.tour_active = True
                st.session_state.tour_completed = False
                if last_step and last_step.get("nav_key") in NAV_KEYS:
                    st.session_state.tour_pending_nav = last_step["nav_key"]
                st.rerun()

            if restart_clicked:
                st.session_state.tour_active = True
                st.session_state.tour_completed = False
                st.session_state.tour_step_index = 0
                if TOUR_STEPS:
                    st.session_state.tour_pending_nav = TOUR_STEPS[0]["nav_key"]
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def apply_tour_highlight(step: Optional[Dict[str, str]]) -> None:
    payload = {
        "key": step.get("key") if step else "",
        "navKey": step.get("nav_key") if step else "",
        "label": step.get("label") if step else "",
        "heading": step.get("heading") if step else "",
    }
    script = f"""
    <script>
    const STEP = {json.dumps(payload, ensure_ascii=False)};
    const normalize = (text) => (text || '').replace(/\s+/g, ' ').trim();
    const doc = window.parent.document;
    const run = () => {{
        const root = doc.documentElement;
        if (STEP.key) {{
            root.setAttribute('data-tour-key', STEP.key);
        }} else {{
            root.removeAttribute('data-tour-key');
        }}

        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {{
            sidebar.querySelectorAll('.tour-highlight-nav').forEach((el) => el.classList.remove('tour-highlight-nav'));
            let target = null;
            if (STEP.navKey) {{
                target = sidebar.querySelector(`label[data-nav-key="${{STEP.navKey}}"]`);
            }}
            if (!target && STEP.label) {{
                const labels = Array.from(sidebar.querySelectorAll('label'));
                target = labels.find((el) => normalize(el.innerText) === normalize(STEP.label));
            }}
            if (target) {{
                target.classList.add('tour-highlight-nav');
                target.scrollIntoView({{ block: 'nearest' }});
            }}
        }}

        doc.querySelectorAll('.tour-highlight-heading').forEach((el) => el.classList.remove('tour-highlight-heading'));
        if (STEP.heading) {{
            const headings = Array.from(doc.querySelectorAll('h1, h2, h3'));
            const targetHeading = headings.find((el) => normalize(el.innerText) === normalize(STEP.heading));
            if (targetHeading) {{
                const container = targetHeading.closest('.mck-section-header') || targetHeading.parentElement;
                if (container) {{
                    container.classList.add('tour-highlight-heading');
                    container.scrollIntoView({{ block: 'start', behavior: 'smooth' }});
                }}
            }}
        }}

        const hints = Array.from(doc.querySelectorAll('div, span')).filter((el) => normalize(el.textContent).includes('→キーで次へ'));
        hints.forEach((el) => el.remove());
    }};
    setTimeout(run, 120);
    </script>
    """
    components.html(script, height=0)
def section_header(
    title: str, subtitle: Optional[str] = None, icon: Optional[str] = None
):
    icon_html = f"<span class='mck-section-icon'>{icon}</span>" if icon else ""
    subtitle_html = (
        f"<p class='mck-section-subtitle'>{subtitle}</p>" if subtitle else ""
    )
    st.markdown(
        f"""
        <div class=\"mck-section-header\">
            {icon_html}
            <div>
                <h2>{title}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clip_text(value: str, width: int = 220) -> str:
    if not value:
        return ""
    return textwrap.shorten(value, width=width, placeholder="…")


# ---------------- Helpers ----------------
def require_data():
    if st.session_state.data_year is None or st.session_state.data_monthly is None:
        st.stop()


def month_options(df: pd.DataFrame) -> List[str]:
    return sorted(df["month"].dropna().unique().tolist())


def end_month_selector(
    df: pd.DataFrame,
    key: str = "end_month",
    label: str = "終端月（年計の計算対象）",
    sidebar: bool = False,
    help_text: Optional[str] = None,
    default: Optional[str] = None,
):
    """Month selector that can be rendered either in the main area or sidebar."""

    mopts = month_options(df)
    widget = st.sidebar if sidebar else st
    if not mopts:
        widget.caption("対象となる月がありません。")
        return None
    default_value = default
    if default_value is None:
        default_value = st.session_state.get("filters", {}).get("end_month")
    session_value = st.session_state.get(key)
    if session_value is None and default_value in mopts:
        st.session_state[key] = default_value
        session_value = default_value
    if session_value not in mopts:
        fallback = default_value if default_value in mopts else mopts[-1]
        st.session_state[key] = fallback
        session_value = fallback
    index = mopts.index(session_value)
    return widget.selectbox(
        label,
        mopts,
        index=index,
        key=key,
        help=help_text or "集計結果を確認したい基準月を選択します。",
    )


def download_excel(df: pd.DataFrame, filename: str) -> bytes:
    import xlsxwriter  # noqa

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()


def download_pdf_overview(kpi: dict, top_df: pd.DataFrame, filename: str) -> bytes:
    # Minimal PDF using reportlab (text only)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "年計KPIサマリー")
    y -= 24
    c.setFont("Helvetica", 11)
    for k, v in kpi.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "TOP10（年計）")
    y -= 18
    c.setFont("Helvetica", 10)
    cols = ["product_code", "product_name", "year_sum"]
    for _, row in top_df[cols].head(10).iterrows():
        c.drawString(
            40,
            y,
            f"{row['product_code']}  {row['product_name']}  {int(row['year_sum']):,}",
        )
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def process_long_dataframe(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize long-form sales data and update session state tables."""

    settings = st.session_state.settings
    policy = settings.get("missing_policy", "zero_fill")
    window = int(settings.get("window", 12) or 12)
    last_n = int(settings.get("last_n", 12) or 12)

    normalized = fill_missing_months(long_df.copy(), policy=policy)
    year_df = compute_year_rolling(normalized, window=window, policy=policy)
    year_df = compute_slopes(year_df, last_n=last_n)

    st.session_state.data_monthly = normalized
    st.session_state.data_year = year_df
    return normalized, year_df


def ingest_wide_dataframe(
    df_raw: pd.DataFrame,
    *,
    product_name_col: str,
    product_code_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a wide table to normalized long/year tables and persist them."""

    long_df = parse_uploaded_table(
        df_raw,
        product_name_col=product_name_col,
        product_code_col=product_code_col,
    )
    return process_long_dataframe(long_df)


def format_amount(val: Optional[float], unit: str) -> str:
    """Format a numeric value according to currency unit."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    scale = UNIT_MAP.get(unit, 1)
    return f"{format_int(val / scale)} {unit}".strip()


def format_int(val: float | int) -> str:
    """Format a number with commas and no decimal part."""
    try:
        return f"{int(round(val)):,}"
    except (TypeError, ValueError):
        return "0"


def render_dataset_metric_cards(
    data_year: Optional[pd.DataFrame], end_month: Optional[str]
) -> None:
    if data_year is None or getattr(data_year, "empty", True) or not end_month:
        return
    unit = st.session_state.settings.get("currency_unit", "円")
    kpi = aggregate_overview(data_year, end_month)
    hhi_val = compute_hhi(data_year, end_month)
    sku_count = int(data_year["product_code"].nunique()) if "product_code" in data_year.columns else 0
    yoy_val = kpi.get("yoy")
    def _card(title: str, subtitle: str, value: str, icon: str) -> Dict[str, object]:
        meta = METRIC_EXPLANATIONS.get(title, {})
        tooltip = meta.get("tooltip") or meta.get("footnote")
        return {
            "title": title,
            "subtitle": subtitle,
            "value": value,
            "icon": icon,
            "footnote": meta.get("footnote"),
            "tooltip": tooltip,
        }

    cards: List[Dict[str, object]] = [
        _card(
            "年計総額",
            f"{end_month} 基準",
            format_amount(kpi.get("total_year_sum"), unit),
            "dataset",
        ),
        _card(
            "年計YoY",
            "前年比",
            f"{yoy_val * 100:.1f}%" if yoy_val is not None else "—",
            "trend",
        ),
        _card(
            "前月差(Δ)",
            "モメンタム",
            format_amount(kpi.get("delta"), unit),
            "delta",
        ),
    ]
    if hhi_val is not None:
        cards.append(
            _card("HHI(集中度)", "シェア分散", f"{hhi_val:.3f}", "metrics")
        )
    cards.append(
        _card("SKU数", "アクティブ件数", f"{sku_count:,}", "sku")
    )
    render_icon_label(
        "metrics",
        "主要指標サマリー",
        "Key KPI snapshot",
        help_text="年計基準のKPIをカード形式で表示します。ダッシュボードに移動する前に全体感を把握できます。",
    )
    render_metric_cards(cards, columns=min(4, len(cards)))


def _detect_column(
    df: Optional[pd.DataFrame], candidates: List[str]
) -> Optional[str]:
    if df is None or getattr(df, "empty", True):
        return None
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _resolve_store_options(
    df: Optional[pd.DataFrame],
) -> Tuple[List[str], Optional[str]]:
    store_column = _detect_column(
        df,
        ["店舗", "店舗名", "store", "Store", "支店", "location", "branch"],
    )
    if not store_column:
        return ["全体"], None
    values = (
        df[store_column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": None})
        .dropna()
        .unique()
        .tolist()
    )
    values = sorted(values)
    return ["全体"] + values, store_column


def _detect_channel_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    return _detect_column(
        df,
        [
            "チャネル",
            "販売チャネル",
            "channel",
            "Channel",
            "チャネル区分",
        ],
    )


def _filter_monthly_data(
    df: Optional[pd.DataFrame],
    *,
    end_month: Optional[str],
    months: Optional[int],
    store_column: Optional[str] = None,
    store_value: Optional[str] = None,
) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(
            columns=["product_code", "product_name", "month", "sales_amount_jpy"]
        )

    filtered = df.copy()
    filtered["month"] = filtered["month"].astype(str)
    if store_column and store_value and store_value != "全体":
        filtered = filtered[
            filtered[store_column].astype(str).str.strip() == str(store_value)
        ]

    filtered["month_dt"] = pd.to_datetime(filtered["month"], errors="coerce")
    filtered = filtered.dropna(subset=["month_dt"])
    filtered = filtered.sort_values("month_dt")

    if filtered.empty:
        return filtered

    if end_month:
        end_dt = pd.to_datetime(end_month, errors="coerce")
    else:
        end_dt = filtered["month_dt"].max()

    if pd.isna(end_dt):
        end_dt = filtered["month_dt"].max()

    if months and months > 0:
        start_dt = end_dt - pd.DateOffset(months=months - 1)
        mask = (filtered["month_dt"] >= start_dt) & (filtered["month_dt"] <= end_dt)
        filtered = filtered.loc[mask]

    return filtered.reset_index(drop=True)


def _prepare_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "month_dt",
                "month",
                "sales_amount_jpy",
                "delta",
                "yoy",
            ]
        )
    monthly = (
        df.groupby("month_dt", as_index=False)["sales_amount_jpy"].sum().sort_values(
            "month_dt"
        )
    )
    monthly["month"] = monthly["month_dt"].dt.strftime("%Y-%m")
    monthly["delta"] = monthly["sales_amount_jpy"].diff()
    monthly["yoy"] = monthly["sales_amount_jpy"].pct_change(periods=12)
    return monthly


def _sorted_months(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or getattr(df, "empty", True) or "month" not in df.columns:
        return []
    months = df["month"].dropna().astype(str).unique().tolist()
    return sorted(months)


def _previous_month(months: List[str], current: Optional[str]) -> Optional[str]:
    if not months or not current:
        return None
    try:
        idx = months.index(current)
    except ValueError:
        return None
    if idx <= 0:
        return None
    return months[idx - 1]


def _find_ratio(items: Iterable[Dict[str, object]], keywords: Iterable[str]) -> float:
    for item in items or []:
        label = str(item.get("item", "")).lower()
        for keyword in keywords:
            if keyword.lower() in label:
                try:
                    return float(item.get("ratio", 0.0))
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def _monthly_year_totals(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["month", "year_sum", "month_dt", "delta"])
    totals = (
        df.groupby("month", as_index=False)["year_sum"].sum().sort_values("month")
    )
    totals["month_dt"] = pd.to_datetime(totals["month"], errors="coerce")
    totals["delta"] = totals["year_sum"].diff()
    return totals


def _compute_financial_snapshot(
    year_df: Optional[pd.DataFrame],
    month: Optional[str],
    profile: Optional[Dict[str, object]],
) -> Dict[str, object]:
    base_snapshot = {
        "revenue": 0.0,
        "cogs": 0.0,
        "gross_profit": 0.0,
        "gross_margin_rate": None,
        "assets_total": 0.0,
        "cash_balance": 0.0,
        "inventory_balance": 0.0,
        "cash_flows": [],
        "net_cash_flow": 0.0,
    }
    if (
        year_df is None
        or getattr(year_df, "empty", True)
        or not month
        or not profile
    ):
        return base_snapshot

    snapshot = year_df[year_df["month"] == month].dropna(subset=["year_sum"])
    total_revenue = float(snapshot["year_sum"].sum())
    if total_revenue <= 0:
        return base_snapshot

    cogs_ratio = float(profile.get("cogs_ratio", 0.6) or 0.0)
    asset_turnover = float(profile.get("asset_turnover", 2.5) or 0.0)
    gross_profit = total_revenue * (1 - cogs_ratio)
    gross_margin_rate = gross_profit / total_revenue if total_revenue else None
    assets_total = (
        total_revenue / asset_turnover if asset_turnover else total_revenue
    )
    cash_ratio = _find_ratio(profile.get("balance_assets", []), ["現金", "cash"])
    inventory_ratio = _find_ratio(
        profile.get("balance_assets", []),
        ["棚卸", "inventory", "在庫"],
    )
    cash_balance = assets_total * cash_ratio
    inventory_balance = assets_total * inventory_ratio

    cash_flows: List[Dict[str, object]] = []
    net_cash = 0.0
    for item in profile.get("cash_flow", []):
        label = item.get("item", "キャッシュフロー")
        try:
            ratio = float(item.get("ratio", 0.0))
        except (TypeError, ValueError):
            ratio = 0.0
        amount = total_revenue * ratio
        net_cash += amount
        cash_flows.append({"item": label, "amount": amount, "ratio": ratio})

    return {
        "revenue": total_revenue,
        "cogs": total_revenue * cogs_ratio,
        "gross_profit": gross_profit,
        "gross_margin_rate": gross_margin_rate,
        "assets_total": assets_total,
        "cash_balance": cash_balance,
        "inventory_balance": inventory_balance,
        "cash_flows": cash_flows,
        "net_cash_flow": net_cash,
    }


def _render_sales_tab(
    *,
    filtered_monthly: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    unit: str,
    end_month: Optional[str],
    year_df: Optional[pd.DataFrame],
    channel_column: Optional[str],
) -> None:
    unit_scale = UNIT_MAP.get(unit, 1)
    st.markdown("##### 指標カード")
    metric_cols = st.columns(3)

    snapshot_month = end_month
    if not snapshot_month and not monthly_trend.empty:
        snapshot_month = monthly_trend["month"].iloc[-1]

    monthly_value = 0.0
    delta_label = None
    yoy_label = "—"
    if not monthly_trend.empty:
        latest = monthly_trend.iloc[-1]
        prev = monthly_trend.iloc[-2] if len(monthly_trend) > 1 else None
        monthly_value = float(latest.get("sales_amount_jpy", 0.0) or 0.0)
        delta_value = latest.get("delta")
        yoy_value = latest.get("yoy")
        delta_label = (
            format_amount(delta_value, unit) if delta_value is not None else None
        )
        metric_cols[0].metric(
            "月次売上",
            format_amount(monthly_value, unit),
            delta=delta_label,
        )
        yoy_label = f"{yoy_value * 100:.1f}%" if pd.notna(yoy_value) else "—"
        yoy_delta = None
        if (
            prev is not None
            and pd.notna(prev.get("yoy"))
            and pd.notna(yoy_value)
        ):
            yoy_delta = f"{(yoy_value - prev.get('yoy', 0.0)) * 100:.1f}pt"
        metric_cols[1].metric("前年同月比", yoy_label, delta=yoy_delta)
    else:
        for col in metric_cols[:2]:
            col.metric("—", "—")

    snapshot = pd.DataFrame()
    if snapshot_month:
        snapshot = filtered_monthly[filtered_monthly["month"] == snapshot_month]
    if snapshot.empty and not filtered_monthly.empty:
        fallback_month = filtered_monthly["month"].iloc[-1]
        snapshot = filtered_monthly[filtered_monthly["month"] == fallback_month]

    top_share = None
    if not snapshot.empty:
        product_totals = (
            snapshot.groupby(["product_code", "product_name"], as_index=False)[
                "sales_amount_jpy"
            ]
            .sum()
            .sort_values("sales_amount_jpy", ascending=False)
        )
        total_snapshot = float(product_totals["sales_amount_jpy"].sum())
        if total_snapshot > 0 and not product_totals.empty:
            top_share = (
                product_totals.iloc[0]["sales_amount_jpy"] / total_snapshot * 100.0
            )
    metric_cols[2].metric(
        "トップ商品構成比",
        f"{top_share:.1f}%" if top_share is not None else "—",
    )

    snapshot_year = pd.DataFrame()
    if (
        year_df is not None
        and not getattr(year_df, "empty", True)
        and snapshot_month
    ):
        snapshot_year = year_df[year_df["month"] == snapshot_month].dropna(
            subset=["year_sum"]
        )

    month_totals = pd.DataFrame()
    if not snapshot.empty:
        month_totals = snapshot.groupby(
            ["product_code", "product_name"], as_index=False
        )["sales_amount_jpy"].sum()

    detail_display_df: Optional[pd.DataFrame] = None
    detail_formatters: Dict[str, str] = {}
    detail_csv_data: Optional[bytes] = None
    pdf_table_df = pd.DataFrame()
    detail_available = False
    if not snapshot_year.empty or not month_totals.empty:
        detail_available = True
        if snapshot_year.empty:
            detail_df = month_totals.copy()
            detail_df["year_sum"] = np.nan
            detail_df["yoy"] = np.nan
            detail_df["delta"] = np.nan
        else:
            detail_df = snapshot_year[
                ["product_code", "product_name", "year_sum", "yoy", "delta"]
            ].copy()
            if not month_totals.empty:
                detail_df = detail_df.merge(
                    month_totals,
                    on=["product_code", "product_name"],
                    how="left",
                )
            else:
                detail_df["sales_amount_jpy"] = np.nan

        detail_df["sales_amount_jpy"] = detail_df["sales_amount_jpy"].fillna(0.0)
        total_month = float(detail_df["sales_amount_jpy"].sum())
        detail_df["share"] = (
            detail_df["sales_amount_jpy"] / total_month if total_month > 0 else 0.0
        )

        detail_display_df = pd.DataFrame(
            {
                "商品コード": detail_df["product_code"],
                "商品名": detail_df["product_name"],
                f"月次売上({unit})": detail_df["sales_amount_jpy"] / unit_scale,
                f"年計({unit})": detail_df["year_sum"] / unit_scale,
                "シェア(%)": detail_df["share"] * 100.0,
                "前年同月比(%)": detail_df["yoy"] * 100.0,
                f"前月差({unit})": detail_df["delta"] / unit_scale,
            }
        )
        detail_formatters = {
            f"月次売上({unit})": "{:,.0f}",
            f"年計({unit})": "{:,.0f}",
            "シェア(%)": "{:.1f}%",
            "前年同月比(%)": "{:.1f}%",
            f"前月差({unit})": "{:,.0f}",
        }
        detail_csv_data = detail_display_df.to_csv(index=False).encode("utf-8-sig")

        pdf_table_df = detail_df[
            ["product_code", "product_name", "year_sum", "sales_amount_jpy"]
        ].copy()
        pdf_table_df["year_sum"] = pdf_table_df["year_sum"].fillna(
            pdf_table_df["sales_amount_jpy"]
        )
        pdf_table_df = pdf_table_df[
            ["product_code", "product_name", "year_sum"]
        ]

    pdf_kpi = {
        "対象月": snapshot_month or "最新月",
        "月次売上": format_amount(monthly_value, unit),
        "前年同月比": yoy_label,
        "前月差": delta_label or "—",
        "トップ商品構成比": f"{top_share:.1f}%" if top_share is not None else "—",
    }

    st.markdown("##### トレンド")
    if monthly_trend.empty:
        render_status_message(
            "empty",
            key="sales_trend_empty",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="データ取込や期間設定を確認してください。",
        )
    else:
        trend_display = monthly_trend.copy()
        trend_display["売上"] = trend_display["sales_amount_jpy"] / unit_scale
        fig = px.line(trend_display, x="month", y="売上", markers=True)
        fig.update_yaxes(title=f"売上 ({unit})", tickformat=",.0f")
        fig.update_xaxes(title="月")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "light"))
        render_plotly_with_spinner(
            fig, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
        )

    st.markdown("##### 構成分析")
    comp_cols = st.columns(2)

    with comp_cols[0]:
        st.markdown("###### 商品別")
        if snapshot.empty:
            render_status_message(
                "empty",
                key="sales_product_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="期間や店舗の条件を変更して再表示してください。",
            )
        else:
            product_comp = (
                snapshot.groupby(["product_code", "product_name"], as_index=False)[
                    "sales_amount_jpy"
                ]
                .sum()
                .sort_values("sales_amount_jpy", ascending=False)
            )
            total_amount = float(product_comp["sales_amount_jpy"].sum())
            if total_amount <= 0:
                render_status_message(
                    "empty",
                    key="sales_product_zero",
                    on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                    guide="売上が発生している期間を選択してください。",
                )
            else:
                product_comp["シェア"] = (
                    product_comp["sales_amount_jpy"] / total_amount * 100.0
                )
                product_comp["表示額"] = (
                    product_comp["sales_amount_jpy"] / unit_scale
                )
                top_products = product_comp.head(10)
                fig_prod = px.bar(
                    top_products.sort_values("表示額"),
                    x="表示額",
                    y="product_name",
                    orientation="h",
                    text=top_products["シェア"].map(lambda v: f"{v:.1f}%"),
                )
                fig_prod.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=f"売上 ({unit})",
                    yaxis_title="",
                )
                fig_prod = apply_elegant_theme(
                    fig_prod, theme=st.session_state.get("ui_theme", "light")
                )
                render_plotly_with_spinner(
                    fig_prod, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
                )

    with comp_cols[1]:
        st.markdown("###### チャネル別")
        if not channel_column or channel_column not in snapshot.columns:
            st.info("チャネル情報が含まれていません。")
        else:
            channel_comp = (
                snapshot.groupby(channel_column, as_index=False)["sales_amount_jpy"].sum()
            )
            total_channel = float(channel_comp["sales_amount_jpy"].sum())
            if total_channel <= 0:
                render_status_message(
                    "empty",
                    key="sales_channel_zero",
                    on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                    guide="チャネル別データが含まれる期間を選択してください。",
                )
            else:
                channel_comp["シェア"] = (
                    channel_comp["sales_amount_jpy"] / total_channel * 100.0
                )
                channel_comp["表示額"] = (
                    channel_comp["sales_amount_jpy"] / unit_scale
                )
                fig_channel = px.pie(
                    channel_comp,
                    names=channel_column,
                    values="表示額",
                    hole=0.35,
                )
                fig_channel.update_traces(
                    textposition="inside",
                    texttemplate="%{label}<br>%{percent:.1%}",
                )
                fig_channel.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
                fig_channel = apply_elegant_theme(
                    fig_channel, theme=st.session_state.get("ui_theme", "light")
                )
                render_plotly_with_spinner(
                    fig_channel, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
                )

    st.markdown("##### 明細テーブル")
    csv_clicked = False
    pdf_clicked = False
    pdf_bytes: bytes = b""
    pdf_filename = f"sales_detail_{snapshot_month or 'latest'}.pdf"
    if detail_available:
        output_cols = st.columns(2)
        with output_cols[0]:
            csv_clicked = st.download_button(
                "CSVダウンロード",
                data=detail_csv_data or b"",
                file_name="sales_detail.csv",
                mime="text/csv",
                disabled=detail_csv_data is None,
                help="表を開かなくても最新の売上明細CSVを保存できます。",
                key="sales_detail_csv",
            )
        with output_cols[1]:
            pdf_enabled = not pdf_table_df.empty
            if pdf_enabled:
                pdf_bytes = download_pdf_overview(pdf_kpi, pdf_table_df, pdf_filename)
            pdf_clicked = st.download_button(
                "PDFダウンロード",
                data=pdf_bytes if pdf_enabled else b"",
                file_name=pdf_filename,
                mime="application/pdf",
                disabled=not pdf_enabled,
                help="KPIサマリー付きPDFをワンクリックで出力します。",
                key="sales_detail_pdf",
            )
        if csv_clicked:
            render_status_message(
                "completed",
                key="sales_csv_download",
                guide="粗利タブでも同様にCSV出力できます。",
            )
        if pdf_clicked:
            render_status_message(
                "completed",
                key="sales_pdf_download",
                guide="ダウンロードしたPDFを会議資料として共有してください。",
            )

    with st.expander("売上明細を表示", expanded=False):
        if not detail_available or detail_display_df is None:
            render_status_message(
                "empty",
                key="sales_detail_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="期間や店舗を切り替えてデータを再取得してください。",
            )
        else:
            st.dataframe(
                detail_display_df.style.format(detail_formatters),
                use_container_width=True,
            )


def _render_gross_profit_tab(
    *,
    filtered_monthly: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    unit: str,
    end_month: Optional[str],
    profile: Optional[Dict[str, object]],
    year_df: Optional[pd.DataFrame],
) -> None:
    unit_scale = UNIT_MAP.get(unit, 1)
    gross_ratio = 1.0 - float(profile.get("cogs_ratio", 0.6) or 0.0)
    st.markdown("##### 指標カード")
    metric_cols = st.columns(3)

    snapshot_month = end_month
    if not snapshot_month and not monthly_trend.empty:
        snapshot_month = monthly_trend["month"].iloc[-1]

    gross_trend = monthly_trend.copy()
    if not gross_trend.empty:
        gross_trend["gross_amount"] = gross_trend["sales_amount_jpy"] * gross_ratio
        gross_trend["gross_display"] = gross_trend["gross_amount"] / unit_scale
        gross_trend["gross_delta"] = gross_trend["gross_amount"].diff()
        gross_trend["margin_pct"] = np.where(
            gross_trend["sales_amount_jpy"] > 0,
            gross_trend["gross_amount"] / gross_trend["sales_amount_jpy"] * 100.0,
            np.nan,
        )

        latest = gross_trend.iloc[-1]
        prev = gross_trend.iloc[-2] if len(gross_trend) > 1 else None

        metric_cols[0].metric(
            "月次粗利",
            format_amount(float(latest.get("gross_amount", 0.0)), unit),
            delta=(
                format_amount(latest.get("gross_delta"), unit)
                if pd.notna(latest.get("gross_delta"))
                else None
            ),
        )

        margin_label = (
            f"{latest.get('margin_pct', 0.0):.1f}%"
            if pd.notna(latest.get("margin_pct"))
            else "—"
        )
        margin_delta = None
        if (
            prev is not None
            and pd.notna(prev.get("margin_pct"))
            and pd.notna(latest.get("margin_pct"))
        ):
            margin_delta = f"{latest.get('margin_pct', 0.0) - prev.get('margin_pct', 0.0):.1f}pt"
        metric_cols[1].metric("粗利率", margin_label, delta=margin_delta)
    else:
        for col in metric_cols[:2]:
            col.metric("—", "—")

    snapshot_year = pd.DataFrame()
    if (
        year_df is not None
        and not getattr(year_df, "empty", True)
        and snapshot_month
    ):
        snapshot_year = year_df[year_df["month"] == snapshot_month].dropna(
            subset=["year_sum"]
        )

    if not snapshot_year.empty:
        year_gross = snapshot_year["year_sum"].sum() * gross_ratio
        prev_month = _previous_month(_sorted_months(year_df), snapshot_month)
        prev_year = pd.DataFrame()
        if prev_month:
            prev_year = year_df[year_df["month"] == prev_month].dropna(
                subset=["year_sum"]
            )
        prev_gross = prev_year["year_sum"].sum() * gross_ratio if not prev_year.empty else None
        gross_delta = None
        if prev_gross is not None:
            gross_delta = format_amount(year_gross - prev_gross, unit)
        metric_cols[2].metric(
            "粗利年計",
            format_amount(year_gross, unit),
            delta=gross_delta,
        )
    else:
        metric_cols[2].metric("粗利年計", "—")

    st.markdown("##### 粗利額と粗利率の推移")
    if gross_trend.empty:
        render_status_message(
            "empty",
            key="gross_trend_empty",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="売上データを確認し、粗利計算に必要な期間を選択してください。",
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=gross_trend["month"],
                y=gross_trend["gross_display"],
                name="粗利額",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=gross_trend["month"],
                y=gross_trend["margin_pct"],
                name="粗利率",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title=f"粗利額 ({unit})", tickformat=",.0f"),
            yaxis2=dict(title="粗利率(%)", overlaying="y", side="right"),
            barmode="relative",
        )
        fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "light"))
        render_plotly_with_spinner(
            fig, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
        )

    snapshot = pd.DataFrame()
    if snapshot_month:
        snapshot = filtered_monthly[filtered_monthly["month"] == snapshot_month]
    if snapshot.empty and not filtered_monthly.empty:
        fallback = filtered_monthly["month"].iloc[-1]
        snapshot = filtered_monthly[filtered_monthly["month"] == fallback]

    st.markdown("##### 構成分析")
    comp_cols = st.columns(2)
    with comp_cols[0]:
        st.markdown("###### 商品別粗利")
        if snapshot.empty:
            render_status_message(
                "empty",
                key="gross_product_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="期間を変えると商品別粗利が表示されます。",
            )
        else:
            prod_gross = (
                snapshot.groupby(["product_code", "product_name"], as_index=False)[
                    "sales_amount_jpy"
                ]
                .sum()
                .sort_values("sales_amount_jpy", ascending=False)
            )
            prod_gross["gross_amount"] = prod_gross["sales_amount_jpy"] * gross_ratio
            total_gross = float(prod_gross["gross_amount"].sum())
            if total_gross <= 0:
                render_status_message(
                    "empty",
                    key="gross_product_zero",
                    on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                    guide="粗利が発生している期間を選択してください。",
                )
            else:
                prod_gross["表示額"] = prod_gross["gross_amount"] / unit_scale
                prod_gross["シェア"] = prod_gross["gross_amount"] / total_gross * 100.0
                fig_prod = px.bar(
                    prod_gross.head(10).sort_values("表示額"),
                    x="表示額",
                    y="product_name",
                    orientation="h",
                    text=prod_gross.head(10)["シェア"].map(lambda v: f"{v:.1f}%"),
                )
                fig_prod.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title=f"粗利額 ({unit})",
                    yaxis_title="",
                )
                fig_prod = apply_elegant_theme(
                    fig_prod, theme=st.session_state.get("ui_theme", "light")
                )
                render_plotly_with_spinner(
                    fig_prod, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
                )

    with comp_cols[1]:
        st.markdown("###### 粗利率の推移 (トップ商品)")
        if snapshot.empty or snapshot_month is None:
            render_status_message(
                "empty",
                key="gross_margin_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="対象月を変更すると粗利率の推移を確認できます。",
            )
        else:
            top_codes = (
                snapshot.groupby("product_code")["sales_amount_jpy"].sum()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
            if not top_codes:
                render_status_message(
                    "empty",
                    key="gross_margin_top_empty",
                    on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                    guide="粗利が大きい商品が存在する期間を選択してください。",
                )
            else:
                history = filtered_monthly[
                    filtered_monthly["product_code"].isin(top_codes)
                ].copy()
                history["gross_margin"] = np.where(
                    history["sales_amount_jpy"] > 0,
                    gross_ratio * 100.0,
                    np.nan,
                )
                if history.empty:
                    render_status_message(
                        "empty",
                        key="gross_margin_history_empty",
                        on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                        guide="対象期間内に粗利が発生していることを確認してください。",
                    )
                else:
                    fig_margin = px.line(
                        history,
                        x="month",
                        y="gross_margin",
                        color="product_name",
                        markers=True,
                    )
                    fig_margin.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=30, b=10),
                        yaxis_title="粗利率(%)",
                        xaxis_title="月",
                    )
                    fig_margin = apply_elegant_theme(
                        fig_margin, theme=st.session_state.get("ui_theme", "light")
                    )
                    render_plotly_with_spinner(
                        fig_margin,
                        config=PLOTLY_CONFIG,
                        spinner_text=SPINNER_MESSAGE,
                    )

    st.markdown("##### 明細テーブル")
    with st.expander("粗利明細を表示", expanded=False):
        if snapshot_year.empty:
            render_status_message(
                "empty",
                key="gross_detail_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="期間やテンプレートの設定を見直してください。",
            )
        else:
            detail_df = snapshot_year[[
                "product_code",
                "product_name",
                "year_sum",
                "yoy",
                "delta",
            ]].copy()
            detail_df["gross_year"] = detail_df["year_sum"] * gross_ratio
            detail_df["gross_margin"] = np.where(
                detail_df["year_sum"] > 0,
                detail_df["gross_year"] / detail_df["year_sum"] * 100.0,
                np.nan,
            )
            month_totals = (
                snapshot.groupby(["product_code", "product_name"], as_index=False)[
                    "sales_amount_jpy"
                ]
                .sum()
                if not snapshot.empty
                else pd.DataFrame()
            )
            if not month_totals.empty:
                month_totals["monthly_gross"] = (
                    month_totals["sales_amount_jpy"] * gross_ratio
                )
                detail_df = detail_df.merge(
                    month_totals[[
                        "product_code",
                        "product_name",
                        "monthly_gross",
                    ]],
                    on=["product_code", "product_name"],
                    how="left",
                )
            else:
                detail_df["monthly_gross"] = np.nan

            display_df = pd.DataFrame(
                {
                    "商品コード": detail_df["product_code"],
                    "商品名": detail_df["product_name"],
                    f"月次粗利({unit})": detail_df["monthly_gross"] / unit_scale,
                    f"年計粗利({unit})": detail_df["gross_year"] / unit_scale,
                    "粗利率(%)": detail_df["gross_margin"],
                    "前年同月比(%)": detail_df["yoy"] * 100.0,
                    f"前月差({unit})": detail_df["delta"] * gross_ratio / unit_scale,
                }
            )

            st.dataframe(
                display_df.style.format(
                    {
                        f"月次粗利({unit})": "{:,.0f}",
                        f"年計粗利({unit})": "{:,.0f}",
                        "粗利率(%)": "{:.1f}%",
                        "前年同月比(%)": "{:.1f}%",
                        f"前月差({unit})": "{:,.0f}",
                    }
                ),
                use_container_width=True,
            )

            csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
            clicked = st.download_button(
                "CSVダウンロード",
                data=csv_data,
                file_name="gross_profit_detail.csv",
                mime="text/csv",
                key="gross_detail_csv",
            )
            if clicked:
                render_status_message(
                    "completed",
                    key="gross_detail_download",
                    guide="在庫タブでも同様に明細をダウンロードできます。",
                )


def _render_inventory_tab(
    *,
    year_df: Optional[pd.DataFrame],
    financial_snapshot: Dict[str, object],
    unit: str,
    end_month: Optional[str],
    profile: Optional[Dict[str, object]],
) -> None:
    unit_scale = UNIT_MAP.get(unit, 1)
    year_totals = _monthly_year_totals(year_df)
    if year_totals.empty:
        render_status_message(
            "empty",
            key="inventory_missing",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="データ取込ページで在庫関連のカラムを設定してください。",
        )
        return

    cogs_ratio = float(profile.get("cogs_ratio", 0.6) or 0.0)
    asset_turnover = float(profile.get("asset_turnover", 2.5) or 0.0)
    inventory_ratio = _find_ratio(
        profile.get("balance_assets", []),
        ["棚卸", "inventory", "在庫"],
    )

    if asset_turnover <= 0:
        asset_turnover = 1.0

    year_totals["inventory"] = (
        year_totals["year_sum"] / asset_turnover * inventory_ratio
    )
    delta_ratio = (
        year_totals["delta"]
        / year_totals["year_sum"].replace(0, np.nan)
    ).fillna(0.0)
    adjust = (1 + delta_ratio * 0.5).clip(0.7, 1.3)
    year_totals["inventory"] = year_totals["inventory"] * adjust
    year_totals["cogs"] = year_totals["year_sum"] * cogs_ratio
    year_totals["turnover"] = np.where(
        year_totals["inventory"] > 0,
        year_totals["cogs"] / year_totals["inventory"],
        np.nan,
    )

    snapshot_month = end_month or year_totals["month"].iloc[-1]
    snapshot_row = year_totals[year_totals["month"] == snapshot_month]
    prev_month = _previous_month(year_totals["month"].tolist(), snapshot_month)
    prev_row = year_totals[year_totals["month"] == prev_month]

    inv_value = float(financial_snapshot.get("inventory_balance") or 0.0)
    if inv_value == 0.0 and not snapshot_row.empty:
        inv_value = float(snapshot_row["inventory"].iloc[0])

    turnover_value = (
        float(snapshot_row["turnover"].iloc[0])
        if not snapshot_row.empty
        else None
    )
    turnover_delta = None
    if not prev_row.empty and turnover_value is not None:
        turnover_delta = turnover_value - float(prev_row["turnover"].iloc[0])

    st.markdown("##### 指標カード")
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "推定在庫残高",
        format_amount(inv_value, unit),
        delta=(
            format_amount(
                inv_value - float(prev_row["inventory"].iloc[0]), unit
            )
            if not prev_row.empty
            else None
        ),
    )
    metric_cols[1].metric(
        "在庫回転率",
        f"{turnover_value:.2f} 回" if turnover_value is not None else "—",
        delta=(
            f"{turnover_delta:.2f}pt" if turnover_delta is not None else None
        ),
    )

    st.markdown("##### 在庫・回転率の推移")
    inv_display = year_totals["inventory"] / unit_scale
    turnover_series = year_totals["turnover"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=year_totals["month"], y=inv_display, name="推定在庫残高")
    )
    fig.add_trace(
        go.Scatter(
            x=year_totals["month"],
            y=turnover_series,
            name="在庫回転率",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title=f"在庫残高 ({unit})", tickformat=",.0f"),
        yaxis2=dict(title="回転率(回)", overlaying="y", side="right"),
    )
    fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "light"))
    render_plotly_with_spinner(
        fig, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
    )

    st.markdown("##### アラート")
    alerts_year = pd.DataFrame()
    if (
        year_df is not None
        and not getattr(year_df, "empty", True)
        and snapshot_month
    ):
        alerts_year = year_df[year_df["month"] == snapshot_month].dropna(
            subset=["year_sum"]
        )

    stockout_alerts = pd.DataFrame()
    excess_alerts = pd.DataFrame()
    if not alerts_year.empty and inv_value > 0:
        total_year_sum = alerts_year["year_sum"].sum()
        alerts_year = alerts_year.copy()
        alerts_year["inventory_value"] = np.where(
            total_year_sum > 0,
            alerts_year["year_sum"] / total_year_sum * inv_value,
            0.0,
        )
        stockout_alerts = alerts_year[
            (alerts_year["yoy"] > 0.15)
            & (alerts_year["inventory_value"] < inv_value * 0.02)
        ]
        excess_alerts = alerts_year[
            (alerts_year["yoy"] < -0.1)
            & (alerts_year["inventory_value"] > inv_value * 0.05)
        ]

    alert_cols = st.columns(3)
    alert_cols[0].metric("品切れリスク", f"{len(stockout_alerts)} 件")
    alert_cols[1].metric("過剰在庫リスク", f"{len(excess_alerts)} 件")
    alert_cols[2].metric("評価対象SKU", f"{len(alerts_year)} 件")

    if alerts_year.empty or inv_value <= 0:
        render_status_message(
            "empty",
            key="inventory_alert_empty",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="在庫金額と売上指標を含むデータを読み込んでください。",
        )
    else:
        if not stockout_alerts.empty:
            st.warning(
                f"品切れリスク: {len(stockout_alerts)} 件 — 売上が伸びる一方で在庫が少ない商品があります。"
            )
        if not excess_alerts.empty:
            st.error(
                f"過剰在庫リスク: {len(excess_alerts)} 件 — 売上が減速しているのに在庫が積み上がっている商品があります。"
            )
        if stockout_alerts.empty and excess_alerts.empty:
            st.success("在庫バランスは良好です。")

    category_column = None
    for candidate in ("category", "カテゴリ", "category_name", "カテゴリー"):
        if candidate in alerts_year.columns:
            category_column = candidate
            break
    if category_column and not alerts_year.empty:
        st.markdown("##### カテゴリー別在庫")
        category_df = alerts_year.copy()
        if "inventory_value" not in category_df.columns:
            category_df["inventory_value"] = category_df["year_sum"]
        category_df = (
            category_df.groupby(category_column, as_index=False)["inventory_value"].sum()
        )
        category_df = category_df.sort_values("inventory_value", ascending=False)
        category_df["表示額"] = category_df["inventory_value"] / unit_scale
        fig_category = px.bar(
            category_df.head(12).sort_values("表示額"),
            x="表示額",
            y=category_column,
            orientation="h",
            text=category_df.head(12)["表示額"].map(lambda v: f"{v:,.0f}"),
        )
        fig_category.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=f"在庫金額 ({unit})",
            yaxis_title="カテゴリー",
        )
        fig_category = apply_elegant_theme(
            fig_category, theme=st.session_state.get("ui_theme", "light")
        )
        render_plotly_with_spinner(
            fig_category, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
        )

    st.markdown("##### 在庫構成と明細")
    comp_cols = st.columns(2)
    with comp_cols[0]:
        st.markdown("###### 商品別在庫構成")
        if alerts_year.empty or inv_value <= 0:
            render_status_message(
                "empty",
                key="inventory_product_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="在庫金額が集計できるようデータを再確認してください。",
            )
        else:
            product_inv = alerts_year[[
                "product_code",
                "product_name",
                "inventory_value",
            ]].copy()
            product_inv = product_inv.sort_values(
                "inventory_value", ascending=False
            )
            product_inv["表示額"] = product_inv["inventory_value"] / unit_scale
            product_inv["シェア"] = product_inv["inventory_value"] / inv_value * 100.0
            fig_inv = px.bar(
                product_inv.head(10).sort_values("表示額"),
                x="表示額",
                y="product_name",
                orientation="h",
                text=product_inv.head(10)["シェア"].map(lambda v: f"{v:.1f}%"),
            )
            fig_inv.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title=f"在庫金額 ({unit})",
                yaxis_title="",
            )
            fig_inv = apply_elegant_theme(
                fig_inv, theme=st.session_state.get("ui_theme", "light")
            )
            render_plotly_with_spinner(
                fig_inv, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
            )

    with comp_cols[1]:
        st.markdown("###### 在庫回転率 (商品別)")
        if alerts_year.empty:
            render_status_message(
                "empty",
                key="inventory_turnover_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="在庫指標が計算できる期間を選択してください。",
            )
        else:
            product_turnover = alerts_year[[
                "product_code",
                "product_name",
                "yoy",
                "delta",
            ]].copy()
            product_turnover = product_turnover.sort_values("yoy", ascending=False)
            fig_turnover = px.bar(
                product_turnover.head(10),
                x="product_name",
                y="yoy",
                labels={"yoy": "YoY", "product_name": "商品"},
            )
            fig_turnover.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_tickformat="+.0%",
            )
            fig_turnover = apply_elegant_theme(
                fig_turnover, theme=st.session_state.get("ui_theme", "light")
            )
            render_plotly_with_spinner(
                fig_turnover, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
            )

    with st.expander("在庫明細を表示", expanded=False):
        if alerts_year.empty:
            render_status_message(
                "empty",
                key="inventory_detail_empty",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="在庫データを再読み込みしてからご確認ください。",
            )
        else:
            detail_df = alerts_year[[
                "product_code",
                "product_name",
                "inventory_value",
                "yoy",
                "delta",
            ]].copy()
            detail_df["シェア"] = detail_df["inventory_value"] / inv_value * 100.0
            display_df = pd.DataFrame(
                {
                    "商品コード": detail_df["product_code"],
                    "商品名": detail_df["product_name"],
                    f"在庫金額({unit})": detail_df["inventory_value"] / unit_scale,
                    "シェア(%)": detail_df["シェア"],
                    "前年同月比(%)": detail_df["yoy"] * 100.0,
                    f"前月差({unit})": detail_df["delta"] / unit_scale,
                }
            )
            st.dataframe(
                display_df.style.format(
                    {
                        f"在庫金額({unit})": "{:,.0f}",
                        "シェア(%)": "{:.1f}%",
                        "前年同月比(%)": "{:.1f}%",
                        f"前月差({unit})": "{:,.0f}",
                    }
                ),
                use_container_width=True,
            )
            csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
            clicked = st.download_button(
                "CSVダウンロード",
                data=csv_data,
                file_name="inventory_detail.csv",
                mime="text/csv",
                key="inventory_detail_csv",
            )
            if clicked:
                render_status_message(
                    "completed",
                    key="inventory_detail_download",
                    guide="ダウンロードした明細を補充計画に活用してください。",
                )


def _render_funds_tab(
    *,
    year_df: Optional[pd.DataFrame],
    financial_snapshot: Dict[str, object],
    unit: str,
    end_month: Optional[str],
    profile: Optional[Dict[str, object]],
) -> None:
    unit_scale = UNIT_MAP.get(unit, 1)
    cash_items = profile.get("cash_flow", []) or []
    year_totals = _monthly_year_totals(year_df)
    if year_totals.empty:
        render_status_message(
            "empty",
            key="funds_missing",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="キャッシュフローの前提が設定されているか確認してください。",
        )
        return

    cash_long: List[Dict[str, object]] = []
    for item in cash_items:
        label = item.get("item", "キャッシュフロー")
        try:
            ratio = float(item.get("ratio", 0.0))
        except (TypeError, ValueError):
            ratio = 0.0
        amounts = year_totals["year_sum"] * ratio
        for month, amount in zip(year_totals["month"], amounts):
            cash_long.append({"month": month, "category": label, "amount": amount})

    cash_df = pd.DataFrame(cash_long)
    net_series = pd.DataFrame()
    if not cash_df.empty:
        net_series = cash_df.groupby("month", as_index=False)["amount"].sum()

    flows = financial_snapshot.get("cash_flows", []) or []

    def _flow_amount(keyword: str) -> float:
        for entry in flows:
            if keyword in str(entry.get("item", "")):
                try:
                    return float(entry.get("amount", 0.0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    operating_cf = _flow_amount("営業")
    investing_cf = _flow_amount("投資")
    financing_cf = _flow_amount("財務")

    st.markdown("##### 指標カード")
    metric_cols = st.columns(3)
    metric_cols[0].metric("営業キャッシュフロー", format_amount(operating_cf, unit))
    metric_cols[1].metric("投資キャッシュフロー", format_amount(investing_cf, unit))
    metric_cols[2].metric("財務キャッシュフロー", format_amount(financing_cf, unit))

    st.markdown("##### キャッシュフロー推移")
    if cash_df.empty:
        render_status_message(
            "empty",
            key="funds_flow_data_empty",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="キャッシュフロー設定に必要な比率と金額を確認してください。",
        )
    else:
        display_df = cash_df.copy()
        display_df["表示額"] = display_df["amount"] / unit_scale
        fig = px.bar(
            display_df,
            x="month",
            y="表示額",
            color="category",
            barmode="relative",
        )
        if not net_series.empty:
            net_series["表示額"] = net_series["amount"] / unit_scale
            fig.add_trace(
                go.Scatter(
                    x=net_series["month"],
                    y=net_series["表示額"],
                    name="純キャッシュフロー",
                    mode="lines+markers",
                )
            )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title=f"金額 ({unit})",
        )
        fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "light"))
        render_plotly_with_spinner(
            fig, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
        )

    st.markdown("##### 入出金構成")
    latest_month = end_month or (net_series["month"].iloc[-1] if not net_series.empty else None)
    latest_flows = pd.DataFrame()
    if latest_month and not cash_df.empty:
        latest_flows = cash_df[cash_df["month"] == latest_month]
    if latest_flows.empty:
        render_status_message(
            "empty",
            key="funds_latest_flow_empty",
            on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
            guide="キャッシュフロー項目と金額を入力して再計算してください。",
        )
    else:
        latest_flows = latest_flows.copy()
        latest_flows["表示額"] = latest_flows["amount"] / unit_scale
        fig_latest = px.bar(
            latest_flows,
            x="category",
            y="表示額",
            text=latest_flows["表示額"].map(lambda v: f"{v:,.0f}"),
        )
        fig_latest.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="キャッシュフロー項目",
            yaxis_title=f"金額 ({unit})",
        )
        fig_latest = apply_elegant_theme(
            fig_latest, theme=st.session_state.get("ui_theme", "light")
        )
        render_plotly_with_spinner(
            fig_latest, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
        )

    with st.expander("資金繰り計算書を表示", expanded=False):
        if not flows:
            render_status_message(
                "empty",
                key="funds_flow_missing",
                on_modify=lambda: set_active_page("settings", rerun_on_lock=True),
                guide="設定ページでキャッシュフロー比率を入力してください。",
            )
        else:
            table_df = pd.DataFrame(flows)
            table_df["金額({unit})"] = table_df["amount"].astype(float) / unit_scale
            table_df["構成比(%)"] = table_df["ratio"].astype(float) * 100.0
            display_df = table_df[["item", f"金額({unit})", "構成比(%)"]].rename(
                columns={"item": "項目"}
            )
            st.dataframe(
                display_df.style.format(
                    {
                        f"金額({unit})": "{:,.0f}",
                        "構成比(%)": "{:.1f}%",
                    }
                ),
                use_container_width=True,
            )
            csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
            clicked = st.download_button(
                "CSVダウンロード",
                data=csv_data,
                file_name="cash_flow_statement.csv",
                mime="text/csv",
                key="funds_statement_csv",
            )
            if clicked:
                render_status_message(
                    "completed",
                    key="funds_statement_download",
                    guide="キャッシュフロー表を財務チームと共有しましょう。",
                )


def nice_slider_step(max_value: int, target_steps: int = 40) -> int:
    """Return an intuitive step size so sliders move in round increments."""
    if max_value <= 0:
        return 1

    raw_step = max_value / target_steps
    if raw_step <= 1:
        return 1

    exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    base = raw_step / (10 ** exponent) if raw_step > 0 else 1

    for nice in (1, 2, 5, 10):
        if base <= nice:
            step = nice * (10 ** exponent)
            return int(step) if step >= 1 else 1

    return int(10 ** (exponent + 1))


def choose_amount_slider_unit(max_amount: int) -> tuple[int, str]:
    """Choose a unit so the slider operates in easy-to-understand scales."""
    units = [
        (1, "円"),
        (1_000, "千円"),
        (10_000, "万円"),
        (1_000_000, "百万円"),
        (100_000_000, "億円"),
    ]

    if max_amount <= 0:
        return units[0]

    for scale, label in units:
        if max_amount / scale <= 300:
            return scale, label

    return units[-1]


def int_input(label: str, value: int) -> int:
    """Text input for integer values displayed with thousands separators."""
    text = st.text_input(label, format_int(value))
    try:
        return int(text.replace(",", ""))
    except ValueError:
        return value


def render_sidebar_summary() -> Optional[str]:
    template_label = get_template_config().get("label", get_active_template_key())
    st.sidebar.caption(f"テンプレート: {template_label}（推奨指標を自動反映）")

    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        st.sidebar.caption("データを取り込むと最新サマリーが表示されます。")
        return None

    months = month_options(year_df)
    if not months:
        st.sidebar.caption("月次データが存在しません。")
        return None

    end_m = months[-1]
    unit = st.session_state.settings.get("currency_unit", "円")
    kpi = aggregate_overview(year_df, end_m)
    hhi_val = compute_hhi(year_df, end_m)
    sku_cnt = int(year_df["product_code"].nunique())
    rec_cnt = int(len(year_df))

    total_txt = format_amount(kpi.get("total_year_sum"), unit)
    yoy_val = kpi.get("yoy")
    yoy_txt = f"{yoy_val * 100:.1f}%" if yoy_val is not None else "—"
    delta_txt = format_amount(kpi.get("delta"), unit)
    hhi_txt = f"{hhi_val:.3f}" if hhi_val is not None else "—"

    st.sidebar.markdown(
        f"""
        <div class=\"mck-sidebar-summary\">
            <strong>最新月:</strong> {end_m}<br>
            <strong>年計総額:</strong> {total_txt}<br>
            <strong>YoY:</strong> {yoy_txt}<br>
            <strong>Δ:</strong> {delta_txt}<br>
            <strong>HHI:</strong> {hhi_txt}<br>
            <strong>SKU数:</strong> {sku_cnt:,}<br>
            <strong>レコード:</strong> {rec_cnt:,}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return end_m


def build_copilot_context(
    focus: str, end_month: Optional[str] = None, top_n: int = 5
) -> str:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        return "データが取り込まれていません。"

    months = month_options(year_df)
    if not months:
        return "月度情報が存在しません。"

    end_m = end_month or months[-1]
    snap = (
        year_df[year_df["month"] == end_m]
        .dropna(subset=["year_sum"])
        .copy()
    )
    if snap.empty:
        return f"{end_m}の年計スナップショットが空です。"

    kpi = aggregate_overview(year_df, end_m)
    hhi_val = compute_hhi(year_df, end_m)

    def fmt_amt(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "—"
        return f"{format_int(val)}円"

    def fmt_pct(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "—"
        return f"{val * 100:.1f}%"

    lines = [
        f"対象月: {end_m}",
        f"年計総額: {fmt_amt(kpi.get('total_year_sum'))}",
        f"年計YoY: {fmt_pct(kpi.get('yoy'))}",
        f"前月差Δ: {fmt_amt(kpi.get('delta'))}",
        f"SKU数: {snap['product_code'].nunique():,}",
    ]
    if hhi_val is not None:
        lines.append(f"HHI: {hhi_val:.3f}")

    if focus == "伸びているSKU":
        subset = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(top_n)
        )
        label = "伸長SKU"
    elif focus == "苦戦しているSKU":
        subset = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=True)
            .head(top_n)
        )
        label = "苦戦SKU"
    else:
        subset = snap.sort_values("year_sum", ascending=False).head(top_n)
        label = "主要SKU"

    if not subset.empty:
        bullets = []
        for _, row in subset.iterrows():
            name = row.get("product_name") or row.get("product_code")
            yoy_txt = fmt_pct(row.get("yoy"))
            delta_txt = fmt_amt(row.get("delta"))
            bullets.append(
                f"{name} (年計 {fmt_amt(row.get('year_sum'))}, YoY {yoy_txt}, Δ {delta_txt})"
            )
        lines.append(f"{label}: " + " / ".join(bullets))

    worst = (
        snap.dropna(subset=["yoy"])
        .sort_values("yoy", ascending=True)
        .head(1)
    )
    best = (
        snap.dropna(subset=["yoy"])
        .sort_values("yoy", ascending=False)
        .head(1)
    )
    if not best.empty:
        b = best.iloc[0]
        lines.append(
            f"YoY最高: {(b['product_name'] or b['product_code'])} ({fmt_pct(b['yoy'])})"
        )
    if not worst.empty:
        w = worst.iloc[0]
        lines.append(
            f"YoY最低: {(w['product_name'] or w['product_code'])} ({fmt_pct(w['yoy'])})"
        )

    return " ｜ ".join(lines)


def marker_step(dates, target_points=24):
    n = len(pd.unique(dates))
    return max(1, round(n / target_points))


NAME_MAP = {
    "year_sum": "年計（12ヶ月累計）",
    "yoy": "YoY（前年同月比）",
    "delta": "Δ（前月差）",
    "slope6m": "直近6ヶ月の傾き",
    "std6m": "直近6ヶ月の変動",
    "slope_beta": "直近Nの傾き",
    "hhi_share": "HHI寄与度",
}



# ---------------- Sidebar ----------------
st.sidebar.markdown(
    f"""
    <div class="sidebar-app-brand">
        <div class="sidebar-app-brand__title">{APP_TITLE}</div>
        <p class="sidebar-app-brand__caption">メニューは色分けされ、各機能の役割がひと目で分かります。</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title(t("sidebar.navigation_title"))

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .sidebar-app-brand{
      background:linear-gradient(135deg, rgba(255,255,255,0.24), rgba(255,255,255,0.06));
      border-radius:18px;
      padding:1rem 1.1rem;
      border:1px solid rgba(255,255,255,0.2);
      box-shadow:0 14px 32px rgba(7,32,54,0.32);
      margin-bottom:1.1rem;
    }
    [data-testid="stSidebar"] .sidebar-app-brand__title{
      font-size:1.18rem;
      font-weight:800;
      letter-spacing:.08em;
      margin:0 0 .35rem;
      color:#ffffff;
    }
    [data-testid="stSidebar"] .sidebar-app-brand__caption{
      font-size:0.9rem;
      line-height:1.55;
      color:rgba(255,255,255,0.86);
      margin:0;
    }
    [data-testid="stSidebar"] .sidebar-legend{
      background:rgba(255,255,255,0.08);
      border-radius:14px;
      border:1px solid rgba(255,255,255,0.2);
      padding:0.75rem 0.85rem;
      margin:0 0 0.9rem;
      box-shadow:0 8px 18px rgba(7,32,54,0.28);
    }
    [data-testid="stSidebar"] .sidebar-legend__title{
      font-size:0.78rem;
      letter-spacing:.12em;
      text-transform:uppercase;
      margin:0 0 0.55rem;
      color:rgba(255,255,255,0.72);
      font-weight:700;
    }
    [data-testid="stSidebar"] .sidebar-legend__items{
      display:flex;
      flex-wrap:wrap;
      gap:0.4rem;
    }
    [data-testid="stSidebar"] .sidebar-legend__item{
      display:inline-flex;
      align-items:center;
      gap:0.35rem;
      padding:0.25rem 0.6rem;
      border-radius:999px;
      background:rgba(255,255,255,0.12);
      color:#ffffff;
      font-size:0.82rem;
      font-weight:600;
      box-shadow:0 4px 10px rgba(7,32,54,0.22);
    }
    [data-testid="stSidebar"] .sidebar-legend__item::before{
      content:"";
      width:0.55rem;
      height:0.55rem;
      border-radius:50%;
      background:var(--legend-color,#71b7d4);
      box-shadow:0 0 0 3px rgba(255,255,255,0.15);
    }
    [data-testid="stSidebar"] .sidebar-legend__hint{
      margin:0.6rem 0 0;
      font-size:0.78rem;
      color:rgba(255,255,255,0.7);
    }
    [data-testid="stSidebar"] label.nav-pill{
      display:flex;
      align-items:flex-start;
      gap:0.75rem;
      padding:0.85rem 0.95rem;
      border-radius:16px;
      border:1px solid rgba(255,255,255,0.16);
      background:rgba(255,255,255,0.06);
      margin-bottom:0.5rem;
      box-shadow:0 14px 26px rgba(7,32,54,0.28);
      position:relative;
      transition:transform .12s ease, border-color .12s ease, background-color .12s ease, box-shadow .12s ease;
    }
    [data-testid="stSidebar"] label.nav-pill:hover{
      transform:translateY(-2px);
      border-color:rgba(255,255,255,0.4);
      background:rgba(255,255,255,0.12);
      box-shadow:0 18px 32px rgba(7,32,54,0.34);
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__icon{
      width:2rem;
      height:2rem;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:1rem;
      background:rgba(var(--nav-accent-rgb,71,183,212),0.18);
      border:2px solid rgba(var(--nav-accent-rgb,71,183,212),0.45);
      box-shadow:0 10px 20px rgba(var(--nav-accent-rgb,71,183,212),0.35);
      color:#ffffff;
      flex-shrink:0;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__icon svg{
      width:18px;
      height:18px;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__body{
      display:flex;
      flex-direction:column;
      gap:0.2rem;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__badge{
      display:inline-flex;
      align-items:center;
      justify-content:flex-start;
      gap:0.3rem;
      font-size:0.75rem;
      font-weight:700;
      padding:0.18rem 0.55rem;
      border-radius:999px;
      background:rgba(var(--nav-accent-rgb,71,183,212),0.28);
      color:#ffffff;
      width:max-content;
      letter-spacing:.06em;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__badge:empty{
      display:none;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__title{
      font-size:1rem;
      font-weight:700;
      color:#f8fbff !important;
      line-height:1.2;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__desc{
      font-size:0.85rem;
      line-height:1.35;
      color:rgba(255,255,255,0.82) !important;
    }
    [data-testid="stSidebar"] label.nav-pill .nav-pill__desc:empty{
      display:none;
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active{
      border-color:rgba(var(--nav-accent-rgb,71,183,212),0.65);
      background:rgba(var(--nav-accent-rgb,71,183,212),0.25);
      box-shadow:0 20px 36px rgba(var(--nav-accent-rgb,71,183,212),0.48);
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__icon{
      background:rgba(var(--nav-accent-rgb,71,183,212),0.35);
      border-color:rgba(var(--nav-accent-rgb,71,183,212),0.85);
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__badge{
      background:rgba(var(--nav-accent-rgb,71,183,212),0.55);
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__title{
      color:#ffffff !important;
    }
    [data-testid="stSidebar"] label.nav-pill.nav-pill--active .nav-pill__desc{
      color:rgba(255,255,255,0.92) !important;
    }
    .has-tooltip{
      position:relative;
    }
    .has-tooltip::after,
    .has-tooltip::before{
      pointer-events:none;
      opacity:0;
      transition:opacity .15s ease, transform .15s ease;
    }
    .has-tooltip[data-tooltip=""]::after,
    .has-tooltip[data-tooltip=""]::before{
      display:none;
    }
    .has-tooltip::after{
      content:attr(data-tooltip);
      position:absolute;
      left:50%;
      bottom:calc(100% + 8px);
      transform:translate(-50%, 0);
      background:rgba(11,23,38,0.92);
      color:#ffffff;
      padding:0.45rem 0.7rem;
      border-radius:10px;
      font-size:0.78rem;
      line-height:1.4;
      max-width:260px;
      text-align:center;
      box-shadow:0 12px 28px rgba(7,32,54,0.38);
      white-space:pre-wrap;
      z-index:60;
    }
    .has-tooltip::before{
      content:"";
      position:absolute;
      left:50%;
      bottom:calc(100% + 2px);
      transform:translate(-50%, 0);
      border:6px solid transparent;
      border-top-color:rgba(11,23,38,0.92);
      z-index:60;
    }
    .has-tooltip:hover::after,
    .has-tooltip:hover::before,
    .has-tooltip:focus-visible::after,
    .has-tooltip:focus-visible::before{
      opacity:1;
      transform:translate(-50%, -4px);
    }
    .tour-step-guide{
      display:flex;
      flex-wrap:wrap;
      gap:0.9rem;
      margin:0 0 1.2rem;
    }
    .tour-step-guide__item{
      display:flex;
      flex-direction:column;
      align-items:center;
      gap:0.45rem;
      padding:0.75rem 0.9rem;
      border-radius:14px;
      border:1px solid var(--border);
      background:var(--panel);
      box-shadow:0 12px 26px rgba(11,44,74,0.14);
      min-width:120px;
      position:relative;
      transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease;
      cursor:pointer;
    }
    .tour-step-guide__item:hover{
      transform:translateY(-3px);
      box-shadow:0 18px 32px rgba(11,44,74,0.18);
    }
    .tour-step-guide__item[data-active="true"]{
      border-color:rgba(15,76,129,0.55);
      box-shadow:0 20px 40px rgba(15,76,129,0.22);
    }
    .tour-step-guide__item:focus-visible{
      outline:3px solid rgba(15,76,129,0.35);
      outline-offset:3px;
    }
    .tour-step-guide__icon{
      width:34px;
      height:34px;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:1.05rem;
      background:rgba(15,76,129,0.1);
      color:var(--accent-strong);
      box-shadow:0 10px 20px rgba(15,76,129,0.12);
    }
    .tour-step-guide__icon svg{
      width:18px;
      height:18px;
    }
    .tour-step-guide__label{
      font-size:0.95rem;
      font-weight:700;
      color:var(--accent-strong);
      text-align:center;
      line-height:1.3;
    }
    .mck-import-stepper{
      display:flex;
      flex-direction:column;
      gap:0.75rem;
      margin:0 0 1.5rem;
    }
    .mck-import-stepper__item{
      display:flex;
      gap:1rem;
      align-items:flex-start;
      padding:0.85rem 1rem;
      border-radius:14px;
      border:1px solid var(--border);
      background:var(--panel);
      box-shadow:0 12px 26px rgba(11,44,74,0.12);
      position:relative;
      transition:border-color .16s ease, box-shadow .16s ease, transform .16s ease;
    }
    .mck-import-stepper__item[data-status="current"]{
      border-color:rgba(79,154,184,0.6);
      box-shadow:0 18px 36px rgba(79,154,184,0.22);
      transform:translateY(-2px);
    }
    .mck-import-stepper__item[data-status="complete"]{
      border-color:rgba(18,58,95,0.4);
    }
    .mck-import-stepper__status{
      width:36px;
      height:36px;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:1rem;
      font-weight:700;
      color:#ffffff;
      background:#dbe8f5;
      box-shadow:0 10px 20px rgba(15,60,105,0.12);
      flex-shrink:0;
    }
    .mck-import-stepper__status svg{
      width:18px;
      height:18px;
    }
    .mck-import-stepper__status[data-status="complete"]{
      background:#2d6f8e;
    }
    .mck-import-stepper__status[data-status="current"]{
      background:#4f9ab8;
    }
    .mck-import-stepper__status[data-status="upcoming"]{
      background:#dbe8f5;
      color:#123a5f;
    }
    .mck-import-stepper__content{
      flex:1;
      display:flex;
      flex-direction:column;
      gap:0.35rem;
    }
    .mck-import-stepper__header{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:0.75rem;
    }
    .mck-import-stepper__title{
      font-weight:700;
      color:var(--accent-strong);
      font-size:1.05rem;
      line-height:1.25;
    }
    .mck-import-stepper__title .en{
      display:block;
      font-weight:600;
      font-size:0.82rem;
      color:var(--muted);
      letter-spacing:0.04em;
    }
    .mck-import-stepper__state{
      font-size:0.78rem;
      font-weight:700;
      text-transform:uppercase;
      letter-spacing:0.08em;
      padding:0.3rem 0.6rem;
      border-radius:999px;
      border:1px solid transparent;
      color:#ffffff;
      background:#2d6f8e;
      align-self:flex-start;
    }
    .mck-import-stepper__state[data-status="current"]{
      background:#4f9ab8;
    }
    .mck-import-stepper__state[data-status="upcoming"]{
      background:#dbe8f5;
      color:#123a5f;
      border-color:rgba(18,58,95,0.18);
    }
    .mck-import-stepper__desc{
      color:var(--muted);
      font-size:0.9rem;
      line-height:1.55;
    }
    .mck-import-stepper__desc .en{
      display:block;
      font-size:0.82rem;
      color:var(--muted);
    }
    .mck-flow-stepper{
      display:flex;
      align-items:center;
      gap:0.75rem;
      padding:0.85rem 1.1rem;
      margin:1rem 0 1.5rem;
      border-radius:18px;
      border:1px solid var(--border);
      background:var(--panel);
      box-shadow:0 18px 36px rgba(11,44,74,0.12);
      flex-wrap:wrap;
    }
    .mck-flow-stepper__item{
      flex:1 1 160px;
      display:flex;
      flex-direction:column;
      align-items:center;
      text-align:center;
      gap:0.45rem;
      position:relative;
    }
    .mck-flow-stepper__indicator{
      width:36px;
      height:36px;
      border-radius:50%;
      display:flex;
      align-items:center;
      justify-content:center;
      background:rgba(var(--accent-rgb,30,136,229),0.12);
      color:var(--accent-strong);
      border:2px solid rgba(var(--accent-rgb,30,136,229),0.22);
      box-shadow:0 12px 26px rgba(11,44,74,0.16);
      font-size:0.95rem;
    }
    .mck-flow-stepper__indicator svg{
      width:18px;
      height:18px;
    }
    .mck-flow-stepper__item[data-state="complete"] .mck-flow-stepper__indicator{
      background:var(--accent,#1E88E5);
      border-color:var(--accent,#1E88E5);
      color:#ffffff;
    }
    .mck-flow-stepper__item[data-state="active"] .mck-flow-stepper__indicator{
      background:#4f9ab8;
      border-color:#4f9ab8;
      color:#ffffff;
      box-shadow:0 18px 40px rgba(79,154,184,0.26);
    }
    .mck-flow-stepper__item[data-state="pending"] .mck-flow-stepper__indicator{
      background:rgba(var(--accent-rgb,30,136,229),0.08);
      border-color:rgba(var(--accent-rgb,30,136,229),0.18);
      color:var(--muted);
    }
    .mck-flow-stepper__label{
      display:flex;
      flex-direction:column;
      gap:0.2rem;
      font-weight:700;
      color:var(--accent-strong);
      font-size:1rem;
      line-height:1.25;
    }
    .mck-flow-stepper__label .en{
      font-size:0.8rem;
      font-weight:600;
      color:var(--muted);
      letter-spacing:0.05em;
    }
    .mck-flow-stepper__hint{
      margin:0;
      font-size:0.82rem;
      color:var(--muted);
      line-height:1.4;
    }
    .mck-flow-stepper__item[data-state="active"] .mck-flow-stepper__hint{
      color:var(--accent-strong);
      font-weight:600;
    }
    .mck-flow-stepper__connector{
      flex:0 0 32px;
      height:3px;
      border-radius:999px;
      background:rgba(var(--accent-rgb,30,136,229),0.2);
    }
    .mck-flow-stepper__connector[data-state="complete"]{
      background:rgba(var(--accent-rgb,30,136,229),0.6);
      box-shadow:0 4px 12px rgba(var(--accent-rgb,30,136,229),0.35);
    }
    @media (max-width: 920px){
      .mck-flow-stepper{
        gap:0.6rem;
        padding:0.75rem;
      }
      .mck-flow-stepper__connector{
        display:none;
      }
      .mck-flow-stepper__item{
        flex:1 1 calc(50% - 0.6rem);
        padding:0.6rem 0.4rem;
        border-radius:16px;
        border:1px solid var(--border);
        background:var(--panel-alt);
      }
      .mck-flow-stepper__indicator{
        width:38px;
        height:38px;
      }
    }
    .mck-import-section{
      background:var(--panel);
      border:1px solid var(--border);
      border-radius:18px;
      padding:1.1rem 1.25rem;
      margin-bottom:1.2rem;
      box-shadow:0 16px 32px rgba(11,44,74,0.1);
    }
    .mck-import-section__header{
      display:flex;
      align-items:center;
      gap:0.75rem;
      margin-bottom:0.6rem;
    }
    .mck-import-section__badge{
      width:32px;
      height:32px;
      border-radius:50%;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      font-size:0.85rem;
      color:#ffffff;
      background:#123a5f;
      box-shadow:0 10px 20px rgba(18,58,95,0.2);
    }
    .mck-import-section__icon{
      width:30px;
      height:30px;
      border-radius:50%;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      background:rgba(79,154,184,0.16);
      font-size:0.95rem;
      color:var(--accent-strong);
    }
    .mck-import-section__icon svg{
      width:18px;
      height:18px;
    }
    .mck-import-section__titles{
      display:flex;
      flex-direction:column;
      gap:0.1rem;
    }
    .mck-import-section__titles .jp{
      font-weight:700;
      font-size:1.1rem;
      color:var(--accent-strong);
    }
    .mck-import-section__titles .en{
      font-size:0.85rem;
      color:var(--muted);
      letter-spacing:0.04em;
    }
    .mck-import-section__body{
      display:flex;
      flex-direction:column;
      gap:0.7rem;
    }
    .mck-import-section__hint{
      display:block;
      font-size:0.8rem;
      color:var(--muted);
      margin-top:0.2rem;
    }
    .mck-import-section__note{
      display:block;
      font-size:0.78rem;
      color:var(--muted);
      margin-left:1rem;
    }
    .mck-import-section--template .mck-import-section__badge{
      background:#123a5f;
    }
    .mck-import-section--metrics .mck-import-section__badge{
      background:#2d6f8e;
    }
    .mck-import-section--upload .mck-import-section__badge{
      background:#4f9ab8;
    }
    .mck-import-section--quality .mck-import-section__badge{
      background:#71b7d4;
      color:#0b2f4c;
    }
    .mck-import-section--quality{
      border-color:rgba(79,154,184,0.32);
    }
    .mck-breadcrumb{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:1rem;
      padding:0.75rem 1rem;
      margin:0 0 1rem;
      border-radius:12px;
      background:var(--panel);
      border:1px solid var(--border);
      box-shadow:0 10px 20px rgba(11,44,74,0.08);
      font-size:0.9rem;
    }
    .mck-breadcrumb__trail{
      display:flex;
      align-items:center;
      flex-wrap:wrap;
      gap:0.45rem;
      color:var(--muted);
      font-weight:600;
      letter-spacing:.04em;
    }
    .mck-breadcrumb__sep{
      color:var(--muted);
      opacity:0.6;
    }
    .mck-breadcrumb__item--current{
      color:var(--accent-strong);
    }
    .mck-breadcrumb__meta{
      color:var(--muted);
      font-weight:600;
      letter-spacing:.08em;
      font-variant-numeric:tabular-nums;
      white-space:nowrap;
    }
    .mck-breadcrumb__desc{
      margin:-0.4rem 0 1.2rem;
      color:var(--muted);
      font-size:0.85rem;
      letter-spacing:.02em;
    }
    .nav-overlay{
      position:fixed;
      inset:0;
      background:rgba(7,22,39,0.55);
      backdrop-filter:blur(6px);
      opacity:0;
      pointer-events:none;
      transition:opacity .3s ease;
      z-index:1000;
    }
    .mobile-nav-toggle{
      position:fixed;
      top:1rem;
      right:1rem;
      width:46px;
      height:46px;
      border-radius:14px;
      background:rgba(var(--primary-rgb,11,31,59),0.82);
      border:1px solid rgba(255,255,255,0.35);
      box-shadow:0 18px 36px rgba(var(--primary-rgb,11,31,59),0.35);
      display:none;
      align-items:center;
      justify-content:center;
      gap:6px;
      z-index:1100;
      cursor:pointer;
    }
    .mobile-nav-toggle span{
      display:block;
      width:22px;
      height:2px;
      background:#ffffff;
      border-radius:999px;
      transition:transform .25s ease, opacity .25s ease;
    }
    .page-transition-fade{
      animation:pageFade .38s ease;
    }
    @keyframes pageFade{
      from{ opacity:0; transform:translateY(8px); }
      to{ opacity:1; transform:translateY(0); }
    }
    .nav-open .nav-overlay{
      opacity:1;
      pointer-events:auto;
    }
    .nav-open .mobile-nav-toggle span:nth-child(1){ transform:translateY(6px) rotate(45deg); }
    .nav-open .mobile-nav-toggle span:nth-child(2){ opacity:0; }
    .nav-open .mobile-nav-toggle span:nth-child(3){ transform:translateY(-6px) rotate(-45deg); }
    @media (max-width: 880px){
      html{ font-size:105%; }
      body{ overflow-x:hidden; }
      [data-testid="stSidebar"]{
        position:fixed;
        top:0;
        left:0;
        height:100vh;
        width:min(320px,82vw);
        max-width:86vw;
        transform:translateX(-110%);
        transition:transform .3s ease;
        z-index:1001;
        box-shadow:18px 0 32px rgba(var(--primary-rgb,11,31,59),0.35);
      }
      .nav-open [data-testid="stSidebar"]{ transform:translateX(0); }
      .mobile-nav-toggle{ display:flex; }
      [data-testid="stSidebar"] .sidebar-app-brand{ margin-top:3.2rem; }
      [data-testid="stSidebar"] label.nav-pill{ padding:1rem 1.1rem; }
      [data-testid="stSidebar"] label.nav-pill .nav-pill__icon{ width:2.2rem; height:2.2rem; font-size:1.1rem; }
      [data-testid="stSidebar"] label.nav-pill .nav-pill__title{ font-size:1.05rem; }
      [data-testid="stSidebar"] label.nav-pill .nav-pill__desc{ font-size:0.95rem; }
      .stButton>button{
        padding:0.75rem 1.6rem;
        font-size:1.05rem;
        min-height:3.1rem;
      }
      select,
      .stTextInput>div>input,
      .stSelectbox>div>div>input{
        font-size:1.02rem !important;
      }
      [data-testid="stHorizontalBlock"]{
        flex-direction:column !important;
        gap:0.8rem !important;
      }
      [data-testid="column"]{
        width:100% !important;
        flex-basis:100% !important;
      }
      .mck-metric-card{
        min-height:auto;
        padding:1.2rem 1.3rem;
      }
      .mck-metric-card__value{ font-size:1.65rem; }
      .mck-metric-card__title{ font-size:1.05rem; }
      .mck-metric-card__subtitle{ font-size:0.9rem; }
      .mck-breadcrumb{ flex-direction:column; align-items:flex-start; }
      .tour-step-guide{ justify-content:center; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SIDEBAR_CATEGORY_STYLES = {
    "input": {
        "label": "データ入力",
        "color": "#4f9ab8",
        "description": "テンプレート選択からアップロード、設定までの初期操作を集約しています。",
    },
    "report": {
        "label": "分析レポート",
        "color": "#123a5f",
        "description": "ダッシュボードやランキングなど、分析結果を確認するページです。",
    },
    "simulation": {
        "label": "シナリオシミュレーション",
        "color": "#f2994a",
        "description": "異常検知やアラートで将来シナリオやリスクを検証します。",
    },
}
SIDEBAR_CATEGORY_ORDER = ["input", "report", "simulation"]

SIDEBAR_PAGES = [
    {
        "key": "dashboard",
        "page": "ダッシュボード",
        "icon": "🏠",
        "title": "ホーム",
        "tagline": "分析ダッシュボード",
        "tooltip": "主要KPIとトレンドを俯瞰できるダッシュボードです。",
        "category": "report",
    },
    {
        "key": "ranking",
        "page": "ランキング",
        "icon": "📊",
        "title": "ランキング",
        "tagline": "指標別トップ・ボトム",
        "tooltip": "指定月の上位・下位SKUを指標別に比較して勢いを捉えます。",
        "category": "report",
    },
    {
        "key": "compare",
        "page": "比較ビュー",
        "icon": "🔁",
        "title": "比較ビュー",
        "tagline": "SKU横断の推移比較",
        "tooltip": "複数SKUの推移を重ね合わせ、変化の違いを見比べます。",
        "category": "report",
    },
    {
        "key": "detail",
        "page": "SKU詳細",
        "icon": "🧾",
        "title": "SKU詳細",
        "tagline": "個別SKUの深掘り",
        "tooltip": "個別SKUの時系列やAIサマリーで背景を確認します。",
        "category": "report",
    },
    {
        "key": "correlation",
        "page": "相関分析",
        "icon": "🔗",
        "title": "相関分析",
        "tagline": "指標のつながり分析",
        "tooltip": "散布図と相関係数で指標同士やSKU間の関係を把握します。",
        "category": "report",
    },
    {
        "key": "category",
        "page": "併買カテゴリ",
        "icon": "🛍️",
        "title": "併買カテゴリ",
        "tagline": "併買パターンの探索",
        "tooltip": "購買ネットワークのクラスタリングでクロスセル候補を探します。",
        "category": "report",
    },
    {
        "key": "import",
        "page": "データ取込",
        "icon": "📥",
        "title": "データ取込",
        "tagline": "CSV/Excelアップロード",
        "tooltip": "CSVやExcelの月次データを取り込み、分析用データを整えます。",
        "category": "input",
    },
    {
        "key": "anomaly",
        "page": "異常検知",
        "icon": "⚠️",
        "title": "異常検知",
        "tagline": "異常値とリスク検知",
        "tooltip": "回帰残差を基にした異常値スコアでリスク兆候を洗い出します。",
        "category": "simulation",
    },
    {
        "key": "alert",
        "page": "アラート",
        "icon": "🚨",
        "title": "アラート",
        "tagline": "しきい値ベースの監視",
        "tooltip": "設定した条件に該当するSKUをリスト化し、対応優先度を整理します。",
        "category": "simulation",
    },
    {
        "key": "settings",
        "page": "設定",
        "icon": "⚙️",
        "title": "設定",
        "tagline": "集計条件の設定",
        "tooltip": "年計ウィンドウや通貨単位など、分析前提を調整します。",
        "category": "input",
    },
    {
        "key": "saved",
        "page": "保存ビュー",
        "icon": "💾",
        "title": "保存ビュー",
        "tagline": "条件の保存と共有",
        "tooltip": "現在の設定や比較条件を保存し、ワンクリックで再現します。",
        "category": "input",
    },
]

SIDEBAR_PAGE_LOOKUP = {page["key"]: page for page in SIDEBAR_PAGES}
NAV_KEYS = [page["key"] for page in SIDEBAR_PAGES]
NAV_TITLE_LOOKUP = {page["key"]: page["title"] for page in SIDEBAR_PAGES}
page_lookup = {page["key"]: page["page"] for page in SIDEBAR_PAGES}

PRIMARY_NAV_MENU = [
    {
        "key": "home",
        "label": SIDEBAR_PAGE_LOOKUP["dashboard"]["title"],
        "icon": SIDEBAR_PAGE_LOOKUP["dashboard"].get("icon", "🏠"),
        "description": SIDEBAR_PAGE_LOOKUP["dashboard"].get("tooltip", ""),
        "pages": ["dashboard"],
    },
    {
        "key": "ranking",
        "label": SIDEBAR_PAGE_LOOKUP["ranking"]["title"],
        "icon": SIDEBAR_PAGE_LOOKUP["ranking"].get("icon", "📊"),
        "description": SIDEBAR_PAGE_LOOKUP["ranking"].get("tooltip", ""),
        "pages": ["ranking"],
    },
    {
        "key": "analysis",
        "label": "分析ツール",
        "icon": SIDEBAR_PAGE_LOOKUP["compare"].get("icon", "🔍"),
        "description": "比較ビューやSKU詳細、相関分析などの深掘りページをまとめています。",
        "pages": ["compare", "detail", "correlation", "category"],
    },
    {
        "key": "monitor",
        "label": "監視アラート",
        "icon": SIDEBAR_PAGE_LOOKUP["anomaly"].get("icon", "⚠️"),
        "description": "異常検知とアラート機能でリスクを素早く把握します。",
        "pages": ["anomaly", "alert"],
    },
    {
        "key": "data",
        "label": "データ設定",
        "icon": SIDEBAR_PAGE_LOOKUP["import"].get("icon", "📥"),
        "description": "データ取込や設定、保存ビューを一箇所にまとめました。",
        "pages": ["import", "settings", "saved"],
    },
]

PRIMARY_NAV_LOOKUP = {item["key"]: item for item in PRIMARY_NAV_MENU}
PAGE_TO_PRIMARY_LOOKUP: Dict[str, str] = {}
for item in PRIMARY_NAV_MENU:
    pages = list(dict.fromkeys(item.get("pages", [])))
    item["pages"] = pages
    for page_key in pages:
        if page_key in SIDEBAR_PAGE_LOOKUP:
            PAGE_TO_PRIMARY_LOOKUP[page_key] = item["key"]

PRIMARY_NAV_CLIENT_DATA: List[Dict[str, object]] = []
for item in PRIMARY_NAV_MENU:
    page_titles = {
        page_key: NAV_TITLE_LOOKUP.get(
            page_key,
            SIDEBAR_PAGE_LOOKUP.get(page_key, {}).get("title", page_key),
        )
        for page_key in item["pages"]
    }
    page_tooltips = {
        page_key: (
            SIDEBAR_PAGE_LOOKUP.get(page_key, {}).get("tooltip")
            or SIDEBAR_PAGE_LOOKUP.get(page_key, {}).get("tagline", "")
        )
        for page_key in item["pages"]
    }
    PRIMARY_NAV_CLIENT_DATA.append(
        {
            "key": item["key"],
            "label": item["label"],
            "icon": item.get("icon", ""),
            "description": item.get("description", ""),
            "pages": item["pages"],
            "page_titles": page_titles,
            "page_tooltips": page_tooltips,
        }
    )

NAV_CATEGORY_STATE_KEY = "nav_category"
PENDING_NAV_CATEGORY_KEY = "_pending_nav_category"
PENDING_NAV_PAGE_KEY = "_pending_nav_page"
NAV_PRIMARY_STATE_KEY = "nav_primary"
PENDING_NAV_PRIMARY_KEY = "_pending_nav_primary"
PENDING_NAV_SUB_PREFIX = "_pending_nav_sub_"


def _queue_nav_category(category: Optional[str], *, rerun_on_lock: bool = False) -> None:
    if not category:
        return
    current_category = st.session_state.get(NAV_CATEGORY_STATE_KEY)
    pending_category = st.session_state.get(PENDING_NAV_CATEGORY_KEY)
    if current_category == category:
        if pending_category:
            st.session_state.pop(PENDING_NAV_CATEGORY_KEY, None)
        return
    if pending_category != category:
        st.session_state[PENDING_NAV_CATEGORY_KEY] = category
    if NAV_CATEGORY_STATE_KEY not in st.session_state:
        st.session_state[NAV_CATEGORY_STATE_KEY] = category
    if rerun_on_lock:
        st.rerun()


def set_active_page(page_key: str, *, rerun_on_lock: bool = False) -> None:
    meta = SIDEBAR_PAGE_LOOKUP.get(page_key)
    if not meta:
        return
    category = meta.get("category")
    rerun_required = False
    try:
        st.session_state["nav_page"] = page_key
    except StreamlitAPIException:
        st.session_state[PENDING_NAV_PAGE_KEY] = page_key
        rerun_required = True
    primary_key = PAGE_TO_PRIMARY_LOOKUP.get(page_key)
    if primary_key:
        sub_state_key = f"nav_sub_{primary_key}"
        try:
            st.session_state[sub_state_key] = page_key
        except StreamlitAPIException:
            st.session_state[f"{PENDING_NAV_SUB_PREFIX}{primary_key}"] = page_key
            rerun_required = True
        try:
            st.session_state[NAV_PRIMARY_STATE_KEY] = primary_key
        except StreamlitAPIException:
            st.session_state[PENDING_NAV_PRIMARY_KEY] = primary_key
            rerun_required = True
    if category:
        _queue_nav_category(
            category,
            rerun_on_lock=rerun_on_lock and not rerun_required,
        )
    if rerun_required and rerun_on_lock:
        st.rerun()


def _hex_to_rgb_string(color: str) -> str:
    stripped = color.lstrip("#")
    if len(stripped) == 6:
        try:
            r, g, b = (int(stripped[i : i + 2], 16) for i in (0, 2, 4))
            return f"{r}, {g}, {b}"
        except ValueError:
            pass
    return "71, 183, 212"


NAV_HOVER_LOOKUP: Dict[str, str] = {}
nav_client_data: List[Dict[str, str]] = []
for page in SIDEBAR_PAGES:
    category_info = SIDEBAR_CATEGORY_STYLES.get(page["category"], {})
    color = category_info.get("color", "#71b7d4")
    hover_lines = [page.get("title", "").strip()]
    tooltip_text = page.get("tooltip", "").strip()
    tagline_text = page.get("tagline", "").strip()
    if tooltip_text:
        hover_lines.append(tooltip_text)
    elif tagline_text:
        hover_lines.append(tagline_text)
    hover_text = "\n".join(filter(None, hover_lines))
    nav_client_data.append(
        {
            "key": page["key"],
            "title": page["title"],
            "tagline": page.get("tagline", ""),
            "icon": page.get("icon", ""),
            "tooltip": page.get("tooltip", ""),
            "category": page["category"],
            "category_label": category_info.get("label", ""),
            "color": color,
            "rgb": _hex_to_rgb_string(color),
            "hover_text": hover_text,
        }
    )
    NAV_HOVER_LOOKUP[page["key"]] = hover_text

used_category_keys = [
    cat for cat in SIDEBAR_CATEGORY_ORDER if any(p["category"] == cat for p in SIDEBAR_PAGES)
]
if used_category_keys:
    legend_items_html = "".join(
        f"<span class='sidebar-legend__item' style='--legend-color:{SIDEBAR_CATEGORY_STYLES[cat]['color']};'>{SIDEBAR_CATEGORY_STYLES[cat]['label']}</span>"
        for cat in used_category_keys
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-legend">
            <p class="sidebar-legend__title">色でカテゴリを表示しています</p>
            <div class="sidebar-legend__items">{legend_items_html}</div>
            <p class="sidebar-legend__hint">アイコンにカーソルを合わせると各機能の説明が表示されます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

TOUR_STEPS: List[Dict[str, str]] = [
    {
        "key": "import",
        "nav_key": "import",
        "label": SIDEBAR_PAGE_LOOKUP["import"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["import"]["page"],
        "heading": "データ取込",
        "title": "データ取込",
        "section": "基礎編",
        "description": "最初に月次売上データをアップロードし、分析ダッシュボードを有効化します。",
        "details": "テンプレートのマッピングを完了すると基礎編の残りステップをすぐに確認できます。",
    },
    {
        "key": "dashboard",
        "nav_key": "dashboard",
        "label": SIDEBAR_PAGE_LOOKUP["dashboard"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["dashboard"]["page"],
        "heading": "ダッシュボード",
        "title": "ダッシュボード",
        "section": "基礎編",
        "description": "年計KPIと総合トレンドを俯瞰し、AIサマリーで直近の動きを素早く把握します。",
        "details": "ハイライト/ランキングタブで主要SKUの変化を数クリックでチェック。",
    },
    {
        "key": "ranking",
        "nav_key": "ranking",
        "label": SIDEBAR_PAGE_LOOKUP["ranking"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["ranking"]["page"],
        "heading": "ランキング",
        "title": "ランキング",
        "section": "基礎編",
        "description": "指定月の上位・下位SKUを指標別に比較し、勢いのある商品を短時間で把握します。",
        "details": "並び順や指標を切り替えて気になるSKUを絞り込み、必要に応じてCSV/Excelで共有。",
    },
    {
        "key": "compare",
        "nav_key": "compare",
        "label": SIDEBAR_PAGE_LOOKUP["compare"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["compare"]["page"],
        "heading": "マルチ商品比較",
        "title": "比較ビュー",
        "section": "応用編",
        "description": "条件で絞った複数SKUの推移を重ね合わせ、帯やバンドで素早く切り替えます。",
        "details": "操作バーで期間や表示を選び、スモールマルチプルで個別の動きを確認。",
    },
    {
        "key": "detail",
        "nav_key": "detail",
        "label": SIDEBAR_PAGE_LOOKUP["detail"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["detail"]["page"],
        "heading": "SKU 詳細",
        "title": "SKU詳細",
        "section": "応用編",
        "description": "個別SKUの時系列と指標を確認し、メモやタグでアクションを記録します。",
        "details": "単品/複数比較モードとAIサマリーで詳細な解釈を補助。",
    },
    {
        "key": "anomaly",
        "nav_key": "anomaly",
        "label": SIDEBAR_PAGE_LOOKUP["anomaly"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["anomaly"]["page"],
        "heading": "異常検知",
        "title": "異常検知",
        "section": "応用編",
        "description": "回帰残差ベースで異常な月次を検知し、スコアの高い事象を優先的に確認します。",
        "details": "窓幅・閾値を調整し、AI異常サマリーで発生背景を把握。",
    },
    {
        "key": "correlation",
        "nav_key": "correlation",
        "label": SIDEBAR_PAGE_LOOKUP["correlation"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["correlation"]["page"],
        "heading": "相関分析",
        "title": "相関分析",
        "section": "応用編",
        "description": "指標間の関係性やSKU同士の動きを散布図と相関係数で分析します。",
        "details": "相関指標や対象SKUを選び、外れ値の注釈からインサイトを発見。",
    },
    {
        "key": "category",
        "nav_key": "category",
        "label": SIDEBAR_PAGE_LOOKUP["category"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["category"]["page"],
        "heading": "購買カテゴリ探索",
        "title": "併買カテゴリ",
        "section": "応用編",
        "description": "購買ネットワークをクラスタリングしてクロスセル候補のグルーピングを見つけます。",
        "details": "入力データや閾値・検出法を変え、ネットワーク可視化をチューニング。",
    },
    {
        "key": "alert",
        "nav_key": "alert",
        "label": SIDEBAR_PAGE_LOOKUP["alert"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["alert"]["page"],
        "heading": "アラート",
        "title": "アラート",
        "section": "応用編",
        "description": "設定した閾値に該当するリスクSKUを一覧化し、優先度の高い対応を整理します。",
        "details": "CSVダウンロードで日次の共有や監視に活用。",
    },
    {
        "key": "settings",
        "nav_key": "settings",
        "label": SIDEBAR_PAGE_LOOKUP["settings"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["settings"]["page"],
        "heading": "設定",
        "title": "設定",
        "section": "応用編",
        "description": "年計ウィンドウやアラート条件など、分析の前提を調整します。",
        "details": "変更後は再計算ボタンでデータを更新し、全ページに反映します。",
    },
    {
        "key": "saved",
        "nav_key": "saved",
        "label": SIDEBAR_PAGE_LOOKUP["saved"]["title"],
        "page": SIDEBAR_PAGE_LOOKUP["saved"]["page"],
        "heading": "保存ビュー",
        "title": "保存ビュー",
        "section": "応用編",
        "description": "現在の設定や比較条件を名前付きで保存し、ワンクリックで再現できます。",
        "details": "設定と比較条件を共有し、分析の再現性を高めます。",
    },
]


TOUR_SECTION_ORDER: List[str] = []
TOUR_SECTION_COUNTS: Dict[str, int] = {}
for step in TOUR_STEPS:
    section_name = step.get("section") or "応用編"
    if section_name not in TOUR_SECTION_COUNTS:
        TOUR_SECTION_ORDER.append(section_name)
        TOUR_SECTION_COUNTS[section_name] = 0
    TOUR_SECTION_COUNTS[section_name] += 1
    step["section"] = section_name

section_positions: Dict[str, int] = {section: 0 for section in TOUR_SECTION_ORDER}
for step in TOUR_STEPS:
    section_name = step.get("section") or "応用編"
    section_positions[section_name] = section_positions.get(section_name, 0) + 1
    step["section_index"] = section_positions[section_name]
    step["section_total"] = TOUR_SECTION_COUNTS.get(section_name, len(TOUR_STEPS))


def render_step_guide(active_nav_key: str) -> None:
    if not TOUR_STEPS:
        return

    items_html: List[str] = []
    for step in TOUR_STEPS:
        nav_key = step.get("nav_key")
        if not nav_key:
            continue

        nav_meta = SIDEBAR_PAGE_LOOKUP.get(nav_key)
        if not nav_meta:
            continue

        label_text = (
            nav_meta.get("title")
            or step.get("title")
            or step.get("label")
            or nav_key
        )
        icon_text = nav_meta.get("icon", "")

        tooltip_candidates = [
            NAV_HOVER_LOOKUP.get(nav_key, "").strip(),
            (
                f"{step.get('section', '').strip()} {step.get('section_index', 0)} / {step.get('section_total', 0)}"
                if step.get("section")
                else ""
            ),
            step.get("description", "").strip(),
            step.get("details", "").strip(),
        ]
        tooltip_parts: List[str] = []
        for candidate in tooltip_candidates:
            if candidate and candidate not in tooltip_parts:
                tooltip_parts.append(candidate)

        tooltip_text = "\n".join(tooltip_parts)
        tooltip_attr = html.escape(tooltip_text, quote=True).replace("\n", "&#10;")
        title_text = tooltip_text.replace("\n", " ") if tooltip_text else label_text
        title_attr = html.escape(title_text, quote=True)
        aria_label_text = tooltip_text.replace("\n", " ") if tooltip_text else label_text
        aria_label_attr = html.escape(aria_label_text, quote=True)

        icon_html = html.escape(icon_text)
        label_html = html.escape(label_text)
        data_active = "true" if nav_key == active_nav_key else "false"
        aria_current_attr = ' aria-current="step"' if nav_key == active_nav_key else ""

        category_key = html.escape(nav_meta.get("category", ""), quote=True)

        item_html = (
            f'<div class="tour-step-guide__item has-tooltip" data-step="{nav_key}" '
            f'data-category="{category_key}" '
            f'data-active="{data_active}" data-tooltip="{tooltip_attr}" title="{title_attr}" '
            f'tabindex="0" role="listitem" aria-label="{aria_label_attr}"{aria_current_attr}>'
            f'<span class="tour-step-guide__icon" aria-hidden="true">{icon_html}</span>'
            f'<span class="tour-step-guide__label">{label_html}</span>'
            "</div>"
        )
        items_html.append(item_html)

    if not items_html:
        return

    st.markdown(
        f'<div class="tour-step-guide" role="list" aria-label="主要ナビゲーションステップ">'
        f'{"".join(items_html)}</div>',
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const findSidebarInput = (navKey) => {
                if (!navKey) return null;
                const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
                if (!sidebar) return null;
                const label = sidebar.querySelector('label[data-nav-key="' + navKey + '"]');
                if (!label) return null;
                return label.querySelector('input[type="radio"]');
            };
            const findCategoryInput = (categoryKey) => {
                if (!categoryKey) return null;
                const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
                if (!sidebar) return null;
                const label = sidebar.querySelector('label[data-category-key="' + categoryKey + '"]');
                if (label) {
                    const input = label.querySelector('input[type="radio"]');
                    if (input) {
                        return input;
                    }
                }
                const radios = Array.from(sidebar.querySelectorAll('input[type="radio"]'));
                return radios.find((input) => input.value === categoryKey) || null;
            };
            const bind = (attempt = 0) => {
                const items = Array.from(doc.querySelectorAll('.tour-step-guide__item[data-step]'));
                if (!items.length) {
                    if (attempt < 10) {
                        setTimeout(() => bind(attempt + 1), 120);
                    }
                    return;
                }
                items.forEach((item) => {
                    if (item.dataset.navBound === '1') return;
                    item.dataset.navBound = '1';
                    const activate = () => {
                        const navKey = item.dataset.step;
                        const categoryKey = item.dataset.category;
                        const input = findSidebarInput(navKey);
                        const categoryInput = findCategoryInput(categoryKey);
                        if (categoryInput && !categoryInput.checked) {
                            categoryInput.click();
                            setTimeout(() => {
                                const refreshed = findSidebarInput(navKey);
                                if (refreshed && !refreshed.checked) {
                                    refreshed.click();
                                }
                            }, 80);
                            return;
                        }
                        if (input && !input.checked) {
                            input.click();
                        }
                    };
                    item.addEventListener('click', activate);
                    item.addEventListener('keydown', (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            activate();
                        }
                    });
                });
            };
            bind();
        })();
        </script>
        """,
        height=0,
    )


def render_getting_started_intro() -> None:
    """Show a short how-to guide with step hints and a demo video."""

    st.markdown("### 3分でわかるスターターガイド")
    steps = [
        {
            "title": "データをアップロード",
            "body": "売上・仕入れ・経費のCSVを取り込み、商品名と月度を確認します。",
        },
        {
            "title": "KPIをチェック",
            "body": "ダッシュボードで年計・YoY・Δを確認し、気になるSKUをブックマーク。",
        },
        {
            "title": "深掘り分析",
            "body": "ランキングや相関分析で伸長/苦戦領域を深堀りし、AIコメントを参考に次のアクションを検討します。",
        },
    ]

    for idx, step in enumerate(steps, start=1):
        st.markdown(f"**STEP {idx}. {step['title']}**")
        st.write(step["body"])

    with st.expander("操作のポイントを詳しく見る", expanded=False):
        st.markdown(
            "- **サイドバー**からページを移動すると、関連するヒントが自動でハイライトされます。\n"
            "- **AIコパイロット**に質問すると、最新の年計スナップショットを踏まえた要約と推奨アクションが得られます。\n"
            "- チャートの右上にあるツールバーから画像保存やデータダウンロードが可能です。",
        )


def render_sample_data_hub() -> None:
    """Provide downloadable CSV samples and one-click loaders."""

    sample_metas = list_sample_csv_meta()
    if not sample_metas:
        return

    meta_lookup = {meta.key: meta for meta in sample_metas}
    sample_keys = [meta.key for meta in sample_metas]
    default_key = st.session_state.get("sample_data_selector", sample_metas[0].key)
    sample_key = st.selectbox(
        "サンプルデータの種類",
        options=sample_keys,
        index=sample_keys.index(default_key) if default_key in meta_lookup else 0,
        format_func=lambda key: meta_lookup[key].title,
        key="sample_data_selector",
        help="業務別のCSVテンプレートを選び、列構成と数値例を確認します。",
    )
    selected_meta: SampleCSVMeta = meta_lookup.get(sample_key, sample_metas[0])

    st.caption(selected_meta.description)
    sample_df = load_sample_csv_dataframe(selected_meta.key)
    st.dataframe(sample_df.head(5), use_container_width=True)

    col_download, col_load = st.columns(2)
    with col_download:
        st.download_button(
            f"{selected_meta.title}をダウンロード",
            data=get_sample_csv_bytes(selected_meta.key),
            file_name=selected_meta.download_name,
            mime="text/csv",
            help="CSVテンプレートをローカルに保存し、実データを追記してアップロードできます。",
        )
    with col_load:
        if st.button(
            f"{selected_meta.title}を読み込む",
            key=f"load_sample_{selected_meta.key}",
            help="サンプルCSVをアプリに読み込み、ダッシュボードをすぐに体験します。",
        ):
            try:
                with loading_message("サンプルデータを初期化しています…"):
                    ingest_wide_dataframe(
                        sample_df.copy(),
                        product_name_col=selected_meta.name_column,
                        product_code_col=selected_meta.code_column,
                    )
                st.session_state.sample_data_notice = True
                st.session_state.sample_data_message = (
                    f"{selected_meta.title}を読み込みました。ダッシュボードでサンプル指標を確認できます。"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"サンプルデータの読込に失敗しました: {exc}")

    with st.expander("ワンクリックでデモデータを試す", expanded=False):
        st.markdown(
            "時間がない場合は、あらかじめ用意した合成データセットを読み込み、全ページの動作を確認できます。"
        )
        if st.button(
            "デモデータを読み込む",
            key="load_demo_dataset",
            help="売上トレンドを再現した合成データで、主要な分析ページをすぐに表示します。",
        ):
            try:
                with loading_message("デモデータを準備しています…"):
                    demo_long = load_sample_dataset()
                    process_long_dataframe(demo_long)
                st.session_state.sample_data_notice = True
                st.session_state.sample_data_message = (
                    "デモデータを読み込みました。ダッシュボードで操作感を確認してください。"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"デモデータの準備に失敗しました: {exc}")


@contextmanager
def import_section(
    number: int,
    title_ja: str,
    title_en: str,
    icon: str,
    *,
    accent: str = "template",
):
    """Render a structured panel with numbered headers for the import workflow."""

    accent_class = f" mck-import-section--{accent}" if accent else ""
    outer = st.container()
    with outer:
        icon_html = icon if icon else ""
        st.markdown(
            """
            <section class="mck-import-section{accent_class}" data-section="{number:02d}">
              <header class="mck-import-section__header">
                <span class="mck-import-section__badge">{number:02d}</span>
                <span class="mck-import-section__icon" aria-hidden="true">{icon}</span>
                <div class="mck-import-section__titles">
                  <span class="jp">{title_ja}</span>
                  <span class="en">{title_en}</span>
                </div>
              </header>
              <div class="mck-import-section__body">
            """.format(
                accent_class=accent_class,
                number=number,
                icon=icon_html,
                title_ja=html.escape(title_ja),
                title_en=html.escape(title_en),
            ),
            unsafe_allow_html=True,
        )
        inner = st.container()
        with inner:
            yield
        st.markdown("</div></section>", unsafe_allow_html=True)


def build_import_progress_steps() -> tuple[List[Dict[str, object]], Dict[str, str]]:
    """Assemble import workflow steps and their current statuses."""

    template_key = get_active_template_key()
    template_config = get_template_config(template_key)
    template_label = template_config.get("label", template_key)
    data_monthly = st.session_state.get("data_monthly")
    data_year = st.session_state.get("data_year")
    quality_summary = st.session_state.get("import_quality_summary")
    uploaded_name = st.session_state.get("import_uploaded_file_name", "")
    uploaded_at_raw = st.session_state.get("import_last_uploaded")
    uploaded_at_disp = ""
    if uploaded_at_raw:
        try:
            uploaded_dt = datetime.fromisoformat(uploaded_at_raw)
            uploaded_at_disp = uploaded_dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            uploaded_at_disp = str(uploaded_at_raw)

    def _has_rows(df: Optional[pd.DataFrame]) -> bool:
        return bool(df is not None and not getattr(df, "empty", False))

    if not quality_summary and _has_rows(data_monthly):
        try:
            derived = {
                "missing": int(
                    data_monthly["is_missing"].sum()
                    if "is_missing" in data_monthly.columns
                    else 0
                ),
                "total": int(len(data_monthly)),
                "sku_count": int(
                    data_monthly["product_code"].nunique()
                    if "product_code" in data_monthly.columns
                    else 0
                ),
                "period_start": str(data_monthly["month"].min()),
                "period_end": str(data_monthly["month"].max()),
            }
            quality_summary = derived
        except Exception:
            quality_summary = None

    template_ready = bool(template_label)
    upload_ready = _has_rows(data_monthly)
    quality_ready = bool(quality_summary) or _has_rows(data_year)
    report_ready = _has_rows(data_year)
    report_done = bool(st.session_state.get("import_report_completed"))

    status_template = "complete" if template_ready else "current"
    status_upload = (
        "complete"
        if upload_ready
        else ("current" if status_template == "complete" else "upcoming")
    )
    status_quality = (
        "complete"
        if quality_ready
        else ("current" if status_upload == "complete" else "upcoming")
    )
    if report_ready:
        status_report = "complete" if report_done else (
            "current" if status_quality == "complete" else "upcoming"
        )
    else:
        status_report = "upcoming"

    steps: List[Dict[str, object]] = []

    step_template_desc = [
        (
            "業種テンプレートを選ぶと推奨指標が自動反映されます。",
            "Selecting an industry template preloads recommended metrics.",
        )
    ]
    if template_label:
        step_template_desc.append(
            (
                f"適用中テンプレート: {template_label}",
                f"Active template: {template_label}",
            )
        )
    steps.append(
        {
            "key": "template",
            "icon": icon_svg("template"),
            "label": "テンプレート選択",
            "label_en": "Template Selection",
            "status": status_template,
            "desc": step_template_desc,
        }
    )

    step_upload_desc = [
        (
            "CSV/Excelファイルをドラッグ＆ドロップしてください。",
            "Drag & drop CSV or Excel files to begin the import.",
        ),
        (
            "月度列は YYYY-MM 形式が推奨です。",
            "Month columns in YYYY-MM format are recommended.",
        ),
    ]
    if uploaded_name:
        step_upload_desc.append(
            (
                f"最新アップロード: {uploaded_name}",
                f"Latest upload: {uploaded_name}",
            )
        )
    if uploaded_at_disp:
        step_upload_desc.append(
            (
                f"更新日時: {uploaded_at_disp}",
                f"Updated at: {uploaded_at_disp}",
            )
        )
    steps.append(
        {
            "key": "upload",
            "icon": icon_svg("upload"),
            "label": "データアップロード",
            "label_en": "Data Upload",
            "status": status_upload,
            "desc": step_upload_desc,
        }
    )

    if quality_summary:
        missing = quality_summary.get("missing", 0)
        total = quality_summary.get("total", 0)
        sku_cnt = quality_summary.get("sku_count", 0)
        period_start = quality_summary.get("period_start", "—")
        period_end = quality_summary.get("period_end", "—")
        quality_desc = [
            (
                f"欠測セル: {missing:,} / {total:,}",
                f"Missing cells: {missing:,} / {total:,}",
            ),
            (
                f"SKU数: {sku_cnt:,}",
                f"SKU count: {sku_cnt:,}",
            ),
            (
                f"期間: {period_start} 〜 {period_end}",
                f"Coverage: {period_start} to {period_end}",
            ),
        ]
    else:
        quality_desc = [
            (
                "欠測や型の揺れをチェックし、エラーを先に解消します。",
                "Review missing values and format issues before continuing.",
            )
        ]
    steps.append(
        {
            "key": "quality",
            "icon": icon_svg("quality"),
            "label": "データチェック",
            "label_en": "Data Quality Review",
            "status": status_quality,
            "desc": quality_desc,
        }
    )

    step_report_desc = [
        (
            "ダッシュボードとランキングで結果を確認しましょう。",
            "Open dashboards and rankings to explore the results.",
        ),
        (
            "PDF/CSV出力でレポートを共有できます。",
            "Share insights with PDF or CSV exports.",
        ),
    ]
    if report_done:
        step_report_desc.append(
            (
                "レポート出力を完了しました。",
                "Report export completed.",
            )
        )
    steps.append(
        {
            "key": "report",
            "icon": icon_svg("download"),
            "label": "レポート出力",
            "label_en": "Report Output",
            "status": status_report,
            "desc": step_report_desc,
        }
    )

    status_lookup = {step["key"]: step.get("status", "upcoming") for step in steps}
    return steps, status_lookup


def render_import_stepper() -> None:
    """Show progress for the import workflow with bilingual labels."""

    steps, _ = build_import_progress_steps()
    if not steps:
        return

    STATUS_LABELS = {
        "complete": "完了 / Done",
        "current": "進行中 / In Progress",
        "upcoming": "未着手 / Pending",
    }

    items_html: List[str] = []
    for step in steps:
        status = step.get("status", "upcoming")
        aria_current = " aria-current=\"step\"" if status == "current" else ""
        state_label = STATUS_LABELS.get(status, "")
        desc_lines = step.get("desc", []) or []
        desc_html_parts: List[str] = []
        for jp, en in desc_lines:
            desc_html_parts.append(
                f"{html.escape(str(jp))}<span class='en'>{html.escape(str(en))}</span>"
            )
        desc_html = "<br>".join(desc_html_parts) if desc_html_parts else "&nbsp;"
        items_html.append(
            """
            <div class="mck-import-stepper__item" data-status="{status}" role="listitem"{aria_current}>
              <div class="mck-import-stepper__status" data-status="{status}" aria-hidden="true">{icon}</div>
              <div class="mck-import-stepper__content">
                <div class="mck-import-stepper__header">
                  <div class="mck-import-stepper__title">
                    <span class="jp">{label}</span>
                    <span class="en">{label_en}</span>
                  </div>
                  <span class="mck-import-stepper__state" data-status="{status}">{state}</span>
                </div>
                <p class="mck-import-stepper__desc">{desc}</p>
              </div>
            </div>
            """.format(
                status=html.escape(str(status)),
                aria_current=aria_current,
                icon=step.get("icon", ""),
                label=html.escape(str(step.get("label", ""))),
                label_en=html.escape(str(step.get("label_en", ""))),
                state=html.escape(state_label),
                desc=desc_html,
            )
        )

    if items_html:
        st.markdown(
            """
            <div class="mck-import-stepper" role="list" aria-label="データ取込ステップ">
              {items}
            </div>
            """.format(items="".join(items_html)),
            unsafe_allow_html=True,
        )


FLOW_STEP_SEQUENCE: List[Dict[str, str]] = [
    {
        "key": "template",
        "label": "テンプレート選択",
        "label_en": "Template",
        "icon": icon_svg("template"),
        "hint": "業種テンプレートと推奨指標を選択",
    },
    {
        "key": "upload",
        "label": "データ入力",
        "label_en": "Data Input",
        "icon": icon_svg("upload"),
        "hint": "CSV/Excelをアップロードして整形",
    },
    {
        "key": "quality",
        "label": "データ確認",
        "label_en": "Data Review",
        "icon": icon_svg("quality"),
        "hint": "欠測チェックと年計生成",
    },
    {
        "key": "report",
        "label": "指標表示",
        "label_en": "KPI View",
        "icon": icon_svg("metrics"),
        "hint": "ダッシュボードで指標を確認",
    },
]

FLOW_PAGE_OVERRIDES: Dict[str, str] = {
    "settings": "template",
    "saved": "report",
}

FLOW_CATEGORY_DEFAULT: Dict[str, str] = {
    "input": "upload",
    "report": "report",
    "simulation": "report",
}


def render_process_step_bar(page_key: str) -> None:
    """Render a compact horizontal step bar showing the overall workflow."""

    steps, status_lookup = build_import_progress_steps()
    if not steps:
        return

    page_meta = SIDEBAR_PAGE_LOOKUP.get(page_key, {})
    category = page_meta.get("category", "")

    active_key = FLOW_PAGE_OVERRIDES.get(page_key)
    if not active_key:
        if page_key == "import":
            active_key = next(
                (step.get("key") for step in steps if step.get("status") == "current"),
                None,
            )
            if not active_key:
                if status_lookup.get("report") == "complete":
                    active_key = "report"
                elif status_lookup.get("quality") == "complete":
                    active_key = "quality"
                elif status_lookup.get("upload") == "complete":
                    active_key = "upload"
                else:
                    active_key = "template"
        else:
            active_key = FLOW_CATEGORY_DEFAULT.get(category)
            if not active_key:
                if status_lookup.get("report") == "complete":
                    active_key = "report"
                else:
                    next_incomplete = next(
                        (step.get("key") for step in steps if step.get("status") != "complete"),
                        None,
                    )
                    active_key = next_incomplete or "template"

    state_lookup: Dict[str, str] = {}
    for step_def in FLOW_STEP_SEQUENCE:
        key = step_def["key"]
        status = status_lookup.get(key, "upcoming")
        if key == active_key:
            state_lookup[key] = "active"
        elif status == "complete":
            state_lookup[key] = "complete"
        else:
            state_lookup[key] = "pending"

    items_html: List[str] = []
    for idx, step_def in enumerate(FLOW_STEP_SEQUENCE):
        key = step_def["key"]
        state = state_lookup.get(key, "pending")
        label_jp = html.escape(step_def["label"])
        label_en = html.escape(step_def["label_en"])
        hint = html.escape(step_def["hint"])
        aria_label = html.escape(f"{step_def['label']} - {step_def['hint']}", quote=True)
        items_html.append(
            """
            <div class="mck-flow-stepper__item" data-state="{state}" role="listitem" aria-label="{aria}">
              <span class="mck-flow-stepper__indicator" aria-hidden="true">{icon}</span>
              <div class="mck-flow-stepper__label">
                <span class="jp">{label}</span>
                <span class="en">{label_en}</span>
              </div>
              <p class="mck-flow-stepper__hint">{hint}</p>
            </div>
            """.format(
                state=state,
                aria=aria_label,
                icon=step_def.get("icon", icon_svg("info")),
                label=label_jp,
                label_en=label_en,
                hint=hint,
            )
        )
        if idx < len(FLOW_STEP_SEQUENCE) - 1:
            connector_state = (
                "complete"
                if state_lookup.get(key) in ("complete", "active")
                else "pending"
            )
            items_html.append(
                """
                <div class="mck-flow-stepper__connector" data-state="{state}" aria-hidden="true"></div>
                """.format(state=connector_state)
            )

    st.markdown(
        """
        <div class="mck-flow-stepper" role="list" aria-label="主要作業ステップ">
          {items}
        </div>
        """.format(items="".join(items_html)),
        unsafe_allow_html=True,
    )


def render_breadcrumbs(category_key: str, page_key: str) -> None:
    page_meta = SIDEBAR_PAGE_LOOKUP.get(page_key)
    if not page_meta:
        return

    category_meta = SIDEBAR_CATEGORY_STYLES.get(category_key, {})
    category_label = category_meta.get("label", category_key or "")
    page_title = page_meta.get("title") or page_meta.get("page") or page_key

    trail_items: List[tuple[str, bool]] = [("ホーム", False)]
    if category_label:
        trail_items.append((category_label, False))
    trail_items.append((page_title, True))

    trail_html: List[str] = []
    for idx, (label, is_current) in enumerate(trail_items):
        item_class = "mck-breadcrumb__item"
        if is_current:
            item_class += " mck-breadcrumb__item--current"
        trail_html.append(
            f"<span class='{item_class}'>{html.escape(label)}</span>"
        )
        if idx < len(trail_items) - 1:
            trail_html.append("<span class='mck-breadcrumb__sep'>›</span>")

    category_pages = [
        meta for meta in SIDEBAR_PAGES if meta.get("category") == category_key
    ]
    source_pages = category_pages or SIDEBAR_PAGES
    total = len(source_pages)
    position = 1
    for idx, meta in enumerate(source_pages):
        if meta.get("key") == page_key:
            position = idx + 1
            break

    meta_text = f"ページ {position} / {total}" if total else ""
    breadcrumb_html = (
        "<div class='mck-breadcrumb'>"
        f"<div class='mck-breadcrumb__trail'>{''.join(trail_html)}</div>"
    )
    if meta_text:
        breadcrumb_html += f"<div class='mck-breadcrumb__meta'>{html.escape(meta_text)}</div>"
    else:
        breadcrumb_html += "<div class='mck-breadcrumb__meta'></div>"
    breadcrumb_html += "</div>"
    st.markdown(breadcrumb_html, unsafe_allow_html=True)

    desc_parts: List[str] = []
    tagline = page_meta.get("tagline")
    if tagline:
        desc_parts.append(tagline)
    category_desc = category_meta.get("description")
    if category_desc and category_desc not in desc_parts:
        desc_parts.append(category_desc)
    if desc_parts:
        desc_html = " ｜ ".join(html.escape(part) for part in desc_parts)
        st.markdown(
            f"<div class='mck-breadcrumb__desc'>{desc_html}</div>",
            unsafe_allow_html=True,
        )


def render_quick_nav_tabs(active_page_key: str) -> None:
    if not used_category_keys:
        return

    tab_labels = [
        SIDEBAR_CATEGORY_STYLES.get(cat, {}).get("label", cat) for cat in used_category_keys
    ]
    tabs = st.tabs(tab_labels)
    for tab, cat in zip(tabs, used_category_keys):
        category_meta = SIDEBAR_CATEGORY_STYLES.get(cat, {})
        category_desc = category_meta.get("description", "")
        pages_in_category = [
            meta for meta in SIDEBAR_PAGES if meta.get("category") == cat
        ]
        with tab:
            if category_desc:
                st.caption(category_desc)
            for page_meta in pages_in_category:
                page_key = page_meta.get("key")
                button_label_parts = [
                    (page_meta.get("icon") or "").strip(),
                    page_meta.get("title") or page_meta.get("page") or page_key,
                ]
                button_label = " ".join(part for part in button_label_parts if part)
                clicked = st.button(
                    button_label,
                    key=f"quick_nav_{cat}_{page_key}",
                    use_container_width=True,
                    disabled=page_key == active_page_key,
                )
                if clicked:
                    set_active_page(page_key, rerun_on_lock=True)
                    st.rerun()
                caption_parts: List[str] = []
                tagline = page_meta.get("tagline")
                if tagline:
                    caption_parts.append(tagline)
                if page_key == active_page_key:
                    caption_parts.append("現在表示中")
                if caption_parts:
                    st.caption(" ｜ ".join(caption_parts))
if st.session_state.get("tour_active", True) and TOUR_STEPS:
    initial_idx = max(0, min(st.session_state.get("tour_step_index", 0), len(TOUR_STEPS) - 1))
    default_key = TOUR_STEPS[initial_idx]["nav_key"]
    if default_key not in NAV_KEYS:
        default_key = NAV_KEYS[0]
else:
    default_key = NAV_KEYS[0]

if "nav_page" not in st.session_state:
    set_active_page(default_key)

pending_nav_page = st.session_state.pop(PENDING_NAV_PAGE_KEY, None)
if pending_nav_page in NAV_KEYS:
    set_active_page(pending_nav_page)

if "tour_pending_nav" in st.session_state:
    pending = st.session_state.pop("tour_pending_nav")
    if pending in NAV_KEYS:
        set_active_page(pending)

current_page_key = st.session_state.get("nav_page", default_key)
current_meta = SIDEBAR_PAGE_LOOKUP.get(current_page_key, {})
default_category = current_meta.get("category")
pending_nav_category = st.session_state.pop(PENDING_NAV_CATEGORY_KEY, None)
if pending_nav_category:
    st.session_state[NAV_CATEGORY_STATE_KEY] = pending_nav_category
if NAV_CATEGORY_STATE_KEY not in st.session_state:
    if default_category:
        st.session_state[NAV_CATEGORY_STATE_KEY] = default_category
    elif used_category_keys:
        st.session_state[NAV_CATEGORY_STATE_KEY] = used_category_keys[0]

pending_nav_primary = st.session_state.pop(PENDING_NAV_PRIMARY_KEY, None)
if pending_nav_primary in PRIMARY_NAV_LOOKUP:
    st.session_state[NAV_PRIMARY_STATE_KEY] = pending_nav_primary

current_primary_default = PAGE_TO_PRIMARY_LOOKUP.get(
    current_page_key, PRIMARY_NAV_MENU[0]["key"]
)
if (
    NAV_PRIMARY_STATE_KEY not in st.session_state
    or st.session_state[NAV_PRIMARY_STATE_KEY] not in PRIMARY_NAV_LOOKUP
):
    st.session_state[NAV_PRIMARY_STATE_KEY] = current_primary_default

for item in PRIMARY_NAV_MENU:
    state_key = f"nav_sub_{item['key']}"
    pending_sub_key = f"{PENDING_NAV_SUB_PREFIX}{item['key']}"
    pending_value = st.session_state.pop(pending_sub_key, None)
    if state_key not in st.session_state and item["pages"]:
        st.session_state[state_key] = item["pages"][0]
    if pending_value is not None and pending_value in item["pages"]:
        st.session_state[state_key] = pending_value
    if current_page_key in item["pages"]:
        st.session_state[state_key] = current_page_key

def _format_primary_label(key: str) -> str:
    item = PRIMARY_NAV_LOOKUP.get(key, {})
    icon = (item.get("icon") or "").strip()
    label = item.get("label", key)
    active_page = st.session_state.get("nav_page", current_page_key)
    if item.get("pages") and len(item["pages"]) > 1 and active_page in item["pages"]:
        sub_label = NAV_TITLE_LOOKUP.get(active_page, active_page)
        combined = f"{label}｜{sub_label}" if sub_label else label
    else:
        combined = label
    return f"{icon} {combined}".strip()

primary_keys = [item["key"] for item in PRIMARY_NAV_MENU]
selected_primary = st.sidebar.radio(
    "メインメニュー",
    primary_keys,
    key=NAV_PRIMARY_STATE_KEY,
    format_func=_format_primary_label,
)

primary_item = PRIMARY_NAV_LOOKUP.get(selected_primary, PRIMARY_NAV_MENU[0])
primary_pages = primary_item.get("pages", [])
if not primary_pages:
    primary_pages = [current_page_key]

target_page_key = primary_pages[0]
sub_state_key = f"nav_sub_{selected_primary}"
if len(primary_pages) > 1:
    current_sub_value = st.session_state.get(sub_state_key, primary_pages[0])
    if current_sub_value not in primary_pages:
        current_sub_value = primary_pages[0]
        st.session_state[sub_state_key] = current_sub_value
    target_page_key = st.sidebar.selectbox(
        "表示する機能",
        primary_pages,
        key=sub_state_key,
        format_func=lambda key: NAV_TITLE_LOOKUP.get(key, key),
        help=primary_item.get("description", "このメニューに含まれる機能を選択します。"),
    )
else:
    if primary_item.get("description"):
        st.sidebar.caption(primary_item["description"])
    target_page_key = primary_pages[0]

if target_page_key not in NAV_KEYS:
    target_page_key = current_page_key

if st.session_state.get("nav_page") != target_page_key:
    set_active_page(target_page_key)

page_key = st.session_state.get("nav_page", target_page_key)
page = page_lookup[page_key]
page_meta = SIDEBAR_PAGE_LOOKUP.get(page_key, {})
current_category = page_meta.get("category")
if current_category and st.session_state.get("nav_category") != current_category:
    _queue_nav_category(current_category, rerun_on_lock=True)
if page_meta.get("tagline"):
    st.sidebar.caption(page_meta["tagline"])
current_page_key = page_key

nav_script_payload = json.dumps(
    {
        "items": PRIMARY_NAV_CLIENT_DATA,
        "activePage": page_key,
        "activePrimary": PAGE_TO_PRIMARY_LOOKUP.get(page_key, selected_primary),
    },
    ensure_ascii=False,
)
nav_script_template = """
<script>
const NAV_PRIMARY = {payload};
(function() {
    const doc = window.parent.document;
    const ensureToggle = () => {
        const root = doc.documentElement;
        if (!root) return;
        let toggle = doc.querySelector('.mobile-nav-toggle');
        if (!toggle) {
            toggle = doc.createElement('button');
            toggle.className = 'mobile-nav-toggle';
            toggle.type = 'button';
            toggle.setAttribute('aria-label', 'メニューを開閉');
            toggle.innerHTML = '<span></span><span></span><span></span>';
            toggle.addEventListener('click', () => {
                root.classList.toggle('nav-open');
            });
            const host = doc.querySelector('header') || doc.body;
            host.appendChild(toggle);
        }
        let overlay = doc.querySelector('.nav-overlay');
        if (!overlay) {
            overlay = doc.createElement('div');
            overlay.className = 'nav-overlay';
            doc.body.appendChild(overlay);
            overlay.addEventListener('click', () => {
                root.classList.remove('nav-open');
            });
        }
    };
    const apply = (attempt = 0) => {
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) {
            if (attempt < 12) {
                setTimeout(() => apply(attempt + 1), 140);
            }
            return;
        }
        const radioGroup = sidebar.querySelector('div[data-baseweb="radio"]');
        if (radioGroup) {
            const labels = Array.from(radioGroup.querySelectorAll('label'));
            const metaByKey = Object.fromEntries(NAV_PRIMARY.items.map((item) => [item.key, item]));
            const updateActive = () => {
                labels.forEach((label) => {
                    const input = label.querySelector('input[type="radio"]');
                    if (!input) return;
                    const key = input.value;
                    label.classList.toggle('nav-pill--active', input.checked || key === NAV_PRIMARY.activePrimary);
                });
            };
            labels.forEach((label) => {
                const input = label.querySelector('input[type="radio"]');
                if (!input) return;
                const key = input.value;
                const meta = metaByKey[key];
                if (!meta) return;
                label.classList.add('nav-pill');
                let iconSpan = label.querySelector('.nav-pill__icon');
                if (!iconSpan) {
                    iconSpan = doc.createElement('span');
                    iconSpan.className = 'nav-pill__icon';
                    iconSpan.setAttribute('aria-hidden', 'true');
                    label.insertBefore(iconSpan, label.firstChild);
                }
                iconSpan.textContent = meta.icon || '';
                let bodySpan = label.querySelector('.nav-pill__body');
                if (!bodySpan) {
                    bodySpan = doc.createElement('span');
                    bodySpan.className = 'nav-pill__body';
                    while (label.childNodes.length > 1) {
                        bodySpan.appendChild(label.childNodes[1]);
                    }
                    label.appendChild(bodySpan);
                }
                let titleEl = bodySpan.querySelector('.nav-pill__title');
                if (!titleEl) {
                    titleEl = doc.createElement('span');
                    titleEl.className = 'nav-pill__title';
                    bodySpan.insertBefore(titleEl, bodySpan.firstChild);
                }
                titleEl.textContent = meta.label || '';
                let descEl = bodySpan.querySelector('.nav-pill__desc');
                if (!descEl) {
                    descEl = doc.createElement('span');
                    descEl.className = 'nav-pill__desc';
                    bodySpan.appendChild(descEl);
                }
                const activePage = NAV_PRIMARY.activePage;
                if (meta.pages && meta.pages.length > 1 && meta.pages.includes(activePage)) {
                    const subLabel = meta.page_titles ? (meta.page_titles[activePage] || '') : '';
                    descEl.textContent = subLabel;
                    label.classList.add('nav-pill--has-sub');
                } else {
                    descEl.textContent = '';
                    label.classList.remove('nav-pill--has-sub');
                }
                const tooltipParts = [];
                if (meta.description) {
                    tooltipParts.push(meta.description);
                }
                if (meta.pages && meta.pages.length > 1) {
                    const titles = meta.pages
                        .map((page) => (meta.page_titles ? (meta.page_titles[page] || '') : ''))
                        .filter(Boolean);
                    if (titles.length) {
                        tooltipParts.push(titles.join(' / '));
                    }
                } else if (meta.pages && meta.pages.length === 1) {
                    const single = meta.pages[0];
                    const tip = meta.page_tooltips ? (meta.page_tooltips[single] || '') : '';
                    if (tip) {
                        tooltipParts.push(tip);
                    }
                }
                const tooltipText = tooltipParts.join('\n').trim();
                const ariaLabel = tooltipText ? `${meta.label}: ${tooltipText}` : meta.label;
                label.setAttribute('title', tooltipText || meta.label || '');
                label.dataset.tooltip = tooltipText;
                input.setAttribute('aria-label', ariaLabel || meta.label || '');
                input.setAttribute('title', tooltipText || meta.label || '');
                if (!input.dataset.navEnhanced) {
                    input.addEventListener('change', () => {
                        updateActive();
                        doc.documentElement.classList.remove('nav-open');
                    });
                    input.dataset.navEnhanced = 'true';
                }
            });
            updateActive();
        }
        const selects = Array.from(sidebar.querySelectorAll('select'));
        selects.forEach((select) => {
            if (!select.dataset.navEnhanced) {
                select.addEventListener('change', () => {
                    doc.documentElement.classList.remove('nav-open');
                });
                select.dataset.navEnhanced = 'true';
            }
        });
        const root = doc.documentElement;
        if (root && NAV_PRIMARY.activePage) {
            if (root.getAttribute('data-active-page') !== NAV_PRIMARY.activePage) {
                root.setAttribute('data-active-page', NAV_PRIMARY.activePage);
            }
            const container = doc.querySelector('main .block-container');
            if (container) {
                container.classList.remove('page-transition-fade');
                void container.offsetWidth;
                container.classList.add('page-transition-fade');
            }
        }
    };
    ensureToggle();
    apply();
})();
</script>
"""
nav_script = nav_script_template.replace("{payload}", nav_script_payload)
components.html(nav_script, height=0)

if st.session_state.get("tour_active", True):
    for idx, step in enumerate(TOUR_STEPS):
        if step["nav_key"] == page_key:
            st.session_state.tour_step_index = idx
            break


latest_month = render_sidebar_summary()

sidebar_state: Dict[str, object] = {}
year_df = st.session_state.get("data_year")

if year_df is not None and not year_df.empty:
    if page == "ダッシュボード":
        pass
    elif page == "ランキング":
        st.sidebar.subheader("期間選択")
        sidebar_state["rank_end_month"] = end_month_selector(
            year_df,
            key="end_month_rank",
            label="ランキング対象月",
            sidebar=True,
        )
        if sidebar_state["rank_end_month"]:
            st.session_state.filters["end_month"] = sidebar_state["rank_end_month"]
        st.sidebar.subheader("評価指標")
        metric_options = [
            ("年計（12カ月累計）", "year_sum"),
            ("前年同月比（YoY）", "yoy"),
            ("前月差（Δ）", "delta"),
            ("直近傾き（β）", "slope_beta"),
        ]
        selected_metric = st.sidebar.selectbox(
            "表示指標",
            metric_options,
            format_func=lambda opt: opt[0],
            key="sidebar_rank_metric",
        )
        sidebar_state["rank_metric"] = selected_metric[1]
        order_options = [
            ("降順 (大きい順)", "desc"),
            ("昇順 (小さい順)", "asc"),
        ]
        selected_order = st.sidebar.selectbox(
            "並び順",
            order_options,
            format_func=lambda opt: opt[0],
            key="sidebar_rank_order",
        )
        sidebar_state["rank_order"] = selected_order[1]
        sidebar_state["rank_hide_zero"] = st.sidebar.checkbox(
            "年計ゼロを除外",
            value=True,
            key="sidebar_rank_hide_zero",
        )
    elif page == "比較ビュー":
        st.sidebar.subheader("期間選択")
        sidebar_state["compare_end_month"] = end_month_selector(
            year_df,
            key="compare_end_month",
            label="比較対象月",
            sidebar=True,
        )
        if sidebar_state["compare_end_month"]:
            st.session_state.filters["end_month"] = sidebar_state["compare_end_month"]
    elif page == "SKU詳細":
        st.sidebar.subheader("期間選択")
        sidebar_state["detail_end_month"] = end_month_selector(
            year_df,
            key="end_month_detail",
            label="詳細確認月",
            sidebar=True,
        )
        if sidebar_state["detail_end_month"]:
            st.session_state.filters["end_month"] = sidebar_state["detail_end_month"]
    elif page == "相関分析":
        st.sidebar.subheader("期間選択")
        sidebar_state["corr_end_month"] = end_month_selector(
            year_df,
            key="corr_end_month",
            label="分析対象月",
            sidebar=True,
        )
        if sidebar_state["corr_end_month"]:
            st.session_state.filters["end_month"] = sidebar_state["corr_end_month"]
    elif page == "アラート":
        st.sidebar.subheader("期間選択")
        sidebar_state["alert_end_month"] = end_month_selector(
            year_df,
            key="end_month_alert",
            label="評価対象月",
            sidebar=True,
        )
        if sidebar_state["alert_end_month"]:
            st.session_state.filters["end_month"] = sidebar_state["alert_end_month"]

st.sidebar.divider()

with st.sidebar.expander("AIコパイロット", expanded=False):
    st.caption("最新の年計スナップショットを使って質問できます。")
    default_question = st.session_state.get(
        "copilot_question",
        "直近の売上トレンドと注目SKUを教えて",
    )
    st.text_area(
        "聞きたいこと",
        value=default_question,
        key="copilot_question",
        height=90,
        placeholder="例：前年同月比が高いSKUや、下落しているSKUを教えて",
        help="AIに知りたい内容を入力します。質問例もそのまま実行できます。",
    )
    focus = st.selectbox(
        "フォーカス",
        ["全体サマリー", "伸びているSKU", "苦戦しているSKU"],
        key="copilot_focus",
        help="回答の視点を選択します。伸びているSKUを選ぶと成長している商品に絞った要約が得られます。",
    )
    if st.button(
        "AIに質問",
        key="ask_ai",
        use_container_width=True,
        help="年計スナップショットに基づくAI分析を実行します。",
    ):
        question = st.session_state.get("copilot_question", "").strip()
        if not question:
            st.warning("質問を入力してください。")
        else:
            context = build_copilot_context(focus, end_month=latest_month)
            answer = _ai_answer(question, context)
            st.session_state.copilot_answer = answer
            st.session_state.copilot_context = context
    if st.session_state.copilot_answer:
        st.markdown(
            f"<div class='mck-ai-answer'><strong>AI回答</strong><br>{st.session_state.copilot_answer}</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.copilot_context:
            st.caption("コンテキスト: " + clip_text(st.session_state.copilot_context, 220))
st.sidebar.divider()

render_app_hero()

render_process_step_bar(page_key)

render_onboarding_modal()

render_tour_banner()

render_step_guide(page_key)

active_category = (
    st.session_state.get("nav_category")
    or page_meta.get("category")
    or ""
)
render_breadcrumbs(active_category, page_key)
render_quick_nav_tabs(page_key)

if st.session_state.get("sample_data_notice"):
    notice_text = st.session_state.get("sample_data_message") or (
        "サンプルデータを読み込みました。ダッシュボードからすぐに分析を確認できます。"
    )
    st.success(notice_text)
    st.session_state.sample_data_notice = False
    st.session_state.sample_data_message = ""

if (
    st.session_state.data_year is None
    or st.session_state.data_monthly is None
):
    st.info(
        "左メニューの「データ取込」からCSVまたはExcelファイルをアップロードしてください。\n"
        "Upload CSV or Excel files from the “Data Import” menu on the left.\n\n"
        "サンプルテンプレートを活用すると、初期セットアップを数分で体験できます。\n"
        "Use the sample template to experience the initial setup in minutes."
    )
    with st.expander("アップロード前のチェックリスト / Pre-upload checklist", expanded=False):
        st.markdown(
            "- 行は商品（または費目）単位で、列には12ヶ月以上の月度を配置してください。\n"
            "  - Ensure each row represents an item or cost category and include at least 12 months of columns.\n"
            "- 月度列名は `2023-01` や `2023/01/01` など日付として解釈できる形式にしてください。\n"
            "  - Use month headers that can be parsed as dates (e.g., `2023-01`).\n"
            "- 数値は円単位で入力し、欠損がある場合は空欄または0で埋めてください。\n"
            "  - Provide values in JPY and fill missing months with blank cells or 0."
        )
    render_getting_started_intro()
    render_sample_data_hub()

# ---------------- Pages ----------------

# 1) データ取込
if page == "データ取込":
    section_header(
        "データ取込", "ファイルのマッピングと品質チェックを行います。", icon="📥"
    )

    render_import_stepper()

    template_options = [
        key for key in INDUSTRY_TEMPLATE_ORDER if key in INDUSTRY_TEMPLATES
    ]
    active_template = get_active_template_key()
    if active_template not in template_options:
        template_options.append(active_template)
    template_index = (
        template_options.index(active_template)
        if active_template in template_options
        else 0
    )

    template_config = get_template_config(active_template)

    with import_section(
        1, "テンプレート選択", "Template Selection", icon_svg("template"), accent="template"
    ):
        st.caption(
            """業種テンプレートを選ぶと推奨科目や指標、閾値が自動セットされます。
Select an industry template to preload recommended fields, metrics, and thresholds."""
        )
        render_icon_label(
            "template",
            "業種別テンプレート",
            "Industry template",
            help_text="業種を選ぶと推奨科目や指標、閾値が自動セットされ入力工数を削減できます。",
        )
        selected_template = st.selectbox(
            "業種別テンプレート / Industry template",
            template_options,
            index=template_index,
            format_func=lambda key: INDUSTRY_TEMPLATES.get(key, {}).get(
                "label", key
            ),
            help="業種を選ぶと推奨科目や指標、閾値が自動セットされ入力工数を削減できます。/ Choosing an industry applies recommended KPIs and thresholds automatically.",
            label_visibility="collapsed",
        )
        if selected_template != active_template:
            apply_industry_template(selected_template)
            template_label = INDUSTRY_TEMPLATES.get(selected_template, {}).get(
                "label", selected_template
            )
            st.success(
                f"""{template_label} テンプレートを適用しました。閾値と推奨KPIが更新されています。
Applied the {template_label} template. Thresholds and KPI presets are refreshed."""
            )
            active_template = selected_template
            template_config = get_template_config(active_template)
        goal_text = template_config.get("goal")
        if goal_text:
            st.caption(
                f"{goal_text}<br><span class='mck-import-section__hint'>Goal: Reduce manual setup time with predefined metrics.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                """テンプレート適用で入力工数を約50％削減（30分→15分以内）することを目指します。
Applying the template aims to cut manual setup time by about 50% (30 min → under 15 min)."""
            )

    with import_section(
        2, "推奨項目と指標", "Recommended Fields & Metrics", icon_svg("metrics"), accent="metrics"
    ):
        st.caption(
            """推奨項目・指標をチェックリストとして活用し、入力漏れを防ぎましょう。
Use the recommended fields and metrics as a checklist to prevent omissions."""
        )
        fields = template_config.get("fields", [])
        metrics_list = template_config.get("recommended_metrics", [])
        col_fields, col_metrics = st.columns([1, 2])
        with col_fields:
            render_icon_label(
                "template",
                "推奨項目",
                "Recommended columns",
                help_text="テンプレートに含めるべき列を確認し、アップロード前の抜け漏れを防ぎます。",
            )
            if fields:
                for field in fields:
                    field_html = html.escape(str(field))
                    st.markdown(
                        f"- {field_html}<br><span class='mck-import-section__hint'>Suggested column name / Recommended field</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(
                    """テンプレートに推奨項目は設定されていません。
No additional recommended columns in this template."""
                )
        with col_metrics:
            render_icon_label(
                "metrics",
                "自動計算される指標",
                "Auto-calculated metrics",
                help_text="テンプレートが想定する代表的な指標です。カードで値を確認し、ターゲットのイメージを掴みましょう。",
            )
            if metrics_list:
                metric_cards: List[Dict[str, object]] = []
                for metric in metrics_list:
                    metric_cards.append(
                        {
                            "title": metric.get("name", "指標"),
                            "subtitle": "Template KPI",
                            "value": format_template_metric(metric),
                            "icon": detect_metric_icon(metric.get("name", "")),
                            "footnote": metric.get("description", ""),
                            "tooltip": metric.get("description", ""),
                        }
                    )
                render_metric_cards(metric_cards, columns=min(2, len(metric_cards)))
            else:
                st.caption(
                    """テンプレートに紐づく指標はありません。
No auto-calculated metrics are linked to this template."""
                )
        if metrics_list:
            render_metric_bar_chart(metrics_list)
        template_bytes = build_industry_template_csv(active_template)
        render_icon_label(
            "download",
            "CSVテンプレートをダウンロード",
            "Download CSV template",
            help_text="推奨科目と月度列を含むテンプレートをダウンロードし、社内で共有できます。",
        )
        template_clicked = st.download_button(
            "CSVテンプレートをダウンロード / Download CSV template",
            data=template_bytes,
            file_name=f"{active_template}_template.csv",
            mime="text/csv",
            help="推奨科目と12ヶ月の月度列を含むテンプレートをダウンロードします。/ Download a starter CSV with recommended columns and 12 month headers.",
            key="template_csv_download",
        )
        if template_clicked:
            render_status_message(
                "completed",
                key="template_csv_downloaded",
                guide="テンプレートをチームに共有してデータ入力を統一しましょう。",
            )

    missing_policy_options = ["zero_fill", "mark_missing"]
    current_policy = st.session_state.settings.get("missing_policy", "zero_fill")
    policy_index = (
        missing_policy_options.index(current_policy)
        if current_policy in missing_policy_options
        else 0
    )

    with import_section(
        3, "ファイルアップロード", "Data Upload", icon_svg("upload"), accent="upload"
    ):
        st.markdown(
            "**Excel(.xlsx) / CSV をアップロードしてください。**<br>"
            "<span class='mck-import-section__hint'>Upload Excel (.xlsx) or CSV files that include monthly sales, cost, or inventory data.</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "列に `YYYY-MM` 形式の月度を含め、数値は半角で記録してください。<br>"
            "<span class='mck-import-section__hint'>Include month columns in YYYY-MM format and provide numeric values in standard digits.</span>",
            unsafe_allow_html=True,
        )
        if "import_layout_expanded" not in st.session_state:
            st.session_state.import_layout_expanded = True
        with st.expander(
            "CSVの列構成を確認する / Review column layout",
            expanded=st.session_state.get("import_layout_expanded", True),
        ):
            st.markdown(
                "例: `商品名, 商品コード, 2022-01, 2022-02, ...` のように名称・コード列の後に月次列を並べてください。<br>"
                "Example: place month columns after name/code columns such as `Product Name, Product Code, 2022-01, 2022-02, ...`.",
                unsafe_allow_html=True,
            )
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            render_icon_label(
                "upload",
                "ファイル選択",
                "Choose file",
                help_text="月次の売上・仕入れ・経費などを含むCSV/Excelファイルを指定します。",
            )
            file = st.file_uploader(
                "ファイル選択 / Choose file",
                type=["xlsx", "csv"],
                help="月次の売上・仕入れ・経費などを含むCSV/Excelファイルを指定します。/ Select the monthly dataset to import.",
                label_visibility="collapsed",
            )
        with col_u2:
            render_icon_label(
                "policy",
                "欠測月ポリシー",
                "Missing month policy",
                help_text="ゼロ補完は欠測を0で補完し、欠測保持は空欄のまま残して計算対象外とします。",
            )
            missing_policy = st.selectbox(
                "欠測月ポリシー / Missing month policy",
                options=missing_policy_options,
                index=policy_index,
                format_func=lambda x: "ゼロ補完(推奨) / Fill missing months with 0"
                if x == "zero_fill"
                else "欠測を保持 / Keep missing months as blank",
                help="欠測月の扱いを選択します。/ Choose how missing months should be treated during calculations.",
                label_visibility="collapsed",
            )
            st.session_state.settings["missing_policy"] = missing_policy
            st.session_state.import_policy_expanded = missing_policy == "mark_missing"

        policy_help_expanded = st.session_state.get("import_policy_expanded", False)
        with st.expander(
            "欠測月ポリシーの比較 / Missing policy guide",
            expanded=policy_help_expanded,
        ):
            st.markdown(
                "- **ゼロ補完**: 欠測月を 0 で補完し、年計の継続性を保ちます。<br>"
                "- **欠測保持**: 欠測月を空欄のまま残し、該当期間の年計を除外します。",
                unsafe_allow_html=True,
            )

        if file is not None:
            try:
                with loading_message("ファイルを読み込んでいます…"):
                    if file.name.lower().endswith(".csv"):
                        df_raw = pd.read_csv(file)
                    else:
                        df_raw = pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                st.error(
                    f"読込エラー: {e}\nFailed to load the file. Please check the format and encoding."
                )
                st.stop()

            with st.expander(
                "アップロードプレビュー（先頭100行） / Preview first 100 rows",
                expanded=True,
            ):
                st.dataframe(df_raw.head(100), use_container_width=True)

            cols = df_raw.columns.tolist()
            product_name_col = st.selectbox(
                "商品名列の選択 / Select product name column",
                options=cols,
                index=0,
                help="可視化に使用する名称列を選択します。/ Choose the column to display as product names.",
            )
            product_code_col = st.selectbox(
                "商品コード列の選択（任意） / Select product code column (optional)",
                options=["<なし>"] + cols,
                index=0,
                help="SKUコードなど識別子の列がある場合に選択します。/ Select an identifier column if available.",
            )
            code_col = None if product_code_col == "<なし>" else product_code_col

            st.markdown("<div class='mobile-sticky-actions'>", unsafe_allow_html=True)
            convert_clicked = st.button(
                "変換＆取込 / Convert & ingest",
                type="primary",
                help="年計・YoY・Δを自動計算し、ダッシュボード各ページで利用できる形式に整えます。/ Convert the file into yearly KPIs for the dashboard.",
                use_container_width=True,
            )
            st.markdown(
                "<p class='mobile-action-caption'>テンプレートの推奨科目を使うと入力から取り込みまでを15分以内で完了できます。<br>"
                "Using the recommended template helps you finish setup within 15 minutes.</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if convert_clicked:
                try:
                    with loading_message("年計データを計算中…"):
                        long_df, year_df = ingest_wide_dataframe(
                            df_raw,
                            product_name_col=product_name_col,
                            product_code_col=code_col,
                        )

                    st.success(
                        """取込完了。ダッシュボードへ移動して可視化を確認してください。
Import completed. Open the dashboard pages to review the visuals."""
                    )
                    quality_summary = {
                        "missing": int(long_df["is_missing"].sum()),
                        "total": int(len(long_df)),
                        "sku_count": int(long_df["product_code"].nunique()),
                        "period_start": str(long_df["month"].min()),
                        "period_end": str(long_df["month"].max()),
                    }
                    st.session_state.import_quality_summary = quality_summary
                    st.session_state.import_uploaded_file_name = file.name
                    st.session_state.import_last_uploaded = datetime.now().isoformat()
                    st.session_state.import_report_completed = False
                except Exception as e:
                    st.exception(e)

            st.session_state.import_layout_expanded = False
        else:
            st.session_state.import_layout_expanded = True
            st.caption(
                """CSV/XLSXの形式チェックとカラムマッピングを行ったあと、年計データを生成します。
After validating and mapping the CSV/XLSX, yearly KPIs will be calculated automatically."""
            )

    quality_summary = st.session_state.get("import_quality_summary")
    data_year = st.session_state.get("data_year")

    with import_section(
        4, "データチェック", "Data Quality Review", icon_svg("quality"), accent="quality"
    ):
        latest_month = None
        if data_year is not None and not getattr(data_year, "empty", True) and "month" in data_year.columns:
            try:
                latest_month = data_year["month"].astype(str).max()
            except Exception:
                latest_month = None

        if quality_summary:
            render_quality_summary_panel(quality_summary)
            if data_year is not None and not data_year.empty:
                render_dataset_metric_cards(data_year, latest_month)
                render_icon_label(
                    "download",
                    "年計テーブルをCSVでダウンロード",
                    "Download yearly table",
                    help_text="年計やYoYの計算結果をCSVで保存し、他部署と共有できます。",
                )
                download_clicked = st.download_button(
                    "年計テーブルをCSVでダウンロード / Download yearly table (CSV)",
                    data=data_year.to_csv(index=False).encode("utf-8-sig"),
                    file_name="year_rolling.csv",
                    mime="text/csv",
                    help="年計やYoYなどの計算結果をCSVで保存し、他システムと共有できます。/ Export yearly KPIs as CSV for sharing.",
                )
                if download_clicked:
                    st.session_state.import_report_completed = True
                    render_status_message(
                        "completed",
                        key="import_year_csv_ready",
                        guide="ダウンロードしたCSVを共有し、最新の年計指標を連携できます。",
                    )
            st.caption(
                """ダッシュボードやランキングに移動して、AIサマリーやPDF出力を活用しましょう。
Move to the dashboard or ranking pages to use AI summaries and PDF exports."""
            )
        elif data_year is not None and not data_year.empty:
            render_dataset_metric_cards(data_year, latest_month)
            render_icon_label(
                "download",
                "年計テーブルをCSVでダウンロード",
                "Download yearly table",
                help_text="品質サマリー計算中でも最新の年計テーブルを取得できます。",
            )
            download_clicked = st.download_button(
                "年計テーブルをCSVでダウンロード / Download yearly table (CSV)",
                data=data_year.to_csv(index=False).encode("utf-8-sig"),
                file_name="year_rolling.csv",
                mime="text/csv",
            )
            if download_clicked:
                st.session_state.import_report_completed = True
                render_status_message(
                    "completed",
                    key="import_year_csv_ready",
                    guide="ダウンロードしたCSVを共有し、最新の年計指標を連携できます。",
                )
            st.caption(
                """現在のデータセットに基づく品質サマリーを取得しています。
Quality metrics are available for the current dataset."""
            )
        else:
            st.caption(
                """ファイルをアップロードすると欠測状況や期間のサマリーが表示されます。
Upload a file to view missing values and coverage summaries here."""
            )

# 2) ダッシュボード
elif page == "ダッシュボード":
    require_data()
    section_header("ダッシュボード", "年計KPIと成長トレンドを俯瞰します。", icon="📈")

    year_df = st.session_state.data_year
    data_monthly = st.session_state.data_monthly
    template_key = get_active_template_key()
    template_config = get_template_config(template_key)
    profile = template_config.get("financial_profile", {})

    months_available = _sorted_months(year_df)
    if not months_available:
        st.warning("表示できる月次データがありません。データ取込を確認してください。")
        st.stop()

    latest_month = months_available[-1]
    period_options = [12, 24, 36]
    default_period = st.session_state.settings.get("window", 12)
    if default_period not in period_options:
        default_period = 12

    unit_options = list(UNIT_MAP.keys())
    default_unit = st.session_state.settings.get("currency_unit", "円")
    if default_unit not in unit_options:
        default_unit = unit_options[0]

    store_options, store_column = _resolve_store_options(data_monthly)
    default_store = st.session_state.get("dashboard_store", store_options[0])
    if default_store not in store_options:
        default_store = store_options[0]
        st.session_state.dashboard_store = default_store

    control_cols = st.columns([5.0, 1.5, 1.4, 1.4, 1.4])

    with control_cols[1]:
        current_end = st.session_state.get("end_month_dash", latest_month)
        if current_end not in months_available:
            current_end = latest_month
        end_index = months_available.index(current_end)
        end_m = st.selectbox(
            "表示月",
            months_available,
            index=end_index,
            key="end_month_dash",
        )

    with control_cols[2]:
        period_index = period_options.index(default_period)
        period_value = st.selectbox(
            "期間",
            period_options,
            index=period_index,
            key="sidebar_period",
            format_func=lambda v: f"{v}ヶ月",
        )

    with control_cols[3]:
        store_index = store_options.index(default_store)
        store_value = st.selectbox(
            "店舗",
            store_options,
            index=store_index,
            key="dashboard_store",
        )

    with control_cols[4]:
        unit_index = unit_options.index(default_unit)
        unit_value = st.selectbox(
            "単位",
            unit_options,
            index=unit_index,
            key="sidebar_unit",
        )

    active_end_month = end_m or latest_month
    sidebar_state["dashboard_end_month"] = active_end_month

    st.session_state.settings["window"] = period_value
    st.session_state.settings["currency_unit"] = unit_value
    st.session_state.filters.update(
        {
            "period": period_value,
            "currency_unit": unit_value,
            "store": store_value,
            "end_month": active_end_month,
        }
    )

    filtered_monthly = _filter_monthly_data(
        data_monthly,
        end_month=active_end_month,
        months=period_value,
        store_column=store_column,
        store_value=store_value,
    )
    monthly_trend = _prepare_monthly_trend(filtered_monthly)
    channel_column = _detect_channel_column(filtered_monthly)

    kpi = aggregate_overview(year_df, active_end_month)
    financial_snapshot = _compute_financial_snapshot(
        year_df, active_end_month, profile
    )
    prev_month = _previous_month(months_available, active_end_month)
    prev_snapshot = _compute_financial_snapshot(year_df, prev_month, profile)

    st.markdown(
        f"**表示条件**：{store_value} ｜ 過去 {period_value} ヶ月 ｜ 単位 {unit_value}"
    )

    kpi_cols = st.columns(3)
    total_sales = kpi.get("total_year_sum")
    delta_sales = kpi.get("delta")
    kpi_cols[0].metric(
        "売上総額 (年計)",
        format_amount(total_sales, unit_value),
        delta=format_amount(delta_sales, unit_value) if delta_sales is not None else None,
    )

    gross_margin = financial_snapshot.get("gross_margin_rate")
    prev_margin = prev_snapshot.get("gross_margin_rate")
    margin_label = (
        f"{gross_margin * 100:.1f}%" if gross_margin is not None else "—"
    )
    margin_delta = None
    if gross_margin is not None and prev_margin is not None:
        margin_delta = f"{(gross_margin - prev_margin) * 100:.1f}pt"
    kpi_cols[1].metric("粗利率", margin_label, delta=margin_delta)

    cash_balance = financial_snapshot.get("cash_balance")
    prev_cash = prev_snapshot.get("cash_balance")
    cash_delta = None
    if cash_balance is not None and prev_cash is not None:
        cash_delta = cash_balance - prev_cash
    kpi_cols[2].metric(
        "キャッシュ残高",
        format_amount(cash_balance, unit_value),
        delta=(
            format_amount(cash_delta, unit_value) if cash_delta is not None else None
        ),
    )

    tabs = st.tabs(["売上", "粗利", "在庫", "資金"])
    with tabs[0]:
        _render_sales_tab(
            filtered_monthly=filtered_monthly,
            monthly_trend=monthly_trend,
            unit=unit_value,
            end_month=active_end_month,
            year_df=year_df,
            channel_column=channel_column,
        )

    with tabs[1]:
        _render_gross_profit_tab(
            filtered_monthly=filtered_monthly,
            monthly_trend=monthly_trend,
            unit=unit_value,
            end_month=active_end_month,
            profile=profile,
            year_df=year_df,
        )

    with tabs[2]:
        _render_inventory_tab(
            year_df=year_df,
            financial_snapshot=financial_snapshot,
            unit=unit_value,
            end_month=active_end_month,
            profile=profile,
        )

    with tabs[3]:
        _render_funds_tab(
            year_df=year_df,
            financial_snapshot=financial_snapshot,
            unit=unit_value,
            end_month=active_end_month,
            profile=profile,
        )

elif page == "ランキング":
    require_data()
    section_header("ランキング", "上位と下位のSKUを瞬時に把握します。", icon="🏆")
    end_m = sidebar_state.get("rank_end_month") or latest_month
    metric = sidebar_state.get("rank_metric", "year_sum")
    order = sidebar_state.get("rank_order", "desc")
    hide_zero = sidebar_state.get("rank_hide_zero", True)

    ai_on = st.toggle(
        "AIサマリー",
        value=st.session_state.get("rank_ai_toggle", False),
        key="rank_ai_toggle",
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    snap = st.session_state.data_year[
        st.session_state.data_year["month"] == end_m
    ].copy()
    total = len(snap)
    zero_cnt = int((snap["year_sum"] == 0).sum())
    if hide_zero:
        snap = snap[snap["year_sum"] > 0]
    snap = snap.dropna(subset=[metric])
    snap = snap.sort_values(metric, ascending=(order == "asc"))
    st.caption(f"除外 {zero_cnt} 件 / 全 {total} 件")

    fig_bar = px.bar(snap.head(20), x="product_name", y=metric)
    fig_bar = apply_elegant_theme(
        fig_bar, theme=st.session_state.get("ui_theme", "light")
    )
    render_plotly_with_spinner(
        fig_bar, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
    )

    with st.expander("AIサマリー", expanded=ai_on):
        if ai_on and not snap.empty:
            st.info(_ai_sum_df(snap[["year_sum", "yoy", "delta"]].head(200)))
            st.caption(_ai_comment("上位と下位の入替やYoYの極端値に注意"))

    st.dataframe(
        snap[
            ["product_code", "product_name", "year_sum", "yoy", "delta", "slope_beta"]
        ].head(100),
        use_container_width=True,
    )

    csv_clicked = st.download_button(
        "CSVダウンロード",
        data=snap.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ranking_{metric}_{end_m}.csv",
        mime="text/csv",
        key="ranking_csv_download",
    )
    if csv_clicked:
        render_status_message(
            "completed",
            key="ranking_csv_downloaded",
            guide="ランキングCSVを共有して商談準備に活用してください。",
        )
    excel_clicked = st.download_button(
        "Excelダウンロード",
        data=download_excel(snap, f"ranking_{metric}_{end_m}.xlsx"),
        file_name=f"ranking_{metric}_{end_m}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="ranking_excel_download",
    )
    if excel_clicked:
        render_status_message(
            "completed",
            key="ranking_excel_downloaded",
            guide="Excel出力を使って詳細な並び替えや共有を行いましょう。",
        )

    # 4) 比較ビュー（マルチ商品バンド）
elif page == "比較ビュー":
    require_data()
    section_header("マルチ商品比較", "条件を柔軟に切り替えてSKUを重ね合わせます。", icon="🔍")
    params = st.session_state.compare_params
    year_df = st.session_state.data_year
    end_m = sidebar_state.get("compare_end_month") or latest_month

    snapshot = latest_yearsum_snapshot(year_df, end_m)
    snapshot["display_name"] = snapshot["product_name"].fillna(snapshot["product_code"])

    search = st.text_input("検索ボックス", "")
    if search:
        snapshot = snapshot[
            snapshot["display_name"].str.contains(search, case=False, na=False)
        ]
    # ---- 操作バー＋グラフ密着カード ----

    band_params_initial = params.get("band_params", {})
    band_params = band_params_initial
    amount_slider_cfg = None
    max_amount = int(snapshot["year_sum"].max()) if not snapshot.empty else 0
    low0 = int(
        band_params_initial.get(
            "low_amount", int(snapshot["year_sum"].min()) if not snapshot.empty else 0
        )
    )
    high0 = int(band_params_initial.get("high_amount", max_amount))

    st.markdown(
        """
  <style>
  .chart-card { position: relative; margin:.35rem 0 1.2rem; border-radius:16px;
    border:1px solid var(--border, rgba(var(--primary-rgb,11,31,59),0.18)); background:var(--panel,#ffffff);
    box-shadow:0 16px 32px rgba(var(--primary-rgb,11,31,59),0.08); }
  .chart-toolbar { position: sticky; top:-1px; z-index:5;
    display:flex; gap:.6rem; flex-wrap:wrap; align-items:center;
    padding:.45rem .75rem; background: linear-gradient(180deg, rgba(var(--accent-rgb,30,136,229),0.08), rgba(var(--accent-rgb,30,136,229),0.02));
    border-bottom:1px solid var(--border, rgba(var(--primary-rgb,11,31,59),0.18)); }
  /* Streamlit標準の下マージンを除去（ここが距離の主因） */
  .chart-toolbar .stRadio, .chart-toolbar .stSelectbox, .chart-toolbar .stSlider,
  .chart-toolbar .stMultiSelect, .chart-toolbar .stCheckbox { margin-bottom:0 !important; }
  .chart-toolbar .stRadio > label, .chart-toolbar .stCheckbox > label { color:var(--ink,var(--primary,#0B1F3B)); font-weight:600; }
  .chart-toolbar .stSlider label { color:var(--ink,var(--primary,#0B1F3B)); }
  .chart-body { padding:.15rem .4rem .4rem; }
  </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<section class="chart-card" id="line-compare">', unsafe_allow_html=True
    )

    st.markdown('<div class="chart-toolbar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 1.1, 1.0, 0.9])
    with c1:
        period = st.radio(
            "期間", ["12ヶ月", "24ヶ月", "36ヶ月"], horizontal=True, index=1
        )
    with c2:
        node_mode = st.radio(
            "ノード表示",
            ["自動", "主要ノードのみ", "すべて", "非表示"],
            horizontal=True,
            index=0,
        )
    with c3:
        hover_mode = st.radio(
            "ホバー", ["個別", "同月まとめ"], horizontal=True, index=0
        )
    with c4:
        op_mode = st.radio("操作", ["パン", "ズーム", "選択"], horizontal=True, index=0)
    with c5:
        peak_on = st.checkbox("ピーク表示", value=False)

    c6, c7, c8 = st.columns([2.0, 1.9, 1.6])
    with c6:
        band_mode = st.radio(
            "バンド",
            ["金額指定", "商品指定(2)", "パーセンタイル", "順位帯", "ターゲット近傍"],
            horizontal=True,
            index=[
                "金額指定",
                "商品指定(2)",
                "パーセンタイル",
                "順位帯",
                "ターゲット近傍",
            ].index(params.get("band_mode", "金額指定")),
        )
    with c7:
        if band_mode == "金額指定":
            if not snapshot.empty:
                unit_scale, unit_label = choose_amount_slider_unit(max_amount)
                slider_max = int(
                    math.ceil(
                        max(
                            max_amount,
                            band_params_initial.get("high_amount", high0),
                        )
                        / unit_scale
                    )
                )
                slider_max = max(slider_max, 1)

                default_low = int(
                    round(band_params_initial.get("low_amount", low0) / unit_scale)
                )
                default_high = int(
                    round(band_params_initial.get("high_amount", high0) / unit_scale)
                )
                default_low = max(0, min(default_low, slider_max))
                default_high = max(default_low, min(default_high, slider_max))

                step = nice_slider_step(slider_max)

                amount_slider_cfg = dict(
                    label=f"金額レンジ（{unit_label}単位）",
                    min_value=0,
                    max_value=slider_max,
                    value=(default_low, default_high),
                    step=step,
                    unit_scale=unit_scale,
                    unit_label=unit_label,
                    max_amount=max_amount,
                )
            else:
                band_params = {"low_amount": low0, "high_amount": high0}
        elif band_mode == "商品指定(2)":
            if not snapshot.empty:
                opts = (
                    snapshot["product_code"].fillna("")
                    + " | "
                    + snapshot["display_name"].fillna("")
                ).tolist()
                opts = [o for o in opts if o.strip() != "|"]
                prod_a = st.selectbox("商品A", opts, index=0)
                prod_b = st.selectbox("商品B", opts, index=1 if len(opts) > 1 else 0)
                band_params = {
                    "prod_a": prod_a.split(" | ")[0],
                    "prod_b": prod_b.split(" | ")[0],
                }
            else:
                band_params = band_params_initial
        elif band_mode == "パーセンタイル":
            if not snapshot.empty:
                p_low = band_params_initial.get("p_low", 0)
                p_high = band_params_initial.get("p_high", 100)
                p_low, p_high = st.slider(
                    "百分位(%)", 0, 100, (int(p_low), int(p_high))
                )
                band_params = {"p_low": p_low, "p_high": p_high}
            else:
                band_params = {
                    "p_low": band_params_initial.get("p_low", 0),
                    "p_high": band_params_initial.get("p_high", 100),
                }
        elif band_mode == "順位帯":
            if not snapshot.empty:
                max_rank = int(snapshot["rank"].max()) if not snapshot.empty else 1
                r_low = band_params_initial.get("r_low", 1)
                r_high = band_params_initial.get("r_high", max_rank)
                r_low, r_high = st.slider(
                    "順位", 1, max_rank, (int(r_low), int(r_high))
                )
                band_params = {"r_low": r_low, "r_high": r_high}
            else:
                band_params = {
                    "r_low": band_params_initial.get("r_low", 1),
                    "r_high": band_params_initial.get("r_high", 1),
                }
        else:
            opts = (
                snapshot["product_code"] + " | " + snapshot["display_name"]
            ).tolist()
            tlabel = st.selectbox("基準商品", opts, index=0) if opts else ""
            tcode = tlabel.split(" | ")[0] if tlabel else ""
            by_default = band_params_initial.get("by", "amt")
            by_index = 0 if by_default == "amt" else 1
            by = st.radio("幅指定", ["金額", "%"], horizontal=True, index=by_index)
            if by == "金額":
                width_default = 100000
                width = int_input(
                    "幅", int(band_params_initial.get("width", width_default))
                )
                band_params = {"target_code": tcode, "by": "amt", "width": int(width)}
            else:
                width_default = 0.1
                width = st.number_input(
                    "幅",
                    value=float(band_params_initial.get("width", width_default)),
                    step=width_default / 10 if width_default else 0.01,
                )
                band_params = {"target_code": tcode, "by": "pct", "width": width}
    with c8:
        quick = st.radio(
            "クイック絞り込み",
            ["なし", "Top5", "Top10", "最新YoY上位", "直近6M伸長上位"],
            horizontal=True,
            index=0,
        )
    c9, c10, c11, c12 = st.columns([1.2, 1.5, 1.5, 1.5])
    with c9:
        enable_label_avoid = st.checkbox("ラベル衝突回避", value=True)
    with c10:
        label_gap_px = st.slider("ラベル最小間隔(px)", 8, 24, 12)
    with c11:
        label_max = st.slider("ラベル最大件数", 5, 20, 12)
    with c12:
        alternate_side = st.checkbox("ラベル左右交互配置", value=True)
    c13, c14, c15, c16, c17 = st.columns([1.0, 1.4, 1.2, 1.2, 1.2])
    with c13:
        unit = st.radio("単位", ["円", "千円", "百万円"], horizontal=True, index=1)
    with c14:
        n_win = st.slider(
            "傾きウィンドウ（月）",
            0,
            12,
            6,
            1,
            help="0=自動（系列の全期間で判定）",
        )
    with c15:
        cmp_mode = st.radio("傾き条件", ["以上", "未満"], horizontal=True)
    with c16:
        thr_type = st.radio(
            "しきい値の種類", ["円/月", "%/月", "zスコア"], horizontal=True
        )
    with c17:
        if thr_type == "円/月":
            thr_val = int_input("しきい値", 0)
        else:
            thr_val = st.number_input("しきい値", value=0.0, step=0.01, format="%.2f")
    c18, c19, c20 = st.columns([1.6, 1.2, 1.8])
    with c18:
        sens = st.slider("形状抽出の感度", 0.0, 1.0, 0.5, 0.05)
    with c19:
        z_thr = st.slider("急勾配 zスコア", 0.0, 3.0, 0.0, 0.1)
    with c20:
        shape_pick = st.radio(
            "形状抽出",
            ["（なし）", "急勾配", "山（への字）", "谷（逆への字）"],
            horizontal=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-body">', unsafe_allow_html=True)
    ai_summary_container = st.container()

    if amount_slider_cfg:
        low_scaled, high_scaled = st.slider(
            amount_slider_cfg["label"],
            min_value=amount_slider_cfg["min_value"],
            max_value=amount_slider_cfg["max_value"],
            value=amount_slider_cfg["value"],
            step=amount_slider_cfg["step"],
        )
        low = int(low_scaled * amount_slider_cfg["unit_scale"])
        high = int(high_scaled * amount_slider_cfg["unit_scale"])
        high = min(high, amount_slider_cfg["max_amount"])
        low = min(low, high)
        st.caption(f"選択中: {format_int(low)}円 〜 {format_int(high)}円")
        band_params = {"low_amount": low, "high_amount": high}
    elif band_mode == "金額指定":
        band_params = {"low_amount": low0, "high_amount": high0}

    params = {
        "end_month": end_m,
        "band_mode": band_mode,
        "band_params": band_params,
        "quick": quick,
    }
    st.session_state.compare_params = params

    mode_map = {
        "金額指定": "amount",
        "商品指定(2)": "two_products",
        "パーセンタイル": "percentile",
        "順位帯": "rank",
        "ターゲット近傍": "target_near",
    }
    low, high = resolve_band(snapshot, mode_map[band_mode], band_params)
    codes = filter_products_by_band(snapshot, low, high)

    if quick == "Top5":
        codes = snapshot.nlargest(5, "year_sum")["product_code"].tolist()
    elif quick == "Top10":
        codes = snapshot.nlargest(10, "year_sum")["product_code"].tolist()
    elif quick == "最新YoY上位":
        codes = (
            snapshot.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(10)["product_code"]
            .tolist()
        )
    elif quick == "直近6M伸長上位":
        codes = top_growth_codes(year_df, end_m, window=6, top=10)

    snap = slopes_snapshot(year_df, n=n_win)
    if thr_type == "円/月":
        key, v = "slope_yen", float(thr_val)
    elif thr_type == "%/月":
        key, v = "slope_ratio", float(thr_val)
    else:
        key, v = "slope_z", float(thr_val)
    mask = (snap[key] >= v) if cmp_mode == "以上" else (snap[key] <= v)
    codes_by_slope = set(snap.loc[mask, "product_code"])

    eff_n = n_win if n_win > 0 else 12
    shape_df = shape_flags(
        year_df,
        window=max(6, eff_n * 2),
        alpha_ratio=0.02 * (1.0 - sens),
        amp_ratio=0.06 * (1.0 - sens),
    )
    codes_steep = set(snap.loc[snap["slope_z"].abs() >= z_thr, "product_code"])
    codes_mtn = set(shape_df.loc[shape_df["is_mountain"], "product_code"])
    codes_val = set(shape_df.loc[shape_df["is_valley"], "product_code"])
    shape_map = {
        "（なし）": None,
        "急勾配": codes_steep,
        "山（への字）": codes_mtn,
        "谷（逆への字）": codes_val,
    }
    codes_by_shape = shape_map[shape_pick] or set(snap["product_code"])

    codes_from_band = set(codes)
    target_codes = list(codes_from_band & codes_by_slope & codes_by_shape)

    scale = {"円": 1, "千円": 1_000, "百万円": 1_000_000}[unit]
    snapshot_disp = snapshot.copy()
    snapshot_disp["year_sum_disp"] = snapshot_disp["year_sum"] / scale
    hist_fig = px.histogram(snapshot_disp, x="year_sum_disp")
    hist_fig.update_xaxes(title_text=f"年計（{unit}）")

    df_long, _ = get_yearly_series(year_df, target_codes)
    df_long["month"] = pd.to_datetime(df_long["month"])
    df_long["display_name"] = df_long["product_name"].fillna(df_long["product_code"])

    main_codes = target_codes
    max_lines = 30
    if len(main_codes) > max_lines:
        top_order = (
            snapshot[snapshot["product_code"].isin(main_codes)]
            .sort_values("year_sum", ascending=False)["product_code"]
            .tolist()
        )
        main_codes = top_order[:max_lines]

    df_main = df_long[df_long["product_code"].isin(main_codes)]

    with ai_summary_container:
        ai_on = st.toggle(
            "AIサマリー",
            value=st.session_state.get("compare_ai_toggle", False),
            key="compare_ai_toggle",
            help="要約・コメント・自動説明を表示（オンデマンド計算）",
        )
        with st.expander("AIサマリー", expanded=ai_on):
            if ai_on and not df_main.empty:
                pos = len(codes_steep)
                mtn = len(codes_mtn & set(main_codes))
                val = len(codes_val & set(main_codes))
                explain = _ai_explain(
                    {
                        "対象SKU数": len(main_codes),
                        "中央値(年計)": float(
                            snapshot_disp.loc[
                                snapshot_disp["product_code"].isin(main_codes),
                                "year_sum_disp",
                            ].median()
                        ),
                        "急勾配数": pos,
                        "山数": mtn,
                        "谷数": val,
                    }
                )
                st.info(f"**AI比較コメント**：{explain}")

    tb_common = dict(
        period=period,
        node_mode=node_mode,
        hover_mode=hover_mode,
        op_mode=op_mode,
        peak_on=peak_on,
        unit=unit,
        enable_avoid=enable_label_avoid,
        gap_px=label_gap_px,
        max_labels=label_max,
        alt_side=alternate_side,
        slope_conf=None,
        forecast_method="なし",
        forecast_window=12,
        forecast_horizon=6,
        forecast_k=2.0,
        forecast_robust=False,
        anomaly="OFF",
    )
    fig = build_chart_card(
        df_main,
        selected_codes=None,
        multi_mode=True,
        tb=tb_common,
        band_range=(low, high),
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)

    st.caption(
        "凡例クリックで表示切替、ダブルクリックで単独表示。ドラッグでズーム/パン、右上メニューからPNG/CSV取得可。"
    )
    st.markdown(
        """
傾き（円/月）：直近 n ヶ月の回帰直線の傾き。+は上昇、−は下降。

%/月：傾き÷平均年計。規模によらず比較可能。

zスコア：全SKUの傾き分布に対する標準化。|z|≥1.5で急勾配の目安。

山/谷：前半と後半の平均変化率の符号が**＋→−（山）／−→＋（谷）かつ振幅が十分**。
"""
    )

    snap_export = snapshot[snapshot["product_code"].isin(main_codes)].copy()
    snap_export[f"year_sum_{unit}"] = snap_export["year_sum"] / scale
    snap_export = snap_export.drop(columns=["year_sum"])
    csv_band_clicked = st.download_button(
        "CSVエクスポート",
        data=snap_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"band_snapshot_{end_m}.csv",
        mime="text/csv",
        key="compare_band_csv",
    )
    if csv_band_clicked:
        render_status_message(
            "completed",
            key="compare_band_csv_downloaded",
            guide="比較ビューのCSVを共有してチーム分析に役立てましょう。",
        )
    try:
        png_bytes = fig.to_image(format="png")
        png_clicked = st.download_button(
            "PNGエクスポート",
            data=png_bytes,
            file_name=f"band_overlay_{end_m}.png",
            mime="image/png",
            key="compare_band_png",
        )
        if png_clicked:
            render_status_message(
                "completed",
                key="compare_band_png_downloaded",
                guide="可視化画像を資料に貼り付けて共有できます。",
            )
    except Exception:
        pass

    with st.expander("分布（オプション）", expanded=False):
        hist_fig = apply_elegant_theme(
            hist_fig, theme=st.session_state.get("ui_theme", "light")
        )
        render_plotly_with_spinner(
            hist_fig, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
        )

    # ---- Small Multiples ----
    df_nodes = df_main.iloc[0:0].copy()
    HALO = "#ffffff" if st.get_option("theme.base") == "dark" else "#222222"
    SZ = 6
    dtick = "M1"
    drag = {"ズーム": "zoom", "パン": "pan", "選択": "select"}[op_mode]

    st.subheader("スモールマルチプル")
    share_y = st.checkbox("Y軸共有", value=False)
    show_keynode_labels = st.checkbox("キーノードラベル表示", value=False)
    per_page = st.radio("1ページ表示枚数", [8, 12], horizontal=True, index=0)
    total_pages = max(1, math.ceil(len(main_codes) / per_page))
    page_idx = st.number_input("ページ", min_value=1, max_value=total_pages, value=1)
    start = (page_idx - 1) * per_page
    page_codes = main_codes[start : start + per_page]
    col_count = 4
    cols = st.columns(col_count)
    ymax = (
        df_long[df_long["product_code"].isin(main_codes)]["year_sum"].max()
        / UNIT_MAP[unit]
        if share_y
        else None
    )
    for i, code in enumerate(page_codes):
        g = df_long[df_long["product_code"] == code]
        disp = g["display_name"].iloc[0] if not g.empty else code
        palette = fig.layout.colorway or px.colors.qualitative.Safe
        fig_s = px.line(
            g,
            x="month",
            y="year_sum",
            color_discrete_sequence=[palette[i % len(palette)]],
            custom_data=["display_name"],
        )
        fig_s.update_traces(
            mode="lines",
            line=dict(width=1.5),
            opacity=0.8,
            showlegend=False,
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
        )
        fig_s.update_xaxes(tickformat="%Y-%m", dtick=dtick, title_text="月（YYYY-MM）")
        fig_s.update_yaxes(
            tickformat="~,d",
            range=[0, ymax] if ymax else None,
            title_text=f"売上 年計（{unit}）",
        )
        fig_s.update_layout(font=dict(family="Noto Sans JP, Meiryo, Arial", size=12))
        fig_s.update_layout(
            hoverlabel=dict(
                bgcolor="rgba(30,30,30,0.92)", font=dict(color="#fff", size=12)
            )
        )
        fig_s.update_layout(dragmode=drag)
        if hover_mode == "個別":
            fig_s.update_layout(hovermode="closest")
        else:
            fig_s.update_layout(hovermode="x unified", hoverlabel=dict(align="left"))
        last_val = (
            g.sort_values("month")["year_sum"].iloc[-1] / UNIT_MAP[unit]
            if not g.empty
            else np.nan
        )
        with cols[i % col_count]:
            st.metric(
                disp, f"{last_val:,.0f} {unit}" if not np.isnan(last_val) else "—"
            )
            fig_s = apply_elegant_theme(
                fig_s, theme=st.session_state.get("ui_theme", "light")
            )
            fig_s.update_layout(height=225)
            render_plotly_with_spinner(
                fig_s, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
            )

    # 5) SKU詳細
elif page == "SKU詳細":
    require_data()
    section_header("SKU 詳細", "個別SKUのトレンドとメモを一元管理。", icon="🗂️")
    end_m = sidebar_state.get("detail_end_month") or latest_month
    prods = (
        st.session_state.data_year[["product_code", "product_name"]]
        .drop_duplicates()
        .sort_values("product_code")
    )
    mode = st.radio("表示モード", ["単品", "複数比較"], horizontal=True)
    tb = toolbar_sku_detail(multi_mode=(mode == "複数比較"))
    df_year = st.session_state.data_year.copy()
    df_year["display_name"] = df_year["product_name"].fillna(df_year["product_code"])

    ai_on = st.toggle(
        "AIサマリー",
        value=st.session_state.get("sku_detail_ai_toggle", False),
        key="sku_detail_ai_toggle",
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    chart_rendered = False
    modal_codes: List[str] | None = None
    modal_is_multi = False

    if mode == "単品":
        prod_label = st.selectbox(
            "SKU選択", options=prods["product_code"] + " | " + prods["product_name"]
        )
        code = prod_label.split(" | ")[0]
        build_chart_card(
            df_year,
            selected_codes=[code],
            multi_mode=False,
            tb=tb,
            height=600,
        )
        chart_rendered = True
        modal_codes = [code]
        modal_is_multi = False

        g_y = df_year[df_year["product_code"] == code].sort_values("month")
        row = g_y[g_y["month"] == end_m]
        if not row.empty:
            rr = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "年計", f"{int(rr['year_sum']) if not pd.isna(rr['year_sum']) else '—'}"
            )
            c2.metric(
                "YoY", f"{rr['yoy']*100:.1f} %" if not pd.isna(rr["yoy"]) else "—"
            )
            c3.metric("Δ", f"{int(rr['delta'])}" if not pd.isna(rr["delta"]) else "—")

        with st.expander("AIサマリー", expanded=ai_on):
            if ai_on and not row.empty:
                st.info(
                    _ai_explain(
                        {
                            "年計": (
                                float(rr["year_sum"])
                                if not pd.isna(rr["year_sum"])
                                else 0.0
                            ),
                            "YoY": float(rr["yoy"]) if not pd.isna(rr["yoy"]) else 0.0,
                            "Δ": float(rr["delta"]) if not pd.isna(rr["delta"]) else 0.0,
                        }
                    )
                )

        st.subheader("メモ / タグ")
        note = st.text_area(
            "メモ（保存で保持）", value=st.session_state.notes.get(code, ""), height=100
        )
        tags_str = st.text_input(
            "タグ（カンマ区切り）", value=",".join(st.session_state.tags.get(code, []))
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("保存"):
            st.session_state.notes[code] = note
            st.session_state.tags[code] = [
                t.strip() for t in tags_str.split(",") if t.strip()
            ]
            st.success("保存しました")
        if c2.button("CSVでエクスポート"):
            meta = pd.DataFrame(
                [
                    {
                        "product_code": code,
                        "note": st.session_state.notes.get(code, ""),
                        "tags": ",".join(st.session_state.tags.get(code, [])),
                    }
                ]
            )
            st.download_button(
                "ダウンロード",
                data=meta.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"notes_{code}.csv",
                mime="text/csv",
            )
    else:
        opts = (prods["product_code"] + " | " + prods["product_name"]).tolist()
        sel = st.multiselect("SKU選択（最大60件）", options=opts, max_selections=60)
        codes = [s.split(" | ")[0] for s in sel]
        if codes or (tb.get("slope_conf") and tb["slope_conf"].get("quick") != "なし"):
            build_chart_card(
                df_year,
                selected_codes=codes,
                multi_mode=True,
                tb=tb,
                height=600,
            )
            chart_rendered = True
            modal_codes = codes
            modal_is_multi = True
            snap = latest_yearsum_snapshot(df_year, end_m)
            if codes:
                snap = snap[snap["product_code"].isin(codes)]
            with st.expander("AIサマリー", expanded=ai_on):
                if ai_on and not snap.empty:
                    st.info(_ai_sum_df(snap[["year_sum", "yoy", "delta"]]))
            st.dataframe(
                snap[["product_code", "product_name", "year_sum", "yoy", "delta"]],
                use_container_width=True,
            )
            st.download_button(
                "CSVダウンロード",
                data=snap.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"sku_multi_{end_m}.csv",
                mime="text/csv",
            )
        else:
            st.info("SKUを選択してください。")

    if tb.get("expand_mode") and chart_rendered:
        with compat_modal("グラフ拡大モード", key="sku_expand_modal"):
            st.caption("操作パネルは拡大表示中も利用できます。")
            tb_modal = toolbar_sku_detail(
                multi_mode=modal_is_multi,
                key_prefix="sku_modal",
                include_expand_toggle=False,
            )
            build_chart_card(
                df_year,
                selected_codes=modal_codes,
                multi_mode=modal_is_multi,
                tb=tb_modal,
                height=tb_modal.get("chart_height", 760),
            )
            if st.button("閉じる", key="close_expand_modal"):
                st.session_state.setdefault("ui", {})["expand_mode"] = False
                st.session_state["sku_expand_mode"] = False
                st.rerun()

# 5) 異常検知
elif page == "異常検知":
    require_data()
    section_header("異常検知", "回帰残差ベースで異常ポイントを抽出します。", icon="🚨")
    year_df = st.session_state.data_year.copy()
    unit = st.session_state.settings.get("currency_unit", "円")
    scale = UNIT_MAP.get(unit, 1)

    col_a, col_b = st.columns([1.1, 1.1])
    with col_a:
        window = st.slider("学習窓幅（月）", 6, 18, st.session_state.get("anomaly_window", 12), key="anomaly_window")
    with col_b:
        score_method = st.radio("スコア基準", ["zスコア", "MADスコア"], horizontal=True, key="anomaly_score_method")

    if score_method == "zスコア":
        thr_key = "anomaly_thr_z"
        threshold = st.slider(
            "異常判定しきい値",
            2.0,
            5.0,
            value=float(st.session_state.get(thr_key, 3.0)),
            step=0.1,
            key=thr_key,
        )
        robust = False
    else:
        thr_key = "anomaly_thr_mad"
        threshold = st.slider(
            "異常判定しきい値",
            2.5,
            6.0,
            value=float(st.session_state.get(thr_key, 3.5)),
            step=0.1,
            key=thr_key,
        )
        robust = True

    prod_opts = (
        year_df[["product_code", "product_name"]]
        .drop_duplicates()
        .sort_values("product_code")
    )
    prod_opts["label"] = (
        prod_opts["product_code"]
        + " | "
        + prod_opts["product_name"].fillna(prod_opts["product_code"])
    )
    selected_labels = st.multiselect(
        "対象SKU（未選択=全件）",
        options=prod_opts["label"].tolist(),
        key="anomaly_filter_codes",
    )
    selected_codes = [lab.split(" | ")[0] for lab in selected_labels]

    records: List[pd.DataFrame] = []
    for code, g in year_df.groupby("product_code"):
        if selected_codes and code not in selected_codes:
            continue
        s = g.sort_values("month").set_index("month")["year_sum"]
        res = detect_linear_anomalies(
            s,
            window=int(window),
            threshold=float(threshold),
            robust=robust,
        )
        if res.empty:
            continue
        res["product_code"] = code
        res["product_name"] = g["product_name"].iloc[0]
        res = res.merge(
            g[["month", "year_sum", "yoy", "delta"]],
            on="month",
            how="left",
        )
        res["score_abs"] = res["score"].abs()
        records.append(res)

    if not records:
        st.success("異常値は検出されませんでした。窓幅やしきい値を調整してください。")
    else:
        anomalies = pd.concat(records, ignore_index=True)
        anomalies = anomalies.sort_values("score_abs", ascending=False)
        anomalies["year_sum_disp"] = anomalies["year_sum"] / scale
        anomalies["delta_disp"] = anomalies["delta"] / scale
        total_count = len(anomalies)
        sku_count = anomalies["product_code"].nunique()
        pos_cnt = int((anomalies["score"] > 0).sum())
        neg_cnt = int((anomalies["score"] < 0).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("異常件数", f"{total_count:,}")
        m2.metric("対象SKU", f"{sku_count:,}")
        m3.metric("上振れ/下振れ", f"{pos_cnt:,} / {neg_cnt:,}")

        max_top = min(200, total_count)
        top_default = min(50, max_top)
        top_n = int(
            st.slider(
                "表示件数",
                min_value=1,
                max_value=max_top,
                value=top_default,
                key="anomaly_view_top",
            )
        )
        view = anomalies.head(top_n).copy()
        view_table = view[
            [
                "product_code",
                "product_name",
                "month",
                "year_sum_disp",
                "yoy",
                "delta_disp",
                "score",
            ]
        ].rename(
            columns={
                "product_code": "商品コード",
                "product_name": "商品名",
                "month": "月",
                "year_sum_disp": f"年計({unit})",
                "yoy": "YoY",
                "delta_disp": f"Δ({unit})",
                "score": "スコア",
            }
        )
        st.dataframe(view_table, use_container_width=True)
        st.caption("値は指定した単位換算、スコアはローカル回帰残差の標準化値です。")
        st.download_button(
            "CSVダウンロード",
            data=view_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"anomalies_{score_method}_{threshold:.1f}.csv",
            mime="text/csv",
        )

        anomaly_ai_on = st.toggle(
            "AI異常サマリー", value=False, key="anomaly_ai_toggle"
        )
        with st.expander("AI異常サマリー", expanded=anomaly_ai_on):
            if anomaly_ai_on and not view.empty:
                ai_df = view[
                    ["product_name", "month", "score", "year_sum", "yoy", "delta"]
                ].fillna(0)
                st.info(_ai_anomaly_report(ai_df))

        option_labels = [
            f"{row['product_code']}｜{row['product_name'] or row['product_code']}｜{row['month']}"
            for _, row in view.iterrows()
        ]
        if option_labels:
            sel_label = st.selectbox("詳細チャート", options=option_labels, key="anomaly_detail_select")
            code_sel, name_sel, month_sel = sel_label.split("｜")
            g = year_df[year_df["product_code"] == code_sel].sort_values("month").copy()
            g["year_sum_disp"] = g["year_sum"] / scale
            fig_anom = px.line(
                g,
                x="month",
                y="year_sum_disp",
                markers=True,
                title=f"{name_sel} 年計推移",
            )
            fig_anom.update_yaxes(title_text=f"年計（{unit}）", tickformat="~,d")
            fig_anom.update_traces(hovertemplate="月：%{x|%Y-%m}<br>年計：%{y:,.0f} {unit}<extra></extra>")

            code_anoms = anomalies[anomalies["product_code"] == code_sel]
            if not code_anoms.empty:
                fig_anom.add_scatter(
                    x=code_anoms["month"],
                    y=code_anoms["year_sum"] / scale,
                    mode="markers",
                    name="異常値",
                    marker=dict(color="#d94c53", size=10, symbol="triangle-up"),
                    hovertemplate="異常月：%{x|%Y-%m}<br>年計：%{y:,.0f} {unit}<br>スコア：%{customdata[0]:.2f}<extra></extra>",
                    customdata=np.stack([code_anoms["score"]], axis=-1),
                    showlegend=False,
                )
            target = code_anoms[code_anoms["month"] == month_sel]
            if not target.empty:
                tgt = target.iloc[0]
                fig_anom.add_annotation(
                    x=month_sel,
                    y=tgt["year_sum"] / scale,
                    text=f"スコア {tgt['score']:.2f}",
                    showarrow=True,
                    arrowcolor="#d94c53",
                    arrowhead=2,
                )
                yoy_txt = (
                    f"{tgt['yoy'] * 100:.1f}%" if tgt.get("yoy") is not None and not pd.isna(tgt.get("yoy")) else "—"
                )
                delta_txt = format_amount(tgt.get("delta"), unit)
                st.info(
                    f"{name_sel} {month_sel} の年計は {tgt['year_sum_disp']:.0f} {unit}、YoY {yoy_txt}、Δ {delta_txt}。"
                    f" 異常スコアは {tgt['score']:.2f} です。"
                )
            fig_anom = apply_elegant_theme(
                fig_anom, theme=st.session_state.get("ui_theme", "light")
            )
            render_plotly_with_spinner(
                fig_anom, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
            )

# 6) 相関分析
elif page == "相関分析":
    require_data()
    section_header("相関分析", "指標間の関係性からインサイトを発掘。", icon="🧭")
    end_m = sidebar_state.get("corr_end_month") or latest_month
    snapshot = latest_yearsum_snapshot(st.session_state.data_year, end_m)

    metric_opts = [
        "year_sum",
        "yoy",
        "delta",
        "slope_beta",
        "slope6m",
        "std6m",
        "hhi_share",
    ]
    analysis_mode = st.radio(
        "分析対象",
        ["指標間", "SKU間"],
        horizontal=True,
    )
    method = st.radio(
        "相関の種類",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda x: "Pearson" if x == "pearson" else "Spearman",
    )
    r_thr = st.slider("相関 r 閾値（|r|≥）", 0.0, 1.0, 0.0, 0.05)

    if analysis_mode == "指標間":
        metrics = st.multiselect(
            "指標",
            [m for m in metric_opts if m in snapshot.columns],
            default=[
                m
                for m in ["year_sum", "yoy", "delta", "slope_beta"]
                if m in snapshot.columns
            ],
        )
        winsor_pct = st.slider("外れ値丸め(%)", 0.0, 5.0, 1.0)
        log_enable = st.checkbox("ログ変換", value=False)
        ai_on = st.toggle(
            "AIサマリー",
            value=False,
            key="corr_ai_metric",
            help="要約・コメント・自動説明を表示（オンデマンド計算）",
        )

        if metrics:
            df_plot = snapshot.copy()
            df_plot = winsorize_frame(df_plot, metrics, p=winsor_pct / 100)
            df_plot = maybe_log1p(df_plot, metrics, log_enable)
            tbl = corr_table(df_plot, metrics, method=method)
            tbl = tbl[abs(tbl["r"]) >= r_thr]

            st.subheader("相関の要点")
            for line in narrate_top_insights(tbl, NAME_MAP):
                st.write("・", line)
            sig_cnt = int((tbl["sig"] == "有意(95%)").sum())
            weak_cnt = int((tbl["r"].abs() < 0.2).sum())
            st.write(f"統計的に有意な相関: {sig_cnt} 組")
            st.write(f"|r|<0.2 の組み合わせ: {weak_cnt} 組")

            with st.expander("AIサマリー", expanded=ai_on):
                if ai_on and not tbl.empty:
                    r_mean = float(tbl["r"].abs().mean())
                    st.info(
                        _ai_explain(
                            {
                                "有意本数": int((tbl["sig"] == "有意(95%)").sum()),
                                "平均|r|": r_mean,
                            }
                        )
                    )

            st.subheader("相関ヒートマップ")
            st.caption("右上=強い正、左下=強い負、白=関係薄")
            corr = df_plot[metrics].corr(method=method)
            fig_corr = px.imshow(
                corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True
            )
            fig_corr = apply_elegant_theme(
                fig_corr, theme=st.session_state.get("ui_theme", "light")
            )
            render_plotly_with_spinner(
                fig_corr, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
            )

            st.subheader("ペア・エクスプローラ")
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("指標X", metrics, index=0)
            with c2:
                y_col = st.selectbox(
                    "指標Y", metrics, index=1 if len(metrics) > 1 else 0
                )
            df_xy = df_plot[[x_col, y_col, "product_name", "product_code"]].dropna()
            if not df_xy.empty:
                m, b, r2 = fit_line(df_xy[x_col], df_xy[y_col])
                r = df_xy[x_col].corr(df_xy[y_col], method=method)
                lo, hi = fisher_ci(r, len(df_xy))
                fig_sc = px.scatter(
                    df_xy, x=x_col, y=y_col, hover_data=["product_code", "product_name"]
                )
                xs = np.linspace(df_xy[x_col].min(), df_xy[x_col].max(), 100)
                fig_sc.add_trace(
                    go.Scatter(x=xs, y=m * xs + b, mode="lines", name="回帰")
                )
                fig_sc.add_annotation(
                    x=0.99,
                    y=0.01,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="bottom",
                    text=f"r={r:.2f} (95%CI [{lo:.2f},{hi:.2f}])<br>R²={r2:.2f}",
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.6)",
                )
                resid = np.abs(df_xy[y_col] - (m * df_xy[x_col] + b))
                outliers = df_xy.loc[resid.nlargest(3).index]
                for _, row in outliers.iterrows():
                    label = row["product_name"] or row["product_code"]
                    fig_sc.add_annotation(
                        x=row[x_col],
                        y=row[y_col],
                        text=label,
                        showarrow=True,
                        arrowhead=1,
                    )
                fig_sc = apply_elegant_theme(
                    fig_sc, theme=st.session_state.get("ui_theme", "light")
                )
                render_plotly_with_spinner(
                    fig_sc, config=PLOTLY_CONFIG, spinner_text=SPINNER_MESSAGE
                )
                st.caption("rは -1〜+1。0は関連が薄い。CIに0を含まなければ有意。")
                st.caption("散布図の点が右上・左下に伸びれば正、右下・左上なら負。")
        else:
            st.info("指標を選択してください。")
    else:
        df_year = st.session_state.data_year.copy()
        series_metric_opts = [m for m in metric_opts if m in df_year.columns]
        if not series_metric_opts:
            st.info("SKU間相関に利用できる指標がありません。")
        else:
            sku_metric = st.selectbox(
                "対象指標",
                series_metric_opts,
                format_func=lambda x: NAME_MAP.get(x, x),
            )
            months_all = sorted(df_year["month"].unique())
            if not months_all:
                st.info("データが不足しています。")
            else:
                if end_m in months_all:
                    end_idx = months_all.index(end_m)
                else:
                    end_idx = len(months_all) - 1
                if end_idx < 0:
                    st.info("対象期間のデータがありません。")
                else:
                    max_period = end_idx + 1
                    if max_period < 2:
                        st.info("対象期間のデータが不足しています。")
                    else:
                        slider_min = 2
                        slider_max = max_period
                        default_period = max(slider_min, min(12, slider_max))
                        period = int(
                            st.slider(
                                "対象期間（月数）",
                                min_value=slider_min,
                                max_value=slider_max,
                                value=default_period,
                            )
                        )
                        start_idx = max(0, end_idx - period + 1)
                        months_window = months_all[start_idx : end_idx + 1]
                        df_window = df_year[df_year["month"].isin(months_window)]
                        pivot = (
                            df_window.pivot(
                                index="month", columns="product_code", values=sku_metric
                            ).sort_index()
                        )
                        pivot = pivot.dropna(how="all")
                        if pivot.empty:
                            st.info("選択した期間に利用できるデータがありません。")
                        else:
                            top_candidates = [
                                c for c in snapshot["product_code"] if c in pivot.columns
                            ]
                            if len(top_candidates) < 2:
                                st.info("対象SKUが不足しています。")
                            else:
                                top_max = min(60, len(top_candidates))
                                top_default = max(2, min(10, top_max))
                                top_n = int(
                                    st.slider(
                                        "対象SKU数（上位）",
                                        min_value=2,
                                        max_value=top_max,
                                        value=top_default,
                                    )
                                )
                                selected_codes = top_candidates[:top_n]
                                sku_pivot = pivot[selected_codes].dropna(
                                    axis=1, how="all"
                                )
                                available_codes = sku_pivot.columns.tolist()
                                min_periods = 3
                                valid_codes = [
                                    code
                                    for code in available_codes
                                    if sku_pivot[code].count() >= min_periods
                                ]
                                if len(valid_codes) < 2:
                                    st.info(
                                        "有効なSKUが2件未満です。期間やSKU数を調整してください。"
                                    )
                                else:
                                    sku_pivot = sku_pivot[valid_codes]
                                    months_used = sku_pivot.index.tolist()
                                    code_to_name = (
                                        df_year[["product_code", "product_name"]]
                                        .drop_duplicates()
                                        .set_index("product_code")["product_name"]
                                        .to_dict()
                                    )
                                    display_map = {
                                        code: f"{code}｜{code_to_name.get(code, code) or code}"
                                        for code in valid_codes
                                    }
                                    ai_on = st.toggle(
                                        "AIサマリー",
                                        value=False,
                                        key="corr_ai_sku",
                                        help="要約・コメント・自動説明を表示（オンデマンド計算）",
                                    )
                                    tbl_raw = corr_table(
                                        sku_pivot,
                                        valid_codes,
                                        method=method,
                                        pairwise=True,
                                        min_periods=min_periods,
                                    )
                                    tbl = tbl_raw.dropna(subset=["r"])
                                    tbl = tbl[abs(tbl["r"]) >= r_thr]

                                    st.subheader("相関の要点")
                                    if months_used:
                                        st.caption(
                                            f"対象期間: {months_used[0]}〜{months_used[-1]}（{len(months_used)}ヶ月）"
                                        )
                                    st.caption(
                                        "対象SKU: "
                                        + "、".join(display_map[code] for code in valid_codes)
                                    )
                                    for line in narrate_top_insights(tbl, display_map):
                                        st.write("・", line)
                                    sig_cnt = int((tbl["sig"] == "有意(95%)").sum())
                                    weak_cnt = int((tbl["r"].abs() < 0.2).sum())
                                    st.write(f"統計的に有意な相関: {sig_cnt} 組")
                                    st.write(f"|r|<0.2 の組み合わせ: {weak_cnt} 組")
                                    if tbl.empty:
                                        st.info(
                                            "条件に合致するSKU間相関は見つかりませんでした。"
                                        )

                                    with st.expander("AIサマリー", expanded=ai_on):
                                        if ai_on and not tbl.empty:
                                            r_mean = float(tbl["r"].abs().mean())
                                            st.info(
                                                _ai_explain(
                                                    {
                                                        "有意本数": int(
                                                            (tbl["sig"] == "有意(95%)").sum()
                                                        ),
                                                        "平均|r|": r_mean,
                                                    }
                                                )
                                            )

                                    st.subheader("相関ヒートマップ")
                                    st.caption(
                                        "セルは対象期間におけるSKU同士の相関係数を示します。"
                                    )
                                    heatmap = sku_pivot.rename(columns=display_map)
                                    corr = heatmap.corr(
                                        method=method, min_periods=min_periods
                                    )
                                    fig_corr = px.imshow(
                                        corr,
                                        color_continuous_scale="RdBu_r",
                                        zmin=-1,
                                        zmax=1,
                                        text_auto=True,
                                    )
                                    fig_corr = apply_elegant_theme(
                                        fig_corr, theme=st.session_state.get("ui_theme", "light")
                                    )
                                    render_plotly_with_spinner(
                                        fig_corr,
                                        config=PLOTLY_CONFIG,
                                        spinner_text=SPINNER_MESSAGE,
                                    )

                                    st.subheader("SKUペア・エクスプローラ")
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        x_code = st.selectbox(
                                            "SKU X",
                                            valid_codes,
                                            format_func=lambda c: display_map.get(c, c),
                                        )
                                    with c2:
                                        y_default = 1 if len(valid_codes) > 1 else 0
                                        y_code = st.selectbox(
                                            "SKU Y",
                                            valid_codes,
                                            index=y_default,
                                            format_func=lambda c: display_map.get(c, c),
                                        )
                                    df_xy = (
                                        sku_pivot[[x_code, y_code]]
                                        .dropna()
                                        .reset_index()
                                    )
                                    if len(df_xy) >= 2:
                                        x_label = display_map.get(x_code, x_code)
                                        y_label = display_map.get(y_code, y_code)
                                        df_xy = df_xy.rename(
                                            columns={
                                                "month": "月",
                                                x_code: x_label,
                                                y_code: y_label,
                                            }
                                        )
                                        m, b, r2 = fit_line(
                                            df_xy[x_label], df_xy[y_label]
                                        )
                                        r = df_xy[x_label].corr(
                                            df_xy[y_label], method=method
                                        )
                                        lo, hi = fisher_ci(r, len(df_xy))
                                        fig_sc = px.scatter(
                                            df_xy,
                                            x=x_label,
                                            y=y_label,
                                            hover_data=["月"],
                                        )
                                        xs = np.linspace(
                                            df_xy[x_label].min(), df_xy[x_label].max(), 100
                                        )
                                        fig_sc.add_trace(
                                            go.Scatter(
                                                x=xs, y=m * xs + b, mode="lines", name="回帰"
                                            )
                                        )
                                        fig_sc.add_annotation(
                                            x=0.99,
                                            y=0.01,
                                            xref="paper",
                                            yref="paper",
                                            xanchor="right",
                                            yanchor="bottom",
                                            text=f"r={r:.2f} (95%CI [{lo:.2f},{hi:.2f}])<br>R²={r2:.2f}｜n={len(df_xy)}",
                                            showarrow=False,
                                            align="right",
                                            bgcolor="rgba(255,255,255,0.6)",
                                        )
                                        resid = np.abs(
                                            df_xy[y_label] - (m * df_xy[x_label] + b)
                                        )
                                        outliers = df_xy.loc[
                                            resid.nlargest(min(3, len(resid))).index
                                        ]
                                        for _, row in outliers.iterrows():
                                            fig_sc.add_annotation(
                                                x=row[x_label],
                                                y=row[y_label],
                                                text=row["月"],
                                                showarrow=True,
                                                arrowhead=1,
                                            )
                                        fig_sc = apply_elegant_theme(
                                            fig_sc,
                                            theme=st.session_state.get("ui_theme", "light"),
                                        )
                                        render_plotly_with_spinner(
                                            fig_sc,
                                            config=PLOTLY_CONFIG,
                                            spinner_text=SPINNER_MESSAGE,
                                        )
                                        st.caption(
                                            "各点は対象期間の月次値。右上（左下）に伸びれば同時に増加（減少）。"
                                        )
                                    else:
                                        st.info(
                                            "共通する月のデータが不足しています。期間やSKU数を調整してください。"
                                        )

    with st.expander("相関の読み方"):
        st.write("正の相関：片方が大きいほどもう片方も大きい")
        st.write("負の相関：片方が大きいほどもう片方は小さい")
        st.write(
            "|r|<0.2は弱い、0.2-0.5はややあり、0.5-0.8は中~強、>0.8は非常に強い（目安）"
        )

# 7) 併買カテゴリ
elif page == "併買カテゴリ":
    render_correlation_category_module(plot_config=PLOTLY_CONFIG)

# 8) アラート
elif page == "アラート":
    require_data()
    section_header("アラート", "閾値に該当したリスクSKUを自動抽出。", icon="⚠️")
    end_m = sidebar_state.get("alert_end_month") or latest_month
    s = st.session_state.settings
    alerts = build_alerts(
        st.session_state.data_year,
        end_month=end_m,
        yoy_threshold=s["yoy_threshold"],
        delta_threshold=s["delta_threshold"],
        slope_threshold=s["slope_threshold"],
    )
    if alerts.empty:
        st.success("閾値に該当するアラートはありません。")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button(
            "CSVダウンロード",
            data=alerts.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"alerts_{end_m}.csv",
            mime="text/csv",
        )

# 9) 設定
elif page == "設定":
    section_header("設定", "年計計算条件や閾値を調整します。", icon="⚙️")
    s = st.session_state.settings
    c1, c2, c3 = st.columns(3)
    with c1:
        s["window"] = st.number_input(
            "年計ウィンドウ（月）",
            min_value=3,
            max_value=24,
            value=int(s["window"]),
            step=1,
        )
        s["last_n"] = st.number_input(
            "傾き算出の対象点数",
            min_value=3,
            max_value=36,
            value=int(s["last_n"]),
            step=1,
        )
    with c2:
        s["yoy_threshold"] = st.number_input(
            "YoY 閾値（<=）", value=float(s["yoy_threshold"]), step=0.01, format="%.2f"
        )
        s["delta_threshold"] = int_input("Δ 閾値（<= 円）", int(s["delta_threshold"]))
    with c3:
        s["slope_threshold"] = st.number_input(
            "傾き 閾値（<=）",
            value=float(s["slope_threshold"]),
            step=0.1,
            format="%.2f",
        )
        s["currency_unit"] = st.selectbox(
            "通貨単位表記",
            options=["円", "千円", "百万円"],
            index=["円", "千円", "百万円"].index(s["currency_unit"]),
        )

    st.caption("※ 設定変更後は再計算が必要です。")
    if st.button("年計の再計算を実行", type="primary"):
        if st.session_state.data_monthly is None:
            st.warning("先にデータを取り込んでください。")
        else:
            long_df = st.session_state.data_monthly
            year_df = compute_year_rolling(
                long_df, window=s["window"], policy=s["missing_policy"]
            )
            year_df = compute_slopes(year_df, last_n=s["last_n"])
            st.session_state.data_year = year_df
            st.success("再計算が完了しました。")

# 10) 保存ビュー
elif page == "保存ビュー":
    section_header("保存ビュー", "設定や比較条件をブックマーク。", icon="🔖")
    s = st.session_state.settings
    cparams = st.session_state.compare_params
    st.write("現在の設定・選択（閾値、ウィンドウ、単位など）を名前を付けて保存します。")

    name = st.text_input("ビュー名")
    if st.button("保存"):
        if not name:
            st.warning("ビュー名を入力してください。")
        else:
            st.session_state.saved_views[name] = {
                "settings": dict(s),
                "compare": dict(cparams),
            }
            st.success(f"ビュー「{name}」を保存しました。")

    st.subheader("保存済みビュー")
    if not st.session_state.saved_views:
        st.info("保存済みビューはありません。")
    else:
        for k, v in st.session_state.saved_views.items():
            st.write(f"**{k}**: {json.dumps(v, ensure_ascii=False)}")
            if st.button(f"適用: {k}"):
                st.session_state.settings.update(v.get("settings", {}))
                st.session_state.compare_params = v.get("compare", {})
                st.session_state.compare_results = None
                st.success(f"ビュー「{k}」を適用しました。")

current_tour_step = get_current_tour_step()
apply_tour_highlight(current_tour_step)
