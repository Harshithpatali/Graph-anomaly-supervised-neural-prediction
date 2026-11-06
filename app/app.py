"""
Fast interactive Streamlit Dashboard for E-Commerce Graph Anomaly Detection.

Optimised for low latency:
- Loads a single pre-scored lean parquet/pickle (no feature rebuild, no model fit)
- Aggressive sampling for plots
- Cached data loading
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="Fraud Graph Anomaly Intelligence",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual theme — "FraudIntel" dark/graph-intelligence look
# ---------------------------------------------------------------------------
_INK = "#0A0E17"
_PANEL = "#0B1020"
_LINE = "rgba(0, 217, 255, 0.16)"
_TEXT = "#E8ECF4"
_MUTED = "#8B95A7"
_CYAN = "#00D9FF"
_VIOLET = "#7C5CFF"
_DANGER = "#FF3B69"
_GOOD = "#33E6A0"
_AMBER = "#FFB800"

FRAUDINTEL_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=_PANEL,
        plot_bgcolor=_PANEL,
        font=dict(family="Segoe UI, Arial, sans-serif", color=_TEXT, size=13),
        title=dict(font=dict(family="Segoe UI, Arial, sans-serif", size=17, color=_TEXT)),
        colorway=[_CYAN, _VIOLET, _DANGER, _GOOD, _AMBER],
        xaxis=dict(gridcolor=_LINE, zerolinecolor=_LINE, linecolor=_LINE),
        yaxis=dict(gridcolor=_LINE, zerolinecolor=_LINE, linecolor=_LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED)),
        margin=dict(t=48, b=32, l=48, r=24),
        hoverlabel=dict(bgcolor=_INK, font=dict(family="Segoe UI, Arial, sans-serif", color=_TEXT)),
    )
)
pio.templates["fraudintel"] = FRAUDINTEL_TEMPLATE
pio.templates.default = "fraudintel"

st.markdown(
    f"""
    <style>
    /* Self-contained typography: no external font download required. */
    :root {{
        color-scheme: dark;
        --app-bg: #0b1220;
        --surface: #111c2e;
        --surface-raised: #16243a;
        --border: #263750;
        --text: #edf3fb;
        --muted: #a8b7ca;
        --accent: #54c7ec;
        --accent-strong: #2aa8d5;
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        font-family: "Segoe UI", Arial, sans-serif !important;
        color: var(--text) !important;
    }}

    .stApp {{
        background: var(--app-bg);
        background-image:
            radial-gradient(ellipse at 8% 0%, rgba(42, 168, 213, .10), transparent 34%),
            radial-gradient(ellipse at 95% 10%, rgba(104, 95, 220, .08), transparent 30%);
    }}

    [data-testid="stHeader"] {{
        background: rgba(11, 18, 32, .92);
    }}

    [data-testid="stMain"] .block-container {{
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4, p, li, label, small, span, div {{
        font-family: "Segoe UI", Arial, sans-serif;
    }}

    h1, h2, h3, h4 {{
        color: var(--text) !important;
        letter-spacing: -0.025em;
        font-weight: 650 !important;
    }}

    h1 {{
        font-size: clamp(1.8rem, 3vw, 2.45rem) !important;
        line-height: 1.18;
        padding-bottom: .35rem;
        background: none !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    h1::after {{ display: none !important; }}

    p, li, [data-testid="stMarkdownContainer"] {{
        color: #d4deeb;
        line-height: 1.6;
    }}

    [data-testid="stCaptionContainer"], .stCaption, small {{
        color: var(--muted) !important;
    }}

    section[data-testid="stSidebar"] {{
        background: #0e192a;
        border-right: 1px solid var(--border);
    }}

    section[data-testid="stSidebar"] * {{
        color: #dce7f4;
    }}

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: var(--muted) !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] {{
        gap: .3rem;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label {{
        border-radius: 9px;
        padding: .45rem .55rem;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(84, 199, 236, .10);
    }}

    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, #142238, #101b2d);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .12);
    }}

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
        color: #b5c4d7 !important;
        font-size: .88rem !important;
        font-weight: 550 !important;
    }}

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
        color: #f4f8fd !important;
        font-weight: 700 !important;
        letter-spacing: -.03em;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: .82rem;
    }}

    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
    [data-testid="stTable"], [data-testid="stForm"] {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(17, 28, 46, .72);
    }}

    [data-testid="stDataFrame"] {{
        padding: .2rem;
    }}

    [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"],
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
        background: #111e31 !important;
        color: var(--text) !important;
        border-color: #354a66 !important;
        border-radius: 9px !important;
    }}

    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
        color: #dce7f4 !important;
        font-weight: 550 !important;
    }}

    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        border-radius: 9px;
        border: 1px solid #398eae;
        background: #17354b;
        color: #f3f8fc !important;
        font-weight: 600;
        min-height: 2.55rem;
        transition: background .15s ease, border-color .15s ease;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {{
        background: #20516a;
        border-color: var(--accent);
        color: #fff !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: .35rem;
        border-bottom: 1px solid var(--border);
    }}

    .stTabs [data-baseweb="tab"] {{
        color: #b7c5d7;
        font-weight: 550;
        padding: .65rem .9rem;
    }}

    .stTabs [aria-selected="true"] {{
        color: #8cdef5 !important;
        border-bottom: 2px solid var(--accent) !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 11px;
    }}

    hr {{
        border-color: var(--border);
    }}

    code, pre {{
        font-family: Consolas, "SFMono-Regular", monospace !important;
    }}

    @media (max-width: 768px) {{
        [data-testid="stMain"] .block-container {{
            padding: 1.2rem .9rem 2rem;
        }}
        [data-testid="stMetric"] {{
            padding: .8rem;
        }}
    }}
    /* Contrast and typography hardening: use installed system fonts only.
       Avoid broad text-fill/transparent effects that can make labels disappear. */
    :root {{
        font-synthesis: none;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stMetric"],
    [data-testid="stMetric"] *,
    [data-testid="stAlert"],
    [data-testid="stAlert"] *,
    [data-testid="stExpander"],
    [data-testid="stExpander"] * {{
        font-family: "Segoe UI", Arial, sans-serif !important;
    }}

    /* Explicit readable defaults across Streamlit's rendered text. */
    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label,
    [data-testid="stMain"] summary,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {{
        color: #e2eaf5 !important;
        -webkit-text-fill-color: currentColor !important;
    }}

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {{
        color: #f4f7fc !important;
        -webkit-text-fill-color: #f4f7fc !important;
        font-weight: 650 !important;
        letter-spacing: -0.02em;
    }}

    [data-testid="stMain"] h2 {{ margin-top: 1.35rem; }}
    [data-testid="stMain"] h3 {{ margin-top: 1.05rem; }}

    /* Sidebar navigation and labels remain legible in selected/hover states. */
    section[data-testid="stSidebar"] [role="radiogroup"] label,
    section[data-testid="stSidebar"] [role="radiogroup"] label *,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {{
        color: #dce7f4 !important;
        -webkit-text-fill-color: currentColor !important;
    }}
    section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {{
        background: rgba(84, 199, 236, .14);
        border: 1px solid rgba(84, 199, 236, .38);
    }}

    /* Inputs, dropdown menus, date controls, and placeholders. */
    [data-baseweb="input"] *,
    [data-baseweb="select"] *,
    [data-baseweb="textarea"] *,
    [data-testid="stNumberInput"] *,
    [data-testid="stTextInput"] *,
    [data-testid="stDateInput"] *,
    [data-testid="stTimeInput"] {{
        color: #f1f5fb !important;
        -webkit-text-fill-color: currentColor !important;
        caret-color: #70d7f4;
    }}
    input::placeholder, textarea::placeholder {{
        color: #a9b8cb !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #a9b8cb !important;
    }}
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] {{
        color: #eaf1fa !important;
        -webkit-text-fill-color: currentColor !important;
    }}

    /* Structured JSON/code panels: avoid Streamlit's default white JSON viewer. */
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] > div,
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code {{
        background: #0b1422 !important;
        color: #e8f0fa !important;
        border: 1px solid #2b405b !important;
        border-radius: 10px !important;
        font-family: Consolas, "SFMono-Regular", monospace !important;
        font-size: .9rem !important;
        line-height: 1.65 !important;
    }}
    [data-testid="stCodeBlock"] span {{
        opacity: 1 !important;
    }}
    [data-testid="stCodeBlock"] button {{
        color: #e8f0fa !important;
        background: #1a2a40 !important;
        border: 1px solid #38516d !important;
    }}

    /* Metric delta pills need dark text on their pale success backgrounds. */
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] svg {{
        color: #075b45 !important;
        -webkit-text-fill-color: #075b45 !important;
        opacity: 1 !important;
        font-weight: 650 !important;
    }}
    /* Keep dataframe surface and grid controls consistent with the dark UI. */
    [data-testid="stDataFrame"] {{
        background: #101b2d !important;
        border: 1px solid #2b405b !important;
    }}
    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="row"],
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="gridcell"] {{
        background-color: #101b2d !important;
        color: #eaf1fa !important;
        -webkit-text-fill-color: #eaf1fa !important;
    }}
    [data-testid="stDataFrame"] [role="columnheader"] {{
        background-color: #1a2a40 !important;
    }}

    /* Tables: readable header/body and clear row separation. */
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {{
        font-family: "Segoe UI", Arial, sans-serif !important;
    }}
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="columnheader"] *,
    [data-testid="stTable"] thead th {{
        color: #f4f7fc !important;
        -webkit-text-fill-color: currentColor !important;
        font-weight: 650 !important;
    }}

    /* Buttons have consistent contrast, focus, and hover feedback. */
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button,
    [data-testid="stPopover"] > button {{
        color: #f7fbff !important;
        -webkit-text-fill-color: #f7fbff !important;
        background: linear-gradient(135deg, #17415a, #18334d);
        border: 1px solid #438eaf;
        box-shadow: 0 4px 14px rgba(0, 0, 0, .16);
    }}
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] > button:focus-visible,
    input:focus-visible, textarea:focus-visible {{
        outline: 2px solid #79ddf8 !important;
        outline-offset: 2px !important;
    }}

    /* Tabs and expanders: distinct inactive/active states. */
    .stTabs [data-baseweb="tab"] {{
        color: #c4d1e1 !important;
        -webkit-text-fill-color: currentColor !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #8cdef5 !important;
        -webkit-text-fill-color: #8cdef5 !important;
        background: rgba(84, 199, 236, .08);
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {{
        color: #eaf1fa !important;
        -webkit-text-fill-color: currentColor !important;
    }}

    /* Links and inline code are visible without relying on browser defaults. */
    [data-testid="stMarkdownContainer"] a {{
        color: #83dff8 !important;
        text-decoration: underline;
        text-underline-offset: 3px;
    }}
    [data-testid="stMarkdownContainer"] code {{
        color: #d8f3ff !important;
        background: #1a2a40 !important;
        border: 1px solid #304762;
        border-radius: 5px;
        padding: .1rem .3rem;
    }}
    [data-testid="stMarkdownContainer"] pre {{
        background: #0b1422 !important;
        border: 1px solid #2b405b;
        border-radius: 10px;
    }}

    /* Alert / callout panels: Streamlit's default light alert surfaces conflicted
       with the global white alert text, making Stakeholder Findings and Model
       Performance messages appear blank. Keep every callout on a dark surface. */
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div,
    [data-testid="stAlert"] [data-baseweb="notification"] {{
        background: #17263a !important;
        border: 1px solid #38516d !important;
        border-radius: 12px !important;
        color: #f2f6fc !important;
    }}
    [data-testid="stAlert"] *,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] li {{
        color: #f2f6fc !important;
        -webkit-text-fill-color: #f2f6fc !important;
        opacity: 1 !important;
    }}
    [data-testid="stAlert"] a {{
        color: #8be2fb !important;
        -webkit-text-fill-color: #8be2fb !important;
    }}
    [data-testid="stAlert"] svg {{
        fill: currentColor !important;
    }}

    /* Status messages retain accessible text contrast. */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {{
        color: #edf4fc !important;
        -webkit-text-fill-color: currentColor !important;
    }}

    /* ------------------------------------------------------------------
       Global widget/menu audit: Streamlit/BaseWeb menus render in a portal
       outside the widget container. Style the portal itself, not only inputs.
       This fixes invisible selectbox, multiselect, radio and autocomplete
       options across every dashboard page.
    ------------------------------------------------------------------ */
    /* Streamlit/BaseWeb portals are mounted under document.body, not inside
       the widget wrapper. Target the portal, listbox, option and every nested
       text node explicitly so theme defaults cannot restore white surfaces. */
    body > div [data-baseweb="popover"],
    body > div [data-baseweb="popover"] > div,
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stMultiSelectVirtualDropdown"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] ul[role="listbox"],
    [data-baseweb="popover"] div[role="listbox"] {{
        background: #101d30 !important;
        background-color: #101d30 !important;
        color: #f4f7fc !important;
        border: 1px solid #49627f !important;
        box-shadow: 0 16px 40px rgba(0,0,0,.48) !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }}
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] li[role="option"],
    [data-baseweb="menu"] li,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stMultiSelectVirtualDropdown"] [role="option"],
    ul[role="listbox"] > li,
    div[role="listbox"] > div {{
        background: #101d30 !important;
        background-color: #101d30 !important;
        color: #f4f7fc !important;
        -webkit-text-fill-color: #f4f7fc !important;
        min-height: 2.4rem !important;
        opacity: 1 !important;
    }}
    [data-baseweb="popover"] [role="option"] *,
    [data-baseweb="popover"] li[role="option"] *,
    [data-baseweb="menu"] li *,
    [data-testid="stSelectboxVirtualDropdown"] *,
    [data-testid="stMultiSelectVirtualDropdown"] *,
    [role="listbox"] *,
    [role="option"] * {{
        color: #f4f7fc !important;
        -webkit-text-fill-color: #f4f7fc !important;
        opacity: 1 !important;
    }}
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [role="option"][aria-selected="true"],
    [data-baseweb="popover"] li[role="option"]:hover,
    [data-baseweb="popover"] li[role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li[aria-selected="true"],
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [role="option"][data-highlighted="true"] {{
        background: #294967 !important;
        background-color: #294967 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    [role="option"][aria-disabled="true"] {{
        color: #a8b8cc !important;
        -webkit-text-fill-color: #a8b8cc !important;
        opacity: 1 !important;
    }}
    [data-baseweb="popover"] svg,
    [data-baseweb="popover"] [role="option"] svg {{
        color: #dce8f5 !important;
        fill: #dce8f5 !important;
    }}

    /* Selectbox and multiselect closed states, tokens, clear/dropdown icons. */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] [role="combobox"] {{
        background: #111e31 !important;
        color: #f1f5fb !important;
        border-color: #405875 !important;
    }}
    [data-baseweb="select"] input,
    [data-baseweb="select"] [data-testid="stMarkdownContainer"],
    [data-baseweb="select"] span {{
        color: #f1f5fb !important;
        -webkit-text-fill-color: #f1f5fb !important;
    }}
    [data-baseweb="tag"] {{
        background: #24415d !important;
        border: 1px solid #4b6b8c !important;
    }}
    [data-baseweb="tag"] *,
    [data-baseweb="select"] svg {{
        color: #f1f5fb !important;
        fill: #f1f5fb !important;
    }}

    /* Number-input steppers and all form controls: visible at rest and hover. */
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button *,
    [data-testid="stDateInput"] button,
    [data-testid="stTimeInput"] button {{
        background: #1b3048 !important;
        color: #f1f5fb !important;
        -webkit-text-fill-color: #f1f5fb !important;
        border-color: #405875 !important;
        opacity: 1 !important;
    }}
    [data-testid="stNumberInput"] button:hover,
    [data-testid="stDateInput"] button:hover,
    [data-testid="stTimeInput"] button:hover {{
        background: #2a4b6a !important;
    }}
    [data-testid="stCheckbox"] label *,
    [data-testid="stRadio"] label *,
    [data-testid="stToggle"] label * {{
        color: #eaf1fa !important;
        -webkit-text-fill-color: #eaf1fa !important;
    }}

    /* Tables: style both Streamlit's canvas grid and ordinary HTML tables. */
    [data-testid="stTable"] table,
    [data-testid="stTable"] thead,
    [data-testid="stTable"] tbody,
    [data-testid="stTable"] tr,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {{
        background: #101b2d !important;
        color: #eaf1fa !important;
        -webkit-text-fill-color: #eaf1fa !important;
        border-color: #304762 !important;
    }}
    [data-testid="stTable"] th {{
        background: #1a2a40 !important;
        color: #ffffff !important;
        font-weight: 650 !important;
    }}
    [data-testid="stDataFrame"] [role="gridcell"] *,
    [data-testid="stDataFrame"] [role="columnheader"] * {{
        color: #eaf1fa !important;
        -webkit-text-fill-color: #eaf1fa !important;
    }}

    /* Plotly tooltips and controls use the same readable palette. */
    .js-plotly-plot .plotly .modebar,
    .js-plotly-plot .plotly .modebar * {{
        color: #eaf1fa !important;
        fill: #eaf1fa !important;
    }}

    /* Keep layout clean and usable on narrow screens. */
    @media (max-width: 768px) {{
        [data-testid="stMain"] .block-container {{
            padding: 1.1rem .85rem 2rem;
        }}
        [data-testid="stMetric"] {{ padding: .8rem; }}
        .stTabs [data-baseweb="tab"] {{ padding: .55rem .65rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Fast data loader — only precomputed scored table
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading scored data…")
def load_dashboard() -> pd.DataFrame:
    pkl = ROOT / "data" / "processed" / "dashboard.pkl"
    pq = ROOT / "data" / "processed" / "dashboard.parquet"
    if pkl.exists():
        df = pd.read_pickle(pkl)
    elif pq.exists():
        df = pd.read_parquet(pq)
    else:
        st.error("Precomputed dashboard data not found. Run the pipeline first.")
        st.stop()
    # Ensure dtypes that plotly likes
    if "purchase_time" in df.columns:
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], utc=True, errors="coerce")
    if "signup_time" in df.columns:
        df["signup_time"] = pd.to_datetime(df["signup_time"], utc=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def sample_df(df: pd.DataFrame, n: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """Fixed sample for heavy plots — keeps UI responsive."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🕵️ Fraud Graph Anomaly")
st.sidebar.caption("Pre-scored · low-latency dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Data Explorer",
        "🕸️ Graph Insights",
        "🚨 Anomaly Ranking",
        "🔍 Investigation Workbench",
        "📈 Model Performance",
        "🧪 Research Monitor",
        "🧠 Neural Fraud Classifier",
        "📋 Stakeholder Findings",
        "ℹ️ About",
    ],
    label_visibility="collapsed",
)

# Load once (cached)
df = load_dashboard()
df_sample = sample_df(df, 20_000)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Dynamic E-Commerce Graph Anomaly Detection")
    st.markdown(
        "Unsupervised anomaly ranking on a multipartite transaction graph. "
        "Fraud labels are used **only for validation**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud rate (label)", f"{df['class'].mean()*100:.2f}%")
    c3.metric("Unique users", f"{df['user_id'].nunique():,}")
    c4.metric("Unique devices", f"{df['device_id'].nunique():,}")

    pie_left, pie_right = st.columns(2)
    with pie_left:
        class_mix = df["class"].map({0: "Normal", 1: "Fraud label"}).value_counts().rename_axis("Class").reset_index(name="Transactions")
        fig = px.pie(class_mix, names="Class", values="Transactions", hole=0.62,
                     title="Label distribution", color="Class",
                     color_discrete_map={"Normal": _CYAN, "Fraud label": _DANGER})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True)
    with pie_right:
        if "source" in df.columns:
            source_mix = df["source"].fillna("Unknown").value_counts().head(8).rename_axis("Source").reset_index(name="Transactions")
            fig = px.pie(source_mix, names="Source", values="Transactions", hole=0.62,
                         title="Transaction source mix")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

    if "purchase_time" in df.columns:
        daily = (df.dropna(subset=["purchase_time"]).set_index("purchase_time")
                 .resample("D").agg(transactions=("class", "size"), fraud_rate=("class", "mean"))
                 .reset_index())
        trend_left, trend_right = st.columns(2)
        with trend_left:
            fig = px.area(daily, x="purchase_time", y="transactions", title="Daily transaction volume")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with trend_right:
            fig = px.line(daily, x="purchase_time", y="fraud_rate", title="Daily observed fraud-label rate")
            fig.update_yaxes(tickformat=".1%")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Score Distribution")
    plot_df = df_sample.copy()
    plot_df["label"] = plot_df["class"].map({0: "Normal", 1: "Fraud (label)"})
    fig = px.histogram(
        plot_df,
        x="anomaly_score",
        color="label",
        nbins=60,
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={"Normal": _CYAN, "Fraud (label)": _DANGER},
        title="Anomaly scores (higher = more anomalous) — sampled",
    )
    fig.update_layout(height=360, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top-20 Most Anomalous Transactions")
    top = df.nsmallest(20, "anomaly_rank")
    show_cols = [c for c in [
        "anomaly_rank", "anomaly_score", "user_id", "device_id",
        "purchase_value", "account_age_hours", "country", "class"
    ] if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True, height=420)

elif page == "📊 Data Explorer":
    st.title("Data Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        countries = st.multiselect(
            "Country",
            options=sorted(df["country"].dropna().unique())[:80],
            default=[],
        )
    with col2:
        class_filter = st.selectbox("Fraud label", ["All", "Fraud only", "Normal only"])
    with col3:
        max_rows = st.slider("Max rows to show", 100, 2000, 500, 100)

    view = df
    if countries:
        view = view[view["country"].isin(countries)]
    if class_filter == "Fraud only":
        view = view[view["class"] == 1]
    elif class_filter == "Normal only":
        view = view[view["class"] == 0]

    st.caption(f"{len(view):,} rows match filters")

    tab1, tab2, tab3 = st.tabs(["Table", "Purchase Value", "Temporal"])
    with tab1:
        st.dataframe(view.head(max_rows), use_container_width=True, height=400)
    with tab2:
        sample = view if len(view) < 15_000 else view.sample(15_000, random_state=1)
        fig = px.box(
            sample, x="class", y="purchase_value", color="class",
            labels={"class": "Fraud label"}, title="Purchase value by label (sampled)",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if "purchase_time" in view.columns:
            daily = (
                view.dropna(subset=["purchase_time"])
                .set_index("purchase_time")
                .resample("D")
                .size()
                .reset_index(name="count")
            )
            fig = px.line(daily, x="purchase_time", y="count", title="Daily volume")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🕸️ Graph Insights":
    st.title("Graph & Entity Insights")
    st.markdown(
        "Transactions link to **user / device / IP** nodes. "
        "Shared devices and IPs are strong collusion signals."
    )

    # Pre-aggregate once (cached via st.cache_data wrapper below)
    @st.cache_data(show_spinner=False)
    def entity_stats(raw: pd.DataFrame):
        users_per_device = raw.groupby("device_id")["user_id"].nunique()
        users_per_ip = raw.groupby("ip_address")["user_id"].nunique()
        high = (
            raw.groupby("device_id")
            .agg(
                n_users=("user_id", "nunique"),
                n_tx=("user_id", "size"),
                fraud_rate=("class", "mean"),
            )
            .query("n_users > 2")
            .sort_values("fraud_rate", ascending=False)
            .head(20)
        )
        return users_per_device, users_per_ip, high

    users_per_device, users_per_ip, high = entity_stats(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg users / device", f"{users_per_device.mean():.2f}")
    c2.metric("Avg users / IP", f"{users_per_ip.mean():.2f}")
    c3.metric("Devices with >1 user", f"{(users_per_device > 1).sum():,}")

    fig = px.histogram(
        users_per_device.clip(upper=20),
        nbins=20,
        title="Users sharing a device (clipped at 20)",
        labels={"value": "Users per device"},
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-risk shared devices")
    st.dataframe(high, use_container_width=True)

elif page == "🚨 Anomaly Ranking":
    st.title("Anomaly Ranking Workbench")

    top_n = st.slider("Show top-N anomalies", 10, 300, 50, 10)
    q = st.slider(
        "Minimum score percentile",
        50, 99, 90, 1,
        help="Only show transactions above this score percentile",
    )
    min_score = float(np.percentile(df["anomaly_score"], q))

    ranked = df[df["anomaly_score"] >= min_score].nsmallest(top_n, "anomaly_rank")
    st.caption(f"{len(ranked)} transactions · min score {min_score:.4f}")

    display_cols = [c for c in [
        "anomaly_rank", "anomaly_score", "user_id", "device_id", "ip_address",
        "purchase_value", "country", "source", "browser", "class"
    ] if c in ranked.columns]
    st.dataframe(ranked[display_cols], use_container_width=True, height=480)

    csv = ranked.to_csv(index=False).encode("utf-8")
    st.download_button("Download ranked CSV", csv, "anomaly_ranking.csv", "text/csv")

elif page == "🔍 Investigation Workbench":
    st.title("Investigation Workbench")
    st.markdown("Drill into a high-ranking transaction or entity.")

    mode = st.radio("Investigate by", ["Transaction rank", "User ID", "Device ID"], horizontal=True)

    if mode == "Transaction rank":
        rank = st.number_input("Anomaly rank (1 = most anomalous)", 1, int(df["anomaly_rank"].max()), 1)
        rows = df[df["anomaly_rank"] == rank]
        if rows.empty:
            st.warning("Rank not found")
            st.stop()
        row = rows.iloc[0]
    elif mode == "User ID":
        # Limit selectbox options for speed
        top_users = (
            df.nsmallest(3000, "anomaly_rank")["user_id"].unique().tolist()
        )
        uid = st.selectbox("User ID (from top anomalous)", top_users)
        row = df[df["user_id"] == uid].sort_values("anomaly_score", ascending=False).iloc[0]
    else:
        top_devices = (
            df.nsmallest(3000, "anomaly_rank")["device_id"].unique().tolist()
        )
        did = st.selectbox("Device ID (from top anomalous)", top_devices)
        row = df[df["device_id"] == did].sort_values("anomaly_score", ascending=False).iloc[0]

    st.subheader("Selected transaction")
    info = {
        "anomaly_rank": int(row["anomaly_rank"]),
        "anomaly_score": float(row["anomaly_score"]),
        "user_id": int(row["user_id"]),
        "device_id": str(row["device_id"]),
        "purchase_value": float(row["purchase_value"]),
        "country": str(row.get("country", "")),
        "class (label)": int(row["class"]),
    }
    if "account_age_hours" in row:
        info["account_age_hours"] = float(row["account_age_hours"]) if pd.notna(row["account_age_hours"]) else None
    import html as _html
    # Render each field separately so the layout remains readable even if the
    # browser/Streamlit markdown layer collapses whitespace inside <pre>.
    detail_rows = []
    for field, value in info.items():
        safe_field = _html.escape(str(field))
        safe_value = _html.escape(str(value))
        detail_rows.append(
            f'<div style="display:grid;grid-template-columns:minmax(180px, .32fr) 1fr;'
            f'gap:14px;padding:9px 12px;border-bottom:1px solid #263750;">'
            f'<span style="color:#9fb4cc;font-weight:600;">{safe_field}</span>'
            f'<span style="color:#f1f6fc;font-family:Consolas,monospace;'
            f'overflow-wrap:anywhere;">{safe_value}</span></div>'
        )
    st.markdown(
        '<div style="background:#0b1422;border:1px solid #38516d;'
        'border-radius:12px;padding:8px 12px;margin-bottom:12px;">'
        + ''.join(detail_rows) + '</div>',
        unsafe_allow_html=True,
    )

    same_user = df[df["user_id"] == row["user_id"]]
    same_device = df[df["device_id"] == row["device_id"]]
    same_ip = df[df["ip_address"] == row["ip_address"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Tx same user", len(same_user), f"fraud {same_user['class'].mean()*100:.1f}%")
    c2.metric("Tx same device", len(same_device), f"fraud {same_device['class'].mean()*100:.1f}%")
    c3.metric("Tx same IP", len(same_ip), f"fraud {same_ip['class'].mean()*100:.1f}%")

    st.markdown("**Other transactions on the same device**")
    cols = [c for c in ["purchase_time", "user_id", "purchase_value", "anomaly_score", "class"] if c in same_device.columns]
    st.dataframe(
        same_device[cols].sort_values("anomaly_score", ascending=False).head(25),
        use_container_width=True,
        height=320,
    )

elif page == "📈 Model Performance":
    st.title("Validation Against Fraud Labels")
    st.warning(
        "Post-hoc validation only — the model never trained on the `class` label."
    )

    from evaluation import evaluate_anomaly_scores, enrichment_factor
    from sklearn.metrics import precision_recall_curve, roc_curve

    y = df["class"].values
    s = df["anomaly_score"].values
    metrics = evaluate_anomaly_scores(y, s)
    k = max(int(y.sum()), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    c3.metric(f"Precision@{k}", f"{metrics['precision_at_k']:.4f}")
    c4.metric("Enrichment@k", f"{enrichment_factor(y, s, k):.2f}×")

    # Curves on a sample for speed if needed (full data is fine for curves)
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR", line=dict(color=_CYAN, width=2)))
        fig1.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision", height=360)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color=_CYAN, width=2)))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color=_MUTED)))
        fig2.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Score by label (sampled)")
    fig = px.box(
        df_sample, x="class", y="anomaly_score", color="class",
        labels={"class": "Fraud label"}, points=False,
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)




elif page == "🧪 Research Monitor":
    st.title("Temporal Fraud Research Monitor")
    st.caption("Walk-forward model selection, calibration, distribution drift and investigation-capacity diagnostics.")
    import json as _json
    research_path = ROOT / "reports" / "research" / "calibrated_temporal_report.json"
    if not research_path.exists():
        st.info("The calibrated temporal research workflow has not published a report yet. Run the Calibrated temporal fraud research workflow from GitHub Actions.")
    else:
        report = _json.loads(research_path.read_text())
        fold = report["final_fold"]
        metrics = fold["metrics"]
        raw, calibrated = metrics["raw"], metrics["isotonic_calibrated"]
        metric_cols = st.columns(6)
        metric_cols[0].metric("Raw PR-AUC", f"{raw['pr_auc']:.4f}")
        metric_cols[1].metric("Calibrated PR-AUC", f"{calibrated['pr_auc']:.4f}")
        metric_cols[2].metric("Raw ROC-AUC", f"{raw['roc_auc']:.4f}")
        metric_cols[3].metric("Calibrated ROC-AUC", f"{calibrated['roc_auc']:.4f}")
        metric_cols[4].metric("Raw Brier", f"{raw['brier']:.4f}")
        metric_cols[5].metric("Calibrated Brier", f"{calibrated['brier']:.4f}", delta=f"{calibrated['brier']-raw['brier']:.4f}", delta_color="inverse")
        st.caption(f"Final test window: {fold['test']['start']} → {fold['test']['end']} · {fold['test']['rows']:,} transactions · config {report['selected_config_id']} · graph embeddings: {'included' if report['selected_include_graph_embeddings'] else 'excluded'}")
        st.warning("The final test window is a one-shot evaluation. Do not use these test charts to retune the model or thresholds.")

        cap_rows = []
        for capacity, result in metrics["capacity_thresholds"].items():
            topk = result["test_top_k_metrics"]
            cap_rows.extend([
                {"Capacity": f"{float(capacity):.0%}", "Metric": "Precision", "Value": topk["precision"]},
                {"Capacity": f"{float(capacity):.0%}", "Metric": "Recall", "Value": topk["recall"]},
                {"Capacity": f"{float(capacity):.0%}", "Metric": "F1", "Value": topk["f1"]},
            ])
        left, right = st.columns(2)
        with left:
            fig = px.bar(pd.DataFrame(cap_rows), x="Capacity", y="Value", color="Metric",
                         barmode="group", title="Final test: investigation capacity")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            cal = metrics["validation_calibration"]
            cal_rows = []
            for label, key in [("Raw", "raw_bins"), ("Isotonic", "calibrated_bins")]:
                for row in cal[key]:
                    cal_rows.append({"Model": label, "Mean predicted": row["mean_probability"],
                                     "Observed fraud rate": row["observed_rate"], "Count": row["count"]})
            fig = px.scatter(pd.DataFrame(cal_rows), x="Mean predicted", y="Observed fraud rate",
                             color="Model", size="Count", title="Validation reliability diagram")
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                     line=dict(dash="dash", color=_MUTED)))
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        drift = pd.DataFrame([{"Feature": name, "PSI": item["psi"], "Method": item["method"]}
                              for name, item in fold["feature_drift"].items() if item["psi"] is not None])
        if not drift.empty:
            drift = drift.sort_values("PSI", ascending=False).head(20)
            fig = px.bar(drift, x="PSI", y="Feature", color="Method", orientation="h",
                         title="Largest feature shifts (categorical-aware PSI)")
            fig.add_vline(x=0.20, line_dash="dash", line_color=_AMBER, annotation_text="0.20 monitor")
            fig.add_vline(x=0.30, line_dash="dot", line_color=_DANGER, annotation_text="0.30 high")
            fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        emb = fold["graph_embedding_diagnostics"]
        unseen = pd.DataFrame([
            {"Entity": "User", "Unseen rate": emb["user_id_unseen_rate"]},
            {"Entity": "Device", "Unseen rate": emb["device_id_unseen_rate"]},
            {"Entity": "IP", "Unseen rate": emb["ip_address_unseen_rate"]},
        ])
        g1, g2 = st.columns(2)
        with g1:
            fig = px.bar(unseen, x="Entity", y="Unseen rate", title="Future entities absent from training vocabulary")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            fig = px.pie(pd.DataFrame({"Embedding coverage": ["Non-zero projection", "Zero projection"],
                                       "Rows": [1 - emb["future_zero_embedding_rate"], emb["future_zero_embedding_rate"]]}),
                         names="Embedding coverage", values="Rows", hole=0.62,
                         title="Future graph embedding coverage")
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        component_drift = pd.DataFrame(emb.get("component_distribution_drift", []))
        if not component_drift.empty:
            fig = px.bar(component_drift, x="component", y="mean_shift_in_train_std",
                         title="Graph embedding component mean shift (training standard deviations)")
            fig.add_hline(y=0, line_dash="dash", line_color=_MUTED)
            fig.update_layout(height=320, xaxis_title="Frozen SVD component", yaxis_title="Mean shift / train SD")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(component_drift, use_container_width=True)
        with st.expander("Model tuning results (earlier validation folds only)"):
            tuning = pd.DataFrame(report["tuning"])
            tuning["params"] = tuning["params"].astype(str)
            fig = px.box(tuning, x="config_id", y="validation_pr_auc", color="fold",
                         points="all", title="Validation PR-AUC by configuration")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tuning, use_container_width=True)
        expanded_path = ROOT / "reports" / "research" / "expanded_model_comparison.json"
        if expanded_path.exists():
            expanded = _json.loads(expanded_path.read_text())
            final = expanded["final_fold"]
            st.divider()
            st.subheader("Expanded model comparison · CatBoost / LightGBM / XGBoost")
            st.caption(expanded["protocol"])
            ec1, ec2, ec3, ec4 = st.columns(4)
            em = final["metrics"]
            ec1.metric("Champion", final["selected_model"].upper())
            ec2.metric("External test PR-AUC", f"{em['pr_auc']:.4f}")
            ec3.metric("ROC-AUC", f"{em['roc_auc']:.4f}")
            ec4.metric("PR-AUC / prevalence", f"{em['pr_auc_lift_over_prevalence']:.2f}×")
            validation = pd.DataFrame(expanded["validation_rows"])
            fig = px.box(validation, x="model", y="validation_pr_auc", color="model",
                         points="all", title="Earlier-fold validation PR-AUC · model/configuration comparison")
            fig.add_hline(y=expanded["final_test_prevalence"], line_dash="dash",
                          line_color=_AMBER, annotation_text="Final test prevalence baseline")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            comparison = expanded.get("final_test_model_comparison", {})
            if comparison:
                final_compare = pd.DataFrame([
                    {"Model": name.upper(), "PR-AUC": result["test_pr_auc"],
                     "ROC-AUC": result["test_roc_auc"]}
                    for name, result in comparison.items()
                ])
                fig = px.bar(final_compare, x="Model", y="PR-AUC", color="Model",
                             title="Same final chronological holdout · PR-AUC")
                fig.add_hline(y=expanded["final_test_prevalence"], line_dash="dash",
                              line_color=_AMBER, annotation_text="Random-ranking baseline")
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(final_compare, use_container_width=True)
            else:
                st.info("Same-holdout comparison across all three model families is being generated by the latest research run.")
            cap_data = []
            for cap, result in em["capacity"].items():
                for metric in ("precision", "recall", "f1"):
                    cap_data.append({"Capacity": f"{float(cap):.0%}", "Metric": metric.title(),
                                     "Value": result["test_exact_capacity"][metric]})
            fig = px.bar(pd.DataFrame(cap_data), x="Capacity", y="Value", color="Metric",
                         barmode="group", title="Expanded champion · exact test investigation capacity")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(cap_data), use_container_width=True)
            st.info("Independent later-period validation: " + expanded["external_validation"]["status"] +
                    ". " + expanded["external_validation"]["reason"])
        st.caption(report["interpretation"])

elif page == "🧠 Neural Fraud Classifier":
    st.title("Neural Fraud Classifier (MLP)")
    st.markdown(
        """
        PyTorch **MLP** with leakage-safe features.  
        Split: stratified 70/15/15 · Primary metric: **PR-AUC**  
        Artifact scores **raw CSV columns** (same as `Fraud_Data.csv` without `class`).
        """
    )

    @st.cache_resource(show_spinner="Loading neural model…")
    def load_neural():
        path = ROOT / "artifacts" / "final_supervised_neural_model.joblib"
        summary_path = ROOT / "artifacts" / "neural_fraud_summary.json"
        if not path.exists():
            return None, None
        from supervised_fraud import load_artifact
        import json as _json
        art = load_artifact(path)
        summary = _json.loads(summary_path.read_text()) if summary_path.exists() else {}
        return art, summary

    @st.cache_data(show_spinner="Loading test predictions…")
    def load_test_predictions():
        path = ROOT / "artifacts" / "final_test_predictions.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    art, summary = load_neural()
    if art is None:
        st.error("Missing `artifacts/final_supervised_neural_model.joblib`.")
        st.stop()

    tm = summary.get("test_metrics", {})
    thr = float(summary.get("threshold") or art.metadata.get("threshold", 0.5))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion", str(summary.get("champion_type", art.model_type)).upper())
    c2.metric("Test PR-AUC", f"{tm.get('pr_auc', float('nan')):.4f}")
    c3.metric("Test ROC-AUC", f"{tm.get('roc_auc', float('nan')):.4f}")
    c4.metric("Threshold", f"{thr:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Precision", f"{tm.get('precision', float('nan')):.3f}")
    c6.metric("Recall", f"{tm.get('recall', float('nan')):.3f}")
    c7.metric("F1", f"{tm.get('f1', float('nan')):.3f}")
    c8.metric("Accuracy", f"{tm.get('accuracy', float('nan')):.3f}")

    tab_pred, tab_csv, tab_batch = st.tabs([
        "🆕 New prediction (raw fields)",
        "📄 Locked test predictions CSV",
        "📥 Batch CSV upload",
    ])

    with tab_pred:
        st.subheader("Enter one transaction (original raw columns)")
        st.caption("Same fields as `Fraud_Data.csv` except `class` (not required for scoring).")
        with st.form("neural_raw_form"):
            r1, r2 = st.columns(2)
            with r1:
                user_id = st.number_input("user_id", value=22058, step=1)
                signup_time = st.text_input("signup_time", value="2015-02-24 22:55:49")
                purchase_time = st.text_input("purchase_time", value="2015-04-18 02:47:11")
                purchase_value = st.number_input("purchase_value", value=34.0, min_value=0.0)
                device_id = st.text_input("device_id", value="QVPSPJUOCKZAR")
            with r2:
                source = st.selectbox("source", ["SEO", "Ads", "Direct"], index=0)
                browser = st.selectbox("browser", ["Chrome", "IE", "Safari", "FireFox", "Opera"], index=0)
                sex = st.selectbox("sex", ["M", "F"], index=0)
                age = st.number_input("age", value=39, min_value=1, max_value=120)
                ip_address = st.number_input("ip_address (numeric)", value=732758368.0, format="%.0f")
            go = st.form_submit_button("Predict fraud probability", type="primary")

        if go:
            row = pd.DataFrame([{
                "user_id": int(user_id),
                "signup_time": signup_time,
                "purchase_time": purchase_time,
                "purchase_value": float(purchase_value),
                "device_id": str(device_id),
                "source": source,
                "browser": browser,
                "sex": sex,
                "age": int(age),
                "ip_address": float(ip_address),
            }])
            try:
                proba = float(art.predict_proba(row)[0, 1])
                pred = int(proba >= thr)
                if pred:
                    st.error(f"**FRAUD**  ·  probability = **{proba:.4f}**  (threshold {thr:.3f})")
                else:
                    st.success(f"**Normal**  ·  probability = **{proba:.4f}**  (threshold {thr:.3f})")
                import json as _json
                import html as _html
                prediction_payload = {
                    "fraud_probability": proba,
                    "predicted_class": pred,
                    "threshold": thr,
                }
                prediction_json = _html.escape(
                    _json.dumps(prediction_payload, indent=2, ensure_ascii=False)
                )
                st.markdown(
                    '<div style="background:#0b1422;border:1px solid #38516d;'
                    'border-radius:12px;padding:14px 18px;overflow-x:auto;">'
                    f'<pre style="margin:0;background:transparent;color:#f1f6fc;'
                    f'font-family:Consolas,monospace;font-size:14px;line-height:1.7;">'
                    f'{prediction_json}</pre></div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_csv:
        st.subheader("Locked test set — scored predictions")
        pred_df = load_test_predictions()
        if pred_df is None:
            st.warning("`artifacts/final_test_predictions.csv` not found.")
        else:
            st.caption(f"{len(pred_df):,} rows · columns include raw fields + fraud_probability + predicted_class + actual_class")
            filt = st.selectbox(
                "Filter",
                ["All", "Predicted fraud", "Predicted normal", "Correct", "False positive", "False negative"],
            )
            view = pred_df
            if filt == "Predicted fraud":
                view = pred_df[pred_df["predicted_class"] == 1]
            elif filt == "Predicted normal":
                view = pred_df[pred_df["predicted_class"] == 0]
            elif filt == "Correct":
                view = pred_df[pred_df["predicted_class"] == pred_df["actual_class"]]
            elif filt == "False positive":
                view = pred_df[(pred_df["predicted_class"] == 1) & (pred_df["actual_class"] == 0)]
            elif filt == "False negative":
                view = pred_df[(pred_df["predicted_class"] == 0) & (pred_df["actual_class"] == 1)]

            n_show = st.slider("Rows to display", 20, 500, 100, 20)
            st.dataframe(
                view.sort_values("fraud_probability", ascending=False).head(n_show),
                use_container_width=True,
                height=420,
            )
            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full final_test_predictions.csv",
                csv_bytes,
                file_name="final_test_predictions.csv",
                mime="text/csv",
            )

    with tab_batch:
        st.subheader("Batch score a CSV (raw columns)")
        st.markdown(
            "Upload a CSV with columns: "
            "`user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address`"
        )
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                batch = pd.read_csv(up)
                missing = [c for c in [
                    "user_id", "signup_time", "purchase_time", "purchase_value",
                    "device_id", "source", "browser", "sex", "age", "ip_address"
                ] if c not in batch.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    with st.spinner("Scoring…"):
                        proba = art.predict_proba(batch)[:, 1]
                    scored = batch.copy()
                    scored["fraud_probability"] = proba
                    scored["predicted_class"] = (proba >= thr).astype(int)
                    st.dataframe(
                        scored.sort_values("fraud_probability", ascending=False).head(100),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download scored CSV",
                        scored.to_csv(index=False).encode("utf-8"),
                        file_name="scored_transactions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Batch scoring failed: {e}")


elif page == "📋 Stakeholder Findings":
    st.title("Stakeholder Findings — What We Learned")
    import json as _json
    findings_path = ROOT / "artifacts" / "stakeholder_findings.json"
    if findings_path.exists():
        findings = _json.loads(findings_path.read_text())
    else:
        findings = {}

    st.header("1. Executive summary")
    st.markdown(f"**{findings.get('headline', '')}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Majority baseline accuracy", f"{findings.get('majority_baseline_accuracy', 0.95):.1%}")
    c2.metric("Fraud rate in locked test", f"{findings.get('fraud_base_rate_test', 0.046):.1%}")
    model_block = findings.get("model_on_locked_test", {})
    c3.metric("Model enrichment @ top-k", f"{model_block.get('enrichment_at_k', 0):.1f}×")

    st.header("2. Why accuracy looks “worse than random”")
    st.warning(findings.get("why_accuracy_misleads", ""))
    st.markdown(
        """
        | Approach | Approx. accuracy on test | Catches fraud? |
        |----------|--------------------------|----------------|
        | Always predict normal | **~95%** | No |
        | Supervised model (F1 threshold) | ~92% | Partially — ranks risk |
        | Random guessing of fraud labels | ~50% balanced | No |
        """
    )
    st.caption("“Worse than majority accuracy” ≠ “worse than random ranking”. Ranking metrics are the right lens.")

    st.header("3. Critical data insight (concept drift)")
    for bullet in findings.get("key_data_insights", []):
        st.markdown(f"- {bullet}")

    st.header("4. Simple rule vs model")
    rule = findings.get("simple_rule_train", {})
    st.markdown(
        f"""
        **Rule:** `{rule.get('rule', '')}`

        - Train precision: **{rule.get('train_precision', '—')}** (historically perfect on that pattern)
        - Train recall of fraud: **{rule.get('train_recall_of_fraud', '—')}**
        - Val/Test recall: **{rule.get('val_test_recall', '—')}**
        - Note: {rule.get('note', '')}
        """
    )

    st.header("5. Model quality on locked future test")
    st.markdown(
        f"""
        - **Model:** {model_block.get('selected_model', '—')}
        - **PR-AUC:** {model_block.get('pr_auc', '—')} (higher than base rate ≈ 0.046)
        - **ROC-AUC:** {model_block.get('roc_auc', '—')}
        - **Precision@k:** {model_block.get('precision_at_k', '—')} with **{model_block.get('enrichment_at_k', '—')}×** enrichment
        - **Accuracy:** {model_block.get('accuracy', '—')} (below majority baseline — expected)

        {model_block.get('interpretation', '')}
        """
    )

    st.header("6. Recommended use for the business")
    for rec in findings.get("recommended_use", []):
        st.markdown(f"- {rec}")
    og = findings.get("operating_guidance", {})
    st.markdown(f"**Default workflow:** {og.get('default', '')}")
    st.markdown(f"**Threshold policy:** {og.get('threshold_policy', '')}")

    st.header("7. Portfolio takeaway")
    st.success(
        "This project is honest about limits: chronological evaluation, label used only for validation "
        "in the unsupervised track, explicit imbalance handling, and clear separation between "
        "ranking quality and accuracy. The main scientific finding is severe **temporal drift** "
        "in attack patterns — instant signup fraud dominates early data and vanishes later."
    )

elif page == "ℹ️ About":
    st.title("About this project")
    st.markdown(
        """
        ### Scientific stance
        - Fraud label is **validation ground truth only**
        - Chronological train / val / test split
        - Graph features (degree, shared device/IP) + temporal + behavioural
        - This UI loads **pre-scored** data for low latency

        ### Pipeline
        Dataset → Audit → Graph → Features → IsolationForest → Ranking → Validation → Dashboard

        ### Run
        ```bash
        pip install -r requirements.txt
        streamlit run app/app.py
        ```
        """
    )
