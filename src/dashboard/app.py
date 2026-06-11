from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crowd Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS + IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import CrowdAnalysisAgent


FRAME_CSV = PROJECT_ROOT / "results" / "benchmark" / "FULL_01_shinjuku_frame_counts.csv"
ZONE_CSV = PROJECT_ROOT / "results" / "benchmark" / "FULL_02_shinjuku_zone_density_risk.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
VIDEOS_DIR = RESULTS_DIR / "videos"
VIDEOS_DASHBOARD_DIR = RESULTS_DIR / "videos_dashboard"

VIDEO_CANDIDATES = {
    "Localization": [
        VIDEOS_DASHBOARD_DIR / "FULL_01_shinjuku_fidtm_localization_count.mp4",
        VIDEOS_DIR / "FULL_01_shinjuku_fidtm_localization_count.mp4",
    ],
    "Heatmap Overlay": [
        VIDEOS_DASHBOARD_DIR / "FULL_02_shinjuku_fidtm_heatmap_overlay_points.mp4",
        VIDEOS_DIR / "FULL_02_shinjuku_fidtm_heatmap_overlay_points.mp4",
    ],
    "Heatmap Only": [
        VIDEOS_DASHBOARD_DIR / "FULL_03_shinjuku_fidtm_heatmap_only_points.mp4",
        VIDEOS_DIR / "FULL_03_shinjuku_fidtm_heatmap_only_points.mp4",
    ],
    "Zone Density + Risk": [
        VIDEOS_DASHBOARD_DIR / "FULL_04_shinjuku_fidtm_zone_density_risk.mp4",
        VIDEOS_DIR / "FULL_04_shinjuku_fidtm_zone_density_risk.mp4",
        VIDEOS_DIR / "FULL_04_shinjuku_zone_density_risk.mp4",
    ],
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    :root {
        --bg0: #020814;
        --panel: rgba(7, 20, 44, 0.92);
        --panel2: rgba(10, 28, 60, 0.94);
        --border: rgba(87, 146, 255, 0.25);
        --border2: rgba(87, 146, 255, 0.42);
        --text: #f5f9ff;
        --muted: #9fb4d3;
        --blue: #2ea8ff;
        --green: #35e6a1;
        --yellow: #fbbf24;
        --orange: #fb923c;
        --red: #fb7185;
        --purple: #a78bfa;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(46, 168, 255, 0.16), transparent 28%),
            radial-gradient(circle at 90% 2%, rgba(34, 211, 238, 0.10), transparent 28%),
            linear-gradient(135deg, #020814 0%, #031222 45%, #020814 100%);
        color: var(--text);
    }

    .block-container {
        max-width: none !important;
        width: 100% !important;
        padding: 0.45rem 0.55rem 0.65rem 0.55rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.58rem !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.70rem !important;
    }

    /* ============================================================
       LEFT NAV
       ============================================================ */

    .nav-marker {
        display: none;
    }

    .nav-inner-top {
        width: 100%;
    }

    .side-logo {
        width: 66px;
        height: 66px;
        margin: 0.00rem auto 0.64rem auto;
        border-radius: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, rgba(46, 168, 255, 0.80), rgba(11, 44, 96, 0.98));
        border: 1px solid rgba(125, 190, 255, 0.58);
        box-shadow: 0 0 32px rgba(46, 168, 255, 0.36);
        font-size: 1.95rem;
    }

    .nav-label {
        color: #8ca3c8;
        text-align: center;
        font-size: 0.64rem;
        font-weight: 900;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.48rem;
    }

    .nav-status {
        color: var(--muted);
        font-size: 0.70rem;
        line-height: 1.45;
        text-align: center;
        margin-top: 1.10rem;
        padding-top: 0.90rem;
        border-top: 1px solid rgba(87,146,255,0.18);
    }

    .nav-status-ok {
        color: #78f0b7;
        font-weight: 800;
    }

    .status-dot {
        color: var(--green);
        font-weight: 900;
    }

    .version {
        color: #7f91b0;
        text-align: center;
        font-size: 0.68rem;
        margin-top: 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker) {
        height: calc(100vh - 0.95rem) !important;
        min-height: 670px !important;
        max-height: none !important;
        background:
            radial-gradient(circle at 50% 0%, rgba(46, 168, 255, 0.25), transparent 30%),
            linear-gradient(180deg, #071a37 0%, #030a17 100%) !important;
        border: 1px solid rgba(87, 146, 255, 0.25) !important;
        border-radius: 18px !important;
        box-shadow: 0 18px 42px rgba(0,0,0,0.28) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker) > div {
        height: 100% !important;
        padding: 0.72rem 0.54rem !important;
    }

    div[data-testid="stRadio"] > label {
        display: none;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.46rem;
    }

    div[data-testid="stRadio"] label {
        background: rgba(8, 20, 43, 0.82);
        border: 1px solid rgba(87, 146, 255, 0.20);
        border-radius: 11px;
        padding: 0.36rem 0.60rem;
        min-height: 34px;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(46, 168, 255, 0.62), rgba(25, 80, 180, 0.36));
        border-color: rgba(102, 178, 255, 0.70);
        box-shadow: 0 0 20px rgba(46,168,255,0.20);
    }

    div[data-testid="stRadio"] label p {
        color: #dce8ff !important;
        font-size: 0.76rem !important;
        font-weight: 850 !important;
        white-space: nowrap;
    }

    .nav-radio-space div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }

    .nav-radio-space div[data-testid="stRadio"] label {
        width: 100%;
        min-height: 54px;
        padding: 0.76rem 0.58rem;
        border-radius: 15px;
        background: rgba(10, 31, 67, 0.72);
        border: 1px solid rgba(87, 146, 255, 0.25);
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }

    .nav-radio-space div[data-testid="stRadio"] label p {
        font-size: 0.82rem !important;
        font-weight: 900 !important;
    }

    /* ============================================================
       HEADER
       ============================================================ */

    .topbar {
        background: rgba(5, 16, 35, 0.84);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.90rem 1.10rem;
        box-shadow: 0 18px 42px rgba(0,0,0,0.26);
    }

    .topbar-grid {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.82rem;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 17px;
        background: linear-gradient(145deg, rgba(46, 168, 255, 0.78), rgba(8, 35, 78, 1));
        border: 1px solid rgba(122, 190, 255, 0.48);
        box-shadow: 0 0 28px rgba(46, 168, 255, 0.26);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.50rem;
    }

    .brand-title {
        font-size: 1.52rem;
        font-weight: 950;
        color: #ffffff;
        letter-spacing: -0.035em;
        line-height: 1.05;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 0.84rem;
        margin-top: 0.16rem;
    }

    .badges {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 0.52rem;
    }

    .badge {
        border-radius: 999px;
        padding: 0.54rem 0.78rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(8, 20, 43, 0.76);
        color: #dce8ff;
        font-size: 0.78rem;
        font-weight: 850;
        white-space: nowrap;
    }

    .badge.green {
        color: #78f0b7;
        border-color: rgba(53, 230, 161, 0.36);
        background: rgba(12, 88, 57, 0.28);
    }

    .badge.blue {
        color: #a8ccff;
        border-color: rgba(87, 146, 255, 0.36);
        background: rgba(46, 168, 255, 0.13);
    }

    /* ============================================================
       KPI
       ============================================================ */

    .kpi-card {
        height: 116px;
        background:
            radial-gradient(circle at 20% 20%, rgba(46, 168, 255, 0.18), transparent 36%),
            linear-gradient(145deg, rgba(9, 25, 52, 0.98), rgba(5, 16, 35, 0.94));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.78rem 0.84rem;
        box-shadow: 0 14px 34px rgba(0,0,0,0.21);
        overflow: hidden;
    }

    .kpi-layout {
        display: grid;
        grid-template-columns: 48px 1fr;
        gap: 0.70rem;
        align-items: center;
        height: 100%;
    }

    .kpi-icon {
        width: 46px;
        height: 46px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(46, 168, 255, 0.18);
        border: 1px solid rgba(87, 146, 255, 0.30);
        font-size: 1.25rem;
    }

    .kpi-label {
        color: #b4c5df;
        font-size: 0.76rem;
        font-weight: 850;
        margin-bottom: 0.14rem;
        white-space: nowrap;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 1.64rem;
        line-height: 1.04;
        font-weight: 950;
        letter-spacing: -0.035em;
    }

    .kpi-sub {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 0.22rem;
        line-height: 1.20;
        white-space: nowrap;
    }

    .kpi-good {
        color: #7df0b8;
    }

    /* ============================================================
       PANELS
       ============================================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            radial-gradient(circle at 100% 0%, rgba(46, 168, 255, 0.08), transparent 35%),
            rgba(7, 20, 44, 0.88) !important;
        border: 1px solid rgba(87, 146, 255, 0.24) !important;
        border-radius: 17px !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.22) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.82rem !important;
    }

    .panel-title {
        color: #ffffff;
        font-weight: 950;
        font-size: 1.03rem;
        letter-spacing: -0.015em;
        margin-bottom: 0;
    }

    .zone-header-simple {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin: 0.20rem 0 0.58rem 0;
    }

    .zone-name {
        color: #ffffff;
        font-size: 1.14rem;
        font-weight: 950;
        letter-spacing: -0.025em;
    }

    .zone-id {
        margin-left: 0.35rem;
        padding: 0.18rem 0.47rem;
        border-radius: 8px;
        background: rgba(46, 168, 255, 0.24);
        color: #a8ccff;
        border: 1px solid rgba(87, 146, 255, 0.30);
        font-size: 0.70rem;
        font-weight: 900;
    }

    .risk-pill {
        padding: 0.35rem 0.64rem;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 950;
        letter-spacing: 0.03em;
        border: 1px solid;
        white-space: nowrap;
    }

    .risk-low {
        background: rgba(34, 197, 94, 0.13);
        border-color: rgba(34, 197, 94, 0.38);
        color: #86efac;
    }

    .risk-medium {
        background: rgba(251, 191, 36, 0.13);
        border-color: rgba(251, 191, 36, 0.38);
        color: #fde68a;
    }

    .risk-high {
        background: rgba(251, 146, 60, 0.14);
        border-color: rgba(251, 146, 60, 0.42);
        color: #fdba74;
    }

    .risk-critical {
        background: rgba(251, 113, 133, 0.16);
        border-color: rgba(251, 113, 133, 0.45);
        color: #fda4af;
    }

    div[data-testid="stMetric"] {
        background: rgba(5, 16, 35, 0.70) !important;
        border: 1px solid rgba(87, 146, 255, 0.24) !important;
        border-radius: 12px !important;
        padding: 0.60rem 0.66rem !important;
        min-height: 82px !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bad7 !important;
        font-size: 0.70rem !important;
        font-weight: 850 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.20rem !important;
        font-weight: 950 !important;
        line-height: 1.05 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #78f0b7 !important;
        font-size: 0.64rem !important;
    }

    .chart-title {
        color: #ffffff;
        font-size: 0.92rem;
        font-weight: 950;
        margin-bottom: 0.48rem;
        letter-spacing: -0.015em;
        line-height: 1.15;
    }

    .analysis-card {
        background: rgba(7, 20, 44, 0.90);
        border: 1px solid rgba(87, 146, 255, 0.22);
        border-radius: 16px;
        padding: 0.98rem;
        min-height: 120px;
        box-shadow: 0 14px 36px rgba(0,0,0,0.20);
    }

    .analysis-card-title {
        color: #ffffff;
        font-size: 0.94rem;
        font-weight: 950;
        margin-bottom: 0.36rem;
    }

    .analysis-card-text {
        color: #b9c7dd;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .stSelectbox label {
        color: #cbd7ee !important;
        font-size: 0.74rem !important;
        font-weight: 850 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(8, 20, 43, 0.92) !important;
        border: 1px solid rgba(87, 146, 255, 0.28) !important;
        border-radius: 11px !important;
        min-height: 38px !important;
        color: #ffffff !important;
        font-size: 0.78rem !important;
    }

    .stButton button {
        background: rgba(8, 20, 43, 0.88) !important;
        border: 1px solid rgba(87, 146, 255, 0.24) !important;
        border-radius: 10px !important;
        color: #dce8ff !important;
        font-size: 0.76rem !important;
        font-weight: 850 !important;
        min-height: 38px !important;
        padding: 0.38rem 0.60rem !important;
    }

    .stButton button:hover {
        background: rgba(46, 168, 255, 0.24) !important;
        border-color: rgba(102, 178, 255, 0.65) !important;
        color: white !important;
    }

    .stTextInput input {
        background-color: rgba(8, 20, 43, 0.92) !important;
        border: 1px solid rgba(87, 146, 255, 0.22) !important;
        border-radius: 11px !important;
        color: #ffffff !important;
        height: 40px !important;
        font-size: 0.80rem !important;
    }

    div[data-testid="stVideo"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(87, 146, 255, 0.28);
        background: black;
        box-shadow: 0 16px 36px rgba(0,0,0,0.25);
    }

    video {
        max-height: 405px !important;
        object-fit: contain !important;
        background: black !important;
    }

    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }

    /* ============================================================
       AI ASSISTANT
       ============================================================ */

    .agent-shell {
        border-radius: 17px;
        overflow: hidden;
    }

    .agent-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.50rem;
    }

    .agent-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .agent-orb {
        width: 38px;
        height: 38px;
        border-radius: 14px;
        background:
            radial-gradient(circle at 35% 25%, rgba(255,255,255,0.35), transparent 28%),
            linear-gradient(145deg, rgba(46,168,255,0.85), rgba(18,75,160,0.72));
        border: 1px solid rgba(125,190,255,0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 22px rgba(46,168,255,0.22);
        font-size: 1.08rem;
    }

    .agent-title {
        color: #ffffff;
        font-size: 1.04rem;
        font-weight: 950;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .agent-sub {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.10rem;
    }

    .agent-badge {
        color: #78f0b7;
        background: rgba(12,88,57,0.24);
        border: 1px solid rgba(53,230,161,0.28);
        border-radius: 999px;
        padding: 0.38rem 0.62rem;
        font-size: 0.70rem;
        font-weight: 850;
        white-space: nowrap;
    }

    .agent-hint {
        color: #9fb4d3;
        font-size: 0.75rem;
        line-height: 1.35;
        margin: 0.12rem 0 0.54rem 0;
    }

    .agent-divider {
        height: 1px;
        background: rgba(87,146,255,0.16);
        margin: 0.55rem 0 0.55rem 0;
    }

    .agent-message-box {
        height: 220px;
        overflow-y: auto;
        padding-right: 0.25rem;
    }

    .stChatMessage {
        background: rgba(8, 20, 43, 0.56) !important;
        border: 1px solid rgba(87, 146, 255, 0.14) !important;
        border-radius: 14px !important;
        padding: 0.62rem 0.76rem !important;
        font-size: 0.86rem !important;
    }

    .agent-quick-title {
        color: #a9bad7;
        font-size: 0.72rem;
        font-weight: 850;
        margin-bottom: 0.34rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    @media (max-width: 1450px) {
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker) {
            max-height: none !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.nav-marker) > div {
            padding-top: 0.56rem !important;
        }

        .side-logo {
            width: 56px;
            height: 56px;
            font-size: 1.66rem;
            margin-bottom: 0.38rem;
        }

        .nav-label {
            font-size: 0.56rem;
            margin-bottom: 0.30rem;
        }

        .nav-radio-space div[data-testid="stRadio"] label {
            min-height: 42px;
            padding: 0.50rem 0.44rem;
        }

        .nav-radio-space div[data-testid="stRadio"] label p {
            font-size: 0.68rem !important;
        }

        .nav-status {
            font-size: 0.56rem;
            margin-top: 0.70rem;
            padding-top: 0.60rem;
        }

        .brand-title {font-size: 1.18rem;}
        .brand-subtitle {font-size: 0.70rem;}
        .badge {font-size: 0.66rem; padding: 0.40rem 0.54rem;}

        .kpi-card {height: 102px; padding: 0.60rem;}
        .kpi-icon {width: 38px; height: 38px;}
        .kpi-layout {grid-template-columns: 40px 1fr; gap: 0.50rem;}
        .kpi-value {font-size: 1.30rem;}
        .kpi-label {font-size: 0.64rem;}
        .kpi-sub {font-size: 0.60rem;}

        video {max-height: 335px !important;}

        div[data-testid="stMetric"] {
            min-height: 70px !important;
            padding: 0.48rem 0.52rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.02rem !important;
        }

        .stChatMessage {
            font-size: 0.76rem !important;
        }

        .agent-message-box {
            height: 185px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_col(df: pd.DataFrame, names: list[str], default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return default


def fmt_int(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_float(value, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}%"


def fmt_seconds(seconds: float) -> str:
    if pd.isna(seconds):
        return "0:00"

    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    sec = int(round(seconds % 60))

    if sec == 60:
        minutes += 1
        sec = 0

    return f"{minutes}:{sec:02d}"


def fmt_density_score(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 10000:.2f}"


def resolve_video(mode: str) -> Optional[Path]:
    for path in VIDEO_CANDIDATES.get(mode, []):
        if path.exists():
            return path
    return None


def risk_class(risk: str) -> str:
    risk = str(risk).upper()

    if risk == "LOW":
        return "risk-low"
    if risk == "MEDIUM":
        return "risk-medium"
    if risk == "HIGH":
        return "risk-high"

    return "risk-critical"


def estimate_video_fps(frame_df: pd.DataFrame) -> float:
    diffs = frame_df["timestamp_sec"].diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 30.0

    median_dt = float(diffs.median())

    if median_dt <= 0:
        return 30.0

    return 1.0 / median_dt


def kpi_card(label: str, value: str, sub: str, icon: str, good: bool = False) -> None:
    sub_class = "kpi-good" if good else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-layout">
                <div class="kpi-icon">{icon}</div>
                <div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub {sub_class}">{sub}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FRAME_CSV.exists():
        raise FileNotFoundError(f"Frame CSV not found: {FRAME_CSV}")

    if not ZONE_CSV.exists():
        raise FileNotFoundError(f"Zone CSV not found: {ZONE_CSV}")

    frame_df = pd.read_csv(FRAME_CSV)
    zone_df = pd.read_csv(ZONE_CSV)

    frame_time_col = safe_col(frame_df, ["timestamp_sec", "timestamp", "time_sec"])
    frame_id_col = safe_col(frame_df, ["frame_id", "frame", "frame_idx"])
    total_count_col = safe_col(frame_df, ["total_count", "count", "global_count"])
    fps_col = safe_col(frame_df, ["inference_fps", "fps", "pipeline_fps"])

    if total_count_col is None:
        raise ValueError(f"Could not find total count column. Available columns: {list(frame_df.columns)}")

    frame_df["timestamp_sec"] = (
        pd.to_numeric(frame_df[frame_time_col], errors="coerce").fillna(0)
        if frame_time_col
        else np.arange(len(frame_df)) / 30.0
    )

    frame_df["frame_id"] = frame_df[frame_id_col] if frame_id_col else np.arange(len(frame_df))
    frame_df["total_count"] = pd.to_numeric(frame_df[total_count_col], errors="coerce").fillna(0)
    frame_df["inference_fps"] = pd.to_numeric(frame_df[fps_col], errors="coerce") if fps_col else np.nan
    frame_df = frame_df.sort_values("timestamp_sec").reset_index(drop=True)

    zone_name_col = safe_col(zone_df, ["zone_name", "zone", "name"])
    zone_short_col = safe_col(zone_df, ["zone_short_id", "zone_id", "zone_code"])
    zone_count_col = safe_col(zone_df, ["zone_count", "count", "count_in_zone"])
    density_col = safe_col(zone_df, ["zone_density_pixel", "density", "zone_density", "density_pixel"])
    risk_col = safe_col(zone_df, ["risk_level", "risk_level_text", "risk", "zone_risk"])
    ts_col = safe_col(zone_df, ["timestamp_sec", "timestamp", "time_sec"])

    missing = []
    if zone_name_col is None:
        missing.append("zone_name")
    if zone_count_col is None:
        missing.append("zone_count")
    if density_col is None:
        missing.append("zone_density_pixel")
    if risk_col is None:
        missing.append("risk_level")
    if ts_col is None:
        missing.append("timestamp_sec")

    if missing:
        raise ValueError(f"Zone CSV missing required columns: {missing}. Available columns: {list(zone_df.columns)}")

    zone_df["timestamp_sec"] = pd.to_numeric(zone_df[ts_col], errors="coerce").fillna(0)
    zone_df["zone_name"] = zone_df[zone_name_col].astype(str)
    zone_df["zone_count"] = pd.to_numeric(zone_df[zone_count_col], errors="coerce").fillna(0)
    zone_df["zone_density_pixel"] = pd.to_numeric(zone_df[density_col], errors="coerce").fillna(0)
    zone_df["risk_level"] = zone_df[risk_col].astype(str).str.upper()
    zone_df["zone_short_id"] = zone_df[zone_short_col].astype(str) if zone_short_col else ""

    fallback_ids = {
        "crosswalk_main": "CW1",
        "crosswalk_left": "CW2",
        "crosswalk_top": "CW3",
        "crosswalk_bottom": "CW4",
        "sidewalk_top": "SW1",
        "sidewalk_right": "SW2",
        "sidewalk_bottom": "SW3",
        "sidewalk_left": "SW4",
    }

    zone_df["zone_short_id"] = zone_df.apply(
        lambda row: row["zone_short_id"]
        if str(row["zone_short_id"]).strip() not in ["", "nan", "None"]
        else fallback_ids.get(row["zone_name"], row["zone_name"]),
        axis=1,
    )

    zone_df = zone_df.sort_values(["zone_name", "timestamp_sec"]).reset_index(drop=True)

    return frame_df, zone_df


frame_df, zone_df = load_data()


# ============================================================
# DERIVED METRICS
# ============================================================

duration_sec = float(frame_df["timestamp_sec"].max() - frame_df["timestamp_sec"].min())
duration_label = fmt_seconds(duration_sec)
num_frames = int(len(frame_df))
video_fps = estimate_video_fps(frame_df)

avg_crowd = float(frame_df["total_count"].mean())
median_crowd = float(frame_df["total_count"].median())
peak_idx = frame_df["total_count"].idxmax()
peak_count = float(frame_df.loc[peak_idx, "total_count"])
peak_time = float(frame_df.loc[peak_idx, "timestamp_sec"])

avg_pipeline_fps = (
    float(frame_df["inference_fps"].dropna().mean())
    if frame_df["inference_fps"].notna().any()
    else 0.0
)

zone_order = sorted(zone_df["zone_name"].dropna().unique().tolist())
reference_time = duration_sec / 2.0

reference_rows = []
for zone_name, group in zone_df.groupby("zone_name"):
    idx = (group["timestamp_sec"] - reference_time).abs().idxmin()
    reference_rows.append(zone_df.loc[idx])

reference_df = pd.DataFrame(reference_rows)[
    ["zone_name", "zone_count", "zone_density_pixel", "risk_level", "timestamp_sec"]
].rename(
    columns={
        "zone_count": "reference_count",
        "zone_density_pixel": "reference_density",
        "risk_level": "reference_risk",
        "timestamp_sec": "reference_time_sec",
    }
)

zone_summary = (
    zone_df.groupby("zone_name")
    .agg(
        zone_short_id=("zone_short_id", "first"),
        avg_count=("zone_count", "mean"),
        median_count=("zone_count", "median"),
        peak_count=("zone_count", "max"),
        mean_density=("zone_density_pixel", "mean"),
        max_density=("zone_density_pixel", "max"),
    )
    .reset_index()
)

zone_peak_rows = zone_df.loc[zone_df.groupby("zone_name")["zone_count"].idxmax()]
zone_peak_times = zone_peak_rows[["zone_name", "timestamp_sec"]].rename(
    columns={"timestamp_sec": "peak_time_sec"}
)

risk_pct = (
    zone_df.assign(is_high_critical=zone_df["risk_level"].isin(["HIGH", "CRITICAL"]))
    .groupby("zone_name")["is_high_critical"]
    .mean()
    .mul(100)
    .reset_index(name="high_critical_pct")
)

risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

dominant_risk = (
    zone_df.assign(risk_num=zone_df["risk_level"].map(risk_rank).fillna(0))
    .groupby("zone_name")["risk_num"]
    .mean()
    .reset_index(name="risk_score")
)

dominant_risk["dominant_risk"] = dominant_risk["risk_score"].apply(
    lambda x: "CRITICAL"
    if x >= 2.5
    else "HIGH"
    if x >= 1.7
    else "MEDIUM"
    if x >= 0.7
    else "LOW"
)

zone_summary = zone_summary.merge(zone_peak_times, on="zone_name", how="left")
zone_summary = zone_summary.merge(risk_pct, on="zone_name", how="left")
zone_summary = zone_summary.merge(dominant_risk[["zone_name", "dominant_risk"]], on="zone_name", how="left")
zone_summary = zone_summary.merge(reference_df, on="zone_name", how="left")


def build_refined_spike_events(zone_df_in: pd.DataFrame, fps: float) -> pd.DataFrame:
    fps = max(1.0, float(fps))
    window = max(30, int(fps * 5))
    cooldown = max(30, int(fps * 5))
    diff_lag = max(1, int(fps))

    events = []

    for zone_name, group in zone_df_in.groupby("zone_name"):
        g = group.sort_values("timestamp_sec").reset_index(drop=True).copy()

        rolling_mean = g["zone_count"].rolling(window, min_periods=max(10, window // 3)).mean()
        rolling_std = g["zone_count"].rolling(window, min_periods=max(10, window // 3)).std().fillna(0)
        one_sec_change = g["zone_count"].diff(diff_lag).fillna(0)

        candidates = (
            (g["zone_count"] > rolling_mean + 2.5 * rolling_std)
            & (one_sec_change > 8)
            & (g["zone_count"] >= 20)
        ).fillna(False)

        last_event_idx = -cooldown

        for idx, is_candidate in enumerate(candidates):
            if not is_candidate:
                continue

            if idx - last_event_idx < cooldown:
                continue

            row = g.iloc[idx]

            events.append(
                {
                    "zone_name": zone_name,
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "zone_count": float(row["zone_count"]),
                    "density_score": float(row["zone_density_pixel"]) * 10000,
                    "risk_level": str(row["risk_level"]),
                    "event_type": "refined_spike",
                }
            )

            last_event_idx = idx

    return pd.DataFrame(events)


spike_events_df = build_refined_spike_events(zone_df, video_fps)

if spike_events_df.empty:
    spike_counts = pd.DataFrame({"zone_name": zone_order, "spike_events": [0] * len(zone_order)})
else:
    spike_counts = spike_events_df.groupby("zone_name").size().reset_index(name="spike_events")

zone_summary = zone_summary.merge(spike_counts, on="zone_name", how="left")
zone_summary["spike_events"] = zone_summary["spike_events"].fillna(0)

zone_df["density_prev_5s"] = zone_df.groupby("zone_name")["zone_density_pixel"].shift(max(1, int(video_fps * 5)))
zone_df["density_delta_5s"] = zone_df["zone_density_pixel"] - zone_df["density_prev_5s"]

density_trend = (
    zone_df.groupby("zone_name")["density_delta_5s"]
    .mean()
    .fillna(0)
    .mul(10000)
    .reset_index(name="density_trend_score")
)

zone_summary = zone_summary.merge(density_trend, on="zone_name", how="left")

total_alerts = int(zone_summary["spike_events"].sum())
hotspot_row = zone_summary.sort_values("avg_count", ascending=False).iloc[0]
most_risky_row = zone_summary.sort_values("high_critical_pct", ascending=False).iloc[0]


# ============================================================
# CHARTS
# ============================================================

BLUE = "#2ea8ff"
PURPLE = "#a78bfa"
GREEN = "#35e6a1"
YELLOW = "#fbbf24"
ORANGE = "#fb923c"
RED = "#fb7185"
GRID = "rgba(148, 163, 184, 0.14)"
FONT = "#e6eef9"


def base_layout(fig: go.Figure, height: int = 260, show_legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT, size=11),
        margin=dict(l=26, r=14, t=8, b=24),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=9),
        title_font=dict(size=10),
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(size=9),
        title_font=dict(size=10),
    )

    return fig


def fig_global_count(height: int = 230) -> go.Figure:
    rolling_window = max(10, int(video_fps * 5))
    rolling = frame_df["total_count"].rolling(rolling_window, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame_df["timestamp_sec"],
            y=frame_df["total_count"],
            mode="lines",
            line=dict(color=BLUE, width=1.65),
            name="Total",
            hovertemplate="Time %{x:.1f}s<br>Total %{y:.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame_df["timestamp_sec"],
            y=rolling,
            mode="lines",
            line=dict(color=PURPLE, width=1.4, dash="dot"),
            name="5s average",
            hovertemplate="Time %{x:.1f}s<br>Avg %{y:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[peak_time],
            y=[peak_count],
            mode="markers",
            marker=dict(color=RED, size=8),
            name="Peak",
            hovertemplate="Peak %{y:.0f}<br>Time %{x:.1f}s<extra></extra>",
        )
    )

    fig.update_yaxes(title="Count")
    fig.update_xaxes(title=None)

    return base_layout(fig, height=height, show_legend=True)


def fig_zone_compare(height: int = 230) -> go.Figure:
    df = zone_summary.sort_values("avg_count", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["avg_count"],
            y=df["zone_name"],
            orientation="h",
            marker=dict(color=BLUE),
            hovertemplate="%{y}<br>Average count %{x:.1f}<extra></extra>",
        )
    )

    fig.update_xaxes(title="Avg Count")
    fig.update_yaxes(title=None)

    return base_layout(fig, height=height, show_legend=False)


def fig_risk_distribution(height: int = 230) -> go.Figure:
    risk_dist = (
        zone_df.groupby(["zone_name", "risk_level"])
        .size()
        .reset_index(name="n")
    )

    total_per_zone = risk_dist.groupby("zone_name")["n"].transform("sum")
    risk_dist["pct"] = risk_dist["n"] / total_per_zone * 100

    colors = {
        "LOW": GREEN,
        "MEDIUM": YELLOW,
        "HIGH": ORANGE,
        "CRITICAL": RED,
    }

    fig = go.Figure()

    for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        sub = risk_dist[risk_dist["risk_level"] == risk]

        fig.add_trace(
            go.Bar(
                x=sub["zone_name"],
                y=sub["pct"],
                name=risk,
                marker_color=colors[risk],
                hovertemplate="%{x}<br>" + risk + ": %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="% frames", range=[0, 100])
    fig.update_xaxes(title=None, tickangle=-25)

    return base_layout(fig, height=height, show_legend=True)


def fig_zone_corr(height: int = 230) -> go.Figure:
    pivot = zone_df.pivot_table(
        index="timestamp_sec",
        columns="zone_name",
        values="zone_count",
        aggfunc="mean",
    ).fillna(0)

    corr = pivot.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr", thickness=8, tickfont=dict(size=8)),
            hovertemplate="%{y} vs %{x}<br>Corr %{z:.2f}<extra></extra>",
        )
    )

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))

    return base_layout(fig, height=height, show_legend=False)


def fig_zone_trend(selected_zone: str, height: int = 120) -> go.Figure:
    df = zone_df[zone_df["zone_name"] == selected_zone].sort_values("timestamp_sec")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["zone_count"],
            mode="lines",
            line=dict(color=BLUE, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(46,168,255,0.10)",
            hovertemplate="Time %{x:.1f}s<br>Count %{y:.0f}<extra></extra>",
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return base_layout(fig, height=height, show_legend=False)


def fig_count_change(height: int = 305) -> go.Figure:
    df = frame_df.copy()
    df["change_abs"] = df["total_count"].diff().abs().fillna(0)
    df["change_5s"] = df["change_abs"].rolling(max(5, int(video_fps * 5)), min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["change_5s"],
            mode="lines",
            line=dict(color=YELLOW, width=1.7),
            name="5s count-change proxy",
            hovertemplate="Time %{x:.1f}s<br>Change %{y:.2f}<extra></extra>",
        )
    )

    fig.update_yaxes(title="Change")
    fig.update_xaxes(title=None)

    return base_layout(fig, height=height, show_legend=False)


def fig_spike_events(height: int = 305) -> go.Figure:
    df = zone_summary.sort_values("spike_events", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["spike_events"],
            y=df["zone_name"],
            orientation="h",
            marker=dict(color=RED),
            hovertemplate="%{y}<br>Refined spike events: %{x}<extra></extra>",
        )
    )

    fig.update_xaxes(title="Events")
    fig.update_yaxes(title=None)

    return base_layout(fig, height=height, show_legend=False)


def fig_entropy(height: int = 305) -> go.Figure:
    pivot = zone_df.pivot_table(
        index="timestamp_sec",
        columns="zone_name",
        values="zone_count",
        aggfunc="sum",
    ).fillna(0)

    total = pivot.sum(axis=1).replace(0, np.nan)
    probabilities = pivot.div(total, axis=0).fillna(0)
    entropy = -(probabilities * np.log(probabilities.replace(0, np.nan))).sum(axis=1).fillna(0)

    max_entropy = np.log(len(pivot.columns)) if len(pivot.columns) > 1 else 1
    norm_entropy = entropy / max_entropy

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=norm_entropy.index,
            y=norm_entropy.values,
            mode="lines",
            line=dict(color=PURPLE, width=1.75),
            hovertemplate="Time %{x:.1f}s<br>Entropy %{y:.3f}<extra></extra>",
        )
    )

    fig.update_yaxes(title="Entropy", range=[0, 1])
    fig.update_xaxes(title=None)

    return base_layout(fig, height=height, show_legend=False)


def fig_density_by_zone(height: int = 305) -> go.Figure:
    df = zone_summary.sort_values("mean_density", ascending=True).copy()
    df["density_score"] = df["mean_density"] * 10000

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["density_score"],
            y=df["zone_name"],
            orientation="h",
            marker=dict(color=PURPLE),
            hovertemplate="%{y}<br>Density score: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_xaxes(title="Density Score, pixel ×10⁴")
    fig.update_yaxes(title=None)

    return base_layout(fig, height=height, show_legend=False)


# ============================================================
# AI AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_crowd_agent() -> CrowdAnalysisAgent:
    """
    Cached dashboard agent.

    If GOOGLE_API_KEY or OPENAI_API_KEY exists in .env, it can use LLM mode.
    Otherwise it automatically falls back to rule-based data-grounded answers.
    """
    return CrowdAnalysisAgent(project_root=PROJECT_ROOT)


def agent_answer(prompt: str, selected_zone: str) -> str:
    agent = get_crowd_agent()
    return str(agent.answer(prompt, selected_zone=selected_zone))


def agent_quick_answer(action: str, selected_zone: str) -> str:
    agent = get_crowd_agent()
    return str(agent.quick_answer(action, selected_zone=selected_zone))


# ============================================================
# RENDER FUNCTIONS
# ============================================================

def render_fixed_nav() -> str:
    with st.container(border=True):
        st.markdown(
            """
            <div class="nav-marker"></div>
            <div class="nav-inner-top">
                <div class="side-logo">🛡️</div>
                <div class="nav-label">Navigation</div>
            </div>
            <div class="nav-radio-space">
            """,
            unsafe_allow_html=True,
        )

        page_selected = st.radio(
            "Navigation",
            ["🏠 Overview", "📊 Analytics"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            """
            </div>
            <div class="nav-status">
                <div><span class="status-dot">●</span> System Status</div>
                <div class="nav-status-ok">All systems operational</div>
                <div class="version">v1.0.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page_selected


def render_header() -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-grid">
                <div class="brand">
                    <div class="brand-icon">🛡️</div>
                    <div>
                        <div class="brand-title">Crowd Monitoring Dashboard</div>
                        <div class="brand-subtitle">Intelligent Crowd Monitoring and Behavioral Analysis System</div>
                    </div>
                </div>
                <div class="badges">
                    <div class="badge green">● Data Loaded</div>
                    <div class="badge blue">🧪 5-minute Shinjuku Experiment</div>
                    <div class="badge">🕒 {duration_label} · {fmt_int(num_frames)} frames · {len(zone_order)} zones</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis() -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi_card("Avg Crowd", fmt_float(avg_crowd, 1), f"median {fmt_float(median_crowd, 1)}", "👥", True)

    with c2:
        kpi_card("Peak Count", fmt_int(peak_count), f"at {fmt_seconds(peak_time)}", "📈")

    with c3:
        kpi_card("Duration", duration_label, f"{duration_sec:.1f}s", "⏱️")

    with c4:
        kpi_card("Frames", fmt_int(num_frames), f"{video_fps:.2f} video FPS", "🎥")

    with c5:
        kpi_card("Alerts", fmt_int(total_alerts), "refined spike events", "⚠️")

    with c6:
        kpi_card("Avg FPS", fmt_float(avg_pipeline_fps, 2), "offline pipeline", "⚡")


def render_video_panel() -> None:
    with st.container(border=True):
        title_col, tabs_col = st.columns([0.55, 1.45])

        with title_col:
            st.markdown('<div class="panel-title">▣ Video Stream</div>', unsafe_allow_html=True)

        with tabs_col:
            video_mode = st.radio(
                "Video mode",
                ["Localization", "Heatmap Overlay", "Heatmap Only", "Zone Density + Risk"],
                index=3,
                horizontal=True,
                label_visibility="collapsed",
            )

        video_path = resolve_video(video_mode)

        if video_path is not None:
            st.video(str(video_path))
        else:
            st.warning(f"Video file for {video_mode} was not found.")


def render_zone_panel() -> str:
    default_zone = "sidewalk_right" if "sidewalk_right" in zone_order else zone_order[0]

    title_col, select_col = st.columns([0.8, 1.2])

    with title_col:
        st.markdown('<div class="panel-title">▦ Zone Analysis</div>', unsafe_allow_html=True)

    with select_col:
        selected_zone = st.selectbox(
            "Select Zone",
            zone_order,
            index=zone_order.index(default_zone),
            label_visibility="collapsed",
        )

    row = zone_summary[zone_summary["zone_name"] == selected_zone].iloc[0]
    risk = str(row["dominant_risk"]).upper()

    trend_score = float(row["density_trend_score"])
    trend_symbol = "↑" if trend_score > 0 else "↓" if trend_score < 0 else "→"

    st.markdown(
        f"""
        <div class="zone-header-simple">
            <div class="zone-name">
                {selected_zone}
                <span class="zone-id">{row["zone_short_id"]}</span>
            </div>
            <div class="risk-pill {risk_class(risk)}">{risk} RISK</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    z1, z2, z3, z4 = st.columns(4)

    with z1:
        st.metric("Average Count", fmt_float(row["avg_count"], 1), "people/frame")

    with z2:
        st.metric("Peak Count", fmt_int(row["peak_count"]), f"at {fmt_seconds(row['peak_time_sec'])}")

    with z3:
        st.metric("Mean Density", fmt_density_score(row["mean_density"]), "pixel ×10⁴")

    with z4:
        st.metric("High/Critical", fmt_pct(row["high_critical_pct"], 1), "rule frames")

    z5, z6, z7 = st.columns(3)

    with z5:
        st.metric("Current Count", fmt_int(row["reference_count"]), f"at {fmt_seconds(row['reference_time_sec'])}")

    with z6:
        st.metric("Density Trend", f"{trend_symbol} {trend_score:+.2f}", "5s Δ score")

    with z7:
        st.metric("Spike Events", fmt_int(row["spike_events"]), "refined")

    st.plotly_chart(
        fig_zone_trend(selected_zone),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    return selected_zone


def render_chart_card(title: str, fig: go.Figure) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )


def render_ai_chat(selected_zone: str) -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="agent-shell">
                <div class="agent-header">
                    <div class="agent-title-wrap">
                        <div class="agent-orb">✦</div>
                        <div>
                            <div class="agent-title">AI Insights Assistant</div>
                            <div class="agent-sub">Data-grounded agent for crowd analysis, risks, anomalies, and chart interpretation.</div>
                        </div>
                    </div>
                    <div class="agent-badge">● Agent Ready</div>
                </div>
                <div class="agent-hint">
                    Ask freely about the saved results. The assistant uses Python tools to retrieve exact values from the CSV outputs before answering.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "I am ready. Ask me about peak moments, risky zones, zone behavior, anomalies, "
                        "temporal trends, spatial hotspots, or statistical insights."
                    ),
                }
            ]

        st.markdown('<div class="agent-divider"></div>', unsafe_allow_html=True)

        chat_box = st.container(height=235, border=False)

        with chat_box:
            st.markdown('<div class="agent-message-box">', unsafe_allow_html=True)

            for message in st.session_state.chat_messages[-6:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="agent-quick-title">Quick questions</div>', unsafe_allow_html=True)

        q1, q2, q3, q4 = st.columns(4)

        quick_prompt = None

        with q1:
            if st.button("Most risky zone?", use_container_width=True):
                quick_prompt = "Which zone is the most risky and why?"

        with q2:
            if st.button("Peak moment?", use_container_width=True):
                quick_prompt = "When was the peak crowd moment and what happened around it?"

        with q3:
            if st.button("Explain selected zone", use_container_width=True):
                quick_prompt = f"Explain {selected_zone} using the analysis results."

        with q4:
            if st.button("Show anomalies", use_container_width=True):
                quick_prompt = "Summarize the anomaly detection and spike events."

        typed_prompt = st.chat_input("Ask anything about the crowd analysis...")

        prompt = typed_prompt or quick_prompt

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.spinner("Agent is checking the analysis outputs..."):
                if quick_prompt:
                    answer = agent_quick_answer(prompt, selected_zone)
                else:
                    answer = agent_answer(prompt, selected_zone)

            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()


def render_overview() -> None:
    render_header()
    render_kpis()

    main_left, main_right = st.columns([1.34, 1.0], gap="medium")

    with main_left:
        render_video_panel()

    with main_right:
        with st.container(border=True):
            selected_zone = render_zone_panel()

    render_ai_chat(selected_zone)


def analysis_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-card-title">{title}</div>
            <div class="analysis-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analytics() -> None:
    render_header()

    analysis_type = st.radio(
        "Choose analysis type",
        [
            "Temporal analysis",
            "Spatial / zone analysis",
            "Anomaly detection",
            "Statistical insights",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if analysis_type == "Temporal analysis":
        c1, c2 = st.columns([1.35, 1.0], gap="medium")

        with c1:
            render_chart_card("Global Crowd Timeline", fig_global_count(height=305))

        with c2:
            render_chart_card("Rate of Change Proxy", fig_count_change(height=305))

        a1, a2, a3 = st.columns(3)

        with a1:
            analysis_card(
                "Peak moment",
                f"The highest estimated crowd count was {fmt_int(peak_count)} at {fmt_seconds(peak_time)}.",
            )

        with a2:
            analysis_card(
                "Average crowd level",
                f"The experiment average was {fmt_float(avg_crowd, 1)} people/frame, with median {fmt_float(median_crowd, 1)}.",
            )

        with a3:
            analysis_card(
                "Interpretation",
                "Temporal analysis shows when crowd build-up happens and supports peak-time monitoring.",
            )

    elif analysis_type == "Spatial / zone analysis":
        c1, c2 = st.columns([1.0, 1.0], gap="medium")

        with c1:
            render_chart_card("Zone Hotspot Ranking", fig_zone_compare(height=305))

        with c2:
            render_chart_card("Mean Pixel Density by Zone", fig_density_by_zone(height=305))

        a1, a2, a3 = st.columns(3)

        with a1:
            analysis_card(
                "Main count hotspot",
                f"{hotspot_row['zone_name']} has the highest average count: {fmt_float(hotspot_row['avg_count'], 1)}.",
            )

        with a2:
            analysis_card(
                "Most persistent risk zone",
                f"{most_risky_row['zone_name']} has {fmt_pct(most_risky_row['high_critical_pct'], 1)} HIGH/CRITICAL rule-based frames.",
            )

        with a3:
            analysis_card(
                "Density note",
                "Density is image-space pixel density. The dashboard displays it as a scaled pixel ×10⁴ score.",
            )

    elif analysis_type == "Anomaly detection":
        c1, c2 = st.columns([1.0, 1.0], gap="medium")

        with c1:
            render_chart_card("Refined Spike Events", fig_spike_events(height=305))

        with c2:
            render_chart_card("Risk Level Distribution", fig_risk_distribution(height=305))

        spike_zone = zone_summary.sort_values("spike_events", ascending=False).iloc[0]

        a1, a2, a3 = st.columns(3)

        with a1:
            analysis_card(
                "Total refined events",
                f"The prototype refined spike detector found {fmt_int(total_alerts)} events.",
            )

        with a2:
            analysis_card(
                "Most active spike zone",
                f"{spike_zone['zone_name']} had the most refined spike events: {fmt_int(spike_zone['spike_events'])}.",
            )

        with a3:
            analysis_card(
                "Interpretation",
                "Spike events indicate sudden count changes, not certified incidents.",
            )

    else:
        c1, c2 = st.columns([1.0, 1.0], gap="medium")

        with c1:
            render_chart_card("Zone Correlation", fig_zone_corr(height=305))

        with c2:
            render_chart_card("Crowd Distribution Entropy", fig_entropy(height=305))

        a1, a2, a3 = st.columns(3)

        with a1:
            analysis_card(
                "Correlation meaning",
                "Correlation shows which zones tend to fill or empty together over time.",
            )

        with a2:
            analysis_card(
                "Entropy meaning",
                "Entropy estimates whether the crowd is concentrated in few zones or spread more evenly.",
            )

        with a3:
            analysis_card(
                "Interpretation",
                "Statistical analysis supports interpretation beyond simple counting.",
            )


# ============================================================
# APP
# ============================================================

nav_col, main_col = st.columns([0.085, 0.915], gap="small")

with nav_col:
    page = render_fixed_nav()

with main_col:
    if page == "🏠 Overview":
        render_overview()
    else:
        render_analytics()