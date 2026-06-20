from __future__ import annotations

"""
Data tools for the Crowd Monitoring AI Agent.

Pipeline:
  User question
  → detect_intent()           (classify question, parse time, find zones)
  → build_context_for_question()  (call only the tools needed)
  → build_context_text_for_question()  (format as compact text)
  → LLM explains / rule-based fallback answers

This version adds natural language understanding for:
  - "first frame" / "last frame"      -> exact single frame lookup
  - "frame 1234"                       -> exact single frame lookup
  - "first minute" / "last minute"     -> averaged time-range lookup
  - "first 30 seconds" / "last 10 sec" -> averaged time-range lookup
  - "the beginning" / "the start"      -> exact first frame
  - "the end"                          -> exact last frame
  - "whole video" / "entire video"     -> full-duration range
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class LoadedCrowdData:
    frame_df: pd.DataFrame
    zone_df: pd.DataFrame
    zone_summary: pd.DataFrame
    project_root: Path
    analysis_tables_dir: Path
    analysis_insights_dir: Path


# ============================================================
# PATHS
# ============================================================

def resolve_project_root(project_root: Optional[str | Path] = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[2]


def get_default_paths(project_root: Optional[str | Path] = None) -> Dict[str, Path]:
    root = resolve_project_root(project_root)
    return {
        "project_root": root,
        "frame_csv": root / "results" / "benchmark" / "FULL_01_shinjuku_frame_counts.csv",
        "zone_csv": root / "results" / "benchmark" / "FULL_02_shinjuku_zone_density_risk.csv",
        "analysis_tables_dir": root / "results" / "analysis_5min_refined" / "tables",
        "analysis_insights_dir": root / "results" / "analysis_5min_refined" / "insights",
    }


# ============================================================
# FORMAT HELPERS
# ============================================================

def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def fmt_float(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def fmt_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.1f}"


def fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def fmt_density_score(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 10000:.2f}"


def fmt_seconds(seconds: Any) -> str:
    if seconds is None or pd.isna(seconds):
        return "-"
    seconds = float(seconds)
    minutes = int(seconds // 60)
    sec = int(round(seconds % 60))
    if sec == 60:
        minutes += 1
        sec = 0
    return f"{minutes}:{sec:02d}"


def compact_text(text: str, max_chars: int = 900) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_col(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
    label: str = "column",
) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(
            f"Could not find {label}. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_risk(value: Any) -> str:
    value = str(value).strip().upper()
    if value in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return value
    if value in {"MED", "MODERATE"}:
        return "MEDIUM"
    if value in {"CRIT"}:
        return "CRITICAL"
    return value if value else "UNKNOWN"


def risk_score(value: str) -> int:
    value = normalize_risk(value)
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(value, -1)


def normalize_zone_name(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


# ============================================================
# TIME / FRAME PARSING
# ============================================================

def _parse_time_token(token: str) -> Optional[float]:
    """
    Parse a single time token that may be:
      - "1:30" or "1;30"  → 90.0
      - "90"              → 90.0 (seconds)
    """
    token = token.strip()
    m = re.match(r"^(\d{1,2})[;:](\d{2})$", token)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))
    m = re.match(r"^(\d+(?:\.\d+)?)$", token)
    if m:
        return float(m.group(1))
    return None


def parse_frame_reference(question: str) -> Optional[str]:
    """
    Detect requests for an EXACT single frame.

    Returns one of:
      "first"        -> first frame / beginning / start
      "last"         -> last frame / end
      "frame:<int>"  -> explicit numbered frame, e.g. "frame:1234"
      None           -> no exact-frame request found
    """
    q = question.lower()

    # explicit "frame 1234" / "frame #1234" / "frame number 1234"
    m = re.search(r"\bframe\s*(?:number|no\.?|#)?\s*(\d+)\b", q)
    if m:
        return f"frame:{int(m.group(1))}"

    # first frame / very first frame / opening frame / beginning / start
    if re.search(r"\b(first|very first|opening|initial)\s+frame\b", q):
        return "first"
    if re.search(r"\b(the\s+)?(beginning|start)\b", q) and "frame" in q:
        return "first"

    # last frame / final frame / very last frame / end frame
    if re.search(r"\b(last|very last|final|ending)\s+frame\b", q):
        return "last"

    return None


def parse_relative_time_window(question: str) -> Optional[Tuple[float, float, str]]:
    """
    Detect relative-time RANGE requests that should be averaged.

    Returns (start_sec, end_sec, label) or None.

    Handles:
      first minute / first min      -> (0, 60)
      last minute / last min        -> (duration-60, duration)   [duration filled later]
      first 30 seconds / first 30s  -> (0, 30)
      last 10 seconds / last 10 sec -> (duration-10, duration)   [duration filled later]
      first 2 minutes               -> (0, 120)
      last 2 minutes                -> (duration-120, duration)  [duration filled later]
      whole video / entire video    -> (0, duration)             [duration filled later]

    The sentinel value -1.0 means "fill with video duration in build step".
    """
    q = question.lower()

    # whole / entire / full video
    if re.search(r"\b(whole|entire|full|complete)\s+(video|recording|clip|footage)\b", q) \
       or re.search(r"\b(over|across|throughout)\s+the\s+(whole|entire|full)\b", q):
        return (0.0, -1.0, "whole video")

    # first N minutes
    m = re.search(r"\bfirst\s+(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b", q)
    if m:
        secs = float(m.group(1)) * 60.0
        return (0.0, secs, f"first {m.group(1)} minute(s)")

    # last N minutes
    m = re.search(r"\blast\s+(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b", q)
    if m:
        secs = float(m.group(1)) * 60.0
        return (-secs, -1.0, f"last {m.group(1)} minute(s)")

    # first N seconds
    m = re.search(r"\bfirst\s+(\d+(?:\.\d+)?)\s*(?:sec|secs|second|seconds|s)\b", q)
    if m:
        secs = float(m.group(1))
        return (0.0, secs, f"first {m.group(1)} second(s)")

    # last N seconds
    m = re.search(r"\blast\s+(\d+(?:\.\d+)?)\s*(?:sec|secs|second|seconds|s)\b", q)
    if m:
        secs = float(m.group(1))
        return (-secs, -1.0, f"last {m.group(1)} second(s)")

    # bare "first minute" / "first min"
    if re.search(r"\bfirst\s+(?:min|minute)\b", q):
        return (0.0, 60.0, "first minute")

    # bare "last minute" / "last min"
    if re.search(r"\blast\s+(?:min|minute)\b", q):
        return (-60.0, -1.0, "last minute")

    return None


def parse_time_reference(question: str) -> Optional[float]:
    """
    Parse a single time reference from natural-language questions.

    Handles:
      1:00 / 1;00 / at 1:00 / around 2:30
      at minute 1 / minute 2.5
      60 seconds / 60 sec / at 90s
      1 min 30 sec
    """
    q = question.lower()

    m = re.search(
        r"\b(?:at|around|near|minute|min|time)?\s*(\d{1,2})\s*[;:]\s*(\d{2})\b", q
    )
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))

    m = re.search(r"\b(?:at|around|near)?\s*minute\s+(\d+(?:\.\d+)?)\b", q)
    if m:
        return float(m.group(1)) * 60.0

    m = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:min|minute|minutes)\s+(\d+(?:\.\d+)?)\s*(?:sec|second|seconds)\b",
        q,
    )
    if m:
        return float(m.group(1)) * 60.0 + float(m.group(2))

    # "2 minutes" alone — but NOT "first/last N minutes" (handled by relative window)
    if not re.search(r"\b(first|last)\b", q):
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:min|minute|minutes)\b", q)
        if m:
            return float(m.group(1)) * 60.0

    # "60 seconds" / "at 60 sec" — but NOT "first/last N seconds"
    if not re.search(r"\b(first|last)\b", q):
        m = re.search(
            r"\b(?:at|around|near)?\s*(\d+(?:\.\d+)?)\s*(?:sec|second|seconds)\b", q
        )
        if m:
            return float(m.group(1))

    return None


def parse_time_range(question: str) -> Optional[Tuple[float, float]]:
    """
    Parse an explicit time range from natural-language questions.

    Handles:
      between 1:00 and 2:00
      from 1:00 to 2:00
      between 60 seconds and 120 seconds
      from 60s to 120s
      between 1 and 2 minutes
    """
    q = question.lower()

    patterns_mmss = [
        r"between\s+(\d{1,2}[;:]\d{2})\s+and\s+(\d{1,2}[;:]\d{2})",
        r"from\s+(\d{1,2}[;:]\d{2})\s+to\s+(\d{1,2}[;:]\d{2})",
    ]
    for pat in patterns_mmss:
        m = re.search(pat, q)
        if m:
            t1 = _parse_time_token(m.group(1))
            t2 = _parse_time_token(m.group(2))
            if t1 is not None and t2 is not None:
                return (min(t1, t2), max(t1, t2))

    patterns_sec = [
        r"between\s+(\d+(?:\.\d+)?)\s*(?:sec|s|seconds?)\s+and\s+(\d+(?:\.\d+)?)\s*(?:sec|s|seconds?)",
        r"from\s+(\d+(?:\.\d+)?)\s*(?:sec|s|seconds?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:sec|s|seconds?)",
    ]
    for pat in patterns_sec:
        m = re.search(pat, q)
        if m:
            t1 = float(m.group(1))
            t2 = float(m.group(2))
            return (min(t1, t2), max(t1, t2))

    m = re.search(
        r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(?:min|minute|minutes)",
        q,
    )
    if m:
        t1 = float(m.group(1)) * 60.0
        t2 = float(m.group(2)) * 60.0
        return (min(t1, t2), max(t1, t2))

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_frame_csv(frame_csv: Path) -> pd.DataFrame:
    if not frame_csv.exists():
        raise FileNotFoundError(f"Frame CSV not found: {frame_csv}")
    df = pd.read_csv(frame_csv)

    frame_time_col = find_col(df, ["timestamp_sec", "timestamp", "time_sec"], required=False)
    frame_id_col = find_col(df, ["frame_id", "frame", "frame_idx"], required=False)
    total_count_col = find_col(df, ["total_count", "count", "global_count"], label="total count column")
    fps_col = find_col(df, ["inference_fps", "fps", "pipeline_fps"], required=False)

    if frame_time_col is None:
        df["timestamp_sec"] = np.arange(len(df)) / 30.0
    else:
        df["timestamp_sec"] = pd.to_numeric(df[frame_time_col], errors="coerce").fillna(0)

    if frame_id_col is None:
        df["frame_id"] = np.arange(len(df))
    else:
        df["frame_id"] = pd.to_numeric(df[frame_id_col], errors="coerce").fillna(0).astype(int)

    df["total_count"] = pd.to_numeric(df[total_count_col], errors="coerce").fillna(0)
    df["fps"] = pd.to_numeric(df[fps_col], errors="coerce") if fps_col else np.nan

    return df.sort_values("timestamp_sec").reset_index(drop=True)


def load_zone_csv(zone_csv: Path) -> pd.DataFrame:
    if not zone_csv.exists():
        raise FileNotFoundError(f"Zone CSV not found: {zone_csv}")
    df = pd.read_csv(zone_csv)

    zone_name_col = find_col(df, ["zone_name", "zone", "name"], label="zone name column")
    zone_count_col = find_col(df, ["zone_count", "count", "count_in_zone"], label="zone count column")
    density_col = find_col(
        df,
        ["zone_density_pixel", "density", "zone_density", "zone_density_pixel_cleaned", "density_pixel"],
        label="density column",
    )
    risk_col = find_col(df, ["risk_level", "risk_level_text", "risk", "zone_risk"], label="risk column")
    ts_col = find_col(df, ["timestamp_sec", "timestamp", "time_sec"], label="timestamp column")
    zid_col = find_col(df, ["zone_short_id", "zone_id", "zone_code", "zone_label"], required=False)
    fid_col = find_col(df, ["frame_id", "frame", "frame_idx"], required=False)
    area_col = find_col(df, ["zone_area_pixels", "zone_area", "area_pixels"], required=False)

    df["zone_name"] = df[zone_name_col].astype(str)
    df["zone_name_normalized"] = df["zone_name"].map(normalize_zone_name)
    df["zone_count"] = pd.to_numeric(df[zone_count_col], errors="coerce").fillna(0)
    df["density"] = pd.to_numeric(df[density_col], errors="coerce").fillna(0)
    df["risk_level"] = df[risk_col].map(normalize_risk)
    df["timestamp_sec"] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0)

    if fid_col is not None:
        df["frame_id"] = pd.to_numeric(df[fid_col], errors="coerce").fillna(0).astype(int)
    else:
        df["frame_id"] = -1

    if area_col is not None:
        df["zone_area_pixels"] = pd.to_numeric(df[area_col], errors="coerce").fillna(0)
    else:
        df["zone_area_pixels"] = np.nan

    fallback_ids = {
        "crosswalk_main": "CW1", "crosswalk_left": "CW2",
        "crosswalk_top": "CW3", "crosswalk_bottom": "CW4",
        "sidewalk_top": "SW1", "sidewalk_right": "SW2",
        "sidewalk_bottom": "SW3", "sidewalk_left": "SW4",
    }

    if zid_col is None:
        df["zone_id"] = ""
    else:
        df["zone_id"] = df[zid_col].astype(str)

    df["zone_id"] = df.apply(
        lambda row: row["zone_id"]
        if str(row["zone_id"]).strip() not in ["", "nan", "None"]
        else fallback_ids.get(row["zone_name"], row["zone_name"]),
        axis=1,
    )

    return df.sort_values(["zone_name", "timestamp_sec"]).reset_index(drop=True)


def compute_zone_summary(frame_df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    duration_sec = float(frame_df["timestamp_sec"].max())
    reference_time = duration_sec / 2.0

    reference_rows = []
    for zone_name, group in zone_df.groupby("zone_name"):
        idx = (group["timestamp_sec"] - reference_time).abs().idxmin()
        reference_rows.append(zone_df.loc[idx])

    reference_df = pd.DataFrame(reference_rows)[
        ["zone_name", "zone_count", "density", "risk_level", "timestamp_sec"]
    ].rename(columns={
        "zone_count": "reference_count",
        "density": "reference_density",
        "risk_level": "reference_risk",
        "timestamp_sec": "reference_time_sec",
    })

    summary = (
        zone_df.groupby("zone_name")
        .agg(
            zone_id=("zone_id", "first"),
            avg_count=("zone_count", "mean"),
            median_count=("zone_count", "median"),
            peak_count=("zone_count", "max"),
            min_count=("zone_count", "min"),
            std_count=("zone_count", "std"),
            mean_density=("density", "mean"),
            max_density=("density", "max"),
            median_density=("density", "median"),
        )
        .reset_index()
    )

    risk_pct = (
        zone_df.assign(is_hc=zone_df["risk_level"].isin(["HIGH", "CRITICAL"]).astype(int))
        .groupby("zone_name")["is_hc"]
        .mean()
        .mul(100)
        .reset_index(name="high_critical_pct")
    )
    summary = summary.merge(risk_pct, on="zone_name", how="left")

    risk_dist = zone_df.groupby(["zone_name", "risk_level"]).size().reset_index(name="n")
    risk_dist["pct"] = (
        risk_dist["n"] / risk_dist.groupby("zone_name")["n"].transform("sum") * 100
    )
    risk_pivot = risk_dist.pivot_table(
        index="zone_name", columns="risk_level", values="pct", fill_value=0
    ).reset_index()
    for r in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        if r not in risk_pivot.columns:
            risk_pivot[r] = 0.0
    risk_pivot = risk_pivot.rename(columns={
        "LOW": "low_pct", "MEDIUM": "medium_pct",
        "HIGH": "high_pct", "CRITICAL": "critical_pct",
    })
    summary = summary.merge(risk_pivot, on="zone_name", how="left")

    risk_mode = (
        zone_df.groupby("zone_name")["risk_level"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
        .reset_index(name="dominant_risk")
    )
    summary = summary.merge(risk_mode, on="zone_name", how="left")

    peak_idx = zone_df.groupby("zone_name")["zone_count"].idxmax()
    peak_times = zone_df.loc[peak_idx, ["zone_name", "timestamp_sec"]].rename(
        columns={"timestamp_sec": "peak_time_sec"}
    )
    summary = summary.merge(peak_times, on="zone_name", how="left")

    temp = zone_df.copy()
    temp["prev_count_30"] = temp.groupby("zone_name")["zone_count"].shift(30)
    temp["delta_30"] = temp["zone_count"] - temp["prev_count_30"]
    temp["pct_30"] = np.where(
        temp["prev_count_30"] > 0, temp["delta_30"] / temp["prev_count_30"] * 100, 0
    )
    temp["spike_flag"] = (
        (temp["zone_count"] >= 20) & (temp["delta_30"] >= 10) & (temp["pct_30"] >= 50)
    ).astype(int)
    spikes = temp.groupby("zone_name")["spike_flag"].sum().reset_index(name="spike_events")
    summary = summary.merge(spikes, on="zone_name", how="left")

    temp["density_prev_5s"] = temp.groupby("zone_name")["density"].shift(150)
    temp["density_delta_5s"] = temp["density"] - temp["density_prev_5s"]
    trend = (
        temp.groupby("zone_name")["density_delta_5s"]
        .mean()
        .reset_index(name="density_trend_score")
    )
    summary = summary.merge(trend, on="zone_name", how="left")
    summary = summary.merge(reference_df, on="zone_name", how="left")

    fill_cols = [
        "high_critical_pct", "low_pct", "medium_pct", "high_pct",
        "critical_pct", "spike_events", "density_trend_score",
    ]
    for col in fill_cols:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)

    return summary


def load_crowd_data(project_root: Optional[str | Path] = None) -> LoadedCrowdData:
    paths = get_default_paths(project_root)
    frame_df = load_frame_csv(paths["frame_csv"])
    zone_df = load_zone_csv(paths["zone_csv"])
    zone_summary = compute_zone_summary(frame_df, zone_df)
    return LoadedCrowdData(
        frame_df=frame_df,
        zone_df=zone_df,
        zone_summary=zone_summary,
        project_root=paths["project_root"],
        analysis_tables_dir=paths["analysis_tables_dir"],
        analysis_insights_dir=paths["analysis_insights_dir"],
    )


# ============================================================
# ZONE HELPERS
# ============================================================

def available_zones(data: LoadedCrowdData) -> List[str]:
    return sorted(data.zone_df["zone_name"].unique().tolist())


def resolve_zone_name(data: LoadedCrowdData, text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text_norm = normalize_zone_name(text)
    for zone in available_zones(data):
        if normalize_zone_name(zone) == text_norm:
            return zone
    for zone in available_zones(data):
        zone_norm = normalize_zone_name(zone)
        if zone_norm in text_norm or text_norm in zone_norm:
            return zone
    for _, row in data.zone_summary.iterrows():
        zid = str(row.get("zone_id", "")).lower().strip()
        if zid and zid in text_norm:
            return str(row["zone_name"])
    return None


def extract_zone_from_question(data: LoadedCrowdData, question: str) -> Optional[str]:
    q = question.lower()
    q_norm = normalize_zone_name(question)
    for zone in available_zones(data):
        if normalize_zone_name(zone) in q_norm:
            return zone
    for zone in available_zones(data):
        if zone.replace("_", " ").lower() in q:
            return zone
    for _, row in data.zone_summary.iterrows():
        zid = str(row.get("zone_id", "")).lower().strip()
        if zid and re.search(rf"\b{re.escape(zid)}\b", q):
            return str(row["zone_name"])
    return None


def extract_all_zones_from_question(data: LoadedCrowdData, question: str) -> List[str]:
    q = question.lower()
    q_norm = normalize_zone_name(question)
    zones = []
    for zone in available_zones(data):
        if normalize_zone_name(zone) in q_norm or zone.replace("_", " ").lower() in q:
            zones.append(zone)
    for _, row in data.zone_summary.iterrows():
        zid = str(row.get("zone_id", "")).lower().strip()
        if zid and re.search(rf"\b{re.escape(zid)}\b", q):
            zone = str(row["zone_name"])
            if zone not in zones:
                zones.append(zone)
    return zones


# ============================================================
# INTENT DETECTION
# ============================================================

_CHART_KEYWORDS: Dict[str, str] = {
    "global crowd timeline": "global_crowd_timeline",
    "global timeline": "global_crowd_timeline",
    "crowd timeline": "global_crowd_timeline",
    "global count": "global_crowd_timeline",
    "rate of change": "rate_of_change",
    "count change": "rate_of_change",
    "change proxy": "rate_of_change",
    "hotspot ranking": "zone_hotspot_ranking",
    "hotspot": "zone_hotspot_ranking",
    "zone ranking": "zone_hotspot_ranking",
    "pixel density": "mean_pixel_density",
    "mean density": "mean_pixel_density",
    "density by zone": "mean_pixel_density",
    "density chart": "mean_pixel_density",
    "spike events": "refined_spike_events",
    "spike chart": "refined_spike_events",
    "refined spike": "refined_spike_events",
    "risk distribution": "risk_level_distribution",
    "risk level distribution": "risk_level_distribution",
    "risk chart": "risk_level_distribution",
    "stacked risk": "risk_level_distribution",
    "correlation": "zone_correlation",
    "correlation heatmap": "zone_correlation",
    "corr heatmap": "zone_correlation",
    "entropy": "crowd_distribution_entropy",
    "distribution entropy": "crowd_distribution_entropy",
}

_ALL_ZONES_KEYWORDS = [
    "each zone", "all zones", "every zone", "for all zones",
    "each of the zones", "list all", "zone by zone", "every area",
    "all 8", "all eight", "across zones", "for each zone",
    "all the zones", "for every zone",
]


def detect_intent(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify the question into an intent and extract key routing parameters.
    """
    q = question.lower()

    # ── Parse exact frame, relative window, single time, explicit range ──
    frame_ref = parse_frame_reference(question)
    relative_window = parse_relative_time_window(question)
    time_ref = parse_time_reference(question)
    time_range = parse_time_range(question)

    # ── Zone extraction ─────────────────────────────────────
    zone_from_question = extract_zone_from_question(data, question)
    zones_mentioned = extract_all_zones_from_question(data, question)

    # ── All-zones flag ───────────────────────────────────────
    asks_all_zones = any(kw in q for kw in _ALL_ZONES_KEYWORDS)

    # ── Chart detection ──────────────────────────────────────
    chart_name: Optional[str] = None
    for keyword, canonical in _CHART_KEYWORDS.items():
        if keyword in q:
            chart_name = canonical
            break

    # ── Zone to use in context ───────────────────────────────
    if asks_all_zones:
        zone_to_use = None
    else:
        zone_to_use = zone_from_question or (
            resolve_zone_name(data, selected_zone) if selected_zone else None
        )

    # ── Intent classification ────────────────────────────────
    intent: str

    if any(k in q for k in [
        "what are you", "who are you", "what can you do", "your role",
        "how do you work", "explain this dashboard", "about you",
        "what do you do", "capabilities", "help me understand",
    ]):
        intent = "identity"

    elif any(k in q for k in [
        "recommend", "recommendation", "what should", "operator",
        "action", "decision", "what to do", "prioritize",
        "decision support", "advice", "suggest", "evidence-backed",
        "thesis-safe", "thesis recommendation",
    ]):
        intent = "recommendation"

    # exact frame OR relative window OR single time OR explicit range
    elif (
        frame_ref is not None
        or relative_window is not None
        or time_ref is not None
        or time_range is not None
    ):
        intent = "time_specific"

    elif any(k in q for k in [
        "summary", "overview", "experiment", "important results",
        "most important", "give me a summary", "overall results",
        "tell me about", "what happened",
    ]):
        intent = "global_summary"

    elif len(zones_mentioned) >= 2 or "compare" in q or " vs " in q or "versus" in q:
        intent = "comparison"

    elif any(k in q for k in [
        "anomaly", "anomalies", "spike", "sudden", "alert", "alerts",
        "anomaly detection", "spike event",
    ]):
        intent = "anomaly"

    elif any(k in q for k in [
        "correlation", "entropy", "statistical", "distribution",
        "spread", "together", "move together", "fill together",
        "statistical insight",
    ]):
        intent = "statistical"

    elif any(k in q for k in [
        "temporal", "timeline", "global crowd", "rate of change",
        "trend", "time trend", "over time", "build up", "increase",
        "decrease", "temporal analysis",
    ]):
        intent = "temporal"

    elif any(k in q for k in [
        "hotspot", "spatial", "where is", "concentration", "spatial analysis",
        "zone ranking", "most crowded area",
    ]):
        intent = "spatial"

    elif chart_name is not None or any(k in q for k in [
        "chart", "graph", "plot", "visual", "figure", "diagram",
        "explain the", "what does the", "what does this", "show me",
    ]):
        intent = "chart"

    elif any(k in q for k in [
        "risk", "critical", "high risk", "dangerous", "risky", "riskiest",
        "most dangerous",
    ]):
        intent = "risk"

    elif any(k in q for k in [
        "peak", "highest", "maximum", "max", "busiest", "most crowded",
        "peak moment", "peak count",
    ]):
        intent = "peak"

    elif any(k in q for k in [
        "thesis", "academic", "research", "interpretation", "what does it mean",
        "significance",
    ]):
        intent = "thesis"

    elif zone_from_question or any(k in q for k in [
        "zone", "sidewalk", "crosswalk", "explain", "describe", "tell me about",
    ]):
        intent = "zone"

    else:
        intent = "general"

    return {
        "intent": intent,
        "frame_ref": frame_ref,
        "relative_window": relative_window,
        "time_ref": time_ref,
        "time_range": time_range,
        "zone_from_question": zone_from_question,
        "zones_mentioned": zones_mentioned,
        "asks_all_zones": asks_all_zones,
        "zone_to_use": zone_to_use,
        "chart_name": chart_name,
        "selected_zone": selected_zone,
    }


# ============================================================
# CORE FACT TOOLS
# ============================================================

def get_global_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    frame_df = data.frame_df
    zone_df = data.zone_df
    duration_sec = float(frame_df["timestamp_sec"].max())
    processed_frames = int(len(frame_df))
    video_fps = processed_frames / duration_sec if duration_sec > 0 else float("nan")
    peak_idx = frame_df["total_count"].idxmax()
    peak_row = frame_df.loc[peak_idx]
    avg_pipeline_fps = (
        float(frame_df["fps"].dropna().mean())
        if frame_df["fps"].notna().any()
        else float("nan")
    )
    return {
        "duration_sec": duration_sec,
        "duration_label": fmt_seconds(duration_sec),
        "processed_frames": processed_frames,
        "estimated_video_fps": round(video_fps, 2),
        "average_count": round(float(frame_df["total_count"].mean()), 2),
        "median_count": round(float(frame_df["total_count"].median()), 2),
        "maximum_count": int(frame_df["total_count"].max()),
        "minimum_count": int(frame_df["total_count"].min()),
        "std_count": round(float(frame_df["total_count"].std()), 2),
        "peak_frame_id": int(peak_row["frame_id"]),
        "peak_time_sec": float(peak_row["timestamp_sec"]),
        "peak_time_label": fmt_seconds(float(peak_row["timestamp_sec"])),
        "average_pipeline_fps": round(avg_pipeline_fps, 2) if not np.isnan(avg_pipeline_fps) else None,
        "number_of_zones": int(zone_df["zone_name"].nunique()),
    }


def get_zone_summary(data: LoadedCrowdData, zone_name: str) -> Dict[str, Any]:
    resolved = resolve_zone_name(data, zone_name)
    if resolved is None:
        return {"error": f"Zone '{zone_name}' was not found.", "available_zones": available_zones(data)}

    row = data.zone_summary[data.zone_summary["zone_name"] == resolved].iloc[0]
    return {
        "zone_name": resolved,
        "zone_id": str(row.get("zone_id", "")),
        "average_count": round(float(row["avg_count"]), 2),
        "median_count": round(float(row["median_count"]), 2),
        "peak_count": int(row["peak_count"]),
        "min_count": int(row.get("min_count", 0)),
        "std_count": round(float(row.get("std_count", 0)), 2),
        "peak_time_sec": float(row["peak_time_sec"]),
        "peak_time_label": fmt_seconds(float(row["peak_time_sec"])),
        "mean_density_pixel": float(row["mean_density"]),
        "mean_density_score_x10000": round(float(row["mean_density"]) * 10000, 4),
        "max_density_pixel": float(row["max_density"]),
        "high_critical_pct": round(float(row["high_critical_pct"]), 2),
        "low_pct": round(float(row.get("low_pct", 0)), 2),
        "medium_pct": round(float(row.get("medium_pct", 0)), 2),
        "high_pct": round(float(row.get("high_pct", 0)), 2),
        "critical_pct": round(float(row.get("critical_pct", 0)), 2),
        "dominant_risk": str(row.get("dominant_risk", "UNKNOWN")),
        "reference_count": int(row.get("reference_count", 0)),
        "reference_time_sec": float(row.get("reference_time_sec", 0)),
        "reference_time_label": fmt_seconds(float(row.get("reference_time_sec", 0))),
        "reference_risk": str(row.get("reference_risk", "UNKNOWN")),
        "spike_events": int(row.get("spike_events", 0)),
        "density_trend_score_x10000": round(float(row.get("density_trend_score", 0)) * 10000, 4),
    }


def get_all_zone_summaries(data: LoadedCrowdData) -> List[Dict[str, Any]]:
    summaries = []
    for zone_name in available_zones(data):
        s = get_zone_summary(data, zone_name)
        if "error" not in s:
            summaries.append(s)
    return sorted(summaries, key=lambda x: x["high_critical_pct"], reverse=True)


def get_all_zone_rankings(data: LoadedCrowdData) -> Dict[str, Any]:
    summary = data.zone_summary.copy()
    highest_avg = summary.sort_values("avg_count", ascending=False).iloc[0]
    highest_peak = summary.sort_values("peak_count", ascending=False).iloc[0]
    highest_density = summary.sort_values("mean_density", ascending=False).iloc[0]
    most_risky = summary.sort_values("high_critical_pct", ascending=False).iloc[0]
    most_spikes = summary.sort_values("spike_events", ascending=False).iloc[0]

    ranked_by_avg = summary.sort_values("avg_count", ascending=False)[
        ["zone_name", "zone_id", "avg_count", "high_critical_pct", "spike_events", "dominant_risk"]
    ].to_dict(orient="records")

    return {
        "highest_average_count_zone": {
            "zone_name": highest_avg["zone_name"],
            "average_count": round(float(highest_avg["avg_count"]), 2),
        },
        "highest_peak_count_zone": {
            "zone_name": highest_peak["zone_name"],
            "peak_count": int(highest_peak["peak_count"]),
            "peak_time_label": fmt_seconds(float(highest_peak["peak_time_sec"])),
        },
        "highest_mean_density_zone": {
            "zone_name": highest_density["zone_name"],
            "mean_density_score_x10000": round(float(highest_density["mean_density"]) * 10000, 4),
        },
        "most_risky_zone": {
            "zone_name": most_risky["zone_name"],
            "high_critical_pct": round(float(most_risky["high_critical_pct"]), 2),
            "dominant_risk": str(most_risky["dominant_risk"]),
        },
        "most_spike_events_zone": {
            "zone_name": most_spikes["zone_name"],
            "spike_events": int(most_spikes["spike_events"]),
        },
        "all_zones_ranked_by_avg_count": ranked_by_avg,
    }


def get_peak_moment_context(data: LoadedCrowdData, window_sec: float = 5.0) -> Dict[str, Any]:
    frame_df = data.frame_df
    zone_df = data.zone_df
    peak_idx = frame_df["total_count"].idxmax()
    peak_row = frame_df.loc[peak_idx]
    peak_time = float(peak_row["timestamp_sec"])

    nearby_frames = frame_df[
        (frame_df["timestamp_sec"] >= peak_time - window_sec)
        & (frame_df["timestamp_sec"] <= peak_time + window_sec)
    ]

    zone_rows = []
    for zone_name, group in zone_df.groupby("zone_name"):
        idx = (group["timestamp_sec"] - peak_time).abs().idxmin()
        row = group.loc[idx]
        zone_rows.append({
            "zone_name": zone_name,
            "zone_id": str(row.get("zone_id", "")),
            "zone_count": int(row["zone_count"]),
            "density_score_x10000": round(float(row["density"]) * 10000, 4),
            "risk_level": str(row["risk_level"]),
        })

    zone_rows = sorted(
        zone_rows,
        key=lambda x: (risk_score(x["risk_level"]), x["zone_count"]),
        reverse=True,
    )

    return {
        "peak_total_count": int(peak_row["total_count"]),
        "peak_frame_id": int(peak_row["frame_id"]),
        "peak_time_sec": peak_time,
        "peak_time_label": fmt_seconds(peak_time),
        "average_count_in_peak_window": round(float(nearby_frames["total_count"].mean()), 2),
        "zones_at_peak_sorted_by_risk": zone_rows,
        "high_or_critical_zone_count_at_peak": sum(
            1 for r in zone_rows if r["risk_level"] in {"HIGH", "CRITICAL"}
        ),
    }


def get_frame_exact(
    data: LoadedCrowdData,
    which: str,
) -> Dict[str, Any]:
    """
    Return the EXACT zone-by-zone state for one specific frame.

    `which` is one of:
      "first"        -> the first frame in the video
      "last"         -> the last frame in the video
      "frame:<int>"  -> the frame whose frame_id == <int> (nearest if missing)
    """
    frame_df = data.frame_df
    zone_df = data.zone_df

    if which == "first":
        frame_row = frame_df.iloc[0]
        descriptor = "first frame"
    elif which == "last":
        frame_row = frame_df.iloc[-1]
        descriptor = "last frame"
    elif which.startswith("frame:"):
        target_id = int(which.split(":", 1)[1])
        if (frame_df["frame_id"] == target_id).any():
            frame_row = frame_df[frame_df["frame_id"] == target_id].iloc[0]
        else:
            nearest_idx = (frame_df["frame_id"] - target_id).abs().idxmin()
            frame_row = frame_df.loc[nearest_idx]
        descriptor = f"frame {int(frame_row['frame_id'])}"
    else:
        frame_row = frame_df.iloc[0]
        descriptor = "first frame"

    frame_id = int(frame_row["frame_id"])
    actual_time = float(frame_row["timestamp_sec"])

    fzones = zone_df[zone_df["frame_id"] == frame_id]
    if fzones.empty:
        # fall back to nearest timestamp if frame_id is not aligned
        rows = []
        for zone_name, group in zone_df.groupby("zone_name"):
            idx = (group["timestamp_sec"] - actual_time).abs().idxmin()
            rows.append(group.loc[idx])
        fzones = pd.DataFrame(rows)

    zone_rows = []
    for _, row in fzones.iterrows():
        zone_rows.append({
            "zone_name": str(row["zone_name"]),
            "zone_id": str(row.get("zone_id", "")),
            "count": int(row["zone_count"]),
            "density_score_x10000": round(float(row["density"]) * 10000, 4),
            "risk_level": str(row["risk_level"]),
            "risk_score": risk_score(str(row["risk_level"])),
        })

    zone_rows_by_risk = sorted(
        zone_rows, key=lambda x: (x["risk_score"], x["count"]), reverse=True
    )

    return {
        "descriptor": descriptor,
        "frame_id": frame_id,
        "time_sec": actual_time,
        "time_label": fmt_seconds(actual_time),
        "total_count": int(frame_row["total_count"]),
        "is_exact_frame": True,
        "zones_ranked_by_risk": zone_rows_by_risk,
        "zones_ranked_by_count": sorted(zone_rows, key=lambda x: x["count"], reverse=True),
        "high_or_critical_zones": [r for r in zone_rows_by_risk if r["risk_level"] in {"HIGH", "CRITICAL"}],
        "note": (
            "These are exact values for a single frame. "
            "Risk is rule-based. Density is pixel-based relative to polygon area."
        ),
    }


def get_zone_status_at_time(data: LoadedCrowdData, timestamp_sec: float) -> Dict[str, Any]:
    frame_df = data.frame_df
    zone_df = data.zone_df
    duration = float(frame_df["timestamp_sec"].max())
    timestamp_sec = max(0.0, min(float(timestamp_sec), duration))

    frame_idx = (frame_df["timestamp_sec"] - timestamp_sec).abs().idxmin()
    frame_row = frame_df.loc[frame_idx]
    actual_time = float(frame_row["timestamp_sec"])

    rows = []
    for zone_name, group in zone_df.groupby("zone_name"):
        idx = (group["timestamp_sec"] - actual_time).abs().idxmin()
        row = group.loc[idx]
        rows.append({
            "zone_name": zone_name,
            "zone_id": str(row.get("zone_id", "")),
            "count": int(row["zone_count"]),
            "density_pixel": float(row["density"]),
            "density_score_x10000": round(float(row["density"]) * 10000, 4),
            "risk_level": str(row["risk_level"]),
            "risk_score": risk_score(str(row["risk_level"])),
        })

    rows_by_risk = sorted(
        rows, key=lambda x: (x["risk_score"], x["count"]), reverse=True
    )

    return {
        "requested_time_sec": timestamp_sec,
        "nearest_time_sec": actual_time,
        "nearest_time_label": fmt_seconds(actual_time),
        "frame_id": int(frame_row["frame_id"]),
        "total_count": int(frame_row["total_count"]),
        "zones_ranked_by_risk_at_time": rows_by_risk,
        "zones_ranked_by_count_at_time": sorted(rows, key=lambda x: x["count"], reverse=True),
        "high_or_critical_zones_at_time": [r for r in rows_by_risk if r["risk_level"] in {"HIGH", "CRITICAL"}],
    }


def get_all_zone_classifications_at_time(
    data: LoadedCrowdData, timestamp_sec: float
) -> Dict[str, Any]:
    status = get_zone_status_at_time(data, timestamp_sec)
    return {
        "requested_time_sec": status["requested_time_sec"],
        "nearest_time_label": status["nearest_time_label"],
        "total_count_at_time": status["total_count"],
        "note": (
            "All 8 zones listed. Risk is rule-based. "
            "Density is pixel-based relative to polygon area."
        ),
        "all_zone_classifications": [
            {
                "zone_name": z["zone_name"],
                "zone_id": z["zone_id"],
                "risk_level": z["risk_level"],
                "count": z["count"],
                "density_score_x10000": z["density_score_x10000"],
            }
            for z in status["zones_ranked_by_risk_at_time"]
        ],
    }


def get_top_risky_zones_at_time(
    data: LoadedCrowdData, timestamp_sec: float, n: int = 3
) -> Dict[str, Any]:
    status = get_zone_status_at_time(data, timestamp_sec)
    top = status["zones_ranked_by_risk_at_time"][:n]
    return {
        "nearest_time_label": status["nearest_time_label"],
        "total_count_at_time": status["total_count"],
        "top_risky_zones": top,
    }


def get_context_around_time(
    data: LoadedCrowdData, timestamp_sec: float, window_sec: float = 5.0
) -> Dict[str, Any]:
    frame_df = data.frame_df
    duration = float(frame_df["timestamp_sec"].max())
    timestamp_sec = max(0.0, min(float(timestamp_sec), duration))

    window = frame_df[
        (frame_df["timestamp_sec"] >= timestamp_sec - window_sec)
        & (frame_df["timestamp_sec"] <= timestamp_sec + window_sec)
    ]
    status = get_zone_status_at_time(data, timestamp_sec)

    return {
        "time_status": status,
        "window_sec": window_sec,
        "average_total_count_in_window": round(float(window["total_count"].mean()), 2) if not window.empty else None,
        "minimum_total_count_in_window": int(window["total_count"].min()) if not window.empty else None,
        "maximum_total_count_in_window": int(window["total_count"].max()) if not window.empty else None,
    }


def get_time_range_summary(
    data: LoadedCrowdData, start_sec: float, end_sec: float, label: Optional[str] = None
) -> Dict[str, Any]:
    """Summarise (average) crowd activity within a time range."""
    frame_df = data.frame_df
    zone_df = data.zone_df

    fw = frame_df[
        (frame_df["timestamp_sec"] >= start_sec) & (frame_df["timestamp_sec"] <= end_sec)
    ]
    zw = zone_df[
        (zone_df["timestamp_sec"] >= start_sec) & (zone_df["timestamp_sec"] <= end_sec)
    ]

    if fw.empty:
        return {"error": "No frame data found in this time range."}

    peak_idx = fw["total_count"].idxmax()
    peak_row = fw.loc[peak_idx]

    zone_stats = (
        zw.groupby("zone_name")
        .agg(
            avg_count=("zone_count", "mean"),
            max_count=("zone_count", "max"),
            mean_density=("density", "mean"),
        )
        .reset_index()
    )
    risk_modes = (
        zw.groupby("zone_name")["risk_level"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
        .reset_index(name="dominant_risk_in_range")
    )
    hc_pct = (
        zw.assign(is_hc=zw["risk_level"].isin(["HIGH", "CRITICAL"]).astype(int))
        .groupby("zone_name")["is_hc"]
        .mean()
        .mul(100)
        .reset_index(name="high_critical_pct_in_range")
    )
    zone_stats = zone_stats.merge(risk_modes, on="zone_name", how="left")
    zone_stats = zone_stats.merge(hc_pct, on="zone_name", how="left")
    zone_stats["avg_count"] = zone_stats["avg_count"].round(2)
    zone_stats["max_count"] = zone_stats["max_count"].astype(int)
    zone_stats["mean_density_score_x10000"] = (zone_stats["mean_density"] * 10000).round(4)
    zone_stats["high_critical_pct_in_range"] = zone_stats["high_critical_pct_in_range"].round(2)
    zone_stats = zone_stats.drop(columns=["mean_density"])
    zone_stats = zone_stats.sort_values("avg_count", ascending=False)

    return {
        "range_label": label or f"{fmt_seconds(start_sec)}–{fmt_seconds(end_sec)}",
        "start_label": fmt_seconds(start_sec),
        "end_label": fmt_seconds(end_sec),
        "start_sec": round(float(start_sec), 2),
        "end_sec": round(float(end_sec), 2),
        "is_average_over_range": True,
        "frames_in_range": int(len(fw)),
        "avg_total_count": round(float(fw["total_count"].mean()), 2),
        "max_total_count": int(fw["total_count"].max()),
        "min_total_count": int(fw["total_count"].min()),
        "peak_time_label": fmt_seconds(float(peak_row["timestamp_sec"])),
        "peak_count": int(peak_row["total_count"]),
        "zone_summaries_in_range": zone_stats.to_dict(orient="records"),
        "note": (
            "Values are AVERAGED over the time range. "
            "Risk shown per zone is the dominant (most frequent) rule-based label in that range. "
            "Density is pixel-based."
        ),
    }


def get_risk_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    zone_df = data.zone_df
    summary = data.zone_summary
    total_zone_frames = len(zone_df)
    high_critical_frames = int(zone_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
    risk_distribution = (
        zone_df["risk_level"].value_counts(normalize=True).mul(100).round(2).to_dict()
    )
    most_risky = summary.sort_values("high_critical_pct", ascending=False).iloc[0]
    least_risky = summary.sort_values("high_critical_pct", ascending=True).iloc[0]
    top3 = summary.sort_values("high_critical_pct", ascending=False).head(3)
    return {
        "total_zone_frames": total_zone_frames,
        "high_critical_frames": high_critical_frames,
        "high_critical_pct_overall": round(
            high_critical_frames / total_zone_frames * 100 if total_zone_frames else 0, 2
        ),
        "risk_distribution_pct": {str(k): float(v) for k, v in risk_distribution.items()},
        "most_risky_zone": {
            "zone_name": most_risky["zone_name"],
            "high_critical_pct": round(float(most_risky["high_critical_pct"]), 2),
            "dominant_risk": str(most_risky["dominant_risk"]),
            "avg_count": round(float(most_risky["avg_count"]), 2),
            "peak_count": int(most_risky["peak_count"]),
        },
        "least_risky_zone": {
            "zone_name": least_risky["zone_name"],
            "high_critical_pct": round(float(least_risky["high_critical_pct"]), 2),
            "dominant_risk": str(least_risky["dominant_risk"]),
        },
        "top_3_risky_zones": [
            {
                "zone_name": r["zone_name"],
                "high_critical_pct": round(float(r["high_critical_pct"]), 2),
                "dominant_risk": str(r["dominant_risk"]),
                "avg_count": round(float(r["avg_count"]), 2),
                "spike_events": int(r["spike_events"]),
            }
            for _, r in top3.iterrows()
        ],
    }


def get_anomaly_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    summary = data.zone_summary.copy()
    total_spikes = int(summary["spike_events"].sum())
    top_zone = summary.sort_values("spike_events", ascending=False).iloc[0]
    spike_by_zone = summary[["zone_name", "spike_events"]].sort_values(
        "spike_events", ascending=False
    ).to_dict(orient="records")
    return {
        "total_refined_spike_events": total_spikes,
        "top_spike_zone": {
            "zone_name": top_zone["zone_name"],
            "spike_events": int(top_zone["spike_events"]),
        },
        "spike_events_by_zone": spike_by_zone,
        "refined_spike_rule": (
            "count >= 20  AND  absolute_increase >= 10  AND  percent_increase >= 50% "
            "compared to 30 frames earlier (~1 second at 30fps)"
        ),
        "interpretation": (
            "Spike events represent sudden estimated-count increases. "
            "They are analysis signals, not confirmed real-world incidents."
        ),
        "recommended_action": (
            "Review video segments around spike timestamps for any operational concern."
        ),
    }


def get_temporal_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    frame_df = data.frame_df.copy()
    frame_df["count_change"] = frame_df["total_count"].diff().fillna(0)
    frame_df["abs_count_change"] = frame_df["count_change"].abs()
    frame_df["rolling_abs_change_5s"] = (
        frame_df["abs_count_change"].rolling(150, min_periods=1).mean()
    )
    idx_max_change = frame_df["rolling_abs_change_5s"].idxmax()
    row_max_change = frame_df.loc[idx_max_change]

    n = len(frame_df)
    first_quarter = frame_df.iloc[: n // 4]["total_count"].mean()
    last_quarter = frame_df.iloc[-n // 4 :]["total_count"].mean()
    overall_trend = "increasing" if last_quarter > first_quarter * 1.05 else (
        "decreasing" if last_quarter < first_quarter * 0.95 else "stable"
    )

    return {
        "global_summary": get_global_summary(data),
        "peak_context": get_peak_moment_context(data),
        "average_abs_count_change_per_frame": round(float(frame_df["abs_count_change"].mean()), 4),
        "strongest_change_period_time_sec": float(row_max_change["timestamp_sec"]),
        "strongest_change_period_time_label": fmt_seconds(float(row_max_change["timestamp_sec"])),
        "strongest_change_period_value": round(float(row_max_change["rolling_abs_change_5s"]), 4),
        "overall_crowd_trend": overall_trend,
        "first_quarter_avg_count": round(float(first_quarter), 2),
        "last_quarter_avg_count": round(float(last_quarter), 2),
    }


def get_spatial_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    return {
        "zone_rankings": get_all_zone_rankings(data),
        "risk_summary": get_risk_summary(data),
        "number_of_zones": int(data.zone_df["zone_name"].nunique()),
        "zone_names": available_zones(data),
    }


def get_statistical_summary(data: LoadedCrowdData) -> Dict[str, Any]:
    pivot = data.zone_df.pivot_table(
        index="timestamp_sec", columns="zone_name",
        values="zone_count", aggfunc="sum",
    ).fillna(0)

    corr = pivot.corr()
    cols = list(corr.columns)
    pairs = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            pairs.append({"zone_a": a, "zone_b": b, "correlation": round(float(corr.loc[a, b]), 4)})

    pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    strongest = pairs_sorted[0] if pairs_sorted else None
    top5_pairs = pairs_sorted[:5]

    total = pivot.sum(axis=1).replace(0, float("nan"))
    prob = pivot.div(total, axis=0).fillna(0)
    entropy = -(prob * np.log(prob.replace(0, float("nan")))).sum(axis=1).fillna(0)
    max_entropy = np.log(len(pivot.columns)) if len(pivot.columns) > 1 else 1
    norm_entropy = entropy / max_entropy

    return {
        "mean_normalized_entropy": round(float(norm_entropy.mean()), 4),
        "interpretation_entropy": (
            "0.0 = crowd fully concentrated in one zone; "
            "1.0 = crowd evenly spread across all zones"
        ),
        "highest_entropy_time_label": fmt_seconds(float(norm_entropy.idxmax())),
        "lowest_entropy_time_label": fmt_seconds(float(norm_entropy.idxmin())),
        "strongest_zone_correlation": strongest,
        "top_5_zone_correlations": top5_pairs,
        "interpretation_correlation": (
            "Positive correlation (near +1): zones fill and empty together. "
            "Negative correlation (near -1): when one zone fills, the other empties."
        ),
    }


def compare_zones(data: LoadedCrowdData, zone_a: str, zone_b: str) -> Dict[str, Any]:
    a = get_zone_summary(data, zone_a)
    b = get_zone_summary(data, zone_b)
    if "error" in a:
        return a
    if "error" in b:
        return b
    return {
        "zone_a": a,
        "zone_b": b,
        "comparison": {
            "higher_average_count": a["zone_name"] if a["average_count"] >= b["average_count"] else b["zone_name"],
            "higher_peak_count": a["zone_name"] if a["peak_count"] >= b["peak_count"] else b["zone_name"],
            "higher_density": a["zone_name"] if a["mean_density_score_x10000"] >= b["mean_density_score_x10000"] else b["zone_name"],
            "higher_risk_pct": a["zone_name"] if a["high_critical_pct"] >= b["high_critical_pct"] else b["zone_name"],
            "more_spikes": a["zone_name"] if a["spike_events"] >= b["spike_events"] else b["zone_name"],
            "summary": (
                f"{a['zone_name']} avg {a['average_count']:.1f} vs {b['zone_name']} avg {b['average_count']:.1f}. "
                f"Risk: {a['high_critical_pct']:.1f}% vs {b['high_critical_pct']:.1f}% HIGH/CRITICAL."
            ),
        },
    }


# ============================================================
# CHART CONTEXT
# ============================================================

def get_chart_context(data: LoadedCrowdData, chart_name: str) -> Dict[str, Any]:
    cn = chart_name.lower().replace(" ", "_")

    if cn == "global_crowd_timeline":
        gs = get_global_summary(data)
        temp = get_temporal_summary(data)
        return {
            "chart_name": "Global Crowd Timeline",
            "what_it_shows": (
                "Line chart of total estimated crowd count across all zones, "
                "per video frame, over the full recording duration."
            ),
            "axes": {"x": "Time (seconds)", "y": "Total estimated count (people)"},
            "visual_elements": [
                "Blue line: raw per-frame count",
                "Dotted purple line: 5-second rolling average",
                "Red dot: peak moment",
            ],
            "key_numbers": {
                "duration": gs["duration_label"],
                "average_count": gs["average_count"],
                "peak_count": gs["maximum_count"],
                "peak_time": gs["peak_time_label"],
                "overall_trend": temp["overall_crowd_trend"],
                "strongest_change_time": temp["strongest_change_period_time_label"],
            },
            "what_to_look_for": (
                "Peaks indicate when crowd density is highest and may warrant attention. "
                "The rolling average smooths noise and shows genuine trends."
            ),
            "caveat": (
                "Count is an estimated number from FIDTM, not a direct headcount. "
                "This is post-processed offline video, not live CCTV."
            ),
        }

    elif cn == "rate_of_change":
        temp = get_temporal_summary(data)
        return {
            "chart_name": "Rate of Change Proxy",
            "what_it_shows": (
                "5-second rolling average of the absolute frame-to-frame count change. "
                "It estimates how quickly the crowd level is fluctuating."
            ),
            "axes": {"x": "Time (seconds)", "y": "5-second rolling absolute count change"},
            "key_numbers": {
                "average_abs_change_per_frame": temp["average_abs_count_change_per_frame"],
                "strongest_change_time": temp["strongest_change_period_time_label"],
                "strongest_change_value": temp["strongest_change_period_value"],
            },
            "what_to_look_for": (
                "High peaks in this chart indicate periods of rapid crowd fluctuation — "
                "people arriving or leaving quickly. Low values indicate stability."
            ),
            "note": (
                "This is a proxy derived from count differences, not optical flow. "
                "It does not distinguish between people entering vs. leaving."
            ),
            "caveat": (
                "Motion direction requires optical flow or tracking, which is not implemented here."
            ),
        }

    elif cn == "zone_hotspot_ranking":
        rankings = get_all_zone_rankings(data)
        return {
            "chart_name": "Zone Hotspot Ranking",
            "what_it_shows": (
                "Horizontal bar chart ranking all zones by their average estimated crowd count "
                "over the full video duration."
            ),
            "axes": {"x": "Average count per frame", "y": "Zone name"},
            "key_numbers": {
                "main_hotspot": rankings["highest_average_count_zone"],
                "all_zones_ranked": rankings["all_zones_ranked_by_avg_count"],
            },
            "what_to_look_for": (
                "The longest bar is the persistent hotspot — the zone that consistently "
                "holds the most people. This is a planning priority for monitoring."
            ),
            "caveat": (
                "Average count is estimated from FIDTM outputs and zones are manually drawn polygons."
            ),
        }

    elif cn == "mean_pixel_density":
        rankings = get_all_zone_rankings(data)
        all_zones = get_all_zone_summaries(data)
        density_ranked = sorted(all_zones, key=lambda x: x["mean_density_score_x10000"], reverse=True)
        return {
            "chart_name": "Mean Pixel Density by Zone",
            "what_it_shows": (
                "Horizontal bar chart of mean pixel density score per zone, "
                "displayed as pixel density × 10,000 for readability."
            ),
            "axes": {"x": "Density score (pixel density × 10^4)", "y": "Zone name"},
            "key_numbers": {
                "highest_density_zone": rankings["highest_mean_density_zone"],
                "zones_by_density": [
                    {"zone": z["zone_name"], "density_score": z["mean_density_score_x10000"]}
                    for z in density_ranked
                ],
            },
            "what_to_look_for": (
                "High density scores suggest the zone polygon area is small relative to the "
                "number of detected people, or it genuinely receives more concentrated crowds."
            ),
            "critical_caveat": (
                "Density is pixel-based relative to the manually drawn polygon area. "
                "It is NOT real-world persons per square meter and cannot be used for "
                "safety-certified crowd density calculations."
            ),
        }

    elif cn == "refined_spike_events":
        anomaly = get_anomaly_summary(data)
        return {
            "chart_name": "Refined Spike Events",
            "what_it_shows": (
                "Bar chart of refined anomaly/spike event counts per zone. "
                "Each bar shows how many times that zone triggered the spike detector."
            ),
            "axes": {"x": "Number of spike events", "y": "Zone name"},
            "spike_detection_rule": anomaly["refined_spike_rule"],
            "key_numbers": {
                "total_spike_events": anomaly["total_refined_spike_events"],
                "top_spike_zone": anomaly["top_spike_zone"],
                "spike_events_by_zone": anomaly["spike_events_by_zone"],
            },
            "what_to_look_for": (
                "Zones with many spike events had repeated sudden crowd increases. "
                "These are candidate zones for video review."
            ),
            "caveat": (
                "Spike events are rule-based analysis flags, not confirmed real-world incidents. "
                "Each event should be reviewed against the corresponding video segment."
            ),
        }

    elif cn == "risk_level_distribution":
        risk = get_risk_summary(data)
        all_zones = get_all_zone_summaries(data)
        return {
            "chart_name": "Risk Level Distribution",
            "what_it_shows": (
                "Stacked bar chart showing the proportion of LOW/MEDIUM/HIGH/CRITICAL "
                "frames for each zone across the full recording."
            ),
            "axes": {"x": "Zone name", "y": "Percentage of frames (0-100%)"},
            "risk_label_definitions": {
                "LOW": "Low estimated count relative to zone capacity",
                "MEDIUM": "Moderate estimated count",
                "HIGH": "High estimated count",
                "CRITICAL": "Very high estimated count — warrants attention",
            },
            "key_numbers": {
                "most_risky_zone": risk["most_risky_zone"],
                "top_3_risky_zones": risk["top_3_risky_zones"],
                "overall_high_critical_pct": risk["high_critical_pct_overall"],
            },
            "what_to_look_for": (
                "Zones where HIGH or CRITICAL dominate the stack had sustained crowd pressure "
                "throughout the video. Zones with mostly LOW are relatively calm."
            ),
            "caveat": (
                "Risk labels are rule-based prototype thresholds. "
                "They are not certified safety standards."
            ),
        }

    elif cn == "zone_correlation":
        stats = get_statistical_summary(data)
        return {
            "chart_name": "Zone Correlation Heatmap",
            "what_it_shows": (
                "Pearson correlation matrix of zone estimated counts over time. "
                "Each cell shows how strongly two zones co-vary."
            ),
            "color_scale": "RdBu: blue = strong positive correlation, red = strong negative correlation",
            "key_numbers": {
                "strongest_correlation": stats["strongest_zone_correlation"],
                "top_5_pairs": stats["top_5_zone_correlations"],
                "interpretation": stats["interpretation_correlation"],
            },
            "what_to_look_for": (
                "High positive correlation (near +1) means two zones fill and empty together — "
                "they may be adjacent or share the same pedestrian flow. "
                "Negative correlation suggests competing zones or diversion paths."
            ),
            "caveat": (
                "Correlation does not imply causation. "
                "It only shows statistical co-movement in estimated counts."
            ),
        }

    elif cn == "crowd_distribution_entropy":
        stats = get_statistical_summary(data)
        return {
            "chart_name": "Crowd Distribution Entropy",
            "what_it_shows": (
                "Normalized Shannon entropy of the crowd distribution across all zones "
                "computed per frame, plotted over time."
            ),
            "axes": {"x": "Time (seconds)", "y": "Normalized entropy (0 to 1)"},
            "key_numbers": {
                "mean_normalized_entropy": stats["mean_normalized_entropy"],
                "highest_entropy_time": stats["highest_entropy_time_label"],
                "lowest_entropy_time": stats["lowest_entropy_time_label"],
                "interpretation": stats["interpretation_entropy"],
            },
            "what_to_look_for": (
                "High entropy = crowd is spread across many zones (dispersed). "
                "Low entropy = crowd is concentrated in one or few zones (focused). "
                "Sudden drops in entropy indicate crowd concentration events."
            ),
            "caveat": (
                "Entropy is computed from estimated counts, "
                "which are FIDTM model outputs, not a ground-truth headcount."
            ),
        }

    return {
        "error": f"Unknown chart: {chart_name}",
        "available_charts": list(set(_CHART_KEYWORDS.values())),
    }


# ============================================================
# RECOMMENDATION CONTEXT
# ============================================================

def get_recommendation_context(
    data: LoadedCrowdData,
    question: Optional[str] = None,
    selected_zone: Optional[str] = None,
) -> Dict[str, Any]:
    rankings = get_all_zone_rankings(data)
    risk_summary = get_risk_summary(data)
    anomaly_summary = get_anomaly_summary(data)
    peak_context = get_peak_moment_context(data)

    summary = data.zone_summary.copy()
    top3 = summary.sort_values("high_critical_pct", ascending=False).head(3)
    top3_data = [
        {
            "zone_name": r["zone_name"],
            "zone_id": str(r["zone_id"]),
            "high_critical_pct": round(float(r["high_critical_pct"]), 2),
            "avg_count": round(float(r["avg_count"]), 2),
            "peak_count": int(r["peak_count"]),
            "peak_time_label": fmt_seconds(float(r["peak_time_sec"])),
            "dominant_risk": str(r["dominant_risk"]),
            "spike_events": int(r["spike_events"]),
            "mean_density_score_x10000": round(float(r["mean_density"]) * 10000, 4),
        }
        for _, r in top3.iterrows()
    ]

    time_context = None
    if question:
        fr = parse_frame_reference(question)
        rel = parse_relative_time_window(question)
        t = parse_time_reference(question)
        if fr is not None:
            time_context = get_frame_exact(data, fr)
        elif rel is not None:
            s, e, lbl = rel
            duration = float(data.frame_df["timestamp_sec"].max())
            if e < 0:
                e = duration
            if s < 0:
                s = max(0.0, duration + s)
            time_context = get_time_range_summary(data, s, e, label=lbl)
        elif t is not None:
            time_context = get_all_zone_classifications_at_time(data, t)

    selected_zone_data = None
    if selected_zone:
        selected_zone_data = get_zone_summary(data, selected_zone)

    return {
        "top_3_highest_risk_zones": top3_data,
        "overall_risk_summary": risk_summary,
        "anomaly_summary": anomaly_summary,
        "peak_context": {
            "peak_time_label": peak_context["peak_time_label"],
            "peak_total_count": peak_context["peak_total_count"],
            "high_critical_zones_at_peak": peak_context["high_or_critical_zone_count_at_peak"],
        },
        "zone_rankings": rankings,
        "time_specific_context": time_context,
        "selected_zone_summary": selected_zone_data,
        "thesis_safe_guidance": (
            "Risk labels are rule-based prototype labels, not certified safety thresholds. "
            "Density is pixel-based, not real-world persons/m^2. "
            "Recommendations should be framed as monitoring priorities, not safety directives."
        ),
    }


# ============================================================
# CONTEXT ROUTER
# ============================================================

def facts_to_text(title: str, facts: Any, max_depth: int = 5) -> str:
    def render(value: Any, depth: int = 0) -> List[str]:
        if depth > max_depth:
            return [compact_text(str(value), 300)]
        lines = []
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{k}:")
                    lines.extend([f"  {line}" for line in render(v, depth + 1)])
                else:
                    lines.append(f"{k}: {v}")
        elif isinstance(value, list):
            for idx, item in enumerate(value[:15], start=1):
                if isinstance(item, dict):
                    lines.append(f"{idx}.")
                    lines.extend([f"  {line}" for line in render(item, depth + 1)])
                else:
                    lines.append(f"{idx}. {item}")
            if len(value) > 15:
                lines.append(f"... {len(value) - 15} more items")
        else:
            lines.append(str(value))
        return lines

    return f"## {title}\n" + "\n".join(render(facts))


def build_context_for_question(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Smart context router — uses detect_intent to call only the tools needed.
    Always includes global_summary as baseline.
    """
    intent_info = detect_intent(data, question, selected_zone)
    intent = intent_info["intent"]
    frame_ref = intent_info["frame_ref"]
    relative_window = intent_info["relative_window"]
    time_ref = intent_info["time_ref"]
    time_range = intent_info["time_range"]
    zone_to_use = intent_info["zone_to_use"]
    zones_mentioned = intent_info["zones_mentioned"]
    asks_all_zones = intent_info["asks_all_zones"]
    chart_name = intent_info["chart_name"]

    duration = float(data.frame_df["timestamp_sec"].max())

    context: Dict[str, Any] = {}
    context["global_summary"] = get_global_summary(data)

    # ── Identity ───────────────────────────────────────────
    if intent == "identity":
        context["agent_capability"] = {
            "role": "AI Insights Assistant for the Crowd Monitoring Dashboard",
            "data_sources": [
                "FULL_01_shinjuku_frame_counts.csv — per-frame total count and timestamps",
                "FULL_02_shinjuku_zone_density_risk.csv — per-zone count, pixel density, and risk labels",
                "computed zone summaries: averages, peaks, spike events, density trends",
            ],
            "can_answer": [
                "Exact single-frame states: 'first frame', 'last frame', 'frame 1234'",
                "Averaged time windows: 'first minute', 'last 30 seconds', 'whole video'",
                "Temporal analysis: peak moments, count trends, rate of change",
                "Spatial analysis: zone hotspot ranking, density comparison",
                "Anomaly detection: spike events, sudden changes",
                "Statistical insights: zone correlation, crowd entropy",
                "Time-specific queries: zone states at any timestamp",
                "Zone-specific explanations and comparisons",
                "Evidence-backed recommendations for monitoring priorities",
                "Chart explanations for all 8 Analytics page charts",
            ],
            "how_it_works": (
                "Python tools extract exact facts from CSV outputs. "
                "The LLM then explains them in natural language. "
                "No numbers are invented — all values come from the data."
            ),
            "limitations": [
                "Does not directly watch the video",
                "Density is pixel-based, not real-world persons/m^2",
                "Risk labels are rule-based prototypes, not certified thresholds",
                "Motion/stagnation requires optical flow (not yet implemented)",
                "Uses saved offline outputs, not live CCTV unless pipeline is updated",
            ],
        }
        return context

    # ── Time-specific (exact frame / relative window / single time / range) ──
    if intent == "time_specific":
        # 1) exact single frame request wins first
        if frame_ref is not None:
            context["exact_frame_state"] = get_frame_exact(data, frame_ref)
            if zone_to_use:
                context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
            return context

        # 2) relative window -> averaged range
        if relative_window is not None:
            s, e, lbl = relative_window
            if e < 0:
                e = duration
            if s < 0:
                s = max(0.0, duration + s)
            context["time_range_average"] = get_time_range_summary(data, s, e, label=lbl)
            return context

        # 3) explicit "between A and B" range
        if time_range is not None:
            context["time_range_average"] = get_time_range_summary(
                data, time_range[0], time_range[1]
            )
            return context

        # 4) single instant
        if time_ref is not None:
            if asks_all_zones:
                context["all_zone_classifications_at_time"] = get_all_zone_classifications_at_time(
                    data, time_ref
                )
            else:
                context["zone_status_at_time"] = get_zone_status_at_time(data, time_ref)
                context["context_around_time"] = get_context_around_time(data, time_ref, window_sec=5.0)
                if zone_to_use:
                    context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
        return context

    # ── Recommendation ─────────────────────────────────────
    if intent == "recommendation":
        context["recommendation_context"] = get_recommendation_context(
            data, question=question, selected_zone=zone_to_use
        )
        return context

    # ── Chart explanation ──────────────────────────────────
    if intent == "chart":
        if chart_name:
            context["chart_context"] = get_chart_context(data, chart_name)
        else:
            context["temporal_summary"] = get_temporal_summary(data)
            context["zone_rankings"] = get_all_zone_rankings(data)
            context["risk_summary"] = get_risk_summary(data)
            context["anomaly_summary"] = get_anomaly_summary(data)
            context["statistical_summary"] = get_statistical_summary(data)
        return context

    # ── Anomaly ────────────────────────────────────────────
    if intent == "anomaly":
        context["anomaly_summary"] = get_anomaly_summary(data)
        context["risk_summary"] = get_risk_summary(data)
        if zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
        return context

    # ── Statistical ────────────────────────────────────────
    if intent == "statistical":
        context["statistical_summary"] = get_statistical_summary(data)
        if chart_name:
            context["chart_context"] = get_chart_context(data, chart_name)
        return context

    # ── Temporal ───────────────────────────────────────────
    if intent == "temporal":
        context["temporal_summary"] = get_temporal_summary(data)
        if chart_name:
            context["chart_context"] = get_chart_context(data, chart_name)
        return context

    # ── Spatial ────────────────────────────────────────────
    if intent == "spatial":
        context["spatial_summary"] = get_spatial_summary(data)
        if chart_name:
            context["chart_context"] = get_chart_context(data, chart_name)
        return context

    # ── Comparison ─────────────────────────────────────────
    if intent == "comparison":
        if len(zones_mentioned) >= 2:
            context["zone_comparison"] = compare_zones(
                data, zones_mentioned[0], zones_mentioned[1]
            )
            for extra_zone in zones_mentioned[2:4]:
                context[f"zone_summary_{extra_zone}"] = get_zone_summary(data, extra_zone)
        elif zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
            context["zone_rankings"] = get_all_zone_rankings(data)
        return context

    # ── Zone ───────────────────────────────────────────────
    if intent == "zone":
        if zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
        context["zone_rankings"] = get_all_zone_rankings(data)
        return context

    # ── Risk ───────────────────────────────────────────────
    if intent == "risk":
        context["risk_summary"] = get_risk_summary(data)
        context["zone_rankings"] = get_all_zone_rankings(data)
        if zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
        return context

    # ── Peak ───────────────────────────────────────────────
    if intent == "peak":
        context["peak_moment_context"] = get_peak_moment_context(data)
        if zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
        return context

    # ── Global summary ─────────────────────────────────────
    if intent == "global_summary":
        context["temporal_summary"] = get_temporal_summary(data)
        context["zone_rankings"] = get_all_zone_rankings(data)
        context["risk_summary"] = get_risk_summary(data)
        context["anomaly_summary"] = get_anomaly_summary(data)
        return context

    # ── Thesis ─────────────────────────────────────────────
    if intent == "thesis":
        context["temporal_summary"] = get_temporal_summary(data)
        context["zone_rankings"] = get_all_zone_rankings(data)
        context["risk_summary"] = get_risk_summary(data)
        context["anomaly_summary"] = get_anomaly_summary(data)
        context["statistical_summary"] = get_statistical_summary(data)
        if zone_to_use:
            context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)
        return context

    # ── General fallback ───────────────────────────────────
    context["zone_rankings"] = get_all_zone_rankings(data)
    context["risk_summary"] = get_risk_summary(data)
    context["anomaly_summary"] = get_anomaly_summary(data)
    if zone_to_use:
        context["selected_zone_summary"] = get_zone_summary(data, zone_to_use)

    return context


def build_context_text_for_question(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> str:
    context = build_context_for_question(data, question, selected_zone)

    sections = [facts_to_text(title, facts) for title, facts in context.items()]

    sections.append(
        "## Interpretation constraints (MUST respect)\n"
        "- Density is pixel-based relative to manually drawn polygon area — "
        "NOT real-world persons per square meter.\n"
        "- Risk is a rule-based prototype label — "
        "NOT a certified safety threshold.\n"
        "- Dashboard currently uses saved/offline outputs — "
        "not direct live CCTV unless user confirms a live pipeline.\n"
        "- The assistant answers from structured CSV outputs and computed summaries — "
        "it does NOT directly watch the video.\n"
        "- Motion/stagnation requires optical flow or tracking — "
        "do not claim stagnation detection is implemented.\n"
        "- If the context contains 'exact_frame_state', report those values as the EXACT "
        "values for that single frame.\n"
        "- If the context contains 'time_range_average', report those values as AVERAGES "
        "over the stated window, not single-frame values.\n"
        "- Use ONLY facts from this context. Do NOT invent any numbers."
    )

    return "\n\n".join(sections)


# ============================================================
# RULE-BASED FALLBACK
# ============================================================

def answer_rule_based(
    data: LoadedCrowdData,
    question: str,
    selected_zone: Optional[str] = None,
) -> str:
    """
    Deterministic data-grounded fallback used when LLM is unavailable.
    Routes by detected intent and always cites exact numbers.
    """
    intent_info = detect_intent(data, question, selected_zone)
    intent = intent_info["intent"]
    frame_ref = intent_info["frame_ref"]
    relative_window = intent_info["relative_window"]
    time_ref = intent_info["time_ref"]
    time_range = intent_info["time_range"]
    asks_all_zones = intent_info["asks_all_zones"]
    zone_to_use = intent_info["zone_to_use"]
    chart_name = intent_info["chart_name"]

    duration = float(data.frame_df["timestamp_sec"].max())

    # ── Identity ───────────────────────────────────────────
    if intent == "identity":
        return (
            "I am the **AI Insights Assistant** for this crowd monitoring dashboard.\n\n"
            "I can answer questions about:\n"
            "- **Exact frames** — 'first frame', 'last frame', 'frame 1234'\n"
            "- **Time windows** — 'first minute', 'last 30 seconds', 'whole video' (averaged)\n"
            "- **Temporal analysis** — peak moments, count trends, rate of change\n"
            "- **Spatial analysis** — zone hotspots, density rankings\n"
            "- **Anomaly detection** — spike events and sudden changes\n"
            "- **Statistical insights** — zone correlation and crowd entropy\n"
            "- **Time-specific queries** — zone states at any timestamp\n"
            "- **Chart explanations** — all 8 Analytics page charts\n"
            "- **Evidence-backed recommendations** — monitoring priorities\n\n"
            "I answer from CSV outputs and computed summaries, "
            "not directly from the video."
        )

    # ── Exact single frame (first / last / frame N) ────────
    if frame_ref is not None:
        fx = get_frame_exact(data, frame_ref)
        lines = [
            f"At the **{fx['descriptor']}** (frame {fx['frame_id']}, {fx['time_label']}), "
            f"total count was **{fmt_int(fx['total_count'])}**.\n",
            "Exact zone-by-zone classification (sorted by risk level):\n",
        ]
        for z in fx["zones_ranked_by_risk"]:
            lines.append(
                f"- **{z['zone_name']} ({z['zone_id']})**: "
                f"Risk **{z['risk_level']}**, "
                f"Count **{fmt_int(z['count'])}**, "
                f"Density score **{fmt_float(z['density_score_x10000'], 2)}**"
            )
        lines.append(
            "\n*These are exact single-frame values. "
            "Risk is rule-based. Density is pixel-based, not real-world persons/m².*"
        )
        return "\n".join(lines)

    # ── Relative time window (first/last minute, etc.) ─────
    if relative_window is not None:
        s, e, lbl = relative_window
        if e < 0:
            e = duration
        if s < 0:
            s = max(0.0, duration + s)
        tr = get_time_range_summary(data, s, e, label=lbl)
        if "error" in tr:
            return f"No data found for {lbl}."
        lines = [
            f"Averaged over the **{tr['range_label']}** "
            f"({tr['start_label']}–{tr['end_label']}, {tr['frames_in_range']} frames):\n",
            f"- Average total count: **{fmt_count(tr['avg_total_count'])}**",
            f"- Peak total count: **{fmt_int(tr['max_total_count'])}** at **{tr['peak_time_label']}**",
            f"- Minimum total count: **{fmt_int(tr['min_total_count'])}**\n",
            "Per-zone averages in this window (sorted by avg count):\n",
        ]
        for z in tr["zone_summaries_in_range"]:
            lines.append(
                f"- **{z['zone_name']}**: avg count **{fmt_count(z['avg_count'])}**, "
                f"max **{fmt_int(z['max_count'])}**, "
                f"dominant risk **{z['dominant_risk_in_range']}**, "
                f"{fmt_pct(z['high_critical_pct_in_range'])} HIGH/CRITICAL"
            )
        lines.append(
            "\n*Values are averaged over the window. "
            "Risk is rule-based. Density is pixel-based.*"
        )
        return "\n".join(lines)

    # ── Time-specific: all zones at one instant ────────────
    if time_ref is not None and asks_all_zones:
        allz = get_all_zone_classifications_at_time(data, time_ref)
        lines = [
            f"At **{allz['nearest_time_label']}**, total estimated count was "
            f"**{fmt_int(allz['total_count_at_time'])}**.\n",
            "Zone-by-zone classification (sorted by risk level):\n",
        ]
        for z in allz["all_zone_classifications"]:
            lines.append(
                f"- **{z['zone_name']} ({z['zone_id']})**: "
                f"Risk **{z['risk_level']}**, "
                f"Count **{fmt_int(z['count'])}**, "
                f"Density score **{fmt_float(z['density_score_x10000'], 2)}**"
            )
        lines.append(
            "\n*Risk is rule-based. Density is pixel-based, not real-world persons/m².*"
        )
        return "\n".join(lines)

    # ── Time-specific: top zones at one instant ────────────
    if time_ref is not None:
        status = get_zone_status_at_time(data, time_ref)
        top3 = status["zones_ranked_by_risk_at_time"][:3]
        lines = [
            f"At **{status['nearest_time_label']}**, total count was "
            f"**{fmt_int(status['total_count'])}**.\n",
            "Top 3 zones by risk:",
        ]
        for r in top3:
            lines.append(
                f"- **{r['zone_name']} ({r['zone_id']})**: "
                f"Risk **{r['risk_level']}**, "
                f"Count **{fmt_int(r['count'])}**"
            )
        if zone_to_use and zone_to_use not in [r["zone_name"] for r in top3]:
            zs = [
                r for r in status["zones_ranked_by_risk_at_time"]
                if r["zone_name"] == zone_to_use
            ]
            if zs:
                z = zs[0]
                lines.append(
                    f"\n**{z['zone_name']}** (selected zone) at this time: "
                    f"Risk **{z['risk_level']}**, Count **{fmt_int(z['count'])}**"
                )
        lines.append("\n*Risk is rule-based. Density is pixel-based.*")
        return "\n".join(lines)

    # ── Explicit time range ────────────────────────────────
    if time_range is not None:
        tr = get_time_range_summary(data, time_range[0], time_range[1])
        return (
            f"Between **{tr['start_label']}** and **{tr['end_label']}**: "
            f"avg total count **{fmt_count(tr['avg_total_count'])}**, "
            f"peak count **{fmt_int(tr['max_total_count'])}** at **{tr['peak_time_label']}**."
        )

    # ── Recommendation ─────────────────────────────────────
    if intent == "recommendation":
        rec = get_recommendation_context(data, question=question, selected_zone=zone_to_use)
        top3 = rec["top_3_highest_risk_zones"]
        anomaly = rec["anomaly_summary"]
        peak = rec["peak_context"]

        lines = ["**Evidence-backed recommendations:**\n"]

        if top3:
            z1 = top3[0]
            lines.append(f"**1. Prioritize monitoring {z1['zone_name']} ({z1['zone_id']})**")
            lines.append(
                f"   Evidence: {z1['high_critical_pct']:.1f}% HIGH/CRITICAL frames, "
                f"avg count {z1['avg_count']:.1f}, peak {z1['peak_count']} at {z1['peak_time_label']}."
            )
            lines.append(
                f"   Reasoning: Sustained crowd presence indicates a persistent hotspot, "
                f"not just an isolated short-term spike."
            )
            lines.append(
                "   Caveat: Risk labels are rule-based prototype labels, "
                "not certified safety thresholds.\n"
            )

        if len(top3) > 1:
            z2 = top3[1]
            lines.append(f"**2. Secondary attention to {z2['zone_name']} ({z2['zone_id']})**")
            lines.append(
                f"   Evidence: {z2['high_critical_pct']:.1f}% HIGH/CRITICAL frames, "
                f"{z2['spike_events']} spike events detected."
            )
            lines.append(
                "   Reasoning: Recurring anomaly signals indicate unstable crowd dynamics.\n"
            )

        spike_zone = anomaly["top_spike_zone"]
        lines.append(f"**3. Review video segments for {spike_zone['zone_name']}**")
        lines.append(
            f"   Evidence: {spike_zone['spike_events']} refined spike events — "
            f"highest of any zone."
        )
        lines.append(
            "   Reasoning: Frequent sudden increases warrant operational review.\n"
        )

        lines.append(
            f"**4. Enhanced monitoring during the peak period (around {peak['peak_time_label']})**"
        )
        lines.append(
            f"   Evidence: Global count peaked at {fmt_int(peak['peak_total_count'])}, "
            f"with {peak['high_critical_zones_at_peak']} zones at HIGH/CRITICAL simultaneously."
        )

        lines.append(
            "\n*Density is pixel-based. Risk is rule-based. "
            "These recommendations support decision-making, not direct safety certification.*"
        )
        return "\n".join(lines)

    # ── Chart ─────────────────────────────────────────────
    if intent == "chart" and chart_name:
        cc = get_chart_context(data, chart_name)
        if "error" not in cc:
            kn = cc.get("key_numbers", {})
            kn_str = ", ".join(
                f"{k}: {v}" for k, v in kn.items() if not isinstance(v, (dict, list))
            )
            return (
                f"**{cc['chart_name']}**: {cc.get('what_it_shows', '')} "
                f"Key values — {kn_str}. "
                f"{cc.get('what_to_look_for', '')} "
                f"{cc.get('caveat', cc.get('critical_caveat', ''))}"
            )

    # ── Zone-specific ──────────────────────────────────────
    if zone_to_use:
        z = get_zone_summary(data, zone_to_use)
        if "error" not in z:
            return (
                f"**{z['zone_name']} ({z['zone_id']})**: "
                f"avg count **{fmt_count(z['average_count'])}**, "
                f"peak **{fmt_int(z['peak_count'])}** at **{z['peak_time_label']}**, "
                f"density score **{fmt_float(z['mean_density_score_x10000'], 2)}** (pixel×10⁴), "
                f"**{fmt_pct(z['high_critical_pct'])}** HIGH/CRITICAL frames "
                f"(LOW {z['low_pct']:.1f}% / MED {z['medium_pct']:.1f}% / "
                f"HIGH {z['high_pct']:.1f}% / CRIT {z['critical_pct']:.1f}%), "
                f"dominant risk **{z['dominant_risk']}**, "
                f"**{fmt_int(z['spike_events'])}** spike events. "
                f"*Density is pixel-based. Risk is rule-based.*"
            )

    # ── Risk ───────────────────────────────────────────────
    if intent == "risk":
        risk = get_risk_summary(data)
        z = risk["most_risky_zone"]
        top3 = risk["top_3_risky_zones"]
        lines = [
            f"Most risky zone: **{z['zone_name']}** — "
            f"**{fmt_pct(z['high_critical_pct'])}** HIGH/CRITICAL frames, "
            f"dominant risk **{z['dominant_risk']}**, avg count **{z['avg_count']:.1f}**.\n",
            "Top 3 risky zones:",
        ]
        for t in top3:
            lines.append(
                f"- **{t['zone_name']}**: {t['high_critical_pct']:.1f}% HIGH/CRITICAL, "
                f"avg {t['avg_count']:.1f}, {t['spike_events']} spikes"
            )
        lines.append("\n*Risk is a rule-based prototype label, not a certified safety threshold.*")
        return "\n".join(lines)

    # ── Peak ───────────────────────────────────────────────
    if intent == "peak":
        peak = get_peak_moment_context(data)
        top3 = peak["zones_at_peak_sorted_by_risk"][:3]
        lines = [
            f"Peak crowd moment: **{peak['peak_time_label']}**, "
            f"total count **{fmt_int(peak['peak_total_count'])}**. "
            f"{peak['high_or_critical_zone_count_at_peak']} zones were HIGH/CRITICAL at that time.\n",
            "Top zones at peak:",
        ]
        for z in top3:
            lines.append(
                f"- **{z['zone_name']}**: risk **{z['risk_level']}**, "
                f"count **{fmt_int(z['zone_count'])}**"
            )
        return "\n".join(lines)

    # ── Anomaly ────────────────────────────────────────────
    if intent == "anomaly":
        anomaly = get_anomaly_summary(data)
        top = anomaly["top_spike_zone"]
        by_zone = anomaly["spike_events_by_zone"]
        lines = [
            f"Refined spike detector found **{fmt_int(anomaly['total_refined_spike_events'])}** "
            f"spike events across all zones.\n",
            f"Spike rule: {anomaly['refined_spike_rule']}\n",
            "Events by zone:",
        ]
        for row in by_zone:
            lines.append(f"- **{row['zone_name']}**: {fmt_int(row['spike_events'])} events")
        lines.append(
            f"\nInterpretation: {anomaly['interpretation']}"
        )
        return "\n".join(lines)

    # ── Temporal ───────────────────────────────────────────
    if intent == "temporal":
        temp = get_temporal_summary(data)
        gs = temp["global_summary"]
        return (
            f"**Temporal analysis** — Duration: **{gs['duration_label']}**, "
            f"avg count **{fmt_count(gs['average_count'])}**, "
            f"peak **{fmt_int(gs['maximum_count'])}** at **{gs['peak_time_label']}**. "
            f"Overall trend: **{temp['overall_crowd_trend']}**. "
            f"Strongest rate-of-change period: **{temp['strongest_change_period_time_label']}** "
            f"(rolling change value {temp['strongest_change_period_value']:.4f}). "
            f"*Count is estimated from FIDTM; motion direction requires optical flow.*"
        )

    # ── Spatial ────────────────────────────────────────────
    if intent == "spatial":
        spatial = get_spatial_summary(data)
        rk = spatial["zone_rankings"]
        return (
            f"**Spatial analysis** — Main count hotspot: "
            f"**{rk['highest_average_count_zone']['zone_name']}** "
            f"(avg {rk['highest_average_count_zone']['average_count']:.1f}). "
            f"Highest density zone: **{rk['highest_mean_density_zone']['zone_name']}** "
            f"(score {rk['highest_mean_density_zone']['mean_density_score_x10000']:.2f}). "
            f"Most risky: **{rk['most_risky_zone']['zone_name']}** "
            f"({rk['most_risky_zone']['high_critical_pct']:.1f}% HIGH/CRITICAL). "
            f"*Density is pixel-based; zones are manually drawn polygons.*"
        )

    # ── Statistical ────────────────────────────────────────
    if intent == "statistical":
        stats = get_statistical_summary(data)
        corr = stats.get("strongest_zone_correlation")
        corr_str = (
            f"Strongest correlation: **{corr['zone_a']}** vs **{corr['zone_b']}** = {corr['correlation']:.2f}"
            if corr
            else "Correlation data unavailable"
        )
        return (
            f"**Statistical analysis** — {corr_str}. "
            f"Mean normalized entropy: **{stats['mean_normalized_entropy']:.3f}** "
            f"(0 = concentrated, 1 = evenly spread). "
            f"Most concentrated moment: **{stats['lowest_entropy_time_label']}**. "
            f"Most spread moment: **{stats['highest_entropy_time_label']}**."
        )

    # ── General fallback ───────────────────────────────────
    gs = get_global_summary(data)
    rk = get_all_zone_rankings(data)
    return (
        f"Experiment: **{fmt_int(gs['processed_frames'])}** frames over "
        f"**{gs['duration_label']}**. "
        f"Avg count: **{fmt_count(gs['average_count'])}**, "
        f"peak: **{fmt_int(gs['maximum_count'])}** at **{gs['peak_time_label']}**. "
        f"Main hotspot: **{rk['highest_average_count_zone']['zone_name']}**. "
        f"Most risky: **{rk['most_risky_zone']['zone_name']}** "
        f"({rk['most_risky_zone']['high_critical_pct']:.1f}% HIGH/CRITICAL)."
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "LoadedCrowdData",
    "load_crowd_data",
    "available_zones",
    "resolve_zone_name",
    "extract_zone_from_question",
    "extract_all_zones_from_question",
    "parse_frame_reference",
    "parse_relative_time_window",
    "parse_time_reference",
    "parse_time_range",
    "detect_intent",
    "get_global_summary",
    "get_zone_summary",
    "get_all_zone_summaries",
    "get_all_zone_rankings",
    "get_peak_moment_context",
    "get_frame_exact",
    "get_zone_status_at_time",
    "get_all_zone_classifications_at_time",
    "get_top_risky_zones_at_time",
    "get_context_around_time",
    "get_time_range_summary",
    "get_risk_summary",
    "get_anomaly_summary",
    "get_temporal_summary",
    "get_spatial_summary",
    "get_statistical_summary",
    "compare_zones",
    "get_chart_context",
    "get_recommendation_context",
    "build_context_for_question",
    "build_context_text_for_question",
    "answer_rule_based",
    "fmt_int",
    "fmt_float",
    "fmt_count",
    "fmt_pct",
    "fmt_density_score",
    "fmt_seconds",
]