#!/usr/bin/env python3
"""
Build a fuller manager_action_ground_truth.csv for the Balanced Fund VA project.

This version is designed for your current app.js / api_server.py pipeline:
- It accepts both the app.js legacy holdings names and the cleaner names:
  stock_before_2010.csv, stock_between_2010_2014.csv,
  stock_between_2015_2019.csv, stock_between_2020_2026.csv.
- It keeps the frontend Part1-Part5 unchanged and builds a backend ML-ready
  manager-action ground-truth table.
- It tries to reconstruct the same concepts used in the frontend:
  3-year trailing return, S&P500 excess return, manager style, portfolio beta,
  technology exposure, non-individual exposure, future 12-month outcome labels.

Default command on your Windows project:
    python build_manager_action_groundtruth_v2.py --data-root "C:\\Users\\user\\Desktop\\crsp_research__\\data"

Main output:
    <data-root>/outputs/manager_action_groundtruth/manager_action_ground_truth.csv

Audit output:
    <data-root>/outputs/manager_action_groundtruth/manager_action_ground_truth_audit.json

Important thesis wording:
    This table creates historical action-outcome labels. It supports prediction
    and association analysis, but it does not prove causality by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path(r"C:\Users\user\Desktop\crsp_research__\data")
RISK_FREE_RATE = 0.01



def write_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    """Robust CSV writer; vectorizes missing-value cleanup before writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = df.astype(object).where(pd.notna(df), "")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(clean.columns))
        writer.writerows(clean.to_numpy().tolist())

# -----------------------------------------------------------------------------
# App-compatible aliases
# -----------------------------------------------------------------------------

FUND_FILES = [
    ["crsp/fund_level/balanced_before2010.csv", "balanced_before2010.csv"],
    ["crsp/fund_level/balanced_after2010.csv", "balanced_after2010.csv"],
]

SP500_FILES = ["market/sp500_monthly_returns_1871_2026.csv", "sp500_monthly_returns_1871_2026.csv"]
YIELD_FILES = ["market/FRB_H15.csv", "FRB_H15.csv"]
BETA_FILES = ["part5_equity_beta/part5_yearly_trailing_stock_beta.csv", "part5_yearly_trailing_stock_beta.csv"]

# Each row: source_key, candidate relative paths.  The first clean name is the
# one you said you want in holdings_raw; legacy app.js names are also supported.
HOLDINGS_FILES = [
    ("before2010", [
        "crsp/holdings_raw/stock_before_2010.csv",
        "holdings_raw/stock_before_2010.csv",
        "stock_before_2010.csv",
        "crsp/holdings_raw/stock berfore 2010_new___.csv",
        "stock berfore 2010_new___.csv",
    ]),
    ("y2010_2014", [
        "crsp/holdings_raw/stock_between_2010_2014.csv",
        "holdings_raw/stock_between_2010_2014.csv",
        "stock_between_2010_2014.csv",
        "crsp/holdings_raw/stock between 2010_2014_new___.csv",
        "stock between 2010_2014_new___.csv",
    ]),
    ("y2015_2019", [
        "crsp/holdings_raw/stock_between_2015_2019.csv",
        "holdings_raw/stock_between_2015_2019.csv",
        "stock_between_2015_2019.csv",
        "crsp/holdings_raw/stock between 2015_2019_new___.csv",
        "stock between 2015_2019_new___.csv",
    ]),
    ("y2020_2026", [
        "crsp/holdings_raw/stock_between_2020_2026.csv",
        "holdings_raw/stock_between_2020_2026.csv",
        "stock_between_2020_2026.csv",
        "crsp/holdings_raw/stock between 2020_2026_new___.csv",
        "stock between 2020_2026_new___.csv",
    ]),
]

NON_INDIVIDUAL_ENRICHED_FILES = [
    "part5_non_individual_holdings/part5_excluded_two_group_enriched.csv",
    "part5_excluded_two_group_enriched.csv",
]
NON_INDIVIDUAL_PANEL_FILES = [
    "part5_non_individual_holdings/part5_excluded_two_group_active_year_panel.csv",
    "part5_excluded_two_group_active_year_panel.csv",
]

REQUIRED_COLUMNS = [
    "manager", "fund", "report_date", "market_regime", "manager_style_group",
    "stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta",
    "technology_exposure", "bond_money_exposure", "indirect_equity_exposure",
    "delta_stock", "delta_beta", "delta_technology", "delta_bond_money",
    "style_deviation_score", "cross_asset_execution_type", "future_12m_return",
    "future_12m_excess_return", "future_drawdown", "outcome_label",
]

EXTRA_COLUMNS = [
    "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name", "year", "quarter",
    "month_key", "yield10y", "sp500_trailing_3y", "fund_trailing_3y",
    "fund_trailing_3y_excess", "manager_obs_count", "manager_reliability_score",
    "manager_defensive_score", "manager_flow_score", "manager_growth_tilt_score",
    "action_type", "action_strength", "future_12m_sp500_return",
    "stock_allocation_raw", "bond_allocation_raw", "cash_allocation_raw",
    "company_equity_exposure_proxy", "top_holding_concentration",
    "delta_indirect_equity", "allocation_completion_method", "non_individual_source",
    "holding_row_count", "beta_matched_holding_count", "non_individual_matched_holding_count",
    "data_quality_flags",
]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def find_file(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Find a file by expected nested path, flat fallback, then recursive fallback."""
    for rel in candidates:
        p = root / rel
        if p.exists() and p.is_file():
            return p
    names = {Path(rel).name.lower() for rel in candidates}
    for rel in candidates:
        p = root / Path(rel).name
        if p.exists() and p.is_file():
            return p
    try:
        for p in root.rglob("*.csv"):
            if p.name.lower() in names:
                return p
    except Exception:
        pass
    return None


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()




def clean_id(value) -> str:
    """Normalize CRSP-style numeric IDs so 1000057.0 and 1000057 match."""
    t = clean_text(value)
    if not t:
        return ""
    try:
        f = float(t)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    if t.endswith(".0") and t[:-2].isdigit():
        return t[:-2]
    return t


def parse_number(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "."}:
        return np.nan
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    text = text.replace("%", "")
    try:
        return float(text)
    except Exception:
        return np.nan


def parse_percent_value(value) -> float:
    """Normalize frontend-like percent values to decimals: 60 -> .60, .60 -> .60."""
    v = parse_number(value)
    if not np.isfinite(v):
        return np.nan
    return v / 100.0 if abs(v) > 1.5 else v


def clean_allocation(value) -> float:
    """Keep realistic allocation values; preserve modest negatives for cash/short cases."""
    v = parse_percent_value(value)
    if not np.isfinite(v):
        return np.nan
    if v < -0.25 or v > 1.50:
        return np.nan
    return float(v)


def parse_date(value) -> pd.Timestamp:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"\d{6}", text):
        text = f"{text[:4]}-{text[4:6]}-01"
    elif re.fullmatch(r"\d{8}", text):
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return pd.to_datetime(text, errors="coerce")


def month_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    return f"{ts.year:04d}-{ts.month:02d}"


def safe_log1p(x: float) -> float:
    return math.log1p(x) if np.isfinite(x) and x > -1 else np.nan


def safe_expm1(x: float) -> float:
    try:
        return math.expm1(x)
    except Exception:
        return np.nan


def compound_return(values: Iterable[float]) -> float:
    logs = [safe_log1p(v) for v in values if np.isfinite(v)]
    logs = [v for v in logs if np.isfinite(v)]
    return safe_expm1(float(np.sum(logs))) if logs else np.nan


def annualize_log_sum(log_sum: float, count: int) -> float:
    return safe_expm1(log_sum * 12.0 / count) if count and np.isfinite(log_sum) else np.nan


def max_drawdown_from_monthly(values: Sequence[float]) -> float:
    clean = [v for v in values if np.isfinite(v)]
    if not clean:
        return np.nan
    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in clean:
        wealth *= 1.0 + r
        peak = max(peak, wealth)
        if peak > 0:
            max_dd = min(max_dd, wealth / peak - 1.0)
    return float(max_dd)


def mode_text(values: Iterable[str]) -> str:
    vals = [clean_text(v) for v in values if clean_text(v)]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def percentile_rank(s: pd.Series, value: float) -> float:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if len(vals) == 0 or not np.isfinite(value):
        return 0.5
    return float(np.mean(vals <= value))


def normalize_key(value: str) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_ticker(value: str) -> str:
    return clean_text(value).upper().replace(" ", "")


def ticker_aliases(value: str) -> List[str]:
    key = normalize_ticker(value)
    if not key:
        return []
    return sorted({key, key.replace(".", "-"), key.replace("-", ".")})

# -----------------------------------------------------------------------------
# Market / fund-level loading
# -----------------------------------------------------------------------------

def load_sp500(root: Path) -> pd.DataFrame:
    p = find_file(root, SP500_FILES)
    if p is None:
        raise FileNotFoundError("Cannot find sp500_monthly_returns_1871_2026.csv")
    df = pd.read_csv(p, low_memory=False)
    date_col = next((c for c in ["caldt", "date", "month", "Date", "DATE"] if c in df.columns), None)
    ret_col = next((c for c in ["sp500_ret", "sp500_mret", "mret", "ret", "return", "Return"] if c in df.columns), None)
    if date_col is None or ret_col is None:
        raise ValueError(f"Could not identify S&P500 date/return columns in {p}")
    out = pd.DataFrame({"date": df[date_col].map(parse_date), "sp500_ret": df[ret_col].map(parse_number)})
    out = out.dropna(subset=["date", "sp500_ret"])
    out["month_key"] = out["date"].map(month_key)
    out = out.groupby("month_key", as_index=False)["sp500_ret"].mean()
    out["date"] = pd.to_datetime(out["month_key"] + "-01")
    out = out.sort_values("date")
    logs = out["sp500_ret"].map(safe_log1p)
    counts = logs.rolling(36, min_periods=25).count()
    sums = logs.rolling(36, min_periods=25).sum()
    out["sp500_trailing_3y"] = [annualize_log_sum(s, int(c)) if c >= 25 else np.nan for s, c in zip(sums, counts)]
    return out


def load_yield10y(root: Path) -> pd.DataFrame:
    p = find_file(root, YIELD_FILES)
    if p is None:
        return pd.DataFrame(columns=["month_key", "year", "yield10y"])
    # The frontend parses FRB_H15 as headerless, only rows whose first cell is YYYY-MM.
    raw = pd.read_csv(p, header=None, dtype=str)
    rows = []
    for _, r in raw.iterrows():
        dt_text = clean_text(r.iloc[0]) if len(r) else ""
        if not re.fullmatch(r"\d{4}-\d{2}", dt_text):
            continue
        val = parse_number(r.iloc[1] if len(r) > 1 else np.nan)
        if not np.isfinite(val):
            continue
        dt = parse_date(dt_text)
        rows.append({"month_key": month_key(dt), "year": int(dt.year), "yield10y": val})
    return pd.DataFrame(rows)


def compute_net_flow(df: pd.DataFrame) -> pd.Series:
    if all(c in df.columns for c in ["new_sls", "rein_sls", "oth_sls", "redemp"]):
        return (
            df["new_sls"].map(parse_number).fillna(0)
            + df["rein_sls"].map(parse_number).fillna(0)
            + df["oth_sls"].map(parse_number).fillna(0)
            - df["redemp"].map(parse_number).fillna(0)
        )
    if "net_flow" in df.columns:
        return df["net_flow"].map(parse_number)
    return pd.Series(np.nan, index=df.index)


def load_fund_rows(root: Path, sp500: pd.DataFrame) -> pd.DataFrame:
    frames = []
    missing = []
    for candidates in FUND_FILES:
        p = find_file(root, candidates)
        if p is None:
            missing.append(candidates[-1])
            continue
        frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"Missing fund-level files: {missing}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["caldt"], errors="coerce")
    df["month_key"] = df["date"].dt.strftime("%Y-%m")
    df["mret"] = df["mret"].map(parse_number)
    df["sp500_ret"] = np.nan
    df = df.dropna(subset=["date", "mret"])

    df["crsp_fundno"] = df.get("crsp_fundno", pd.Series("", index=df.index)).map(clean_id)
    df["crsp_portno"] = df.get("crsp_portno", pd.Series("", index=df.index)).map(clean_id)
    df["fund"] = df.get("fund_name", pd.Series("", index=df.index)).map(clean_text)
    df["fund_ticker"] = df.get("ticker", pd.Series("", index=df.index)).map(clean_text)
    df["manager"] = df.get("mgr_name", pd.Series("Unknown Manager", index=df.index)).map(clean_text).replace("", "Unknown Manager")
    df["mgmt_name"] = df.get("mgmt_name", pd.Series("", index=df.index)).map(clean_text)
    df["exp_ratio"] = df.get("exp_ratio", pd.Series(np.nan, index=df.index)).map(parse_number)
    df["turn_ratio"] = df.get("turn_ratio", pd.Series(np.nan, index=df.index)).map(parse_number)
    df["mtna"] = df.get("mtna", pd.Series(np.nan, index=df.index)).map(parse_number)
    df["net_flow"] = compute_net_flow(df)
    mgr_dt = pd.to_datetime(df.get("mgr_dt", pd.Series(pd.NaT, index=df.index)), errors="coerce")
    df["tenure"] = ((df["date"] - mgr_dt).dt.days / 365.25).clip(lower=0)
    df = df.merge(sp500[["month_key", "sp500_ret", "sp500_trailing_3y"]], on="month_key", how="left", suffixes=("", "_sp"))
    if "sp500_ret_sp" in df.columns:
        df["sp500_ret"] = df["sp500_ret_sp"]
        df = df.drop(columns=["sp500_ret_sp"])
    df["sp500_ret"] = df["sp500_ret"].fillna(0.10 / 12.0)
    df["excess_ret"] = df["mret"] - df["sp500_ret"]
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    return add_fund_rolling(df)


def add_fund_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["crsp_fundno", "date"]).copy()
    parts = []
    for _, g in df.groupby("crsp_fundno", sort=False):
        g = g.sort_values("date").copy()
        logs = g["mret"].map(safe_log1p)
        counts = logs.rolling(36, min_periods=25).count()
        sums = logs.rolling(36, min_periods=25).sum()
        g["fund_trailing_3y"] = [annualize_log_sum(s, int(c)) if c >= 25 else np.nan for s, c in zip(sums, counts)]
        g["fund_3y_period_return"] = [safe_expm1(s) if c >= 25 else np.nan for s, c in zip(sums, counts)]
        # Fast downside proxy for manager style; true future_drawdown is computed later.
        g["fund_3y_max_drawdown"] = g["mret"].rolling(36, min_periods=25).min()
        g["avg_net_flow_3y"] = g["net_flow"].rolling(36, min_periods=25).mean()
        g["avg_exp_ratio_3y"] = g["exp_ratio"].rolling(36, min_periods=12).mean()
        g["avg_turnover_3y"] = g["turn_ratio"].rolling(36, min_periods=12).mean()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["fund_trailing_3y_excess"] = out["fund_trailing_3y"] - out["sp500_trailing_3y"]
    return out


def first_nonempty(series: pd.Series) -> str:
    for v in series:
        t = clean_text(v)
        if t:
            return t
    return ""


def build_port_month_table(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Fast share-class-to-portfolio monthly aggregation for outcome joins."""
    src = fund_df[fund_df["crsp_portno"].astype(str).str.len() > 0].copy()
    if src.empty:
        return pd.DataFrame()
    agg = src.groupby(["crsp_portno", "month_key"], as_index=False, sort=False).agg(
        manager=("manager", first_nonempty),
        fund=("fund", first_nonempty),
        fund_ticker=("fund_ticker", first_nonempty),
        mgmt_name=("mgmt_name", first_nonempty),
        crsp_fundno=("crsp_fundno", first_nonempty),
        mret=("mret", "mean"),
        sp500_ret=("sp500_ret", "mean"),
        sp500_trailing_3y=("sp500_trailing_3y", "mean"),
        fund_trailing_3y=("fund_trailing_3y", "mean"),
        fund_trailing_3y_excess=("fund_trailing_3y_excess", "mean"),
        avg_exp_ratio_3y=("avg_exp_ratio_3y", "mean"),
        avg_net_flow_3y=("avg_net_flow_3y", "mean"),
    )
    agg["date"] = pd.to_datetime(agg["month_key"] + "-01")
    return agg.sort_values(["crsp_portno", "date"])

# -----------------------------------------------------------------------------
# Manager style
# -----------------------------------------------------------------------------

def build_manager_styles(fund_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    src = fund_df.dropna(subset=["fund_trailing_3y"])
    for manager, g in src.groupby("manager", sort=False):
        returns = g["fund_trailing_3y"].dropna()
        if returns.empty:
            continue
        ann_return = returns.mean()
        ann_vol = returns.std(ddof=1)
        avg_excess = g["fund_trailing_3y_excess"].mean()
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else np.nan
        rows.append({
            "manager": manager,
            "manager_obs_count": int(len(g)),
            "annual_return": ann_return,
            "avg_excess": avg_excess,
            "sharpe": sharpe,
            "beat_rate": float(np.mean(g["fund_trailing_3y_excess"].dropna() > 0)) if g["fund_trailing_3y_excess"].notna().any() else np.nan,
            "max_drawdown": g["fund_3y_max_drawdown"].min(),
            "annual_volatility": ann_vol,
            "avg_fee": g["avg_exp_ratio_3y"].mean(),
            "avg_flow": g["avg_net_flow_3y"].mean(),
            "avg_tenure": g["tenure"].mean(),
            "avg_mtna": g["mtna"].mean(),
        })
    style = pd.DataFrame(rows)
    if style.empty:
        return pd.DataFrame(columns=["manager", "manager_style_group"])

    higher = ["annual_return", "avg_excess", "sharpe", "beat_rate", "max_drawdown", "avg_flow", "avg_tenure", "avg_mtna"]
    lower = ["annual_volatility", "avg_fee"]
    for c in higher:
        style[f"score_{c}"] = style[c].map(lambda v: percentile_rank(style[c], v))
    for c in lower:
        style[f"score_{c}"] = style[c].map(lambda v: 1.0 - percentile_rank(style[c], v))

    style["manager_defensive_score"] = style[["score_max_drawdown", "score_annual_volatility", "score_avg_fee"]].mean(axis=1)
    style["manager_flow_score"] = style[["score_avg_flow", "score_avg_mtna"]].mean(axis=1)
    style["manager_growth_tilt_score"] = style[["score_annual_return", "score_avg_excess", "score_sharpe"]].mean(axis=1)
    max_obs = max(float(style["manager_obs_count"].max()), 1.0)
    style["manager_reliability_score"] = np.minimum(1.0, np.log1p(style["manager_obs_count"]) / np.log1p(max_obs))

    def group_label(r):
        if r["manager_defensive_score"] >= 0.66 and r["manager_growth_tilt_score"] < 0.66:
            return "Defensive / risk-control style"
        if r["manager_growth_tilt_score"] >= 0.66 and r["manager_flow_score"] >= 0.55:
            return "High-return / high-flow style"
        if r["manager_growth_tilt_score"] >= 0.66:
            return "Equity-tilted / growth style"
        if r["manager_flow_score"] >= 0.66:
            return "Flow-supported core style"
        return "Balanced core style"

    style["manager_style_group"] = style.apply(group_label, axis=1)
    return style[[
        "manager", "manager_style_group", "manager_obs_count", "manager_reliability_score",
        "manager_defensive_score", "manager_flow_score", "manager_growth_tilt_score",
    ]]

# -----------------------------------------------------------------------------
# Part5 beta / non-individual lookup / holdings reports
# -----------------------------------------------------------------------------

def load_stock_beta(root: Path) -> Dict[Tuple[str, int], dict]:
    p = find_file(root, BETA_FILES)
    beta_map: Dict[Tuple[str, int], dict] = {}
    if p is None:
        return beta_map
    df = pd.read_csv(p, low_memory=False)
    for _, r in df.iterrows():
        year = parse_number(r.get("year"))
        if not np.isfinite(year):
            continue
        row = {
            "sector": clean_text(r.get("sector")) or "Unknown",
            "industry": clean_text(r.get("industry")) or "Unknown",
            "beta_y1": parse_number(r.get("beta_y1")),
        }
        for col in ["holding_ticker", "yahoo_ticker"]:
            for alias in ticker_aliases(r.get(col)):
                beta_map.setdefault((alias, int(year)), row)
    return beta_map


def lookup_beta(beta_map: Dict[Tuple[str, int], dict], ticker: str, year: int) -> Optional[dict]:
    for y in [year, year - 1, year + 1]:
        for a in ticker_aliases(ticker):
            if (a, y) in beta_map:
                return beta_map[(a, y)]
    return None


def load_non_individual_lookup(root: Path) -> Tuple[Dict[str, str], Dict[str, str], bool]:
    """Return ticker->category and security-name->category lookup."""
    p = find_file(root, NON_INDIVIDUAL_ENRICHED_FILES)
    ticker_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}
    if p is None:
        return ticker_map, name_map, False
    df = pd.read_csv(p, low_memory=False)
    cat_col = "teacher_category" if "teacher_category" in df.columns else "excluded_category" if "excluded_category" in df.columns else None
    if cat_col is None:
        return ticker_map, name_map, False
    for _, r in df.iterrows():
        cat = clean_text(r.get(cat_col))
        if not cat:
            continue
        ticker = normalize_ticker(r.get("holding_ticker"))
        yahoo = normalize_ticker(r.get("yahoo_ticker"))
        name = normalize_key(r.get("holding_security_name"))
        for t in [ticker, yahoo]:
            if t:
                ticker_map.setdefault(t, cat)
        if name:
            name_map.setdefault(name, cat)
    return ticker_map, name_map, True


def lookup_non_individual_category(ticker_map: Dict[str, str], name_map: Dict[str, str], ticker: str, name: str) -> str:
    for t in ticker_aliases(ticker):
        if t in ticker_map:
            return ticker_map[t]
    n = normalize_key(name)
    if n in name_map:
        return name_map[n]
    # Loose fallback for long bond/cash names.
    # Keep this conservative to avoid accidentally classifying operating companies.
    text = f"{normalize_key(ticker)} {n}"
    if any(k in text for k in ["TREASURY", "T-BILL", "TBILL", "FANNIE", "FREDDIE", "MONEY MARKET", "CASH", "GOVERNMENT", "BOND", "NOTE", "TBA", "MBS"]):
        return "Bond / Credit / Money-related"
    if any(k in text for k in ["S&P 500 ETF", "ETF", "INDEX FUND", "ISHARES", "VANGUARD", "SPDR"]):
        return "Equity Fund / Stock-fund-like"
    return ""


def load_non_individual_year_proxy(root: Path) -> pd.DataFrame:
    p = find_file(root, NON_INDIVIDUAL_PANEL_FILES)
    if p is None:
        return pd.DataFrame(columns=["year", "bond_money_exposure_proxy", "indirect_equity_exposure_proxy"])
    df = pd.read_csv(p)
    if "teacher_category" not in df.columns or "year" not in df.columns:
        return pd.DataFrame(columns=["year", "bond_money_exposure_proxy", "indirect_equity_exposure_proxy"])
    df["year"] = df["year"].map(parse_number).astype("Int64")
    df["proxy"] = df.get("holding_record_count_proxy", pd.Series(np.nan, index=df.index)).map(parse_number)
    piv = df.pivot_table(index="year", columns="teacher_category", values="proxy", aggfunc="sum").fillna(0)
    total = piv.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({
        "year": piv.index.astype(int),
        "bond_money_exposure_proxy": piv.get("Bond / Credit / Money-related", pd.Series(0, index=piv.index)) / total,
        "indirect_equity_exposure_proxy": piv.get("Equity Fund / Stock-fund-like", pd.Series(0, index=piv.index)) / total,
    }).reset_index(drop=True)


def normalize_holding_chunk(df: pd.DataFrame, source_key: str, beta_map: Dict[Tuple[str, int], dict], ticker_map: Dict[str, str], name_map: Dict[str, str]) -> pd.DataFrame:
    if "report_dt" not in df.columns or "crsp_portno" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out["report_date"] = pd.to_datetime(df["report_dt"], errors="coerce")
    out = out[out["report_date"].notna()].copy()
    if out.empty:
        return out
    src = df.loc[out.index]
    out["source_key"] = source_key
    out["crsp_portno"] = src.get("crsp_portno", pd.Series("", index=out.index)).map(clean_id)
    out["fund_ticker"] = src.get("fund_ticker", pd.Series("", index=out.index)).map(clean_text)
    out["fund"] = src.get("fund_name", pd.Series("", index=out.index)).map(clean_text)
    out["stock_allocation_raw"] = src.get("fund_percent_common_stock", pd.Series(np.nan, index=out.index)).map(clean_allocation)
    out["bond_allocation_raw"] = src.get("fund_percent_bond", pd.Series(np.nan, index=out.index)).map(clean_allocation)
    out["cash_allocation_raw"] = src.get("fund_percent_cash", pd.Series(np.nan, index=out.index)).map(clean_allocation)
    out["holding_pct"] = src.get("holding_percent_tna", pd.Series(np.nan, index=out.index)).map(parse_percent_value)
    out["security_rank"] = src.get("security_rank", pd.Series(np.nan, index=out.index)).map(parse_number)
    out["holding_ticker"] = src.get("holding_ticker", pd.Series("", index=out.index)).map(clean_text)
    out["holding_security_name"] = src.get("holding_security_name", pd.Series("", index=out.index)).map(clean_text)
    out["year"] = out["report_date"].dt.year.astype(int)
    out["quarter"] = out["report_date"].dt.quarter.astype(int)
    out["month_key"] = out["report_date"].dt.strftime("%Y-%m")
    out["report_key"] = out["crsp_portno"].astype(str) + "|" + out["report_date"].dt.strftime("%Y-%m-%d")

    sectors = []
    betas = []
    noncat = []
    for ticker, name, year in zip(out["holding_ticker"], out["holding_security_name"], out["year"]):
        b = lookup_beta(beta_map, ticker, int(year)) if ticker else None
        sectors.append((b or {}).get("sector", "Unknown"))
        betas.append((b or {}).get("beta_y1", np.nan))
        noncat.append(lookup_non_individual_category(ticker_map, name_map, ticker, name))
    out["sector"] = sectors
    out["stock_beta"] = betas
    out["non_individual_category"] = noncat
    out["weighted_beta"] = out["holding_pct"] * out["stock_beta"]
    out.loc[~np.isfinite(out["weighted_beta"]), "weighted_beta"] = np.nan
    out["tech_weight"] = np.where(out["sector"].str.contains("Technology", case=False, na=False), out["holding_pct"], 0.0)
    out["top10_weight"] = np.where(out["security_rank"].between(1, 10, inclusive="both"), out["holding_pct"], 0.0)
    out["company_equity_weight"] = np.where(out["stock_beta"].notna(), out["holding_pct"], 0.0)
    out["bond_money_weight"] = np.where(out["non_individual_category"].eq("Bond / Credit / Money-related"), out["holding_pct"], 0.0)
    out["indirect_equity_weight"] = np.where(out["non_individual_category"].eq("Equity Fund / Stock-fund-like"), out["holding_pct"], 0.0)
    out["non_individual_match"] = out["non_individual_category"].astype(str).str.len() > 0
    return out


def load_holdings_reports(root: Path, beta_map: Dict[Tuple[str, int], dict], ticker_map: Dict[str, str], name_map: Dict[str, str], chunksize: int) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    parts = []
    missing = []
    found_paths = {}
    for source_key, candidates in HOLDINGS_FILES:
        p = find_file(root, candidates)
        if p is None:
            missing.append(candidates[0])
            continue
        found_paths[source_key] = str(p)
        for chunk in pd.read_csv(p, chunksize=chunksize, low_memory=False):
            h = normalize_holding_chunk(chunk, source_key, beta_map, ticker_map, name_map)
            if h.empty:
                continue
            gcols = ["report_key", "crsp_portno", "report_date", "year", "quarter", "month_key", "source_key"]
            rep = h.groupby(gcols, as_index=False).agg(
                fund=("fund", mode_text),
                fund_ticker=("fund_ticker", mode_text),
                stock_allocation_raw=("stock_allocation_raw", "first"),
                bond_allocation_raw=("bond_allocation_raw", "first"),
                cash_allocation_raw=("cash_allocation_raw", "first"),
                portfolio_beta=("weighted_beta", "sum"),
                technology_exposure=("tech_weight", "sum"),
                top_holding_concentration=("top10_weight", "sum"),
                company_equity_exposure_proxy=("company_equity_weight", "sum"),
                bond_money_exposure_report=("bond_money_weight", "sum"),
                indirect_equity_exposure_report=("indirect_equity_weight", "sum"),
                holding_row_count=("report_key", "size"),
                beta_matched_holding_count=("stock_beta", lambda s: int(s.notna().sum())),
                non_individual_matched_holding_count=("non_individual_match", lambda s: int(s.sum())),
            )
            parts.append(rep)
    if not parts:
        return pd.DataFrame(), missing, found_paths
    reports = pd.concat(parts, ignore_index=True).drop_duplicates("report_key")
    return reports, missing, found_paths

# -----------------------------------------------------------------------------
# Joins / features / labels
# -----------------------------------------------------------------------------

def nearest_port_month_join(reports: pd.DataFrame, port_month: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    reports = reports.sort_values(["crsp_portno", "report_date"])
    for port, g in reports.groupby("crsp_portno", sort=False):
        pm = port_month[port_month["crsp_portno"] == port].sort_values("date")
        if pm.empty:
            gg = g.copy()
            for c in ["manager", "crsp_fundno", "mgmt_name", "fund_trailing_3y", "fund_trailing_3y_excess", "sp500_trailing_3y"]:
                gg[c] = np.nan
            pieces.append(gg)
            continue
        merged = pd.merge_asof(
            g.sort_values("report_date"), pm,
            left_on="report_date", right_on="date", by="crsp_portno",
            direction="backward", tolerance=pd.Timedelta(days=370), suffixes=("", "_fund")
        )
        if "fund_fund" in merged.columns:
            merged["fund"] = merged["fund"].where(merged["fund"].astype(str).str.len() > 0, merged["fund_fund"])
        if "fund_ticker_fund" in merged.columns:
            merged["fund_ticker"] = merged["fund_ticker"].where(merged["fund_ticker"].astype(str).str.len() > 0, merged["fund_ticker_fund"])
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True) if pieces else reports


def add_market_regime(reports: pd.DataFrame, yield_df: pd.DataFrame) -> pd.DataFrame:
    reports = reports.copy()
    if not yield_df.empty:
        ym = yield_df[["month_key", "yield10y"]].drop_duplicates("month_key")
        yy = yield_df.groupby("year", as_index=False)["yield10y"].mean().rename(columns={"yield10y": "yield10y_year"})
        reports = reports.merge(ym, on="month_key", how="left")
        reports = reports.merge(yy, on="year", how="left")
        reports["yield10y"] = reports["yield10y"].fillna(reports["yield10y_year"])
        reports = reports.drop(columns=["yield10y_year"])
    else:
        reports["yield10y"] = np.nan

    def label(r):
        y = r.get("yield10y", np.nan)
        sp = r.get("sp500_trailing_3y", np.nan)
        rate = "unknown_rate"
        if np.isfinite(y):
            rate = "high_rate" if y >= 4.0 else "low_rate" if y <= 2.0 else "mid_rate"
        market = "unknown_market"
        if np.isfinite(sp):
            market = "strong_equity_market" if sp >= 0.08 else "weak_equity_market" if sp <= 0.00 else "moderate_equity_market"
        return f"{rate}__{market}"
    reports["market_regime"] = reports.apply(label, axis=1)
    return reports


def complete_allocations(reports: pd.DataFrame) -> pd.DataFrame:
    reports = reports.copy()
    stock = reports["stock_allocation_raw"].copy()
    # If raw stock allocation is missing, use a conservative company-stock exposure proxy.
    stock = stock.where(stock.notna(), reports["company_equity_exposure_proxy"].where(reports["company_equity_exposure_proxy"] > 0))
    cash = reports["cash_allocation_raw"].copy()
    bond = reports["bond_allocation_raw"].copy()
    method = []
    final_bond = []
    final_cash = []
    final_stock = []
    for sr, br, cr, st in zip(reports["stock_allocation_raw"], reports["bond_allocation_raw"], reports["cash_allocation_raw"], stock):
        s = st if np.isfinite(st) else np.nan
        c = cr if np.isfinite(cr) else np.nan
        b = br if np.isfinite(br) and br > 0 else np.nan
        m = "raw"
        if not np.isfinite(s):
            m = "missing_stock"
        if not np.isfinite(c):
            c = 0.0
            m = "cash_missing_as_zero" if m == "raw" else m + ";cash_missing_as_zero"
        if not np.isfinite(b):
            if np.isfinite(s):
                b = max(0.0, 1.0 - max(0.0, s) - max(0.0, c))
                m = "bond_completed_as_one_minus_stock_cash" if m == "raw" else m + ";bond_completed"
            else:
                b = np.nan
        final_stock.append(s)
        final_bond.append(b)
        final_cash.append(c)
        method.append(m)
    reports["stock_allocation"] = final_stock
    reports["bond_allocation"] = final_bond
    reports["cash_allocation"] = final_cash
    reports["allocation_completion_method"] = method
    return reports


def add_nonindividual_exposure(reports: pd.DataFrame, root: Path) -> pd.DataFrame:
    reports = reports.copy()
    proxy = load_non_individual_year_proxy(root)
    if not proxy.empty:
        reports = reports.merge(proxy, on="year", how="left")
    else:
        reports["bond_money_exposure_proxy"] = np.nan
        reports["indirect_equity_exposure_proxy"] = np.nan
    # Prefer report-specific matched exposure. If no matched exposure and proxy exists,
    # use the year proxy as a weaker market-wide fallback.
    bm_report = reports["bond_money_exposure_report"].replace(0, np.nan)
    ie_report = reports["indirect_equity_exposure_report"].replace(0, np.nan)
    reports["bond_money_exposure"] = bm_report.fillna(reports["bond_money_exposure_proxy"])
    reports["indirect_equity_exposure"] = ie_report.fillna(reports["indirect_equity_exposure_proxy"])
    reports["non_individual_source"] = np.where(bm_report.notna() | ie_report.notna(), "report_matched_holdings", "year_level_proxy")
    return reports


def compute_future_outcomes(reports: pd.DataFrame, port_month: pd.DataFrame) -> pd.DataFrame:
    """Fast future 12-month outcome calculation by portfolio using searchsorted."""
    reports = reports.copy().reset_index(drop=True)
    n = len(reports)
    fut_r = np.full(n, np.nan)
    fut_s = np.full(n, np.nan)
    fut_ex = np.full(n, np.nan)
    fut_dd = np.full(n, np.nan)

    pm_groups = {str(p): g.sort_values("date") for p, g in port_month.groupby("crsp_portno", sort=False)}
    report_groups = reports.groupby("crsp_portno", sort=False).indices

    for port, idxs in report_groups.items():
        g = pm_groups.get(str(port))
        if g is None or g.empty:
            continue
        dates = g["date"].values.astype("datetime64[ns]")
        fund_rets = g["mret"].to_numpy(dtype=float)
        sp_rets = g["sp500_ret"].to_numpy(dtype=float)
        report_dates = pd.to_datetime(reports.loc[idxs, "report_date"], errors="coerce").values.astype("datetime64[ns]")
        positions = np.searchsorted(dates, report_dates, side="right")
        for arr_pos, idx in enumerate(idxs):
            pos = int(positions[arr_pos])
            end = min(pos + 12, len(g))
            if end - pos < 9:
                continue
            fr_slice = fund_rets[pos:end]
            sp_slice = sp_rets[pos:end]
            fr = compound_return(fr_slice)
            fs = compound_return(sp_slice)
            fut_r[idx] = fr
            fut_s[idx] = fs
            fut_ex[idx] = fr - fs if np.isfinite(fr) and np.isfinite(fs) else np.nan
            fut_dd[idx] = max_drawdown_from_monthly(fr_slice)

    reports["future_12m_return"] = fut_r
    reports["future_12m_sp500_return"] = fut_s
    reports["future_12m_excess_return"] = fut_ex
    reports["future_drawdown"] = fut_dd
    return reports

def add_deltas_and_deviation(reports: pd.DataFrame) -> pd.DataFrame:
    reports = reports.sort_values(["manager", "crsp_portno", "report_date"]).copy()
    gcols = ["manager", "crsp_portno"]
    pairs = [
        ("stock_allocation", "delta_stock"),
        ("portfolio_beta", "delta_beta"),
        ("technology_exposure", "delta_technology"),
        ("bond_money_exposure", "delta_bond_money"),
        ("indirect_equity_exposure", "delta_indirect_equity"),
    ]
    for col, dcol in pairs:
        reports[dcol] = reports.groupby(gcols)[col].diff()
    features = ["stock_allocation", "portfolio_beta", "technology_exposure", "bond_money_exposure", "indirect_equity_exposure"]
    means = reports.groupby("manager")[features].transform("mean")
    zparts = []
    for f in features:
        std = reports[f].std(skipna=True)
        if np.isfinite(std) and std > 1e-12:
            zparts.append(((reports[f] - means[f]).abs() / std).replace([np.inf, -np.inf], np.nan))
    reports["style_deviation_score"] = pd.concat(zparts, axis=1).mean(axis=1) if zparts else np.nan
    return reports


def add_labels(reports: pd.DataFrame) -> pd.DataFrame:
    """Vectorized action and outcome labels."""
    reports = reports.copy()
    ds = reports["delta_stock"]
    db = reports["delta_beta"]
    dt = reports["delta_technology"]
    dbm = reports["delta_bond_money"]
    die = reports["delta_indirect_equity"]
    strength = pd.concat([ds.abs(), db.abs(), dt.abs(), dbm.abs(), die.abs()], axis=1).max(axis=1)
    reports["action_strength"] = strength

    stock_thr, beta_thr, exposure_thr = 0.03, 0.05, 0.01
    conditions = [
        (ds >= stock_thr) & (dt >= exposure_thr),
        ds >= stock_thr,
        (ds <= -stock_thr) & (dbm >= exposure_thr),
        db <= -beta_thr,
        dt >= exposure_thr,
        dbm >= exposure_thr,
        die >= exposure_thr,
        strength >= 0.01,
    ]
    action_choices = [
        "increase_equity_plus_technology",
        "increase_stock_allocation",
        "reduce_stock_increase_bond_money",
        "reduce_portfolio_beta",
        "increase_technology_exposure",
        "increase_bond_money_exposure",
        "increase_indirect_equity_exposure",
        "minor_allocation_rotation",
    ]
    execution_choices = [
        "direct_equity_growth_tilt",
        "direct_equity_risk_on",
        "risk_off_bond_money_rotation",
        "beta_reduction_defensive",
        "sector_rotation_technology",
        "bond_money_defensive_execution",
        "indirect_equity_execution",
        "minor_cross_asset_rotation",
    ]
    reports["action_type"] = np.select(conditions, action_choices, default="stable_or_no_clear_action")
    reports["cross_asset_execution_type"] = np.select(conditions, execution_choices, default="stable_or_minor_rebalance")

    ex = reports["future_12m_excess_return"]
    dd = reports["future_drawdown"]
    outcome_conditions = [
        ex.isna(),
        (ex > 0) & (dd.notna()) & (dd <= -0.20),
        ex > 0,
        ex < 0,
    ]
    outcome_choices = [
        "missing_future_outcome",
        "positive_excess_high_drawdown",
        "positive_excess",
        "negative_excess",
    ]
    reports["outcome_label"] = np.select(outcome_conditions, outcome_choices, default="neutral")
    return reports

def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized semicolon-separated quality flags."""
    df = df.copy()
    flags = pd.Series("", index=df.index, dtype="object")

    def add_flag(mask, name):
        nonlocal flags
        mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
        flags.loc[mask] = np.where(flags.loc[mask].astype(str).str.len() > 0, flags.loc[mask] + ";" + name, name)

    add_flag(df.get("manager", pd.Series("", index=df.index)).map(clean_text).isin(["", "Unknown Manager"]), "missing_manager")
    add_flag(~pd.to_numeric(df.get("portfolio_beta", pd.Series(np.nan, index=df.index)), errors="coerce").apply(np.isfinite), "missing_portfolio_beta")
    add_flag(~pd.to_numeric(df.get("stock_allocation", pd.Series(np.nan, index=df.index)), errors="coerce").apply(np.isfinite), "missing_stock_allocation")
    add_flag(df.get("non_individual_source", pd.Series("", index=df.index)).astype(str).eq("year_level_proxy"), "part5b_year_proxy")
    add_flag(~pd.to_numeric(df.get("future_12m_return", pd.Series(np.nan, index=df.index)), errors="coerce").apply(np.isfinite), "missing_future_12m")
    flags = flags.replace("", "ok")
    df["data_quality_flags"] = flags
    return df

# -----------------------------------------------------------------------------
# Optional app-compatible preprocessing copy
# -----------------------------------------------------------------------------

def write_preprocessed_alias_copies(root: Path, output_dir: Path, found_paths: Dict[str, str]) -> Dict[str, str]:
    """Create a small folder with clean/app-compatible aliases, without modifying originals."""
    out = output_dir / "preprocessed_app_compatible_inputs"
    out.mkdir(parents=True, exist_ok=True)
    copied = {}
    alias_targets = {
        "before2010": "stock_before_2010.csv",
        "y2010_2014": "stock_between_2010_2014.csv",
        "y2015_2019": "stock_between_2015_2019.csv",
        "y2020_2026": "stock_between_2020_2026.csv",
    }
    for key, src in found_paths.items():
        dest = out / alias_targets.get(key, Path(src).name)
        try:
            shutil.copyfile(src, dest)
            copied[key] = str(dest)
        except Exception as e:
            copied[key] = f"COPY_FAILED: {e}"
    (out / "README.txt").write_text(
        "These are copied aliases for app/back-end compatibility. Original data files are not modified.\n",
        encoding="utf-8",
    )
    return copied

# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_ground_truth(root: Path, output_dir: Path, chunksize: int = 200_000, write_preprocessed: bool = False) -> Tuple[pd.DataFrame, dict]:
    audit = {"data_root": str(root), "missing_files": [], "warnings": [], "found_holdings_paths": {}}
    print("[1/8] Load S&P500 and 10Y yield")
    sp500 = load_sp500(root)
    yield_df = load_yield10y(root)
    print("[2/8] Load fund-level rows and build 3Y trailing features")
    fund_df = load_fund_rows(root, sp500)
    port_month = build_port_month_table(fund_df)
    print("[3/8] Build manager style groups")
    styles = build_manager_styles(fund_df)
    print("[4/8] Load beta map and non-individual lookup")
    beta_map = load_stock_beta(root)
    ticker_map, name_map, has_nonind = load_non_individual_lookup(root)
    if not has_nonind:
        audit["warnings"].append("No Part5B enriched lookup found; report-level non-individual matching will be limited to keyword fallback.")
    print("[5/8] Load holdings raw files and aggregate report-level actions")
    reports, missing, found_paths = load_holdings_reports(root, beta_map, ticker_map, name_map, chunksize)
    audit["missing_files"].extend(missing)
    audit["found_holdings_paths"] = found_paths
    if reports.empty:
        raise FileNotFoundError("No valid holdings reports were parsed. Check holdings_raw CSV locations/names.")
    if write_preprocessed:
        audit["preprocessed_alias_copies"] = write_preprocessed_alias_copies(root, output_dir, found_paths)

    print("[6/8] Join manager/fund context, market regime, non-individual exposure, and future outcomes")
    reports = nearest_port_month_join(reports, port_month)
    reports = add_market_regime(reports, yield_df)
    reports = complete_allocations(reports)
    reports = add_nonindividual_exposure(reports, root)
    reports = compute_future_outcomes(reports, port_month)
    reports["manager"] = reports.get("manager", pd.Series("", index=reports.index)).map(clean_text).replace("", "Unknown Manager")
    reports = reports.merge(styles, on="manager", how="left")
    reports["manager_style_group"] = reports["manager_style_group"].fillna("Unknown style")

    print("[7/8] Add deltas, action labels, style deviation, quality flags")
    reports = add_deltas_and_deviation(reports)
    reports = add_labels(reports)
    reports = add_quality_flags(reports)

    # Final formatting.
    reports["report_date"] = pd.to_datetime(reports["report_date"]).dt.strftime("%Y-%m-%d")
    reports["manager"] = reports["manager"].fillna("Unknown Manager")
    reports["fund"] = reports["fund"].fillna("")
    reports["fund_ticker"] = reports["fund_ticker"].fillna("")
    for c in REQUIRED_COLUMNS + EXTRA_COLUMNS:
        if c not in reports.columns:
            reports[c] = np.nan
    final_cols = REQUIRED_COLUMNS + [c for c in EXTRA_COLUMNS if c not in REQUIRED_COLUMNS]
    out = reports[final_cols].sort_values(["manager", "crsp_portno", "report_date"]).reset_index(drop=True)

    print("[8/8] Save CSV and audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manager_action_ground_truth.csv"
    audit_path = output_dir / "manager_action_ground_truth_audit.json"
    write_dataframe_csv(csv_path, out)
    audit.update({
        "rows": int(len(out)),
        "columns": final_cols,
        "output_csv": str(csv_path),
        "outcome_label_counts": out["outcome_label"].value_counts(dropna=False).to_dict(),
        "action_type_counts": out["action_type"].value_counts(dropna=False).head(30).to_dict(),
        "execution_type_counts": out["cross_asset_execution_type"].value_counts(dropna=False).head(30).to_dict(),
        "manager_style_group_counts": out["manager_style_group"].value_counts(dropna=False).to_dict(),
        "data_quality_flag_counts": out["data_quality_flags"].value_counts(dropna=False).head(30).to_dict(),
    })
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced fund manager-action ground-truth CSV.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to your data folder")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-root>/outputs/manager_action_groundtruth")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Holdings CSV chunk size")
    parser.add_argument("--write-preprocessed", action="store_true", help="Also copy app-compatible clean holdings aliases into output folder")
    args = parser.parse_args()
    output_dir = args.output_dir or (args.data_root / "outputs" / "manager_action_groundtruth")
    print(f"[INFO] data_root = {args.data_root}")
    print(f"[INFO] output_dir = {output_dir}")
    out, audit = build_ground_truth(args.data_root, output_dir, args.chunksize, args.write_preprocessed)
    print(f"[DONE] rows = {len(out):,}")
    print(f"[DONE] csv = {audit['output_csv']}")
    print("[INFO] outcome labels:")
    for k, v in audit["outcome_label_counts"].items():
        print(f"  {k}: {v}")
    if audit.get("missing_files"):
        print("[WARN] missing files:")
        for m in audit["missing_files"]:
            print(f"  - {m}")
    if audit.get("warnings"):
        print("[WARN] warnings:")
        for w in audit["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
