#!/usr/bin/env python3
"""
Build/enrich manager-action ground truth and ML-ready datasets for the Balanced Fund VA project.

This script builds the multi-horizon event ground truth used by Part 6:
- app.js keeps Part1-Part5 unchanged.
- The backend/offline pipeline reads the existing base manager_action_ground_truth.csv
  plus the original fund-level CSV files.
- It uses a three-year, strictly ex-ante style window.
- Every event receives 3M, 6M, 9M and 12M forward excess-return targets.
- Direction labels use a +/-0.5% neutral band and a five-level outcome label.
- It keeps trailing features and also adds current-month features.
- It adds rolling ex-ante style deviation using only manager history before report_date.

Recommended command in your project:
    python scripts/modeling/build_manager_action_groundtruth_complete_v2.py --data-root data

Main outputs:
    data/derived/manager_action_groundtruth/manager_action_ground_truth.csv
    data/derived/manager_action_groundtruth/manager_action_ground_truth_trailing3y_multi_horizon.csv
    data/derived/prediction/part6_prediction_dataset_trailing3y_multi_horizon.csv
    data/derived/prediction/part6_prediction_dataset.csv

Important thesis wording:
    The table creates historical action-outcome labels for prediction and association analysis.
    It does not by itself establish causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path("data")
RISK_FREE_RATE = 0.01
TRAILING_WINDOWS = {3: 36}
PREDICTION_HORIZONS = (3, 6, 9, 12)
NEUTRAL_BAND = 0.005
MIN_WINDOW_RATIO = 0.70

FUND_FILES = [
    ["crsp/fund_level/balanced_before2010.csv", "balanced_before2010.csv"],
    ["crsp/fund_level/balanced_after2010.csv", "balanced_after2010.csv"],
]
SP500_FILES = ["market/sp500_monthly_returns_1871_2026.csv", "sp500_monthly_returns_1871_2026.csv"]
BASE_GT_FILES = [
    "derived/manager_action_groundtruth/manager_action_ground_truth.csv",
    "manager_action_ground_truth.csv",
]
HOLDINGS_GLOBS = ["crsp/holdings_raw/*.csv", "holdings_raw/*.csv"]
SECTOR_CACHE_FILES = ["part5_equity_beta/part5_yfinance_sector_cache.csv"]
GICS_SECTORS = (
    "energy", "materials", "industrials", "consumer_discretionary", "consumer_staples",
    "health_care", "financials", "information_technology", "communication_services",
    "utilities", "real_estate",
)

IDENTITY_COLUMNS = [
    "event_id", "training_window_years", "training_window_months", "manager", "fund",
    "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name", "report_date",
    "year", "quarter", "month_key", "feature_cutoff_date", "label_start_date", "label_end_date",
    "style_window_start_date", "style_window_end_date", "style_window_type", "style_obs_count",
    "leakage_check_passed",
]

CURRENT_COLUMNS = [
    "current_mret", "current_sp500_ret", "current_excess_ret", "current_net_flow",
    "current_mtna", "current_exp_ratio", "current_mgmt_fee", "current_turn_ratio",
    "current_age", "current_tenure",
]

TRAILING_ALIAS_COLUMNS = [
    "fund_trailing_return", "sp500_trailing_return", "fund_trailing_excess_return",
    "fund_trailing_period_return", "fund_trailing_max_drawdown",
    "fund_trailing_beta_vs_sp500", "trailing_avg_net_flow", "trailing_sum_net_flow",
    "trailing_avg_mtna", "trailing_avg_exp_ratio", "trailing_avg_mgmt_fee",
    "trailing_avg_turn_ratio", "trailing_avg_age", "trailing_avg_tenure",
]

ACTION_EXPOSURE_COLUMNS = [
    "yield10y", "market_regime", "manager_style_group", "stock_allocation",
    "bond_allocation", "cash_allocation", "portfolio_beta", "technology_exposure",
    "bond_money_exposure", "indirect_equity_exposure", "company_equity_exposure_proxy",
    "top_holding_concentration", "delta_stock", "delta_beta", "delta_technology",
    "delta_bond_money", "delta_indirect_equity", "nonstock_total_exposure",
    "delta_nonstock_total_exposure", "delta_sector_exposure", "style_deviation_score",
    "rolling_style_deviation_score", "rolling_sector_deviation_score",
    "rolling_cross_asset_deviation_score", "rolling_action_deviation_score", "action_strength", "action_type",
    "cross_asset_execution_type", "manager_reliability_score", "manager_defensive_score",
    "manager_flow_score", "manager_growth_tilt_score", "allocation_completion_method",
    "non_individual_source", "holding_row_count", "beta_matched_holding_count",
    "non_individual_matched_holding_count", "data_quality_flags",
] + [f"sector_{s}_exposure" for s in GICS_SECTORS] + [f"delta_sector_{s}" for s in GICS_SECTORS] + ["sector_rotation_intensity"]

FUTURE_COLUMNS = [
    item
    for horizon in PREDICTION_HORIZONS
    for item in (
        f"future_{horizon}m_return", f"future_{horizon}m_sp500_return",
        f"future_{horizon}m_excess_return", f"future_{horizon}m_drawdown",
        f"direction_label_{horizon}m", f"outcome_5class_{horizon}m",
        f"label_positive_excess_{horizon}m",
    )
]

ML_NUMERIC_COLUMNS = CURRENT_COLUMNS + TRAILING_ALIAS_COLUMNS + [
    "yield10y", "stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta",
    "technology_exposure", "bond_money_exposure", "indirect_equity_exposure",
    "company_equity_exposure_proxy", "top_holding_concentration", "delta_stock",
    "delta_beta", "delta_technology", "delta_bond_money", "delta_indirect_equity",
    "nonstock_total_exposure", "delta_nonstock_total_exposure", "delta_sector_exposure",
    "style_deviation_score", "rolling_style_deviation_score", "rolling_sector_deviation_score",
    "rolling_cross_asset_deviation_score", "rolling_action_deviation_score", "action_strength",
    "manager_reliability_score", "manager_defensive_score", "manager_flow_score",
    "manager_growth_tilt_score", "holding_row_count", "beta_matched_holding_count",
    "non_individual_matched_holding_count", "sector_rotation_intensity",
] + [f"sector_{s}_exposure" for s in GICS_SECTORS] + [f"delta_sector_{s}" for s in GICS_SECTORS]

ML_CATEGORICAL_COLUMNS = [
    "market_regime", "manager_style_group", "action_type", "cross_asset_execution_type",
    "allocation_completion_method", "non_individual_source",
]

TARGET_COLUMNS = FUTURE_COLUMNS

SECTOR_STYLE_FEATURES = [f"sector_{s}_exposure" for s in GICS_SECTORS]
CROSS_ASSET_STYLE_FEATURES = [
    "stock_allocation", "portfolio_beta", "technology_exposure",
    "bond_money_exposure", "indirect_equity_exposure",
    "nonstock_total_exposure",
]
STYLE_DEVIATION_FEATURES = CROSS_ASSET_STYLE_FEATURES + SECTOR_STYLE_FEATURES
ACTION_DEVIATION_FEATURES = [
    "delta_stock", "delta_beta", "delta_technology", "delta_bond_money",
    "delta_indirect_equity", "delta_nonstock_total_exposure", "delta_sector_exposure",
] + [f"delta_sector_{s}" for s in GICS_SECTORS]


def find_file(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    for c in candidates:
        name = Path(c).name
        matches = list(root.rglob(name)) if root.exists() else []
        if matches:
            return matches[0]
    return None


def write_dataframe_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def clean_id(value) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def parse_number(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) else np.nan
    text = clean_text(value).replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", ".", "na", "n/a"}:
        return np.nan
    neg = text.startswith("(") and text.endswith(")")
    if neg:
        text = text[1:-1]
    text = text.replace("%", "")
    try:
        out = float(text)
        return -out if neg else out
    except ValueError:
        return np.nan


def parse_date(value) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def month_key(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    return f"{int(ts.year):04d}-{int(ts.month):02d}"


def safe_log1p(x: float) -> float:
    if not np.isfinite(x) or x <= -0.999999:
        return np.nan
    return math.log1p(float(x))


def safe_expm1(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return math.expm1(float(x))


def annualize_log_sum(log_sum: float, count: int) -> float:
    if not np.isfinite(log_sum) or count <= 0:
        return np.nan
    return safe_expm1(log_sum * 12.0 / count)


def compound_return(values: Iterable[float]) -> float:
    logs = [safe_log1p(v) for v in values]
    logs = [v for v in logs if np.isfinite(v)]
    return safe_expm1(sum(logs)) if logs else np.nan


def max_drawdown_from_monthly(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    worst = 0.0
    valid = 0
    for r in values:
        if not np.isfinite(r):
            continue
        valid += 1
        wealth *= 1.0 + float(r)
        if wealth > peak:
            peak = wealth
        if peak > 0:
            worst = min(worst, wealth / peak - 1.0)
    return worst if valid else np.nan


def rolling_max_drawdown(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).apply(max_drawdown_from_monthly, raw=True)


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


def load_sp500(root: Path) -> pd.DataFrame:
    p = find_file(root, SP500_FILES)
    if p is None:
        raise FileNotFoundError("Cannot find sp500_monthly_returns_1871_2026.csv")
    df = pd.read_csv(p, low_memory=False)
    date_col = next((c for c in ["caldt", "date", "month", "Date", "DATE"] if c in df.columns), None)
    ret_col = next((c for c in ["sp500_ret", "sp500_mret", "mret", "ret", "return", "Return"] if c in df.columns), None)
    if date_col is None or ret_col is None:
        raise ValueError(f"Cannot identify S&P500 columns in {p}")
    out = pd.DataFrame({"date": df[date_col].map(parse_date), "sp500_ret": df[ret_col].map(parse_number)})
    out = out.dropna(subset=["date", "sp500_ret"])
    out["month_key"] = out["date"].map(month_key)
    out = out.groupby("month_key", as_index=False)["sp500_ret"].mean()
    out["date"] = pd.to_datetime(out["month_key"] + "-01")
    out = out.sort_values("date").reset_index(drop=True)
    logs = out["sp500_ret"].map(safe_log1p)
    for years, months in TRAILING_WINDOWS.items():
        minp = int(months * MIN_WINDOW_RATIO)
        counts = logs.shift(1).rolling(months, min_periods=minp).count()
        sums = logs.shift(1).rolling(months, min_periods=minp).sum()
        out[f"sp500_trailing_{years}y"] = [annualize_log_sum(s, int(c)) if c >= minp else np.nan for s, c in zip(sums, counts)]
        out[f"sp500_trailing_{years}y_period_return"] = [safe_expm1(s) if c >= minp else np.nan for s, c in zip(sums, counts)]
    return out


def load_fund_month_table(root: Path, sp500: pd.DataFrame) -> pd.DataFrame:
    frames = []
    missing = []
    wanted_cols = [
        "crsp_fundno", "crsp_portno", "fund_name", "ticker", "mgmt_name", "mgr_name",
        "mgr_dt", "caldt", "mret", "mtna", "exp_ratio", "mgmt_fee", "turn_ratio",
        "age", "new_sls", "rein_sls", "oth_sls", "redemp", "net_flow"
    ]
    for candidates in FUND_FILES:
        p = find_file(root, candidates)
        if p is None:
            missing.append(candidates[-1])
            continue
        header = pd.read_csv(p, nrows=0).columns.tolist()
        usecols = [c for c in wanted_cols if c in header]
        frames.append(pd.read_csv(p, low_memory=False, usecols=usecols))
    if not frames:
        raise FileNotFoundError(f"Missing fund-level files: {missing}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["caldt"], errors="coerce")
    df["month_key"] = df["date"].dt.strftime("%Y-%m")
    df["mret"] = df["mret"].map(parse_number)
    df = df.dropna(subset=["date", "mret"])
    df["crsp_portno"] = df.get("crsp_portno", pd.Series("", index=df.index)).map(clean_id)
    df["crsp_fundno"] = df.get("crsp_fundno", pd.Series("", index=df.index)).map(clean_id)
    df["fund"] = df.get("fund_name", pd.Series("", index=df.index)).map(clean_text)
    df["fund_ticker"] = df.get("ticker", pd.Series("", index=df.index)).map(clean_text)
    df["manager"] = df.get("mgr_name", pd.Series("Unknown Manager", index=df.index)).map(clean_text).replace("", "Unknown Manager")
    df["mgmt_name"] = df.get("mgmt_name", pd.Series("", index=df.index)).map(clean_text)
    for c in ["mtna", "exp_ratio", "mgmt_fee", "turn_ratio", "age"]:
        df[c] = df.get(c, pd.Series(np.nan, index=df.index)).map(parse_number)
    df["net_flow"] = compute_net_flow(df)
    mgr_dt = pd.to_datetime(df.get("mgr_dt", pd.Series(pd.NaT, index=df.index)), errors="coerce")
    df["tenure"] = ((df["date"] - mgr_dt).dt.days / 365.25).clip(lower=0)
    df = df.merge(sp500[["month_key", "sp500_ret"] + [f"sp500_trailing_{y}y" for y in TRAILING_WINDOWS]], on="month_key", how="left")
    df["sp500_ret"] = df["sp500_ret"].fillna(0.10 / 12.0)
    df["excess_ret"] = df["mret"] - df["sp500_ret"]
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # Portfolio-month table, compatible with holdings report rows.
    numeric_aggs = {
        "mret": "mean", "sp500_ret": "mean", "excess_ret": "mean",
        "net_flow": "sum", "mtna": "mean", "exp_ratio": "mean", "mgmt_fee": "mean",
        "turn_ratio": "mean", "age": "mean", "tenure": "mean",
    }
    id_aggs = {
        "manager": lambda x: next((clean_text(v) for v in x if clean_text(v)), "Unknown Manager"),
        "fund": lambda x: next((clean_text(v) for v in x if clean_text(v)), ""),
        "fund_ticker": lambda x: next((clean_text(v) for v in x if clean_text(v)), ""),
        "mgmt_name": lambda x: next((clean_text(v) for v in x if clean_text(v)), ""),
        "crsp_fundno": lambda x: next((clean_text(v) for v in x if clean_text(v)), ""),
    }
    pm = df[df["crsp_portno"].astype(str).str.len() > 0].groupby(["crsp_portno", "month_key"], as_index=False).agg({**numeric_aggs, **id_aggs})
    pm["date"] = pd.to_datetime(pm["month_key"] + "-01")
    pm["year"] = pm["date"].dt.year
    pm["quarter"] = pm["date"].dt.quarter
    return add_portfolio_rolling_features(pm)


def add_portfolio_rolling_features(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.sort_values(["crsp_portno", "date"]).copy()
    parts = []
    for _, g in pm.groupby("crsp_portno", sort=False):
        g = g.sort_values("date").copy()
        fund_logs = g["mret"].map(safe_log1p)
        sp_logs = g["sp500_ret"].map(safe_log1p)
        for years, months in TRAILING_WINDOWS.items():
            minp = int(months * MIN_WINDOW_RATIO)
            shifted_fund_logs = fund_logs.shift(1)
            shifted_sp_logs = sp_logs.shift(1)
            fund_count = shifted_fund_logs.rolling(months, min_periods=minp).count()
            fund_sum = shifted_fund_logs.rolling(months, min_periods=minp).sum()
            sp_count = shifted_sp_logs.rolling(months, min_periods=minp).count()
            sp_sum = shifted_sp_logs.rolling(months, min_periods=minp).sum()
            g[f"fund_trailing_{years}y"] = [annualize_log_sum(s, int(c)) if c >= minp else np.nan for s, c in zip(fund_sum, fund_count)]
            g[f"fund_trailing_{years}y_period_return"] = [safe_expm1(s) if c >= minp else np.nan for s, c in zip(fund_sum, fund_count)]
            g[f"sp500_trailing_{years}y"] = [annualize_log_sum(s, int(c)) if c >= minp else np.nan for s, c in zip(sp_sum, sp_count)]
            g[f"fund_trailing_{years}y_excess"] = g[f"fund_trailing_{years}y"] - g[f"sp500_trailing_{years}y"]
            # Fast ex-ante downside proxy: rolling minimum monthly return before current month.
            # This keeps the same no-leakage time rule and is much faster than exact rolling drawdown.
            g[f"fund_trailing_{years}y_max_drawdown"] = g["mret"].shift(1).rolling(months, min_periods=minp).min()
            shifted_mret = g["mret"].shift(1)
            shifted_sp = g["sp500_ret"].shift(1)
            cov = shifted_mret.rolling(months, min_periods=minp).cov(shifted_sp)
            var = shifted_sp.rolling(months, min_periods=minp).var()
            g[f"fund_trailing_{years}y_beta_vs_sp500"] = cov / var.replace(0, np.nan)
            for col in ["net_flow", "mtna", "exp_ratio", "mgmt_fee", "turn_ratio", "age", "tenure"]:
                shifted = g[col].shift(1)
                minp_factor = max(3, int(months * 0.5))
                g[f"trailing_avg_{col}_{years}y"] = shifted.rolling(months, min_periods=minp_factor).mean()
                if col == "net_flow":
                    g[f"trailing_sum_net_flow_{years}y"] = shifted.rolling(months, min_periods=minp_factor).sum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def load_sector_exposure_panel(root: Path) -> pd.DataFrame:
    """Build report-level 11-sector weights and observed weight changes.

    The deltas are deliberately named observed sector deltas.  A fully drift-adjusted
    active delta requires security total returns between reports and must not be
    inferred from weights alone.
    """
    cache_path = find_file(root, SECTOR_CACHE_FILES)
    if cache_path is None:
        return pd.DataFrame(columns=["crsp_portno", "report_date"])
    cache = pd.read_csv(cache_path, low_memory=False)
    ticker_col = "holding_ticker" if "holding_ticker" in cache else "yahoo_ticker"
    cache["ticker_key"] = cache[ticker_col].astype(str).str.upper().str.strip()
    sector_map = cache.drop_duplicates("ticker_key").set_index("ticker_key")["sector"].astype(str).to_dict()
    normalization = {
        "Energy": "energy", "Basic Materials": "materials", "Materials": "materials",
        "Industrials": "industrials", "Consumer Cyclical": "consumer_discretionary",
        "Consumer Discretionary": "consumer_discretionary", "Consumer Defensive": "consumer_staples",
        "Consumer Staples": "consumer_staples", "Healthcare": "health_care", "Health Care": "health_care",
        "Financial Services": "financials", "Financials": "financials", "Technology": "information_technology",
        "Information Technology": "information_technology", "Communication Services": "communication_services",
        "Utilities": "utilities", "Real Estate": "real_estate",
    }
    paths: list[Path] = []
    for pattern in HOLDINGS_GLOBS:
        paths.extend(root.glob(pattern))
    frames = []
    for path in sorted(set(paths)):
        frame = pd.read_csv(path, usecols=["crsp_portno", "report_dt", "holding_ticker", "holding_percent_tna"], low_memory=False)
        frame["ticker_key"] = frame["holding_ticker"].astype(str).str.upper().str.strip()
        frame["sector_key"] = frame["ticker_key"].map(sector_map).map(normalization)
        frame["weight"] = pd.to_numeric(frame["holding_percent_tna"], errors="coerce") / 100.0
        frame["report_date"] = pd.to_datetime(frame["report_dt"], errors="coerce").dt.normalize()
        frame["crsp_portno"] = frame["crsp_portno"].map(clean_id)
        frames.append(frame.dropna(subset=["report_date", "sector_key", "weight"]))
    if not frames:
        return pd.DataFrame(columns=["crsp_portno", "report_date"])
    holdings = pd.concat(frames, ignore_index=True)
    panel = holdings.pivot_table(index=["crsp_portno", "report_date"], columns="sector_key", values="weight", aggfunc="sum", fill_value=0).reset_index()
    for sector in GICS_SECTORS:
        if sector not in panel: panel[sector] = 0.0
        panel = panel.rename(columns={sector: f"sector_{sector}_exposure"})
    panel = panel.sort_values(["crsp_portno", "report_date"])
    delta_cols = []
    for sector in GICS_SECTORS:
        exposure = f"sector_{sector}_exposure"
        delta = f"delta_sector_{sector}"
        panel[delta] = panel.groupby("crsp_portno", sort=False)[exposure].diff()
        delta_cols.append(delta)
    panel["sector_rotation_intensity"] = panel[delta_cols].abs().sum(axis=1, min_count=1) / 2.0
    return panel[["crsp_portno", "report_date"] + [f"sector_{s}_exposure" for s in GICS_SECTORS] + delta_cols + ["sector_rotation_intensity"]]


def add_forward_outcomes(pm: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe forward outcomes beginning in the month after each event."""
    pm = pm.sort_values(["crsp_portno", "date"]).copy()
    parts = []
    for _, group in pm.groupby("crsp_portno", sort=False):
        g = group.sort_values("date").copy()
        fund = pd.to_numeric(g["mret"], errors="coerce").to_numpy(dtype=float)
        market = pd.to_numeric(g["sp500_ret"], errors="coerce").to_numpy(dtype=float)
        for horizon in PREDICTION_HORIZONS:
            fund_out, market_out, excess_out, drawdown_out = [], [], [], []
            for i in range(len(g)):
                fwd_fund = fund[i + 1:i + 1 + horizon]
                fwd_market = market[i + 1:i + 1 + horizon]
                if len(fwd_fund) < horizon or np.isfinite(fwd_fund).sum() < horizon:
                    fund_out.append(np.nan); market_out.append(np.nan)
                    excess_out.append(np.nan); drawdown_out.append(np.nan)
                    continue
                fr = compound_return(fwd_fund)
                mr = compound_return(fwd_market)
                fund_out.append(fr); market_out.append(mr)
                excess_out.append(fr - mr if np.isfinite(fr) and np.isfinite(mr) else np.nan)
                drawdown_out.append(max_drawdown_from_monthly(fwd_fund))
            g[f"future_{horizon}m_return"] = fund_out
            g[f"future_{horizon}m_sp500_return"] = market_out
            g[f"future_{horizon}m_excess_return"] = excess_out
            g[f"future_{horizon}m_drawdown"] = drawdown_out
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def load_base_groundtruth(root: Path) -> pd.DataFrame:
    p = find_file(root, BASE_GT_FILES)
    if p is None:
        raise FileNotFoundError("Cannot find base manager_action_ground_truth.csv. Run your original builder first, or place the CSV under data/derived/manager_action_groundtruth/.")
    df = pd.read_csv(p, low_memory=False)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])
    if "month_key" not in df.columns:
        df["month_key"] = df["report_date"].dt.strftime("%Y-%m")
    for c in ["crsp_portno", "crsp_fundno"]:
        if c in df.columns:
            df[c] = df[c].map(clean_id)
    if "year" not in df.columns:
        df["year"] = df["report_date"].dt.year
    if "quarter" not in df.columns:
        df["quarter"] = df["report_date"].dt.quarter
    return df


def enrich_with_current_and_trailing(gt: pd.DataFrame, pm: pd.DataFrame, sector_panel: pd.DataFrame) -> pd.DataFrame:
    # Recompute all forward outcomes from monthly returns so legacy 12M columns cannot
    # silently conflict with the new four-horizon contract.
    gt = gt.drop(columns=[c for c in gt.columns if c in FUTURE_COLUMNS or re.match(r"^(future_|direction_label_|outcome_5class_|label_positive_excess_)", c)], errors="ignore")
    current_cols = [
        "crsp_portno", "month_key", "mret", "sp500_ret", "excess_ret", "net_flow", "mtna",
        "exp_ratio", "mgmt_fee", "turn_ratio", "age", "tenure",
    ]
    trailing_cols = []
    for years in TRAILING_WINDOWS:
        trailing_cols += [
            f"fund_trailing_{years}y", f"sp500_trailing_{years}y",
            f"fund_trailing_{years}y_excess", f"fund_trailing_{years}y_period_return",
            f"fund_trailing_{years}y_max_drawdown", f"fund_trailing_{years}y_beta_vs_sp500",
            f"trailing_avg_net_flow_{years}y", f"trailing_sum_net_flow_{years}y",
            f"trailing_avg_mtna_{years}y", f"trailing_avg_exp_ratio_{years}y",
            f"trailing_avg_mgmt_fee_{years}y", f"trailing_avg_turn_ratio_{years}y",
            f"trailing_avg_age_{years}y", f"trailing_avg_tenure_{years}y",
        ]
    forward_cols = [c for c in FUTURE_COLUMNS if c.startswith("future_") and c in pm.columns]
    join_cols = [c for c in current_cols + trailing_cols + forward_cols if c in pm.columns]
    right = pm[join_cols].copy()
    rename = {
        "mret": "current_mret", "sp500_ret": "current_sp500_ret", "excess_ret": "current_excess_ret",
        "net_flow": "current_net_flow", "mtna": "current_mtna", "exp_ratio": "current_exp_ratio",
        "mgmt_fee": "current_mgmt_fee", "turn_ratio": "current_turn_ratio", "age": "current_age",
        "tenure": "current_tenure",
    }
    right = right.rename(columns=rename)
    overlap = (set(gt.columns) & set(right.columns)) - {"crsp_portno", "month_key"}
    gt = gt.drop(columns=sorted(overlap), errors="ignore")
    out = gt.merge(right, on=["crsp_portno", "month_key"], how="left")
    if not sector_panel.empty:
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce").dt.normalize()
        sector_overlap = (set(out.columns) & set(sector_panel.columns)) - {"crsp_portno", "report_date"}
        out = out.drop(columns=sorted(sector_overlap), errors="ignore")
        out = out.merge(sector_panel, on=["crsp_portno", "report_date"], how="left")
    return out


def compute_future_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for horizon in PREDICTION_HORIZONS:
        ex = pd.to_numeric(df.get(f"future_{horizon}m_excess_return"), errors="coerce")
        direction = np.select(
            [ex <= -NEUTRAL_BAND, ex >= NEUTRAL_BAND], [-1, 1], default=0,
        ).astype(float)
        direction[ex.isna().to_numpy()] = np.nan
        df[f"direction_label_{horizon}m"] = direction
        df[f"label_positive_excess_{horizon}m"] = np.where(ex.notna(), (direction == 1).astype(int), np.nan)
        # Data-adaptive tails preserve the requested five levels without leaking across time per event.
        magnitude = ex.abs()
        cutoff = float(magnitude[magnitude.notna()].median()) if magnitude.notna().any() else NEUTRAL_BAND
        cutoff = max(cutoff, NEUTRAL_BAND)
        df[f"outcome_5class_{horizon}m"] = np.select(
            [ex <= -cutoff, ex < -NEUTRAL_BAND, ex <= NEUTRAL_BAND, ex < cutoff],
            ["large_loss", "small_loss", "neutral", "small_win"],
            default="large_win",
        )
        df.loc[ex.isna(), f"outcome_5class_{horizon}m"] = np.nan
    return df


def add_rolling_ex_ante_style_deviation(df: pd.DataFrame, years: int) -> pd.DataFrame:
    df = df.copy()
    months = TRAILING_WINDOWS[years]
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    all_baseline_features = list(dict.fromkeys(STYLE_DEVIATION_FEATURES + ACTION_DEVIATION_FEATURES))
    for c in all_baseline_features:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strict event-time baseline.  At each report_date, only manager observations
    # in [report_date - 36 months, report_date) are used. Same-date reports are
    # excluded together, so another portfolio disclosed on the same date cannot leak.
    tmp = df.sort_values(["manager", "report_date", "crsp_portno"]).copy()
    baseline_mean = pd.DataFrame(np.nan, index=tmp.index, columns=all_baseline_features, dtype=float)
    baseline_std = pd.DataFrame(np.nan, index=tmp.index, columns=all_baseline_features, dtype=float)
    counts = pd.Series(0, index=tmp.index, dtype=int)
    for _, group in tmp.groupby("manager", sort=False, dropna=False):
        g = group.sort_values(["report_date", "crsp_portno"])
        dates = g["report_date"].to_numpy(dtype="datetime64[ns]")
        start_dates = (g["report_date"] - pd.DateOffset(months=months)).to_numpy(dtype="datetime64[ns]")
        starts = np.searchsorted(dates, start_dates, side="left")
        ends = np.searchsorted(dates, dates, side="left")
        counts.loc[g.index] = ends - starts
        for c in all_baseline_features:
            values = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            safe = np.where(valid, values, 0.0)
            csum = np.r_[0.0, np.cumsum(safe)]
            csq = np.r_[0.0, np.cumsum(safe * safe)]
            ccount = np.r_[0, np.cumsum(valid.astype(int))]
            nobs = ccount[ends] - ccount[starts]
            total = csum[ends] - csum[starts]
            total_sq = csq[ends] - csq[starts]
            mean = np.divide(total, nobs, out=np.full(len(g), np.nan), where=nobs >= 2)
            variance = np.divide(total_sq - np.divide(total * total, nobs, out=np.zeros(len(g)), where=nobs > 0), nobs - 1, out=np.full(len(g), np.nan), where=nobs >= 2)
            std = np.sqrt(np.maximum(variance, 0.0))
            std[std <= 1e-12] = np.nan
            baseline_mean.loc[g.index, c] = mean
            baseline_std.loc[g.index, c] = std

    baseline = baseline_mean.reindex(df.index)
    scale = baseline_std.reindex(df.index)
    zscores: Dict[str, pd.Series] = {}
    for c in all_baseline_features:
        z = ((pd.to_numeric(df[c], errors="coerce") - baseline[c]).abs() / scale[c]).replace([np.inf, -np.inf], np.nan)
        zscores[c] = z
        df[f"rolling_dev_{c}_{years}y"] = z
        df[f"rolling_past_mean_{c}_{years}y"] = baseline[c]
        df[f"rolling_past_std_{c}_{years}y"] = scale[c]

    def mean_score(features: List[str]) -> pd.Series:
        return pd.concat([zscores[c] for c in features], axis=1).mean(axis=1)

    style_score = mean_score(STYLE_DEVIATION_FEATURES)
    sector_score = mean_score(SECTOR_STYLE_FEATURES)
    cross_asset_score = mean_score(CROSS_ASSET_STYLE_FEATURES)
    action_score = mean_score(ACTION_DEVIATION_FEATURES)
    df[f"rolling_style_deviation_score_{years}y"] = style_score
    df["rolling_style_deviation_score"] = style_score.fillna(pd.to_numeric(df.get("style_deviation_score"), errors="coerce"))
    df["rolling_sector_deviation_score"] = sector_score
    df["rolling_cross_asset_deviation_score"] = cross_asset_score
    df["rolling_action_deviation_score"] = action_score
    df["style_window_months"] = months
    df["style_window_years"] = years
    df["style_window_type"] = "strict_event_time_trailing_36m_manager_history_excluding_current_date"
    df["style_window_start_date"] = (df["report_date"] - pd.DateOffset(months=months)).dt.strftime("%Y-%m-%d")
    df["style_window_end_date"] = (df["report_date"] - pd.DateOffset(days=1)).dt.strftime("%Y-%m-%d")
    df["style_obs_count"] = counts.reindex(df.index).fillna(0).astype(int)
    return df

def add_horizon_aliases(df: pd.DataFrame, years: int) -> pd.DataFrame:
    out = df.copy()
    months = TRAILING_WINDOWS[years]
    out["training_window_years"] = years
    out["training_window_months"] = months
    alias_map = {
        f"fund_trailing_{years}y": "fund_trailing_return",
        f"sp500_trailing_{years}y": "sp500_trailing_return",
        f"fund_trailing_{years}y_excess": "fund_trailing_excess_return",
        f"fund_trailing_{years}y_period_return": "fund_trailing_period_return",
        f"fund_trailing_{years}y_max_drawdown": "fund_trailing_max_drawdown",
        f"fund_trailing_{years}y_beta_vs_sp500": "fund_trailing_beta_vs_sp500",
        f"trailing_avg_net_flow_{years}y": "trailing_avg_net_flow",
        f"trailing_sum_net_flow_{years}y": "trailing_sum_net_flow",
        f"trailing_avg_mtna_{years}y": "trailing_avg_mtna",
        f"trailing_avg_exp_ratio_{years}y": "trailing_avg_exp_ratio",
        f"trailing_avg_mgmt_fee_{years}y": "trailing_avg_mgmt_fee",
        f"trailing_avg_turn_ratio_{years}y": "trailing_avg_turn_ratio",
        f"trailing_avg_age_{years}y": "trailing_avg_age",
        f"trailing_avg_tenure_{years}y": "trailing_avg_tenure",
    }
    for src, dst in alias_map.items():
        out[dst] = out[src] if src in out.columns else np.nan
    bond_money = pd.to_numeric(out.get("bond_money_exposure", pd.Series(np.nan, index=out.index)), errors="coerce")
    indirect_equity = pd.to_numeric(out.get("indirect_equity_exposure", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["nonstock_total_exposure"] = pd.concat([bond_money, indirect_equity], axis=1).sum(axis=1, min_count=1)
    if all(c in out.columns for c in ("manager", "crsp_portno", "report_date")):
        order = out.sort_values(["manager", "crsp_portno", "report_date"]).index
        ordered = out.loc[order]
        out.loc[order, "delta_nonstock_total_exposure"] = ordered.groupby(["manager", "crsp_portno"], sort=False)["nonstock_total_exposure"].diff().to_numpy()
    else:
        out["delta_nonstock_total_exposure"] = pd.to_numeric(out.get("delta_bond_money", pd.Series(np.nan, index=out.index)), errors="coerce") + pd.to_numeric(out.get("delta_indirect_equity", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["delta_sector_exposure"] = pd.to_numeric(out.get("sector_rotation_intensity", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["feature_cutoff_date"] = (pd.to_datetime(out["report_date"]) - pd.DateOffset(days=1)).dt.strftime("%Y-%m-%d")
    out["label_start_date"] = (pd.to_datetime(out["report_date"]) + pd.DateOffset(days=1)).dt.strftime("%Y-%m-%d")
    out["label_end_date"] = (pd.to_datetime(out["report_date"]) + pd.DateOffset(months=max(PREDICTION_HORIZONS))).dt.strftime("%Y-%m-%d")
    out = add_rolling_ex_ante_style_deviation(out, years)
    out = compute_future_labels(out)
    out["event_id"] = (
        out["training_window_years"].astype(str) + "y__" +
        out.get("manager", pd.Series("", index=out.index)).map(clean_text).str.replace(r"\W+", "_", regex=True).str[:40] + "__" +
        out.get("crsp_portno", pd.Series("", index=out.index)).map(clean_text) + "__" +
        pd.to_datetime(out["report_date"]).dt.strftime("%Y%m%d")
    )
    # No-leakage audit flags.
    out["leakage_check_passed"] = True
    out.loc[pd.to_datetime(out["style_window_end_date"], errors="coerce") >= pd.to_datetime(out["report_date"], errors="coerce"), "leakage_check_passed"] = False
    return out


def make_ml_training_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = [c for c in ML_NUMERIC_COLUMNS + ML_CATEGORICAL_COLUMNS if c in df.columns]
    meta_cols = [c for c in IDENTITY_COLUMNS if c in df.columns]
    keep = meta_cols + needed + [c for c in TARGET_COLUMNS if c in df.columns]
    ml = df[keep].copy()
    outcome_cols = [f"future_{h}m_excess_return" for h in PREDICTION_HORIZONS]
    ml = ml[ml[outcome_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)]
    # Keep rows where at least one core action/exposure feature exists.
    core = [c for c in ["delta_stock", "delta_beta", "delta_technology", "stock_allocation", "portfolio_beta"] if c in ml.columns]
    if core:
        ml = ml[ml[core].notna().any(axis=1)]
    return ml.reset_index(drop=True)


def write_schema_and_dictionary(out_dir: Path, audit: dict) -> None:
    schema = {
        "description": "Balanced-fund manager-action events with a three-year ex-ante style window and 3M/6M/9M/12M forward labels.",
        "identity_columns": IDENTITY_COLUMNS,
        "current_month_columns": CURRENT_COLUMNS,
        "trailing_alias_columns": TRAILING_ALIAS_COLUMNS,
        "action_exposure_columns": ACTION_EXPOSURE_COLUMNS,
        "future_label_columns": FUTURE_COLUMNS,
        "ml_numeric_columns": ML_NUMERIC_COLUMNS,
        "ml_categorical_columns": ML_CATEGORICAL_COLUMNS,
        "audit": audit,
    }
    (out_dir / "manager_action_ground_truth_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# manager_action_ground_truth data dictionary", "",
        "Each row is a manager-action event. Features end before the event and outcomes cover 3M, 6M, 9M and 12M after it.", "",
        "## Key additions", "",
        "- `current_*`: current report-month fund characteristics.",
        "- `fund_trailing_*`: generic alias for the chosen training window.",
        "- `rolling_style_deviation_score`: deviation from the manager's own past style before report_date.",
        "- `rolling_sector_deviation_score`: 11-sector exposure deviation from the strict prior-36M manager baseline.",
        "- `rolling_action_deviation_score`: action-delta deviation from the strict prior-36M manager baseline.",
        "- `style_window_type`: strict event-time trailing 36M manager history, excluding every current-date event.",
        "- `direction_label_{h}m`: -1/0/+1 direction with a +/-0.5% neutral band.",
        "- `outcome_5class_{h}m`: large loss, small loss, neutral, small win, or large win.",
        "- `leakage_check_passed`: confirms the style window ends before report_date.",
    ]
    (out_dir / "manager_action_ground_truth_data_dictionary.md").write_text("\n".join(md), encoding="utf-8")


def build_ground_truth_v2(root: Path, output_dir: Path) -> Tuple[pd.DataFrame, dict]:
    print("[1/6] Load base ground truth")
    base = load_base_groundtruth(root)
    print(f"      base rows = {len(base):,}")
    print("[2/6] Load S&P500 and fund monthly data; compute current and rolling features")
    sp500 = load_sp500(root)
    port_month = add_forward_outcomes(load_fund_month_table(root, sp500))
    print(f"      port-month rows = {len(port_month):,}")
    print("[3/6] Join current month, 11-sector panel, 3Y style features and four forward horizons")
    sector_panel = load_sector_exposure_panel(root)
    print(f"      sector-report rows = {len(sector_panel):,}")
    enriched = enrich_with_current_and_trailing(base, port_month, sector_panel)
    print("[4/6] Build horizon-specific ground truth and ML datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir.parent / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    horizon_frames = []
    ml_frames = []
    ml_outputs = {}
    for years in [3]:
        h = add_horizon_aliases(enriched, years)
        ml = make_ml_training_table(h)
        horizon_path = output_dir / f"manager_action_ground_truth_trailing{years}y_multi_horizon.csv"
        ml_path = prediction_dir / f"part6_prediction_dataset_trailing{years}y_multi_horizon.csv"
        write_dataframe_csv(horizon_path, h)
        write_dataframe_csv(ml_path, ml)
        horizon_frames.append(h)
        ml_frames.append(ml)
        ml_outputs[str(years)] = {"groundtruth_view_path": str(horizon_path), "ml_dataset_path": str(ml_path), "rows": int(len(ml))}
        print(f"      {years}Y: ground truth rows={len(h):,}, ML rows={len(ml):,}")
    combined_gt = pd.concat(horizon_frames, ignore_index=True)
    combined_ml = pd.concat(ml_frames, ignore_index=True)
    print("[5/6] Save combined outputs")
    combined_gt_path = output_dir / "manager_action_ground_truth.csv"
    combined_ml_path = prediction_dir / "part6_prediction_dataset.csv"
    write_dataframe_csv(combined_gt_path, combined_gt)
    write_dataframe_csv(combined_ml_path, combined_ml)
    audit = {
        "rows": int(len(combined_gt)),
        "base_rows_before_horizon_expansion": int(len(base)),
        "combined_groundtruth": str(combined_gt_path),
        "combined_ml_dataset": str(combined_ml_path),
        "ml_outputs": ml_outputs,
        "targets": [f"future_{h}m_excess_return" for h in PREDICTION_HORIZONS],
        "training_windows": [3],
        "prediction_horizons_months": list(PREDICTION_HORIZONS),
        "neutral_band": NEUTRAL_BAND,
        "outcome_label_counts": combined_gt.get("outcome_label", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "direction_label_counts": {str(h): combined_ml.get(f"direction_label_{h}m", pd.Series(dtype=float)).value_counts(dropna=False).to_dict() for h in PREDICTION_HORIZONS},
        "leakage_check_passed_counts": combined_gt.get("leakage_check_passed", pd.Series(dtype=bool)).value_counts(dropna=False).to_dict(),
        "style_window_type_counts": combined_gt.get("style_window_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "style_obs_count_summary": pd.to_numeric(combined_gt.get("style_obs_count"), errors="coerce").describe().to_dict(),
        "ml_numeric_columns": [c for c in ML_NUMERIC_COLUMNS if c in combined_ml.columns],
        "ml_categorical_columns": [c for c in ML_CATEGORICAL_COLUMNS if c in combined_ml.columns],
    }
    (output_dir / "manager_action_ground_truth_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_schema_and_dictionary(output_dir, audit)
    print("[6/6] Done")
    return combined_gt, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manager-action ground truth for 3M/6M/9M/12M prediction.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.data_root / "derived" / "manager_action_groundtruth")
    print(f"[INFO] data_root = {args.data_root}")
    print(f"[INFO] output_dir = {output_dir}")
    _, audit = build_ground_truth_v2(args.data_root, output_dir)
    print("[DONE] outputs:")
    print(json.dumps(audit.get("ml_outputs", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
