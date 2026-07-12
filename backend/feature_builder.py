from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    metadata: List[Dict[str, Any]]
    warnings: List[str]
    dataset_path: str


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _split_report_key(key: str) -> Tuple[str, str]:
    parts = str(key or "").split("|")
    if len(parts) >= 2:
        return _norm_text(parts[0]), _norm_date(parts[1])
    return "", ""


def _load_dataset(data_root: Path) -> Tuple[pd.DataFrame, Path]:
    candidates = [
        data_root / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_future12m.csv",
        data_root / "derived" / "prediction" / "part6_prediction_dataset.csv",
        data_root / "part6_prediction_dataset_trailing3y_future12m.csv",
        data_root / "part6_prediction_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "report_date" in df.columns:
                df["report_date_norm"] = pd.to_datetime(df["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df["report_date_norm"] = ""
            if "training_window_years" in df.columns:
                df = df[pd.to_numeric(df["training_window_years"], errors="coerce").fillna(3).astype(int) == 3].copy()
            return df, path
    raise FileNotFoundError("Cannot find 3Y prediction dataset. Expected data/derived/prediction/part6_prediction_dataset_trailing3y_future12m.csv")


def _selected_managers(payload: Dict[str, Any]) -> List[str]:
    managers = []
    part5 = payload.get("part5") or {}
    part3 = payload.get("part3") or {}
    part4 = payload.get("part4") or {}
    for value in part5.get("selectedManagerNames") or []:
        if _norm_text(value):
            managers.append(_norm_text(value))
    for value in part3.get("selectedManagers") or []:
        if _norm_text(value):
            managers.append(_norm_text(value))
    for value in part3.get("latestManagers") or []:
        if _norm_text(value):
            managers.append(_norm_text(value))
    for record in part4.get("manager_records_raw") or []:
        manager = _norm_text(record.get("manager"))
        if manager:
            managers.append(manager)
    return sorted(set(managers))


def _selected_report_pairs(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    part5 = payload.get("part5") or {}
    keys = []
    keys.extend(part5.get("brushedReportKeys") or [])
    active = part5.get("activeReportKey")
    if active:
        keys.append(active)
    pairs = [_split_report_key(k) for k in keys]
    for row in part5.get("selected_fund_reports") or []:
        port = _norm_text(row.get("crsp_portno") or row.get("portno"))
        date = _norm_date(row.get("report_dt") or row.get("report_date"))
        if port and date:
            pairs.append((port, date))
    return sorted(set(pair for pair in pairs if pair[0] and pair[1]))


def _selected_funds(payload: Dict[str, Any]) -> List[str]:
    part1 = payload.get("part1") or {}
    funds = []
    for key in ["selectedFundIdsA", "selectedFundIdsB"]:
        for value in part1.get(key) or []:
            if _norm_text(value):
                funds.append(_norm_text(value))
    return sorted(set(funds))


def build_feature_frame(payload: Dict[str, Any], data_root: Path, max_events: int = 50) -> FeatureBuildResult:
    df, dataset_path = _load_dataset(data_root)
    warnings: List[str] = []
    work = df.copy()

    report_pairs = _selected_report_pairs(payload)
    managers = _selected_managers(payload)
    funds = _selected_funds(payload)

    if report_pairs:
        pair_index = set(report_pairs)
        work = work[work.apply(lambda r: (_norm_text(r.get("crsp_portno")), _norm_text(r.get("report_date_norm"))) in pair_index, axis=1)]
        if work.empty:
            warnings.append("Selected Part5 report keys did not match the 3Y prediction dataset; falling back to manager/fund filters.")
            work = df.copy()

    if managers and len(work) == len(df):
        manager_set = set(managers)
        work = work[work["manager"].astype(str).isin(manager_set)] if "manager" in work.columns else work.iloc[0:0]

    if funds and work.empty:
        fund_set = set(funds)
        if "crsp_fundno" in df.columns:
            work = df[df["crsp_fundno"].astype(str).isin(fund_set)].copy()

    if work.empty and managers:
        manager_set = set(managers)
        if "manager" in df.columns:
            work = df[df["manager"].astype(str).isin(manager_set)].copy()

    if work.empty:
        warnings.append("No matching events were found in the 3Y prediction dataset. Returning the most recent rows as a safe demo fallback.")
        work = df.copy()

    if "report_date" in work.columns:
        work = work.sort_values("report_date", ascending=False)
    work = work.head(max_events).copy()

    metadata_cols = [
        "event_id", "manager", "fund", "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name",
        "report_date", "year", "quarter", "month_key", "action_type", "market_regime",
        "manager_style_group", "training_window_years", "label_positive_excess_12m",
        "future_12m_excess_return", "future_drawdown",
    ]
    metadata = []
    for _, row in work.iterrows():
        item = {c: (None if pd.isna(row[c]) else row[c]) for c in metadata_cols if c in work.columns}
        metadata.append(item)

    return FeatureBuildResult(frame=work, metadata=metadata, warnings=warnings, dataset_path=str(dataset_path))
