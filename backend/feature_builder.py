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
    date_domain: Dict[str, str]
    report_manager_map: Dict[str, str]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _report_pair(key: str) -> Tuple[str, str]:
    parts = str(key or "").split("|")
    return (_text(parts[0]), _date(parts[1])) if len(parts) >= 2 else ("", "")


def _load_dataset(data_root: Path) -> Tuple[pd.DataFrame, Path]:
    candidates = [
        data_root / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_multi_horizon.csv",
        data_root / "derived" / "prediction" / "part6_prediction_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["report_date_norm"] = pd.to_datetime(frame.get("report_date"), errors="coerce").dt.strftime("%Y-%m-%d")
            if "training_window_years" in frame:
                frame = frame[pd.to_numeric(frame["training_window_years"], errors="coerce").fillna(3).astype(int).eq(3)]
            return frame, path
    raise FileNotFoundError("Run build_manager_action_groundtruth_complete.py to create the multi-horizon dataset.")


def _selection(payload: Dict[str, Any]) -> tuple[set[Tuple[str, str]], set[str], set[str]]:
    part1, part3, part4, part5 = (payload.get(k) or {} for k in ("part1", "part3", "part4", "part5"))
    keys = list(part5.get("brushedReportKeys") or [])
    if part5.get("activeReportKey"):
        keys.append(part5["activeReportKey"])
    pairs = {_report_pair(k) for k in keys}
    for row in part5.get("selected_fund_reports") or []:
        pairs.add((_text(row.get("crsp_portno") or row.get("portno")), _date(row.get("report_dt") or row.get("report_date"))))
    managers = {_text(v) for key in ("selectedManagers", "latestManagers") for v in (part3.get(key) or [])}
    managers |= {_text(r.get("manager")) for r in (part4.get("manager_records_raw") or [])}
    managers |= {_text(v) for v in (part5.get("selectedManagerNames") or [])}
    funds = {_text(v) for key in ("selectedFundIdsA", "selectedFundIdsB") for v in (part1.get(key) or [])}
    return {p for p in pairs if all(p)}, {m for m in managers if m}, {f for f in funds if f}


def build_feature_frame(payload: Dict[str, Any], data_root: Path, max_events: int = 50) -> FeatureBuildResult:
    frame, path = _load_dataset(data_root)
    warnings: List[str] = []
    valid_dates = pd.to_datetime(frame["report_date_norm"], errors="coerce")
    domain = {
        "min": valid_dates.min().strftime("%Y-%m-%d") if valid_dates.notna().any() else "",
        "max": valid_dates.max().strftime("%Y-%m-%d") if valid_dates.notna().any() else "",
    }
    work = frame.copy()
    part6 = payload.get("part6") or {}
    start, end = _date(part6.get("date_start")), _date(part6.get("date_end"))
    if start:
        work = work[work["report_date_norm"] >= start]
    if end:
        work = work[work["report_date_norm"] <= end]

    pairs, managers, funds = _selection(payload)
    if pairs:
        selected = work.apply(lambda r: (_text(r.get("crsp_portno")), _text(r.get("report_date_norm"))) in pairs, axis=1)
        if selected.any():
            work = work[selected]
        else:
            warnings.append("Part 5 selected report keys did not match the Part 6 event table; using manager/fund fallback.")
            if managers and "manager" in work:
                manager_rows = work["manager"].astype(str).isin(managers)
                if manager_rows.any():
                    work = work[manager_rows]
            elif funds and "crsp_fundno" in work:
                fund_rows = work["crsp_fundno"].astype(str).isin(funds)
                if fund_rows.any():
                    work = work[fund_rows]
    elif managers and "manager" in work:
        selected = work["manager"].astype(str).isin(managers)
        if selected.any():
            work = work[selected]
    elif funds and "crsp_fundno" in work:
        selected = work["crsp_fundno"].astype(str).isin(funds)
        if selected.any():
            work = work[selected]

    if work.empty:
        warnings.append("No event matched the current filters; the most recent events in the selected date range are used.")
        work = frame.copy()
        if start: work = work[work["report_date_norm"] >= start]
        if end: work = work[work["report_date_norm"] <= end]
    work = work.sort_values("report_date", ascending=False)
    report_manager_map: Dict[str, str] = {}
    if all(c in work for c in ("crsp_portno", "report_date_norm", "manager")):
        manager_rows = work[["crsp_portno", "report_date_norm", "manager"]].dropna().drop_duplicates()
        for (portno, report_date), group in manager_rows.groupby(["crsp_portno", "report_date_norm"], sort=False):
            managers_for_report = sorted({str(v).strip() for v in group["manager"] if str(v).strip()})
            report_manager_map[f"{_text(portno)}|{_text(report_date)}"] = " / ".join(managers_for_report) or "Unknown manager"
    if "event_id" in work:
        work = work.drop_duplicates("event_id", keep="first")
    else:
        work = work.drop_duplicates([c for c in ("crsp_portno", "report_date_norm", "manager") if c in work], keep="first")
    work = work.head(max_events).reset_index(drop=True)
    meta_cols = [
        "event_id", "manager", "fund", "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name",
        "report_date", "action_type", "market_regime", "manager_style_group", "training_window_years",
        "style_window_start_date", "style_window_end_date", "style_window_type", "style_obs_count",
        "delta_stock", "delta_beta", "delta_technology", "delta_bond_money", "delta_indirect_equity",
        "delta_nonstock_total_exposure", "delta_sector_exposure", "sector_rotation_intensity",
        "rolling_style_deviation_score", "rolling_sector_deviation_score",
        "rolling_cross_asset_deviation_score", "rolling_action_deviation_score",
    ] + [f"future_{h}m_excess_return" for h in (3, 6, 9, 12)] + [f"outcome_5class_{h}m" for h in (3, 6, 9, 12)]
    metadata = [{c: (None if pd.isna(row[c]) else row[c]) for c in meta_cols if c in work} for _, row in work.iterrows()]
    return FeatureBuildResult(work, metadata, warnings, str(path), domain, report_manager_map)
