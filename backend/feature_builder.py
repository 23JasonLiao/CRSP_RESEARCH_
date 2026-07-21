from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import (
    ANALYSIS_UNIT,
    DISCLOSED_ACTION_DEFINITION,
    EVENT_DEVIATION_DEFINITION,
    MANAGER_STYLE_DEFINITION,
    MANAGER_STYLE_TERM,
    PART6_CLAIM_DEFINITION,
)

FORBIDDEN_PREFIXES = (
    "future_", "direction_label_", "outcome_5class_", "label_positive_excess_",
    "label_start_date_", "label_end_date_", "label_available_",
)
AUDIT_ONLY_PREFIXES = ("current_", "rolling_dev_", "rolling_past_mean_", "rolling_past_std_")
LEGACY_FORBIDDEN_COLUMNS = {
    "manager_reliability_score", "manager_defensive_score", "manager_flow_score",
    "manager_growth_tilt_score", "style_deviation_score",
}
METADATA_COLUMNS = [
    "event_id", "manager", "fund", "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name",
    "report_date", "action_type", "market_regime", "manager_style_group", "training_window_years",
    "style_window_start_date", "style_window_end_date", "style_window_type", "style_obs_count",
    "rolling_history_available", "rolling_history_count", "rolling_history_month_count",
    "manager_history_available", "manager_history_count", "manager_history_month_count",
    "manager_score_window_start", "manager_score_window_end", "feature_source_month",
    "feature_available_at", "availability_check_passed", "data_quality_flags",
    "delta_stock", "delta_beta", "delta_technology", "delta_bond_money", "delta_indirect_equity",
    "delta_nonstock_total_exposure", "delta_sector_exposure", "sector_rotation_intensity",
    "rolling_style_deviation_score", "rolling_sector_deviation_score",
    "rolling_cross_asset_deviation_score", "rolling_action_deviation_score",
]
MODEL_FRAME_METADATA_COLUMNS = {
    "event_id", "manager", "fund", "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name",
    "report_date", "action_type", "market_regime", "manager_style_group", "training_window_years",
    "style_window_start_date", "style_window_end_date", "style_window_type",
    "rolling_history_available", "manager_history_available", "manager_score_window_start",
    "manager_score_window_end", "feature_source_month", "feature_available_at",
    "availability_check_passed", "data_quality_flags",
}
OUTCOME_COLUMNS = [
    column for horizon in (3, 6, 9, 12)
    for column in (
        f"future_{horizon}m_excess_return", f"direction_label_{horizon}m",
        f"outcome_5class_{horizon}m", f"label_available_{horizon}m",
        f"label_start_date_{horizon}m", f"label_end_date_{horizon}m",
    )
]


@dataclass
class FeatureBuildResult:
    model_frame: pd.DataFrame
    metadata: List[Dict[str, Any]]
    realized_outcomes: List[Dict[str, Any]]
    warnings: List[str]
    dataset_path: Path
    date_domain: Dict[str, str]
    report_manager_map: Dict[str, str]
    match_summary: Dict[str, Any]

    @property
    def frame(self) -> pd.DataFrame:
        """Backward-compatible alias used by the existing API route."""
        return self.model_frame


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_id(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict, tuple, set)) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d")


def _report_pair(key: str) -> Tuple[str, str]:
    parts = str(key or "").split("|")
    return (_normalize_id(parts[0]), _date(parts[1])) if len(parts) >= 2 else ("", "")


def _load_dataset(data_root: Path) -> Tuple[pd.DataFrame, Path]:
    candidates = [
        data_root / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_multi_horizon.csv",
        data_root / "derived" / "prediction" / "part6_prediction_dataset.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["report_date_norm"] = pd.to_datetime(frame.get("report_date"), errors="coerce").dt.strftime("%Y-%m-%d")
        for column in ("crsp_portno", "crsp_fundno"):
            if column in frame:
                frame[column] = frame[column].map(_normalize_id)
        if "training_window_years" in frame:
            frame = frame[pd.to_numeric(frame["training_window_years"], errors="coerce").fillna(3).astype(int).eq(3)]
        return frame, path
    raise FileNotFoundError("Run scripts/modeling/build_manager_action_groundtruth_complete.py first.")


def _selection(payload: Dict[str, Any]) -> tuple[set[Tuple[str, str]], set[str], set[str]]:
    part1, part3, part4, part5 = (payload.get(key) or {} for key in ("part1", "part3", "part4", "part5"))
    keys = list(part5.get("brushedReportKeys") or [])
    if part5.get("activeReportKey"):
        keys.append(part5["activeReportKey"])
    pairs = {_report_pair(key) for key in keys}
    for row in part5.get("selected_fund_reports") or []:
        pairs.add((
            _normalize_id(row.get("crsp_portno") or row.get("portno")),
            _date(row.get("report_dt") or row.get("report_date")),
        ))
    managers = {_text(value) for key in ("selectedManagers", "latestManagers") for value in (part3.get(key) or [])}
    managers |= {_text(row.get("manager")) for row in (part4.get("manager_records_raw") or [])}
    managers |= {_text(value) for value in (part5.get("selectedManagerNames") or [])}
    funds = {_normalize_id(value) for key in ("selectedFundIdsA", "selectedFundIdsB") for value in (part1.get(key) or [])}
    return {pair for pair in pairs if all(pair)}, {value for value in managers if value}, {value for value in funds if value}


def _load_required_features(data_root: Path) -> list[str]:
    project_root = data_root.parent
    candidates = [
        project_root / "models" / "action_effectiveness" / "v003" / "feature_columns.json",
        data_root / "derived" / "manager_action_groundtruth" / "manager_action_ground_truth_schema.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = payload.get("numeric_features") or payload.get("feature_columns") or []
        if isinstance(values, list):
            return [str(value) for value in values]
    return []


def validate_feature_schema(frame: pd.DataFrame, required_features: List[str]) -> Dict[str, List[str]]:
    forbidden = [column for column in frame if column.startswith(FORBIDDEN_PREFIXES)]
    missing = [column for column in required_features if column not in frame]
    present = [column for column in required_features if column in frame]
    numeric = frame[present].apply(pd.to_numeric, errors="coerce") if present else pd.DataFrame(index=frame.index)
    all_null = [column for column in present if numeric[column].isna().all()]
    non_numeric = [
        column for column in present
        if frame[column].notna().any() and numeric[column].notna().sum() == 0
    ]
    return {
        "missing_features": missing,
        "all_null_features": all_null,
        "non_numeric_features": non_numeric,
        "forbidden_features": forbidden,
    }


def _safe_record(row: pd.Series, columns: List[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        record[column] = None if pd.isna(value) else value
    return record


def _truthy(value: Any) -> bool | None:
    if value is None or (not isinstance(value, (list, dict, tuple, set)) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "passed", "pass"}:
        return True
    if text in {"false", "0", "no", "n", "failed", "fail"}:
        return False
    return None


def _part7_case_metadata(row: pd.Series) -> Dict[str, Any]:
    """Build a conservative event contract that Part7 can quote verbatim."""
    report_date = _date(row.get("report_date"))
    feature_available_at = _date(row.get("feature_available_at"))
    style_window_end = _date(row.get("style_window_end_date"))
    availability_flag = _truthy(row.get("availability_check_passed"))
    warnings: list[str] = []

    if not row.get("event_id"):
        warnings.append("missing_event_id")
    if not report_date:
        warnings.append("missing_report_date")
    if not feature_available_at:
        warnings.append("missing_feature_available_at")
    elif report_date and feature_available_at > report_date:
        warnings.append("feature_available_after_event_anchor")
    if not style_window_end:
        warnings.append("missing_style_window_end")
    elif report_date and style_window_end >= report_date:
        warnings.append("style_window_not_strictly_ex_ante")
    if availability_flag is False:
        warnings.append("dataset_availability_check_failed")
    elif availability_flag is None:
        warnings.append("dataset_availability_check_unverified")
    if not _truthy(row.get("manager_history_available")):
        warnings.append("manager_history_unavailable_or_unverified")
    try:
        if float(row.get("style_obs_count") or 0) <= 0:
            warnings.append("style_baseline_has_no_observations")
    except (TypeError, ValueError):
        warnings.append("style_observation_count_invalid")

    hard_failures = {
        "missing_event_id", "missing_report_date", "feature_available_after_event_anchor",
        "style_window_not_strictly_ex_ante", "dataset_availability_check_failed",
    }
    if hard_failures.intersection(warnings):
        temporal_status = "failed"
        readiness = "reject_until_temporal_or_identity_issue_is_fixed"
    elif warnings:
        temporal_status = "unverified" if any("unverified" in item or "missing" in item for item in warnings) else "passed"
        readiness = "ready_for_critic_with_explicit_data_warnings"
    else:
        temporal_status = "passed"
        readiness = "ready_for_reliability_aware_critic"

    return {
        "analysis_unit": ANALYSIS_UNIT,
        "observed_action_scope": DISCLOSED_ACTION_DEFINITION,
        "manager_style_term": MANAGER_STYLE_TERM,
        "manager_style_definition": MANAGER_STYLE_DEFINITION,
        "event_deviation_definition": EVENT_DEVIATION_DEFINITION,
        "part6_claim_definition": PART6_CLAIM_DEFINITION,
        "event_anchor_date": report_date or None,
        "part7_evidence_as_of_date": report_date or None,
        "point_in_time_feature_available_at": feature_available_at or None,
        "disclosure_availability_status": (
            "report_date_is_an_event_anchor; a separate public-disclosure available_at timestamp is not present"
        ),
        "temporal_integrity_status": temporal_status,
        "part7_case_readiness": readiness,
        "part7_case_warnings": warnings,
    }


def build_feature_frame(
    payload: Dict[str, Any], data_root: Path, max_events: int = 50,
    allow_unfiltered_fallback: bool = False,
) -> FeatureBuildResult:
    frame, path = _load_dataset(data_root)
    warnings: List[str] = []
    valid_dates = pd.to_datetime(frame["report_date_norm"], errors="coerce")
    date_domain = {
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
    date_filtered_count = int(len(work))

    pairs, managers, funds = _selection(payload)
    matched = False
    matched_by = ""
    attempts: Dict[str, int] = {"report": 0, "manager": 0, "fund": 0}
    if pairs:
        candidate_mask = work.apply(
            lambda row: (_normalize_id(row.get("crsp_portno")), _text(row.get("report_date_norm"))) in pairs,
            axis=1,
        )
        candidate = work[candidate_mask]
        attempts["report"] = int(len(candidate))
        if not candidate.empty:
            work, matched, matched_by = candidate, True, "report"
        else:
            warnings.append("Selected report keys did not match; trying manager fallback.")
    if not matched and managers and "manager" in work:
        candidate = work[work["manager"].astype(str).str.strip().isin(managers)]
        attempts["manager"] = int(len(candidate))
        if not candidate.empty:
            work, matched, matched_by = candidate, True, "manager"
        else:
            warnings.append("Selected managers did not match; trying fund fallback.")
    if not matched and funds and "crsp_fundno" in work:
        candidate = work[work["crsp_fundno"].map(_normalize_id).isin(funds)]
        attempts["fund"] = int(len(candidate))
        if not candidate.empty:
            work, matched, matched_by = candidate, True, "fund"
        else:
            warnings.append("Selected funds did not match the Part 6 event table.")

    selectors_present = bool(pairs or managers or funds)
    if not matched and not (allow_unfiltered_fallback and not selectors_present):
        warnings.append("No selected event matched; unfiltered fallback is disabled.")
        work = frame.iloc[0:0].copy()
        match_status = "failed"
    elif not matched:
        work = work.copy()
        matched_by = "unfiltered"
        match_status = "unfiltered_allowed"
        warnings.append("No selection was supplied; an explicitly allowed unfiltered frame is being used.")
    else:
        match_status = "matched"

    work = work.sort_values(["report_date_norm", "event_id"] if "event_id" in work else ["report_date_norm"], ascending=False)
    if "event_id" in work:
        work = work.drop_duplicates("event_id", keep="first")
    else:
        keys = [column for column in ("crsp_portno", "report_date_norm", "manager") if column in work]
        work = work.drop_duplicates(keys, keep="first")
    work = work.head(max_events).reset_index(drop=True)

    metadata = []
    for _, row in work.iterrows():
        record = _safe_record(row, METADATA_COLUMNS)
        record.update(_part7_case_metadata(row))
        metadata.append(record)
    realized_outcomes = [_safe_record(row, ["event_id", *OUTCOME_COLUMNS]) for _, row in work.iterrows()]
    excluded = {
        column for column in work
        if column.startswith(FORBIDDEN_PREFIXES + AUDIT_ONLY_PREFIXES)
        or column in LEGACY_FORBIDDEN_COLUMNS
        or column in MODEL_FRAME_METADATA_COLUMNS
        or column == "report_date_norm"
    }
    required_features = _load_required_features(data_root)
    candidate_frame = work.drop(columns=sorted(excluded), errors="ignore").copy()
    if required_features:
        model_frame = candidate_frame[
            [column for column in required_features if column in candidate_frame]
        ].copy()
    else:
        # A missing schema is reported below.  Until it is restored, keep only
        # columns that contain at least one numeric value so metadata cannot
        # accidentally become an inference feature.
        model_frame = candidate_frame[
            [
                column for column in candidate_frame
                if pd.to_numeric(candidate_frame[column], errors="coerce").notna().any()
            ]
        ].copy()
        warnings.append("No feature schema was found; model_frame is restricted to numeric candidates.")
    schema_validation = validate_feature_schema(model_frame, required_features)
    for issue, columns in schema_validation.items():
        if columns:
            warnings.append(f"Feature schema {issue}: {columns}")

    report_manager_map: Dict[str, str] = {}
    if all(column in work for column in ("crsp_portno", "report_date_norm", "manager")):
        manager_rows = work[["crsp_portno", "report_date_norm", "manager"]].dropna().drop_duplicates()
        for (portno, report_date), group in manager_rows.groupby(["crsp_portno", "report_date_norm"], sort=False):
            values = sorted({_text(value) for value in group["manager"] if _text(value)})
            report_manager_map[f"{_normalize_id(portno)}|{_date(report_date)}"] = " / ".join(values) or "Unknown manager"

    match_summary = {
        "status": match_status, "matched_by": matched_by,
        "date_filtered_rows": date_filtered_count, "attempt_rows": attempts,
        "returned_rows": int(len(work)), "selectors_present": selectors_present,
        "allow_unfiltered_fallback": bool(allow_unfiltered_fallback),
        "schema_validation": schema_validation,
        "analysis_unit": ANALYSIS_UNIT,
        "manager_style_term": MANAGER_STYLE_TERM,
        "part7_case_readiness_counts": {
            str(key): int(value) for key, value in pd.Series(
                [record["part7_case_readiness"] for record in metadata], dtype="object"
            ).value_counts().items()
        },
    }
    if not (len(model_frame) == len(metadata) == len(realized_outcomes)):
        raise AssertionError("model_frame, metadata, and realized_outcomes lost row alignment.")
    return FeatureBuildResult(
        model_frame=model_frame, metadata=metadata, realized_outcomes=realized_outcomes,
        warnings=warnings, dataset_path=path, date_domain=date_domain,
        report_manager_map=report_manager_map, match_summary=match_summary,
    )
