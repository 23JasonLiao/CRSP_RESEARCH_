from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from backend.feature_builder import build_feature_frame
    from backend.prediction_service import predict_events
    from backend.shap_service import explain_events
    BACKEND_IMPORT_ERROR = None
except Exception as exc:  # keep Part1-Part5 usable even if backend folder is missing
    build_feature_frame = None
    predict_events = None
    explain_events = None
    BACKEND_IMPORT_ERROR = exc

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
BACKEND_PAYLOADS_DIR = OUTPUTS_DIR / "backend_payloads"


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _alias(filename: str, *relative_candidates: str) -> Path:
    candidates = [DATA_DIR / rel for rel in relative_candidates]
    candidates += [DATA_DIR / filename, BASE_DIR / filename]
    return _first_existing(*candidates)


CSV_ALIASES: Dict[str, Path] = {
    "balanced_before2010.csv": _alias("balanced_before2010.csv", "crsp/fund_level/balanced_before2010.csv", "fund_level/balanced_before2010.csv"),
    "balanced_after2010.csv": _alias("balanced_after2010.csv", "crsp/fund_level/balanced_after2010.csv", "fund_level/balanced_after2010.csv"),
    "sp500_monthly_returns_1871_2026.csv": _alias("sp500_monthly_returns_1871_2026.csv", "market/sp500_monthly_returns_1871_2026.csv"),
    "FRB_H15.csv": _alias("FRB_H15.csv", "market/FRB_H15.csv"),
    "part5_yearly_trailing_stock_beta.csv": _alias("part5_yearly_trailing_stock_beta.csv", "part5_equity_beta/part5_yearly_trailing_stock_beta.csv"),
    "stock berfore 2010_new___.csv": _alias("stock berfore 2010_new___.csv", "crsp/holdings_raw/stock berfore 2010_new___.csv", "holdings_raw/stock berfore 2010_new___.csv"),
    "stock between 2010_2014_new___.csv": _alias("stock between 2010_2014_new___.csv", "crsp/holdings_raw/stock between 2010_2014_new___.csv", "holdings_raw/stock between 2010_2014_new___.csv"),
    "stock between 2015_2019_new___.csv": _alias("stock between 2015_2019_new___.csv", "crsp/holdings_raw/stock between 2015_2019_new___.csv", "holdings_raw/stock between 2015_2019_new___.csv"),
    "stock between 2020_2026_new___.csv": _alias("stock between 2020_2026_new___.csv", "crsp/holdings_raw/stock between 2020_2026_new___.csv", "holdings_raw/stock between 2020_2026_new___.csv"),
    "part5_excluded_non_company_holdings_audit.csv": _alias("part5_excluded_non_company_holdings_audit.csv", "part5_non_individual_holdings/part5_excluded_non_company_holdings_audit.csv"),
    "part5_excluded_two_group_enriched.csv": _alias("part5_excluded_two_group_enriched.csv", "part5_non_individual_holdings/part5_excluded_two_group_enriched.csv"),
    "part5_excluded_two_group_summary.csv": _alias("part5_excluded_two_group_summary.csv", "part5_non_individual_holdings/part5_excluded_two_group_summary.csv"),
    "part5_excluded_two_group_top_items.csv": _alias("part5_excluded_two_group_top_items.csv", "part5_non_individual_holdings/part5_excluded_two_group_top_items.csv"),
    "part5_excluded_two_group_active_year_panel.csv": _alias("part5_excluded_two_group_active_year_panel.csv", "part5_non_individual_holdings/part5_excluded_two_group_active_year_panel.csv"),
    "part5_excluded_individual_stock_like_removed_audit.csv": _alias("part5_excluded_individual_stock_like_removed_audit.csv", "part5_non_individual_holdings/part5_excluded_individual_stock_like_removed_audit.csv"),
}

app = FastAPI(
    title="Balanced Fund Visual Analytics API - 3Y Backend ML",
    version="0.2.1",
    description="Serves frontend assets and CSVs, saves Part1-Part5 JSON, and runs 3Y backend ML/SHAP when backend artifacts exist.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeVisualStateRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None
    part1: Optional[Dict[str, Any]] = None
    part2: Optional[Dict[str, Any]] = None
    part3: Optional[Dict[str, Any]] = None
    part4: Optional[Dict[str, Any]] = None
    part5: Optional[Dict[str, Any]] = None
    part6: Optional[Dict[str, Any]] = None


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _compact_list(values: Any, limit: int = 10) -> list[Any]:
    return values[:limit] if isinstance(values, list) else []


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_visual_state_payload(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    BACKEND_PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = BACKEND_PAYLOADS_DIR / f"visual_state_{timestamp}.json"
    latest_path = BACKEND_PAYLOADS_DIR / "visual_state_latest.json"
    save_json_file(output_path, payload_dict)
    save_json_file(latest_path, payload_dict)

    split_paths: Dict[str, str] = {}
    for part_name in ["part1", "part2", "part3", "part4", "part5"]:
        part_payload = {"context": payload_dict.get("context", {}), "part": part_name, "data": payload_dict.get(part_name, {})}
        part_output_path = BACKEND_PAYLOADS_DIR / f"visual_state_{timestamp}_{part_name}.json"
        part_latest_path = BACKEND_PAYLOADS_DIR / f"{part_name}_latest.json"
        save_json_file(part_output_path, part_payload)
        save_json_file(part_latest_path, part_payload)
        split_paths[f"{part_name}_path"] = str(part_output_path.relative_to(BASE_DIR))
        split_paths[f"{part_name}_latest_path"] = str(part_latest_path.relative_to(BASE_DIR))
    return {"saved_payload_path": str(output_path.relative_to(BASE_DIR)), "latest_payload_path": str(latest_path.relative_to(BASE_DIR)), "split_part_paths": split_paths}


def parse_visual_state(payload: AnalyzeVisualStateRequest) -> Dict[str, Any]:
    context = payload.context or {}
    part1 = payload.part1 or {}
    part2 = payload.part2 or {}
    part3 = payload.part3 or {}
    part4 = payload.part4 or {}
    part5 = payload.part5 or {}
    part6 = payload.part6 or {}
    reports = part5.get("reports", []) or []
    holdings = part5.get("holdings_detail_all", []) or part5.get("holdings", []) or []
    tables = part2.get("tables") or {}
    return {
        "context": {
            "horizon": context.get("horizon", "y3"),
            "training_window_years": ((context.get("backend_feature_context") or {}).get("training_window_years", 3)),
            "selection_mode": context.get("selectionMode"),
            "timestamp": context.get("timestamp"),
        },
        "part1": {
            "raw_a_count": part1.get("rawA_count", 0),
            "raw_b_count": part1.get("rawB_count", 0),
            "selected_funds_a_count": _safe_len(part1.get("selectedFundIdsA")),
            "selected_funds_b_count": _safe_len(part1.get("selectedFundIdsB")),
        },
        "part2": {
            "logic": part2.get("logic"),
            "region_count": _safe_len(part2.get("regions")),
            "monthly_rows_a": _safe_len(tables.get("monthly_A")),
            "fund_rows_a": _safe_len(tables.get("fund_level_A")),
            "family_rows_a": _safe_len(tables.get("family_level_A")),
            "regions_sample": _compact_list(part2.get("regions"), 5),
        },
        "part3": {
            "selected_manager_count": _safe_len(part3.get("selectedManagers")),
            "selected_manager_sample": _compact_list(part3.get("selectedManagers")),
            "latest_manager_sample": _compact_list(part3.get("latestManagers")),
            "part3_a_count": part3.get("part3A_count", 0),
            "part3_b_count": part3.get("part3B_count", 0),
        },
        "part4": {
            "manager_record_count": _safe_len(part4.get("manager_records_raw")),
            "style_drift_row_count": _safe_len(part4.get("style_drift_rows")),
        },
        "part5": {
            "analysis_mode": part5.get("analysisMode"),
            "report_count": _safe_len(reports),
            "holding_count": _safe_len(holdings),
            "selected_report_count": _safe_len(part5.get("selected_fund_reports")),
            "brushed_report_count": _safe_len(part5.get("brushedReportKeys")),
            "active_report_key": part5.get("activeReportKey"),
        },
        "part6": {"mode": part6.get("mode", "backend"), "window": part6.get("window", "y3"), "target": part6.get("target", "future12m")},
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "base_dir": str(BASE_DIR),
        "static_dir_exists": STATIC_DIR.exists(),
        "data_dir_exists": DATA_DIR.exists(),
        "backend_import_ok": BACKEND_IMPORT_ERROR is None,
        "backend_import_error": None if BACKEND_IMPORT_ERROR is None else repr(BACKEND_IMPORT_ERROR),
    }


@app.get("/api/files")
def list_known_files() -> Dict[str, Any]:
    files = []
    for public_name, path in CSV_ALIASES.items():
        files.append({"public_name": public_name, "path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None})
    model_dir = BASE_DIR / "models" / "action_effectiveness" / "v001"
    extra = [
        DATA_DIR / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_future12m.csv",
        model_dir / "lightgbm_action_model_trailing3y.pkl",
        model_dir / "feature_columns.json",
    ]
    return {"base_dir": str(BASE_DIR), "files": files, "backend_artifacts": [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else None} for p in extra]}


@app.post("/api/ml/analyze-visual-state")
def analyze_visual_state(payload: AnalyzeVisualStateRequest) -> Dict[str, Any]:
    payload_dict = payload.model_dump()
    saved_files = save_visual_state_payload(payload_dict)
    parsed_state = parse_visual_state(payload)
    warnings: list[str] = []
    ml_result: Dict[str, Any] | None = None
    shap_result: Dict[str, Any] | None = None
    feature_dataset_path = ""

    if BACKEND_IMPORT_ERROR is not None:
        warnings.append(f"Backend modules are not importable: {BACKEND_IMPORT_ERROR}")
    else:
        try:
            build = build_feature_frame(payload_dict, DATA_DIR, max_events=50)  # type: ignore[misc]
            warnings.extend(build.warnings)
            feature_dataset_path = build.dataset_path
            ml_result = predict_events(build.frame, build.metadata, BASE_DIR, model_name="lightgbm")  # type: ignore[misc]
            shap_result = explain_events(build.frame, build.metadata, BASE_DIR, model_name="lightgbm", top_k=8)  # type: ignore[misc]
        except Exception as exc:
            warnings.append(f"ML/SHAP pipeline did not complete: {exc}")

    result = {
        "status": "ok" if ml_result else "partial",
        "mode": "3y_backend_ml_shap",
        "message": "Backend saved Part1-Part5 JSON and attempted 3Y ML/SHAP analysis.",
        "saved_files": saved_files,
        "parsed_state": parsed_state,
        "received_summary": {
            "horizon": "y3",
            "target": "label_positive_excess_12m",
            "part1_a_count": parsed_state["part1"]["raw_a_count"],
            "part1_b_count": parsed_state["part1"]["raw_b_count"],
            "part2_region_count": parsed_state["part2"]["region_count"],
            "part3_selected_manager_count": parsed_state["part3"]["selected_manager_count"],
            "part4_manager_count": parsed_state["part4"]["manager_record_count"],
            "part4_style_drift_row_count": parsed_state["part4"]["style_drift_row_count"],
            "part5_report_count": parsed_state["part5"]["report_count"],
            "part5_selected_report_count": parsed_state["part5"]["selected_report_count"],
            "feature_dataset_path": feature_dataset_path,
        },
        "ml_result": ml_result,
        "shap_result": shap_result,
        "warnings": warnings,
    }
    save_json_file(BACKEND_PAYLOADS_DIR / "backend_ml_latest.json", result)
    return result


@app.get("/{filename:path}")
def serve_file(filename: str):
    if filename in ("", "/"):
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="static/index.html not found")
    static_path = STATIC_DIR / filename
    if static_path.exists() and static_path.is_file():
        return FileResponse(static_path)
    if filename in CSV_ALIASES:
        path = CSV_ALIASES[filename]
        if path.exists() and path.is_file():
            return FileResponse(path)
        raise HTTPException(status_code=404, detail=f"File alias exists but target file not found: {filename} -> {path}")
    raise HTTPException(status_code=404, detail=f"File not found: {filename}")
