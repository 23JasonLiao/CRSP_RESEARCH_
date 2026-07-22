from __future__ import annotations

import json
import hashlib
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import backend.part7_rag_service as part7_service
    from backend.part7_rag_service import part7_status, run_part7_critic
    PART7_IMPORT_ERROR = None
except Exception as exc:
    part7_status = None
    run_part7_critic = None
    part7_service = None
    PART7_IMPORT_ERROR = exc

try:
    from backend.feature_builder import build_feature_frame
    from backend.prediction_service import predict_events
    from backend.shap_service import explain_events
    from backend.expert_collaboration_service import build_expert_collaboration
    BACKEND_IMPORT_ERROR = None
except Exception as exc:  # keep Part1-Part5 usable even if backend folder is missing
    build_feature_frame = None
    predict_events = None
    explain_events = None
    build_expert_collaboration = None
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
    title="Balanced Fund Decision Analytics API",
    version="0.4.0",
    description=(
        "Persists auditable Part1-Part5 research snapshots, produces fallible Part6 model claims, "
        "and orchestrates a reliability-aware Part7 evidence critic."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlexibleState(BaseModel):
    model_config = ConfigDict(extra="allow")


class SnapshotContext(FlexibleState):
    schema_version: str = "visual-state-v2"
    snapshot_id: Optional[str] = None
    generated_at: Optional[str] = None
    timezone: str = "Asia/Taipei"
    horizon: str = "y3"
    selectionMode: Optional[str] = None
    research_purpose: str = "human_guided_manager_action_case_audit"
    analysis_unit: str = "manager × fund/portfolio × report_date disclosed-action event"


class Part1State(FlexibleState):
    rawA_count: int = Field(default=0, ge=0)
    rawB_count: int = Field(default=0, ge=0)
    selectedFundIdsA: list[Any] = Field(default_factory=list)
    selectedFundIdsB: list[Any] = Field(default_factory=list)


class Part2State(FlexibleState):
    logic: Optional[Literal["and", "or"]] = None
    regions: list[Dict[str, Any]] = Field(default_factory=list)


class Part3State(FlexibleState):
    selectedManagers: list[str] = Field(default_factory=list)
    latestManagers: list[str] = Field(default_factory=list)


class Part4State(FlexibleState):
    manager_records_raw: list[Dict[str, Any]] = Field(default_factory=list)
    style_drift_rows: list[Dict[str, Any]] = Field(default_factory=list)


class Part5State(FlexibleState):
    loaded: bool = False
    analysisMode: Optional[str] = None
    activeReportKey: Optional[str] = None
    brushedReportKeys: list[str] = Field(default_factory=list)
    selected_fund_reports: list[Dict[str, Any]] = Field(default_factory=list)
    stock_action_rows: list[Dict[str, Any]] = Field(default_factory=list)


class Part6State(FlexibleState):
    mode: str = "backend"
    horizon_months: int = 12
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    @field_validator("horizon_months")
    @classmethod
    def validate_horizon(cls, value: int) -> int:
        if value not in (3, 6, 9, 12):
            raise ValueError("horizon_months must be one of 3, 6, 9, 12")
        return value


class AnalyzeVisualStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: SnapshotContext = Field(default_factory=SnapshotContext)
    part1: Part1State = Field(default_factory=Part1State)
    part2: Part2State = Field(default_factory=Part2State)
    part3: Part3State = Field(default_factory=Part3State)
    part4: Part4State = Field(default_factory=Part4State)
    part5: Part5State = Field(default_factory=Part5State)
    part6: Part6State = Field(default_factory=Part6State)


class Part7CriticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: Optional[str] = None
    event_ids: list[str] = Field(default_factory=list, max_length=8)
    horizon_months: int = 12
    model: Optional[str] = None
    use_web_search: bool = True
    max_local_chunks: int = Field(default=12, ge=0, le=30)
    question: Optional[str] = None
    orchestration_mode: Literal["auto", "pydantic_ai", "openai_responses"] = "auto"
    evidence_scope: Literal["macro_and_micro", "macro_only", "micro_only"] = "macro_and_micro"
    strict_temporal: bool = True
    require_counterevidence: bool = True
    ticker_focus: list[str] = Field(default_factory=list, max_length=25)

    @field_validator("horizon_months")
    @classmethod
    def validate_part7_horizon(cls, value: int) -> int:
        if value not in (3, 6, 9, 12):
            raise ValueError("horizon_months must be one of 3, 6, 9, 12")
        return value


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    evidence_ids: list[str]
    reasoning: str
    strength: Literal["weak", "moderate", "strong"]


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk: str
    why_it_matters: str
    evidence_ids: list[str]
    recommended_check: str


class CriticCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    title: str
    source: str
    date: str
    url: str


class CriticAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_not_recomputed: bool
    support_and_counterevidence_searched: bool
    uncited_claim_count: int = Field(ge=0)
    notes: str


class Part7CriticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executive_summary: str
    model_claim: str
    supporting_evidence: list[EvidenceAssessment]
    counter_evidence: list[EvidenceAssessment]
    structural_breaks: list[RiskAssessment]
    data_limitations: list[RiskAssessment]
    overinterpretation_risks: list[RiskAssessment]
    questions_for_human: list[str]
    verdict: Literal["supported", "mixed", "contradicted", "insufficient_evidence"]
    confidence: float = Field(ge=0, le=1)
    citations: list[CriticCitation]
    audit: CriticAudit


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _compact_list(values: Any, limit: int = 10) -> list[Any]:
    return values[:limit] if isinstance(values, list) else []


def _canonical_hash(data: Any) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    """Serialize infrastructure values without silently stringifying arbitrary objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value, key=str)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _snapshot_validation(payload: Dict[str, Any]) -> Dict[str, Any]:
    part1, part2, part3, part4, part5, part6 = (
        payload.get(name) or {} for name in ("part1", "part2", "part3", "part4", "part5", "part6")
    )
    report_keys = list(dict.fromkeys([
        *[str(value) for value in (part5.get("brushedReportKeys") or []) if value],
        *([str(part5.get("activeReportKey"))] if part5.get("activeReportKey") else []),
    ]))
    warnings: list[str] = []
    if not report_keys and not (part5.get("selected_fund_reports") or []):
        warnings.append("part5_has_no_selected_report_anchor")
    if not (part3.get("selectedManagers") or part3.get("latestManagers") or part5.get("selectedManagerNames")):
        warnings.append("manager_selection_is_empty")
    if not part4.get("manager_records_raw"):
        warnings.append("manager_archetype_context_is_empty")
    if not part5.get("stock_action_rows"):
        warnings.append("stock_action_rows_are_empty_or_no_previous_report_exists")
    return {
        "status": "ready_with_warnings" if warnings else "ready",
        "warnings": warnings,
        "counts": {
            "part1_selected_funds_a": len(part1.get("selectedFundIdsA") or []),
            "part1_selected_funds_b": len(part1.get("selectedFundIdsB") or []),
            "part2_regions": len(part2.get("regions") or []),
            "part3_selected_managers": len(part3.get("selectedManagers") or []),
            "part4_manager_records": len(part4.get("manager_records_raw") or []),
            "part5_selected_reports": len(part5.get("selected_fund_reports") or []),
            "part5_report_keys": len(report_keys),
            "part5_stock_actions": len(part5.get("stock_action_rows") or []),
        },
        "selected_report_keys": report_keys,
        "selected_horizon_months": part6.get("horizon_months", 12),
    }


def _read_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _model_reliability_summary() -> Dict[str, Any]:
    """Expose the training audit without making the UI parse a very large report."""
    report_path = BASE_DIR / "models" / "action_effectiveness" / "v003" / "stability_report.json"
    report = _read_json_object(report_path)
    horizons: Dict[str, Any] = {}
    for horizon, details in (report.get("by_horizon") or {}).items():
        selection = details.get("window_selection_validation_summary") or {}
        selected_window = selection.get("selected_window")
        selected = (selection.get("candidates") or {}).get(selected_window, {})
        outer = details.get("outer_test_summary") or {}
        manager = details.get("manager_action_feature_conclusion") or {}
        horizons[str(horizon)] = {
            "horizon": details.get("horizon", f"{horizon}m"),
            "status": details.get("horizon_model_status"),
            "feature_count": details.get("feature_count"),
            "selected_window": selected_window,
            "selection_score": selection.get("selected_window_score"),
            "inner_validation": {
                "mean_auc": selected.get("mean_inner_validation_auc"),
                "worst_auc": selected.get("worst_inner_validation_auc"),
                "mean_brier": selected.get("mean_inner_validation_brier"),
            },
            "outer_test": {
                "mean_auc": outer.get("mean_outer_test_auc"),
                "worst_auc": outer.get("worst_outer_test_auc"),
                "mean_brier": outer.get("mean_outer_test_brier"),
                "mean_spearman": outer.get("mean_outer_test_spearman"),
                "calibration_warning_count": outer.get("outer_test_calibration_warning_count"),
            },
            "direction_threshold": details.get("direction_threshold"),
            "manager_action_contribution": manager.get("manager_action_contribution_status"),
            "mean_incremental_manager_action_auc": manager.get("mean_incremental_manager_action_auc"),
            "five_class_status": details.get("five_class_status"),
            "five_class_production_ready": details.get("five_class_production_ready"),
        }
    return {
        "available": bool(horizons),
        "bundle_version": report.get("bundle_version"),
        "evaluation_protocol": report.get("evaluation_protocol"),
        "outer_test_used_for_selection": report.get("outer_test_used_for_selection"),
        "report_path": str(report_path.relative_to(BASE_DIR)),
        "interpretation": "Part6 is a fallible, horizon-specific quantitative claim; scores are not investment recommendations.",
        "horizons": horizons,
    }


def _pydantic_ai_available() -> bool:
    return importlib.util.find_spec("pydantic_ai") is not None


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_visual_state_payload(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    BACKEND_PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    generated_at = now.isoformat().replace("+00:00", "Z")
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    payload_hash = _canonical_hash(payload_dict)
    client_snapshot_id = str((payload_dict.get("context") or {}).get("snapshot_id") or "").strip()
    snapshot_id = client_snapshot_id or f"vs_{timestamp}_{payload_hash[:12]}"
    validation = _snapshot_validation(payload_dict)
    snapshot = {
        "schema_version": "visual-state-v2",
        "snapshot_id": snapshot_id,
        "generated_at_utc": generated_at,
        "payload_sha256": payload_hash,
        "analysis_unit": "manager × fund/portfolio × report_date disclosed-action event",
        "research_purpose": "human_guided_manager_action_case_audit",
        "prediction_is_final_decision": False,
        "validation": validation,
    }
    persisted_payload = dict(payload_dict)
    persisted_payload["_snapshot"] = snapshot
    persisted_payload["context"] = {
        **(persisted_payload.get("context") or {}),
        "schema_version": "visual-state-v2",
        "snapshot_id": snapshot_id,
        "generated_at": generated_at,
    }
    output_path = BACKEND_PAYLOADS_DIR / f"visual_state_{timestamp}.json"
    latest_path = BACKEND_PAYLOADS_DIR / "visual_state_latest.json"
    save_json_file(output_path, persisted_payload)
    save_json_file(latest_path, persisted_payload)

    split_paths: Dict[str, str] = {}
    upstream = {
        "part1": [], "part2": ["part1"], "part3": ["part1", "part2"],
        "part4": ["part1", "part2", "part3"], "part5": ["part1", "part2", "part3", "part4"],
    }
    for part_name in ["part1", "part2", "part3", "part4", "part5"]:
        part_data = persisted_payload.get(part_name, {})
        part_payload = {
            "schema_version": "visual-state-part-v2",
            "snapshot_id": snapshot_id,
            "generated_at_utc": generated_at,
            "part": part_name,
            "upstream_parts": upstream[part_name],
            "content_sha256": _canonical_hash(part_data),
            "context": persisted_payload.get("context", {}),
            "snapshot_validation": validation,
            "data": part_data,
        }
        part_output_path = BACKEND_PAYLOADS_DIR / f"visual_state_{timestamp}_{part_name}.json"
        part_latest_path = BACKEND_PAYLOADS_DIR / f"{part_name}_latest.json"
        save_json_file(part_output_path, part_payload)
        save_json_file(part_latest_path, part_payload)
        split_paths[f"{part_name}_path"] = str(part_output_path.relative_to(BASE_DIR))
        split_paths[f"{part_name}_latest_path"] = str(part_latest_path.relative_to(BASE_DIR))
    return {
        "saved_payload_path": str(output_path.relative_to(BASE_DIR)),
        "latest_payload_path": str(latest_path.relative_to(BASE_DIR)),
        "split_part_paths": split_paths,
        "snapshot": snapshot,
    }


def parse_visual_state(payload: AnalyzeVisualStateRequest) -> Dict[str, Any]:
    raw = payload.model_dump()
    context = raw.get("context") or {}
    part1 = raw.get("part1") or {}
    part2 = raw.get("part2") or {}
    part3 = raw.get("part3") or {}
    part4 = raw.get("part4") or {}
    part5 = raw.get("part5") or {}
    part6 = raw.get("part6") or {}
    reports = part5.get("reports", []) or []
    holdings = part5.get("holdings_detail_all", []) or part5.get("holdings", []) or []
    tables = part2.get("tables") or {}
    return {
        "context": {
            "horizon": context.get("horizon", "y3"),
            "training_window_years": ((context.get("backend_feature_context") or {}).get("training_window_years", 3)),
            "selection_mode": context.get("selectionMode"),
            "timestamp": context.get("timestamp"),
            "schema_version": context.get("schema_version"),
            "snapshot_id": context.get("snapshot_id"),
            "research_purpose": context.get("research_purpose"),
            "analysis_unit": context.get("analysis_unit"),
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
            "stock_action_count": _safe_len(part5.get("stock_action_rows")),
            "availability_contract": part5.get("availability_contract"),
        },
        "part6": {
            "mode": part6.get("mode", "backend"), "window": part6.get("window", "y3"),
            "target": part6.get("target", "future12m"), "horizon_months": part6.get("horizon_months", 12),
            "date_start": part6.get("date_start"), "date_end": part6.get("date_end"),
            "claim_contract": part6.get("claim_contract"),
        },
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
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "pydantic_ai_available": _pydantic_ai_available(),
    }


@app.get("/api/files")
def list_known_files() -> Dict[str, Any]:
    files = []
    for public_name, path in CSV_ALIASES.items():
        files.append({"public_name": public_name, "path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None})
    model_dir = BASE_DIR / "models" / "action_effectiveness" / "v001"
    extra = [
        DATA_DIR / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_multi_horizon.csv",
        BASE_DIR / "models" / "action_effectiveness" / "v002" / "dual_stage_model_12m.pkl",
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
    expert_collaboration: Dict[str, Any] | None = None
    feature_dataset_path = ""
    date_domain: Dict[str, str] = {}

    if BACKEND_IMPORT_ERROR is not None:
        warnings.append(f"Backend modules are not importable: {BACKEND_IMPORT_ERROR}")
    else:
        try:
            build = build_feature_frame(payload_dict, DATA_DIR, max_events=50)  # type: ignore[misc]
            warnings.extend(build.warnings)
            try:
                feature_dataset_path = str(build.dataset_path.relative_to(BASE_DIR))
            except ValueError:
                feature_dataset_path = str(build.dataset_path)
            date_domain = build.date_domain
        except Exception as exc:
            build = None
            warnings.append(f"Part 6 feature matching did not complete: {exc}")

        if build is not None:
            try:
                ml_result = predict_events(build.frame, build.metadata, BASE_DIR, model_name="xgboost")  # type: ignore[misc]
                fallback_horizons = [
                    h for h, info in (ml_result.get("models") or {}).items()
                    if "fallback" in str((info or {}).get("classifier_type", "")).lower()
                ]
                if fallback_horizons:
                    warnings.append(
                        "XGBoost is unavailable; Part 6 is using the portable sklearn tree bundles "
                        f"for {', '.join(fallback_horizons)}M. Install XGBoost and restart the API to use the primary models."
                    )
            except Exception as exc:
                warnings.append(f"Part 6 prediction did not complete: {exc}")

        if build is not None and ml_result is not None:
            selected_horizon = int((payload_dict.get("part6") or {}).get("horizon_months", 12))
            if selected_horizon not in (3, 6, 9, 12):
                selected_horizon = 12
            try:
                expert_collaboration = build_expert_collaboration(  # type: ignore[misc]
                    build.frame, build.metadata, ml_result, payload_dict, selected_horizon,
                )
                expert_collaboration.setdefault("report_manager_map", {}).update(build.report_manager_map)
            except Exception as exc:
                warnings.append(f"Part 6 expert-collaboration analysis did not complete: {exc}")
            try:
                shap_result = explain_events(
                    build.frame, build.metadata, BASE_DIR, model_name="xgboost", top_k=8,
                    selected_horizon=selected_horizon, prediction_result=ml_result,
                )  # type: ignore[misc]
                if shap_result.get("method") != "TreeSHAP":
                    warnings.append(
                        "The SHAP package is unavailable; Part 6 is showing portable tree-importance contributions. "
                        "Install SHAP and restart the API for exact TreeSHAP."
                    )
            except Exception as exc:
                warnings.append(f"Part 6 explanation/clustering did not complete: {exc}")

    used_fallback = bool(ml_result) and any(
        "fallback" in str((info or {}).get("classifier_type", "")).lower()
        for info in ((ml_result or {}).get("models") or {}).values()
    )
    if ml_result:
        message = (
            "Part 6 已連結 Part 5 選取事件；目前使用 sklearn portable 模型。安裝 XGBoost 後請重啟 API。"
            if used_fallback else
            "Part 6 已連結 Part 5 選取事件並完成 3M/6M/9M/12M 分析。"
        )
    else:
        message = "Part 6 未能產生預測，請查看下方錯誤訊息。"

    result = {
        "status": "ok" if ml_result else "partial",
        "mode": "multi_horizon_dual_stage_tree_shap",
        "message": message,
        "saved_files": saved_files,
        "snapshot": saved_files["snapshot"],
        "snapshot_validation": saved_files["snapshot"]["validation"],
        "research_contract": {
            "analysis_unit": "manager × fund/portfolio × report_date disclosed-action event",
            "point_in_time_features": True,
            "prediction_is_final_decision": False,
            "part7_must_audit_with_contemporaneous_evidence": True,
        },
        "model_reliability": _model_reliability_summary(),
        "parsed_state": parsed_state,
        "received_summary": {
            "style_window": "trailing_3y_ex_ante",
            "prediction_horizons_months": [3, 6, 9, 12],
            "date_domain": date_domain,
            "part1_a_count": parsed_state["part1"]["raw_a_count"],
            "part1_b_count": parsed_state["part1"]["raw_b_count"],
            "part2_region_count": parsed_state["part2"]["region_count"],
            "part3_selected_manager_count": parsed_state["part3"]["selected_manager_count"],
            "part4_manager_count": parsed_state["part4"]["manager_record_count"],
            "part4_style_drift_row_count": parsed_state["part4"]["style_drift_row_count"],
            "part5_report_count": parsed_state["part5"]["report_count"],
            "part5_selected_report_count": parsed_state["part5"]["selected_report_count"],
            "feature_dataset_path": feature_dataset_path,
            "snapshot_id": saved_files["snapshot"]["snapshot_id"],
            "snapshot_readiness": saved_files["snapshot"]["validation"]["status"],
            "matched_event_count": len(build.metadata) if build is not None else 0,
            "event_match_summary": build.match_summary if build is not None else {},
        },
        "ml_result": ml_result,
        "shap_result": shap_result,
        "expert_collaboration": expert_collaboration,
        "warnings": warnings,
    }
    save_json_file(BACKEND_PAYLOADS_DIR / "backend_ml_latest.json", result)
    return result


def _part7_context(payload: Part7CriticRequest) -> Dict[str, Any]:
    if part7_service is None:
        raise RuntimeError(f"Part7 service unavailable: {PART7_IMPORT_ERROR}")
    payload_dir = BASE_DIR / "outputs" / "backend_payloads"
    visual = part7_service._read_json(payload_dir / "visual_state_latest.json")
    backend = part7_service._read_json(payload_dir / "backend_ml_latest.json")
    predictions = (backend.get("ml_result") or {}).get("predictions") or []
    available = {str(row.get("event_id")) for row in predictions if row.get("event_id")}
    requested: list[str] = []
    for value in payload.event_ids or ([payload.event_id] if payload.event_id else []):
        clean = str(value or "").strip()
        if clean and clean in available and clean not in requested:
            requested.append(clean)
    requested = requested[:8]
    documents: list[Dict[str, Any]] = []
    events: list[Dict[str, Any]] = []
    for event_id in requested or [None]:
        event_docs, event = part7_service._visual_evidence(visual, backend, event_id, payload.horizon_months)
        if event.get("event_id") and event.get("event_id") not in {x.get("event_id") for x in events}:
            events.append(event)
        documents.extend(event_docs)
    deduplicated = list({(x.get("title"), x.get("date"), x.get("text")): x for x in documents}.values())

    local_docs = part7_service._load_local_documents(BASE_DIR / "data" / "part7_knowledge")
    macro_terms = ("macro", "fomc", "rate", "inflation", "gdp", "yield", "recession", "market")
    micro_terms = ("ticker", "company", "stock", "holding", "industry", "earnings", "fund", "manager")
    if payload.evidence_scope != "macro_and_micro":
        wanted = macro_terms if payload.evidence_scope == "macro_only" else micro_terms
        local_docs = [doc for doc in local_docs if any(term in f"{doc.get('type')} {doc.get('title')} {doc.get('text')}".lower() for term in wanted)]

    report_dates = {str(event.get("event_id")): str(event.get("report_date") or "")[:10] for event in events}
    excluded_future = 0
    excluded_undated = 0
    eligible_local: list[Dict[str, Any]] = []
    for doc in local_docs:
        doc_date = str(doc.get("date") or "")[:10]
        eligible_ids = [event_id for event_id, cutoff in report_dates.items() if doc_date and cutoff and doc_date <= cutoff]
        if payload.strict_temporal and report_dates and not doc_date:
            excluded_undated += 1
            continue
        if payload.strict_temporal and report_dates and not eligible_ids:
            excluded_future += 1
            continue
        enriched = dict(doc)
        enriched["eligible_event_ids"] = eligible_ids or list(report_dates)
        eligible_local.append(enriched)

    query = " ".join(str(value or "") for value in [
        " ".join(str(event.get(key) or "") for event in events for key in ("manager", "fund", "fund_ticker", "report_date", "action_type", "market_regime")),
        " ".join(payload.ticker_focus), payload.question,
        "macroeconomic industry FOMC rate fund report manager commentary support counterevidence structural break",
    ])
    visual_chunks = part7_service._chunk_documents(deduplicated)
    local_chunks = part7_service._chunk_documents(eligible_local)
    retrieved = part7_service._retrieve(local_chunks, query, payload.max_local_chunks) if local_chunks and payload.max_local_chunks else []
    evidence = part7_service._assign_ids(visual_chunks + retrieved)
    instructions = part7_service._prompt_text(BASE_DIR)
    user_input = part7_service._build_input(events, evidence, payload.question, payload.use_web_search)
    user_input += (
        "\n\nAUDIT CONTROLS\n"
        f"Evidence scope: {payload.evidence_scope}. Ticker focus: {payload.ticker_focus or 'none supplied'}. "
        f"Strict point-in-time eligibility: {payload.strict_temporal}. "
        f"Counterevidence required: {payload.require_counterevidence}. "
        "Separate macro/regime evidence from company/ticker evidence; never use a document for an event when its eligible_event_ids excludes that event. "
        "If required evidence is absent, return insufficient_evidence instead of filling gaps."
    )
    return {
        "event": events[0] if events else {}, "events": events, "retrieved_evidence": evidence,
        "instructions": instructions, "input": user_input,
        "retrieval": {
            "visual_evidence_chunks": len(visual_chunks), "local_documents_available": len(local_docs),
            "local_chunks_retrieved": len(retrieved), "web_search_requested": payload.use_web_search,
            "evidence_scope": payload.evidence_scope, "ticker_focus": payload.ticker_focus,
        },
        "temporal_audit": {
            "strict": payload.strict_temporal, "event_cutoffs": report_dates,
            "future_documents_excluded": excluded_future, "undated_documents_excluded": excluded_undated,
            "rule": "local evidence date must be on or before the relevant report_date",
        },
    }


def _run_part7_openai(payload: Part7CriticRequest, context: Dict[str, Any], model: str) -> Dict[str, Any]:
    from openai import OpenAI
    request: Dict[str, Any] = {
        "model": model, "instructions": context["instructions"], "input": context["input"],
        "text": {"format": {"type": "json_schema", "name": "part7_evidence_grounded_critic", "strict": True,
                            "schema": Part7CriticOutput.model_json_schema()}},
    }
    if model.startswith("gpt-5"):
        request["reasoning"] = {"effort": "high"}
    if payload.use_web_search:
        request["tools"] = [{"type": "web_search"}]
    response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).responses.create(**request)
    analysis = Part7CriticOutput.model_validate_json(response.output_text).model_dump()
    return {"analysis": analysis, "response_id": response.id,
            "web_citations": part7_service._extract_web_citations(response.model_dump())}


def _run_part7_pydantic_ai(payload: Part7CriticRequest, context: Dict[str, Any], model: str) -> Dict[str, Any]:
    from pydantic_ai import Agent
    kwargs: Dict[str, Any] = {"output_type": Part7CriticOutput, "instructions": context["instructions"]}
    if payload.use_web_search:
        try:
            from pydantic_ai import WebSearchTool
        except ImportError:
            try:
                from pydantic_ai.builtin_tools import WebSearchTool  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError("Installed PydanticAI does not expose WebSearchTool.") from exc
        kwargs["builtin_tools"] = [WebSearchTool()]
    agent = Agent(f"openai:{model}", **kwargs)
    run = agent.run_sync(context["input"])
    output = run.output if isinstance(run.output, Part7CriticOutput) else Part7CriticOutput.model_validate(run.output)
    return {"analysis": output.model_dump(), "response_id": None, "web_citations": []}


@app.get("/api/part7/status")
def get_part7_status() -> Dict[str, Any]:
    if PART7_IMPORT_ERROR is not None or part7_status is None:
        return {"mode": "unavailable", "error": repr(PART7_IMPORT_ERROR)}
    status = part7_status(BASE_DIR)
    status.update({
        "pydantic_ai_available": _pydantic_ai_available(),
        "orchestration_modes": ["auto", "pydantic_ai", "openai_responses"],
        "auto_resolution": "pydantic_ai when installed; otherwise OpenAI Responses",
        "structured_output_schema": "Part7CriticOutput (Pydantic, extra fields forbidden)",
        "evidence_scopes": ["macro_and_micro", "macro_only", "micro_only"],
        "point_in_time_filter_available": True,
    })
    return status


@app.post("/api/part7/critic")
def analyze_part7(payload: Part7CriticRequest) -> Dict[str, Any]:
    if PART7_IMPORT_ERROR is not None or part7_service is None:
        raise HTTPException(status_code=503, detail=f"Part7 service unavailable: {PART7_IMPORT_ERROR}")
    try:
        context = _part7_context(payload)
        chosen_model = payload.model or os.getenv("OPENAI_MODEL", part7_service.DEFAULT_MODEL)
        requested_mode = payload.orchestration_mode
        resolved_mode = "pydantic_ai" if requested_mode == "auto" and _pydantic_ai_available() else requested_mode
        if resolved_mode == "auto":
            resolved_mode = "openai_responses"
        common = {key: value for key, value in context.items() if key not in ("instructions", "input")}
        common.update({"model": chosen_model, "orchestration": {"requested": requested_mode, "resolved": resolved_mode}})
        if not os.getenv("OPENAI_API_KEY"):
            result = {"status": "preview", "message": "Part7 audit contract and RAG context are ready; OPENAI_API_KEY is not configured.",
                      **common, "analysis": None, "web_citations": [],
                      "prompt_preview": {"instructions": context["instructions"], "input": context["input"]}}
        else:
            if resolved_mode == "pydantic_ai" and not _pydantic_ai_available():
                raise HTTPException(status_code=503, detail="PydanticAI mode requested but pydantic-ai is not installed; use auto/openai_responses or install it.")
            try:
                live = (_run_part7_pydantic_ai(payload, context, chosen_model) if resolved_mode == "pydantic_ai"
                        else _run_part7_openai(payload, context, chosen_model))
            except Exception as exc:
                if requested_mode != "auto" or resolved_mode == "openai_responses":
                    raise
                resolved_mode = "openai_responses"
                common["orchestration"]["resolved"] = resolved_mode
                common["orchestration"]["fallback_reason"] = str(exc)
                live = _run_part7_openai(payload, context, chosen_model)
            result = {"status": "ok", "message": "Part7 evidence-grounded critic completed.", **common,
                      **live, "prompt_preview": None}
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Part7 critic failed: {exc}") from exc
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    part7_dir = OUTPUTS_DIR / "part7"
    save_json_file(part7_dir / f"part7_critic_{timestamp}.json", result)
    save_json_file(part7_dir / "part7_critic_latest.json", result)
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
