#!/usr/bin/env python3
"""Train leakage-controlled Part 6 v003 direction and magnitude bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.prediction_service import (
    extract_model_classes as _shared_extract_model_classes,
    get_positive_class_index as _shared_get_positive_class_index,
    predict_positive_probability as _shared_predict_positive_probability,
    validate_probability_array as _shared_validate_probability_array,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

HORIZONS = (3, 6, 9, 12)
NEUTRAL_RETURN_BAND = 0.005
NEUTRAL_PROBABILITY_LOW = 0.4
NEUTRAL_PROBABILITY_HIGH = 0.6
LARGE_QUANTILE = 0.67
EMBARGO_MONTHS = 1
DATE_COL = "report_date"
TRAINING_WINDOWS: dict[str, int | None] = {
    "expanding": None, "rolling_5y": 5, "rolling_7y": 7, "rolling_10y": 10,
}
THRESHOLD_CANDIDATES = np.round(np.arange(0.05, 0.951, 0.01), 2)
MIN_DIRECTION_THRESHOLD = 0.20
MAX_DIRECTION_THRESHOLD = 0.80
MIN_ALLOWED_PREDICTED_POSITIVE_RATE = 0.02
MAX_ALLOWED_PREDICTED_POSITIVE_RATE = 0.98
MAX_ALLOWED_POSITIVE_RATE_GAP = 0.20
MIN_MAGNITUDE_TRAIN_ROWS = 50
MIN_MAGNITUDE_VALID_ROWS = 20
ISOTONIC_MIN_ROWS = 1000
ISOTONIC_MIN_CLASS_ROWS = 200
ISOTONIC_MIN_UNIQUE_PROBABILITIES = 50
MIN_CLASS_SUPPORT = 50
FORBIDDEN_PREFIXES = (
    "future_", "direction_label_", "outcome_5class_", "label_positive_excess_",
    "label_start_date_", "label_end_date_", "label_available_",
)
FORBIDDEN_COLUMNS = {
    "manager_defensive_score", "manager_flow_score", "manager_growth_tilt_score",
    "manager_reliability_score", "style_deviation_score",
}
GICS_SECTORS = (
    "energy", "materials", "industrials", "consumer_discretionary", "consumer_staples",
    "health_care", "financials", "information_technology", "communication_services",
    "utilities", "real_estate",
)
META_COLUMNS = {
    "event_id", "training_window_years", "training_window_months", "manager", "fund",
    "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name", DATE_COL, "year",
    "quarter", "month_key", "feature_cutoff_date", "feature_source_month",
    "feature_available_at", "availability_check_passed", "dataset_observation_end",
    "style_window_start_date", "style_window_end_date", "style_window_type",
    "manager_score_window_start", "manager_score_window_end", "leakage_check_passed",
    "market_regime", "manager_style_group", "action_type", "cross_asset_execution_type",
    "allocation_completion_method", "non_individual_source", "data_quality_flags",
}

FEATURE_GROUPS = {
    "market_core": [
        "fund_trailing_return", "fund_trailing_excess_return", "fund_trailing_max_drawdown",
        "fund_trailing_beta_vs_sp500", "trailing_avg_net_flow", "trailing_avg_turn_ratio",
        "lag1_mret", "lag1_sp500_ret", "lag1_excess_ret", "lag1_net_flow",
        "lag1_mtna", "lag1_exp_ratio", "lag1_mgmt_fee", "lag1_turn_ratio",
    ],
    "cross_asset_allocation": [
        "stock_allocation", "bond_allocation", "cash_allocation", "bond_money_exposure",
        "indirect_equity_exposure", "company_equity_exposure_proxy", "portfolio_beta",
    ] + [f"sector_{sector}_exposure" for sector in GICS_SECTORS],
    "action_deltas": [
        "delta_stock", "delta_beta", "delta_technology", "delta_bond_money",
        "delta_indirect_equity", "delta_nonstock_total_exposure", "delta_sector_exposure",
        "action_strength", "top_holding_concentration", "sector_rotation_intensity",
    ] + [f"delta_sector_{sector}" for sector in GICS_SECTORS],
    "rolling_deviations": [
        "rolling_style_deviation_score", "rolling_sector_deviation_score",
        "rolling_cross_asset_deviation_score", "rolling_action_deviation_score",
    ],
    "manager_pti_scores": [
        "manager_defensive_score_pti", "manager_flow_score_pti",
        "manager_growth_tilt_score_pti", "manager_reliability_score_pti",
        "manager_history_count", "manager_history_month_count",
    ],
    "holdings_quality": [
        "holding_row_count", "beta_matched_holding_count",
        "non_individual_matched_holding_count",
    ],
    "regime_features": [
        "lag1_interest_rate_level", "lag1_interest_rate_change_3m",
        "lag1_market_return_3m", "lag1_market_return_12m",
        "lag1_market_volatility_12m", "lag1_market_drawdown_12m",
    ],
    "regime_interactions": [
        "stock_allocation_x_rate_change", "portfolio_beta_x_market_volatility",
        "technology_exposure_x_market_trend_12m", "bond_allocation_x_rate_change",
        "rolling_action_deviation_x_market_volatility",
    ],
}
# Retain the old metadata key while centralizing all definitions above.
FEATURE_LAYERS = FEATURE_GROUPS


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _training_data_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.reindex(columns=columns).copy()
    hashed = pd.util.hash_pandas_object(selected, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def validate_feature_list(features: Iterable[str]) -> list[str]:
    feature_list = list(dict.fromkeys(features))
    forbidden = [
        name for name in feature_list
        if name in FORBIDDEN_COLUMNS or name.startswith("current_") or name.startswith(FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Target/leakage columns found in feature list: {forbidden}")
    return feature_list


def infer_features(df: pd.DataFrame) -> list[str]:
    preferred = list(dict.fromkeys(column for layer in FEATURE_LAYERS.values() for column in layer))
    features = [
        column for column in preferred
        if column in df and pd.to_numeric(df[column], errors="coerce").notna().sum() >= 20
    ]
    if not features:
        raise ValueError("No approved numeric features have enough observations.")
    return validate_feature_list(features)


def _numeric_feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def extract_model_classes(model: Any) -> np.ndarray | None:
    return _shared_extract_model_classes(model)


def get_positive_class_index(model: Any, positive_label: Any = 1) -> int:
    return _shared_get_positive_class_index(model, positive_label)


def validate_probability_array(probability: Any, expected_rows: int) -> np.ndarray:
    return _shared_validate_probability_array(probability, expected_rows)


def predict_positive_probability(model: Any, X: np.ndarray, positive_label: Any = 1) -> np.ndarray:
    return _shared_predict_positive_probability(model, X, positive_label)


def build_finite_regression_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    return np.isfinite(matrix).all(axis=1) & np.isfinite(target)


def sanitize_magnitude_predictions(
    prediction: Any, fallback_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(prediction, dtype=float).reshape(-1)
    invalid = ~np.isfinite(values)
    safe_fallback = float(fallback_value) if np.isfinite(fallback_value) else 0.0
    if invalid.any():
        values = values.copy()
        values[invalid] = safe_fallback
    return values, {
        "magnitude_invalid_prediction_count": int(invalid.sum()),
        "magnitude_prediction_fallback_value": safe_fallback,
        "magnitude_prediction_fallback_used": bool(invalid.any()),
    }


def safe_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    valid_n = int(finite.sum())
    if valid_n == 0:
        return {"mae": None, "rmse": None, "spearman": None,
                "spearman_status": "undefined_constant_or_insufficient_data", "spearman_valid_n": 0}
    actual, predicted = actual[finite], predicted[finite]
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(mean_squared_error(actual, predicted) ** 0.5)
    if valid_n < 3 or np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        spearman, status = None, "undefined_constant_or_insufficient_data"
    else:
        correlation = pd.Series(actual).corr(pd.Series(predicted), method="spearman")
        spearman = None if pd.isna(correlation) else float(correlation)
        status = "ok" if spearman is not None else "undefined_constant_or_insufficient_data"
    return {"mae": mae, "rmse": rmse, "spearman": spearman,
            "spearman_status": status, "spearman_valid_n": valid_n}


PRIMARY_PARAMETER_CANDIDATES = [
    {"max_depth": 2, "learning_rate": 0.02, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 1, "reg_lambda": 10},
    {"max_depth": 2, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0, "reg_lambda": 5},
    {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 30, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 2, "reg_lambda": 20},
    {"max_depth": 3, "learning_rate": 0.02, "min_child_weight": 30, "subsample": 0.9, "colsample_bytree": 0.75, "reg_alpha": 2, "reg_lambda": 20},
    {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.9, "reg_alpha": 1, "reg_lambda": 10},
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.6, "reg_alpha": 0, "reg_lambda": 5},
]


def make_primary_tree(
    seed: int, params: dict[str, Any] | None = None,
    class_weight_mode: str = "unweighted", scale_pos_weight: float = 1.0,
) -> tuple[Any, str]:
    params = params or PRIMARY_PARAMETER_CANDIDATES[0]
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=350, **params,
            scale_pos_weight=scale_pos_weight if class_weight_mode == "weighted" else 1.0,
            objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=-1,
        ), "xgboost"
    except ImportError:
        return RandomForestClassifier(
            n_estimators=350, max_depth=9, min_samples_leaf=12,
            class_weight=({0: 1.0, 1: scale_pos_weight} if class_weight_mode == "weighted" else None),
            random_state=seed, n_jobs=-1,
        ), "random_forest_primary_fallback"


def make_portable_tree(
    seed: int, class_weight_mode: str = "unweighted", scale_pos_weight: float = 1.0,
) -> tuple[Any, str]:
    return RandomForestClassifier(
        n_estimators=240, max_depth=9, min_samples_leaf=12,
        class_weight=({0: 1.0, 1: scale_pos_weight} if class_weight_mode == "weighted" else None),
        random_state=seed, n_jobs=-1,
    ), "random_forest_portable"


def _calibrate_prefit(model: Any, X_cal: np.ndarray, y_cal: pd.Series, method: str = "sigmoid") -> Any:
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def _temporal_inner_split(
    train: pd.DataFrame, horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Create base/calibration/validation periods using whole report-date groups."""
    dates = np.asarray(sorted(train[DATE_COL].dropna().unique()))
    if len(dates) < 12:
        raise ValueError("At least twelve report dates are required for temporal inner splits.")
    # Start calibration early enough to preserve a full label-period purge
    # before validation even for the 12M horizon in a 5Y quarterly window.
    calibration_start = pd.Timestamp(dates[max(2, int(len(dates) * 0.45))])
    validation_start = pd.Timestamp(dates[max(3, int(len(dates) * 0.80))])
    label_end = f"label_end_date_{horizon}m"
    direction = f"direction_label_{horizon}m"
    base = train[(train[label_end] < calibration_start) & train[direction].isin([-1, 1])].copy()
    calibration = train[
        train[DATE_COL].ge(calibration_start)
        & train[DATE_COL].lt(validation_start)
        & train[label_end].lt(validation_start)
        & train[direction].isin([-1, 1])
    ].copy()
    validation = train[train[DATE_COL].ge(validation_start) & train[direction].isin([-1, 1])].copy()
    periods = (base, calibration, validation)
    if any(part.empty or part[direction].nunique() < 2 for part in periods):
        raise ValueError("Base, calibration, and validation periods each need both direction classes.")
    date_sets = [set(part[DATE_COL].unique()) for part in periods]
    if any(date_sets[i] & date_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("The same report_date crossed an inner temporal split.")
    if base[label_end].max() >= calibration_start:
        raise AssertionError("Base-training labels overlap calibration.")
    if calibration[DATE_COL].max() >= validation_start:
        raise AssertionError("Calibration report dates overlap validation.")
    return base, calibration, validation, {
        "base_start_date": base[DATE_COL].min().strftime("%Y-%m-%d"),
        "base_end_date": base[DATE_COL].max().strftime("%Y-%m-%d"),
        "calibration_start_date": calibration_start.strftime("%Y-%m-%d"),
        "calibration_end_date": calibration[DATE_COL].max().strftime("%Y-%m-%d"),
        "validation_start_date": validation_start.strftime("%Y-%m-%d"),
        "validation_end_date": validation[DATE_COL].max().strftime("%Y-%m-%d"),
    }


def _windowed_train(
    train: pd.DataFrame, test_start: pd.Timestamp, horizon: int, window_years: int | None,
) -> pd.DataFrame:
    label_end = f"label_end_date_{horizon}m"
    embargo_cutoff = test_start - pd.DateOffset(months=EMBARGO_MONTHS)
    mask = train[label_end].lt(embargo_cutoff)
    if window_years is not None:
        mask &= train[DATE_COL].ge(test_start - pd.DateOffset(years=window_years))
    result = train[mask].copy()
    if not result.empty and result[label_end].max() >= embargo_cutoff:
        raise AssertionError("Rolling window violated purge/embargo.")
    return result


def _threshold_metrics(y: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    labels = pd.Series(y).astype(int).to_numpy()
    predicted = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    actual_positive_rate = float(labels.mean())
    predicted_positive_rate = float(predicted.mean())
    positive_rate_gap = predicted_positive_rate - actual_positive_rate
    return {
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "youden_j": float(recall + specificity - 1.0),
        "recall": float(recall), "specificity": float(specificity),
        "false_positive_rate": float(1.0 - specificity),
        "predicted_positive_rate": predicted_positive_rate,
        "actual_positive_rate": actual_positive_rate,
        "positive_rate_gap": float(positive_rate_gap),
        "absolute_positive_rate_gap": float(abs(positive_rate_gap)),
    }


def select_direction_threshold(y: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    """Select a validation-only threshold with explicit class-collapse protection."""
    actual_rate = float(pd.Series(y).astype(int).mean())
    rows = []
    for threshold in THRESHOLD_CANDIDATES:
        metrics = _threshold_metrics(y, probability, float(threshold))
        rejection_reasons = []
        if not MIN_DIRECTION_THRESHOLD <= threshold <= MAX_DIRECTION_THRESHOLD:
            rejection_reasons.append("outside_safe_threshold_range")
        if not MIN_ALLOWED_PREDICTED_POSITIVE_RATE <= metrics["predicted_positive_rate"] <= MAX_ALLOWED_PREDICTED_POSITIVE_RATE:
            rejection_reasons.append("predicted_positive_rate_collapse")
        if metrics["absolute_positive_rate_gap"] > MAX_ALLOWED_POSITIVE_RATE_GAP:
            rejection_reasons.append("positive_rate_gap_exceeded")
        rows.append({
            "threshold": float(threshold), **metrics,
            "eligible": not rejection_reasons, "rejection_reasons": rejection_reasons,
            "threshold_score": float(metrics["balanced_accuracy"] - 0.25 * metrics["absolute_positive_rate_gap"]),
        })

    eligible_rows = [row for row in rows if row["eligible"]]

    def choose(objective: str) -> dict[str, Any]:
        pool = eligible_rows or [min(rows, key=lambda row: (row["absolute_positive_rate_gap"], abs(row["threshold"] - 0.5)))]
        best_value = max(row[objective] for row in pool)
        tied = [row for row in pool if np.isclose(row[objective], best_value, rtol=0, atol=1e-12)]
        return min(tied, key=lambda row: (
            abs(row["predicted_positive_rate"] - actual_rate),
            abs(row["threshold"] - 0.5),
            row["false_positive_rate"], -row["threshold"],
        ))

    selected = {objective: choose(objective) for objective in ("balanced_accuracy", "f1", "youden_j")}
    if eligible_rows:
        formal = max(eligible_rows, key=lambda row: (
            row["threshold_score"], row["balanced_accuracy"], -row["absolute_positive_rate_gap"],
            -abs(row["threshold"] - 0.5), -row["false_positive_rate"],
        ))
        status, fallback_used, rejection_reasons = "selected_safe_threshold", False, []
    else:
        formal = next((row for row in rows if np.isclose(row["threshold"], 0.5)), selected["balanced_accuracy"])
        status, fallback_used = "fallback_no_eligible_threshold", True
        rejection_reasons = ["all_candidates_failed_safety_constraints"]
    return {
        "candidate_thresholds": [float(value) for value in THRESHOLD_CANDIDATES],
        "threshold_candidate_diagnostics": rows,
        "balanced_accuracy_threshold": selected["balanced_accuracy"]["threshold"],
        "balanced_accuracy_at_threshold": selected["balanced_accuracy"]["balanced_accuracy"],
        "f1_threshold": selected["f1"]["threshold"],
        "f1_at_threshold": selected["f1"]["f1"],
        "youden_j_threshold": selected["youden_j"]["threshold"],
        "youden_j_at_threshold": selected["youden_j"]["youden_j"],
        "direction_threshold": formal["threshold"],
        "selected_threshold": formal["threshold"],
        "selected_threshold_predicted_positive_rate": formal["predicted_positive_rate"],
        "selected_threshold_actual_positive_rate": formal["actual_positive_rate"],
        "selected_threshold_balanced_accuracy": formal["balanced_accuracy"],
        "selected_threshold_positive_rate_gap": formal["positive_rate_gap"],
        "selected_threshold_absolute_positive_rate_gap": formal["absolute_positive_rate_gap"],
        "threshold_selection_status": status,
        "threshold_fallback_used": fallback_used,
        "threshold_rejection_reasons": rejection_reasons,
        "threshold_objective": "balanced_accuracy_minus_0.25_absolute_positive_rate_gap",
        "threshold_validation_metric": formal["threshold_score"],
    }


def _calibration_candidates(
    raw_classifier: Any, X_cal: np.ndarray, y_cal: pd.Series,
) -> tuple[dict[str, Any], dict[str, str]]:
    candidates: dict[str, Any] = {"none": raw_classifier}
    status = {"none": "eligible", "sigmoid": "eligible", "isotonic": "eligible"}
    candidates["sigmoid"] = _calibrate_prefit(raw_classifier, X_cal, y_cal, "sigmoid")
    raw_probability = predict_positive_probability(raw_classifier, X_cal, positive_label=1)
    isotonic_ok = (
        len(y_cal) >= ISOTONIC_MIN_ROWS
        and int(y_cal.sum()) >= ISOTONIC_MIN_CLASS_ROWS
        and int((1 - y_cal).sum()) >= ISOTONIC_MIN_CLASS_ROWS
        and np.unique(raw_probability).size >= ISOTONIC_MIN_UNIQUE_PROBABILITIES
    )
    if isotonic_ok:
        candidates["isotonic"] = _calibrate_prefit(raw_classifier, X_cal, y_cal, "isotonic")
    else:
        status["isotonic"] = "ineligible_insufficient_support"
    return candidates, status


def select_calibration_method(
    candidates: dict[str, Any], status: dict[str, str], X_validation: np.ndarray,
    y_validation: pd.Series,
) -> tuple[str, Any, dict[str, Any]]:
    comparisons: dict[str, Any] = {}
    preference = {"sigmoid": 0, "none": 1, "isotonic": 2}
    ranked = []
    for method in ("none", "sigmoid", "isotonic"):
        if method not in candidates:
            comparisons[method] = {"status": status[method]}
            continue
        probability = np.clip(predict_positive_probability(candidates[method], X_validation, 1), 1e-8, 1 - 1e-8)
        brier = float(brier_score_loss(y_validation, probability))
        loss = float(log_loss(y_validation, probability, labels=[0, 1]))
        threshold_selection = select_direction_threshold(y_validation, probability)
        rate_gap = threshold_selection["selected_threshold_absolute_positive_rate_gap"]
        score = (
            threshold_selection["selected_threshold_balanced_accuracy"]
            - 0.50 * brier - 0.25 * rate_gap
        )
        eligible = not threshold_selection["threshold_fallback_used"]
        comparisons[method] = {
            "status": "eligible" if eligible else "ineligible_threshold_selection_failed",
            "eligible": eligible, "brier": brier, "log_loss": loss,
            "selection_score": score, "threshold_selection": threshold_selection,
            "absolute_positive_rate_gap": rate_gap,
        }
        ranked.append((not eligible, -round(score, 6), preference[method], method))
    ranked.sort()
    selected_method = ranked[0][3]
    return selected_method, candidates[selected_method], comparisons


def classify_prediction(
    probability: float,
    expected_excess: float,
    neutral_probability_low: float,
    neutral_probability_high: float,
    neutral_return_band: float,
    large_threshold: float,
) -> tuple[str, str, str]:
    if (
        neutral_probability_low <= probability <= neutral_probability_high
        or abs(expected_excess) < neutral_return_band
    ):
        return "neutral", "neutral", "neutral"
    direction = "up" if expected_excess > 0 else "down"
    strength = "large" if abs(expected_excess) >= large_threshold else "small"
    predicted_class = (
        "large_win" if direction == "up" and strength == "large" else
        "small_win" if direction == "up" else
        "large_loss" if strength == "large" else "small_loss"
    )
    return direction, strength, predicted_class


def _actual_class(value: float, large_threshold: float) -> str:
    if abs(value) < NEUTRAL_RETURN_BAND:
        return "neutral"
    if value > 0:
        return "large_win" if abs(value) >= large_threshold else "small_win"
    return "large_loss" if abs(value) >= large_threshold else "small_loss"


def _prepare_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = f"future_{horizon}m_excess_return"
    direction = f"direction_label_{horizon}m"
    label_end = f"label_end_date_{horizon}m"
    required = [DATE_COL, target, direction, label_end]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"{horizon}M dataset is missing required columns: {missing}")
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce").dt.normalize()
    work[label_end] = pd.to_datetime(work[label_end], errors="coerce").dt.normalize()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work[direction] = pd.to_numeric(work[direction], errors="coerce")
    if f"label_available_{horizon}m" in work:
        available = work[f"label_available_{horizon}m"].astype(str).str.lower().isin({"true", "1"})
    else:
        available = work[target].notna()
    return work[available & work[target].notna() & work[DATE_COL].notna() & work[label_end].notna()].copy()


def date_grouped_walk_forward(
    work: pd.DataFrame, horizon: int, n_folds: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]]:
    unique_dates = np.asarray(sorted(work[DATE_COL].dropna().unique()))
    if len(unique_dates) < 10:
        raise ValueError(f"{horizon}M needs at least ten unique report dates for walk-forward validation.")
    test_dates = unique_dates[max(3, int(len(unique_dates) * 0.60)):]
    blocks = [block for block in np.array_split(test_dates, min(n_folds, len(test_dates))) if len(block)]
    folds = []
    label_end = f"label_end_date_{horizon}m"
    for block in blocks:
        test_start, test_end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        embargo_cutoff = test_start - pd.DateOffset(months=EMBARGO_MONTHS)
        train = work[work[label_end] < embargo_cutoff].copy()
        test = work[work[DATE_COL].between(test_start, test_end)].copy()
        if train.empty or test.empty:
            continue
        if train[label_end].max() >= embargo_cutoff:
            raise AssertionError("Training label period overlaps the embargo/test boundary.")
        if set(train[DATE_COL].unique()) & set(test[DATE_COL].unique()):
            raise AssertionError("The same report_date crossed train and test.")
        folds.append((train, test, {
            "test_start_date": test_start.strftime("%Y-%m-%d"),
            "test_end_date": test_end.strftime("%Y-%m-%d"),
            "embargo_cutoff": embargo_cutoff.strftime("%Y-%m-%d"),
        }))
    if not folds:
        raise ValueError(f"{horizon}M could not form a leakage-safe walk-forward fold.")
    return folds


def _split_fit_calibration(train: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.asarray(sorted(train[DATE_COL].dropna().unique()))
    if len(dates) < 5:
        raise ValueError("Insufficient dates for a later calibration period.")
    calibration_start = pd.Timestamp(dates[max(1, int(len(dates) * 0.80))])
    label_end = f"label_end_date_{horizon}m"
    fit = train[train[label_end] < calibration_start].copy()
    calibration = train[train[DATE_COL] >= calibration_start].copy()
    direction = f"direction_label_{horizon}m"
    fit = fit[fit[direction].isin([-1, 1])]
    calibration = calibration[calibration[direction].isin([-1, 1])]
    if fit.empty or calibration.empty or fit[direction].nunique() < 2 or calibration[direction].nunique() < 2:
        raise ValueError("Direction fit/calibration periods need both positive and negative observations.")
    return fit, calibration


def _direction_metrics(
    y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5,
) -> dict[str, Any]:
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    if y.empty:
        return {"rows": 0, "status": "insufficient_support"}
    predicted = (p >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result: dict[str, Any] = {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.nunique() > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": matrix.tolist(),
        "actual_positive_rate": float(y.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "raw_probability_mean": float(p.mean()),
        "probability_rate_gap": float(predicted.mean() - y.mean()),
    }
    if y.nunique() > 1:
        observed, predicted_bins = calibration_curve(y, p, n_bins=10, strategy="quantile")
        result["calibration_curve"] = {
            "mean_predicted_probability": predicted_bins.tolist(),
            "fraction_positive": observed.tolist(),
        }
    else:
        result["calibration_curve"] = {"mean_predicted_probability": [], "fraction_positive": []}
    return result


def _five_class_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = ["large_loss", "small_loss", "neutral", "small_win", "large_win"]
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0))
    warning_reasons = []
    if macro_f1 < 0.30:
        warning_reasons.append("low_macro_f1")
    if any(float(report[label]["recall"]) <= 0 for label in ("large_loss", "large_win")):
        warning_reasons.append("zero_recall_for_extreme_classes")
    if len(set(predicted)) < 3:
        warning_reasons.append("prediction_class_collapse")
    production_ready = not warning_reasons
    return {
        "macro_f1": macro_f1,
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                **report[label],
                "status": "ok" if int(report[label]["support"]) >= MIN_CLASS_SUPPORT else "insufficient_support",
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "labels": labels,
        "five_class_status": "production_ready" if production_ready else "experimental",
        "five_class_production_ready": production_ready,
        "five_class_warning_reasons": warning_reasons,
    }


def _magnitude_metrics(
    target: pd.Series, direction: pd.Series, positive: np.ndarray,
    negative: np.ndarray, expected: np.ndarray,
) -> dict[str, Any]:
    actual = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    dirs = pd.to_numeric(direction, errors="coerce").to_numpy(dtype=float)
    pos = dirs == 1
    neg = dirs == -1
    non_neutral = pos | neg
    positive_metrics = safe_regression_metrics(actual[pos], np.asarray(positive)[pos])
    negative_metrics = safe_regression_metrics(actual[neg], np.asarray(negative)[neg])
    overall_metrics = safe_regression_metrics(actual, expected)
    finite_direction = non_neutral & np.isfinite(actual) & np.isfinite(expected)
    return {
        "positive_mae": positive_metrics["mae"], "negative_mae": negative_metrics["mae"],
        "overall_expected_excess_mae": overall_metrics["mae"], "rmse": overall_metrics["rmse"],
        "spearman_correlation": overall_metrics["spearman"],
        "spearman_status": overall_metrics["spearman_status"],
        "spearman_valid_n": overall_metrics["spearman_valid_n"],
        "direction_hit_rate": float(np.mean(np.sign(actual[finite_direction]) == np.sign(np.asarray(expected)[finite_direction]))) if finite_direction.any() else None,
    }


def _fit_magnitude_side(
    fit: pd.DataFrame, validation: pd.DataFrame, features: list[str],
    direction_column: str, target: str, direction_value: int,
) -> dict[str, Any]:
    train_side = fit[fit[direction_column].eq(direction_value)].copy()
    validation_side = validation[validation[direction_column].eq(direction_value)].copy()
    rows_before = int(len(train_side))
    validation_rows_before = int(len(validation_side))
    if not rows_before:
        raise ValueError(f"No magnitude training rows for direction {direction_value}.")
    all_null = [
        feature for feature in features
        if pd.to_numeric(train_side[feature], errors="coerce").isna().all()
    ]
    if all_null:
        raise ValueError(f"Magnitude training subset has all-null features: {all_null}")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(imputer.fit_transform(
        _numeric_feature_frame(train_side, features)
    ))
    X_validation_all = scaler.transform(imputer.transform(
        _numeric_feature_frame(validation_side, features)
    )) if validation_rows_before else np.empty((0, X_train_all.shape[1]))
    y_train_all = pd.to_numeric(train_side[target], errors="coerce").abs().to_numpy(dtype=float)
    y_validation_all = pd.to_numeric(validation_side[target], errors="coerce").abs().to_numpy(dtype=float)
    train_finite = build_finite_regression_mask(X_train_all, y_train_all)
    validation_finite = build_finite_regression_mask(X_validation_all, y_validation_all)
    X_train, y_train = X_train_all[train_finite], y_train_all[train_finite]
    X_validation, y_validation = X_validation_all[validation_finite], y_validation_all[validation_finite]
    fallback_value = float(np.median(y_train)) if len(y_train) else 0.0
    audit = {
        "magnitude_train_rows_before_filter": rows_before,
        "magnitude_train_rows_after_filter": int(len(y_train)),
        "magnitude_train_invalid_row_count": int(rows_before - len(y_train)),
        "magnitude_validation_rows_before_filter": validation_rows_before,
        "magnitude_validation_rows_after_filter": int(len(y_validation)),
        "magnitude_validation_invalid_row_count": int(validation_rows_before - len(y_validation)),
        "magnitude_prediction_fallback_value": fallback_value,
    }

    def fallback_result(reason: str, comparisons: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        dummy = DummyRegressor(strategy="constant", constant=fallback_value)
        dummy.fit(np.zeros((1, X_train_all.shape[1])), np.asarray([fallback_value]))
        return {
            "model": dummy, "imputer": imputer, "scaler": scaler,
            "model_type": "DummyRegressor", "model_params": {"constant": fallback_value},
            "target_winsorized": False, "winsor_lower": fallback_value,
            "winsor_upper": fallback_value, "validation_mae": None,
            "validation_spearman": None,
            "validation_spearman_status": "undefined_constant_or_insufficient_data",
            "candidate_comparison": comparisons or [], "magnitude_status": reason,
            "magnitude_prediction_fallback_used": True, **audit,
        }

    if len(y_train) < MIN_MAGNITUDE_TRAIN_ROWS or len(y_validation) < MIN_MAGNITUDE_VALID_ROWS:
        return fallback_result("insufficient_finite_samples")
    lower, upper = np.quantile(y_train, [0.01, 0.99])
    candidates = [
        ("Ridge", {"alpha": 4.0}, lambda: Ridge(alpha=4.0)),
        ("ElasticNet", {"alpha": 0.0005, "l1_ratio": 0.25}, lambda: ElasticNet(alpha=0.0005, l1_ratio=0.25, max_iter=5000, random_state=42)),
        ("HuberRegressor", {"epsilon": 1.35, "alpha": 0.0001}, lambda: HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500)),
    ]
    comparisons, fitted = [], []
    for model_type, params, factory in candidates:
        for winsorized in (False, True):
            candidate_target = np.clip(y_train, lower, upper) if winsorized else y_train
            model = factory()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_train, candidate_target)
                if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                    comparisons.append({
                        "model_type": model_type, "params": params, "winsorized": winsorized,
                        "status": "removed_nonconverged_candidate",
                    })
                    continue
                prediction = np.maximum(model.predict(X_validation), 0.0)
            except Exception as exc:
                comparisons.append({"model_type": model_type, "params": params, "winsorized": winsorized, "status": f"failed: {exc}"})
                continue
            prediction, prediction_audit = sanitize_magnitude_predictions(prediction, fallback_value)
            if prediction_audit["magnitude_invalid_prediction_count"]:
                comparisons.append({
                    "model_type": model_type, "params": params, "winsorized": winsorized,
                    "status": "removed_invalid_predictions", **prediction_audit,
                })
                continue
            regression = safe_regression_metrics(y_validation, prediction)
            spearman_component = regression["spearman"] if regression["spearman"] is not None else -0.25
            score = regression["mae"] + 0.25 * regression["rmse"] - 0.10 * spearman_component
            comparisons.append({
                "model_type": model_type, "params": params, "winsorized": winsorized,
                "validation_mae": regression["mae"], "validation_rmse": regression["rmse"],
                "validation_spearman": regression["spearman"],
                "spearman_status": regression["spearman_status"],
                "spearman_valid_n": regression["spearman_valid_n"],
                "prediction_sign_validity": float(np.mean(prediction >= 0)),
                "magnitude_selection_score": score, "status": "eligible", **prediction_audit,
            })
            fitted.append((len(comparisons) - 1, model))
    if not fitted:
        return fallback_result("all_candidate_models_failed", comparisons)
    best_row_index, best_model = min(
        fitted, key=lambda item: (comparisons[item[0]]["magnitude_selection_score"], comparisons[item[0]]["validation_mae"])
    )
    selected = comparisons[best_row_index]
    return {
        "model": best_model, "imputer": imputer, "scaler": scaler,
        "model_type": selected["model_type"], "model_params": selected["params"],
        "target_winsorized": selected["winsorized"], "winsor_lower": float(lower),
        "winsor_upper": float(upper), "validation_mae": selected["validation_mae"],
        "validation_spearman": selected["validation_spearman"],
        "validation_spearman_status": selected["spearman_status"],
        "candidate_comparison": comparisons, "magnitude_status": "ok",
        "magnitude_prediction_fallback_used": False, **audit,
    }


def _validation_group_ablation(
    fit: pd.DataFrame, validation: pd.DataFrame, horizon: int, features: list[str],
) -> dict[str, Any]:
    direction = f"direction_label_{horizon}m"
    target = f"future_{horizon}m_excess_return"
    y_train = fit[direction].eq(1).astype(int)
    y_validation = validation[direction].eq(1).astype(int)

    def evaluate(selected_features: list[str]) -> dict[str, Any]:
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(_numeric_feature_frame(fit, selected_features))
        X_validation = imputer.transform(_numeric_feature_frame(validation, selected_features))
        probability = _logistic_baseline(X_train, y_train, X_validation)
        threshold = select_direction_threshold(y_validation, probability)["direction_threshold"]
        metrics = _direction_metrics(y_validation, probability, threshold)
        spearman = pd.Series(pd.to_numeric(validation[target], errors="coerce")).corr(
            pd.Series(probability), method="spearman"
        )
        metrics["spearman"] = None if pd.isna(spearman) else float(spearman)
        return metrics

    full = evaluate(features)
    result: dict[str, Any] = {"full": full, "groups": {}}
    for group, configured in FEATURE_GROUPS.items():
        present = [feature for feature in configured if feature in features]
        if not present:
            result["groups"][group] = {"status": "missing_group_features", "missing": configured}
            continue
        reduced = [feature for feature in features if feature not in present]
        if not reduced:
            result["groups"][group] = {"status": "cannot_remove_all_features"}
            continue
        ablated = evaluate(reduced)
        result["groups"][group] = {
            "status": "ok", "present_features": present,
            "delta_auc": None if full.get("roc_auc") is None or ablated.get("roc_auc") is None else full["roc_auc"] - ablated["roc_auc"],
            "delta_pr_auc": None if full.get("pr_auc") is None or ablated.get("pr_auc") is None else full["pr_auc"] - ablated["pr_auc"],
            "delta_balanced_accuracy": full["balanced_accuracy"] - ablated["balanced_accuracy"],
            "delta_brier": full["brier_score"] - ablated["brier_score"],
            "delta_spearman": None if full["spearman"] is None or ablated["spearman"] is None else full["spearman"] - ablated["spearman"],
        }
    return result


def _feature_importance_snapshot(
    raw_classifier: Any, X_validation: np.ndarray, y_validation: pd.Series,
    features: list[str], seed: int,
) -> dict[str, Any]:
    result = permutation_importance(
        raw_classifier, X_validation, y_validation, scoring="roc_auc",
        n_repeats=3, random_state=seed, n_jobs=1,
    )
    values = np.asarray(result.importances_mean, dtype=float)
    order = np.argsort(values)[::-1]
    return {
        "importance_method": "permutation_importance", "is_exact_shap": False,
        "feature_importance": {features[index]: float(values[index]) for index in range(len(features))},
        "feature_rank": {features[index]: int(rank + 1) for rank, index in enumerate(order)},
        "feature_sign_stability": "unavailable_for_permutation_importance",
    }


def _fit_models(
    train: pd.DataFrame, horizon: int, features: list[str], seed: int, variant: str,
) -> dict[str, Any]:
    fit, calibration, validation, split_contract = _temporal_inner_split(train, horizon)
    direction_column = f"direction_label_{horizon}m"
    target = f"future_{horizon}m_excess_return"
    all_null = [
        feature for feature in features
        if pd.to_numeric(fit[feature], errors="coerce").isna().all()
    ]
    if all_null:
        raise ValueError(f"Base training period has all-null required features: {all_null}")
    imputer = SimpleImputer(strategy="median")
    X_fit = imputer.fit_transform(_numeric_feature_frame(fit, features))
    X_cal = imputer.transform(_numeric_feature_frame(calibration, features))
    X_validation = imputer.transform(_numeric_feature_frame(validation, features))
    y_fit = fit[direction_column].eq(1).astype(int)
    y_cal = calibration[direction_column].eq(1).astype(int)
    y_validation = validation[direction_column].eq(1).astype(int)
    positive_count, negative_count = int(y_fit.sum()), int((1 - y_fit).sum())
    raw_weight = float(negative_count / positive_count) if positive_count else 10.0
    used_weight = float(np.clip(raw_weight, 1.0, 10.0))

    candidate_rows: list[dict[str, Any]] = []
    candidate_models: list[dict[str, Any]] = []
    parameter_candidates = PRIMARY_PARAMETER_CANDIDATES if variant == "primary" else [{}]
    for parameter_index, params in enumerate(parameter_candidates):
        for weight_mode in ("unweighted", "weighted"):
            raw_classifier, classifier_type = (
                make_primary_tree(seed + parameter_index, params, weight_mode, used_weight)
                if variant == "primary"
                else make_portable_tree(seed + parameter_index, weight_mode, used_weight)
            )
            raw_classifier.fit(X_fit, y_fit)
            calibrators, calibration_status = _calibration_candidates(raw_classifier, X_cal, y_cal)
            method, classifier, calibration_comparison = select_calibration_method(
                calibrators, calibration_status, X_validation, y_validation
            )
            probability = predict_positive_probability(classifier, X_validation, 1)
            threshold_selection = select_direction_threshold(y_validation, probability)
            metrics = _direction_metrics(
                y_validation, probability, threshold_selection["direction_threshold"]
            )
            complexity_penalty = 0.005 * float(params.get("max_depth", 3))
            threshold_failed = bool(threshold_selection["threshold_fallback_used"])
            candidate_score = (
                (metrics.get("roc_auc") or 0.0)
                + 0.50 * (metrics.get("pr_auc") or 0.0)
                + 0.30 * (metrics.get("balanced_accuracy") or 0.0)
                - 0.50 * metrics["brier_score"]
                - 0.25 * threshold_selection["selected_threshold_absolute_positive_rate_gap"]
                - (1.0 if threshold_failed else 0.0) - complexity_penalty
            )
            candidate_rows.append({
                "candidate_index": len(candidate_rows), "params": params,
                "class_weight_mode": weight_mode, "raw_scale_pos_weight": raw_weight,
                "used_scale_pos_weight": used_weight, "classifier_type": classifier_type,
                "calibration_method": method, "calibration_comparison": calibration_comparison,
                "threshold_selection": threshold_selection, "validation_metrics": metrics,
                "complexity_penalty": complexity_penalty, "candidate_selection_score": candidate_score,
                "eligible": not threshold_failed,
                "rejection_reasons": threshold_selection["threshold_rejection_reasons"],
            })
            candidate_models.append({"raw_classifier": raw_classifier, "classifier": classifier})
    eligible_candidate_indices = [index for index, row in enumerate(candidate_rows) if row["eligible"]]
    selection_pool = eligible_candidate_indices or list(range(len(candidate_rows)))
    best_index = max(
        selection_pool,
        key=lambda index: (
            candidate_rows[index]["candidate_selection_score"],
            candidate_rows[index]["validation_metrics"].get("pr_auc") or -1.0,
            candidate_rows[index]["validation_metrics"].get("balanced_accuracy") or -1.0,
            -candidate_rows[index]["validation_metrics"]["brier_score"],
        ),
    )
    selected = candidate_rows[best_index]
    raw_classifier = candidate_models[best_index]["raw_classifier"]
    classifier = candidate_models[best_index]["classifier"]
    model_classes = extract_model_classes(classifier)
    positive_class_index = get_positive_class_index(classifier, 1)
    selected_raw_validation_probability = np.clip(predict_positive_probability(raw_classifier, X_validation, 1), 1e-8, 1 - 1e-8)
    selected_calibrated_validation_probability = np.clip(predict_positive_probability(classifier, X_validation, 1), 1e-8, 1 - 1e-8)

    positive_magnitude = _fit_magnitude_side(
        fit, validation, features, direction_column, target, 1
    )
    negative_magnitude = _fit_magnitude_side(
        fit, validation, features, direction_column, target, -1
    )
    # Legacy scaler/key names remain present. New code uses the side-specific
    # preprocessors, which are fit only on the corresponding magnitude subset.
    amplitude_scaler = positive_magnitude["scaler"]
    positive_ridge = positive_magnitude["model"]
    negative_ridge = negative_magnitude["model"]
    large_threshold = float(fit[target].abs().quantile(LARGE_QUANTILE))
    large_threshold = max(large_threshold, NEUTRAL_RETURN_BAND)
    validation_numeric = _numeric_feature_frame(validation, features)
    validation_positive, validation_positive_audit = sanitize_magnitude_predictions(positive_ridge.predict(
        positive_magnitude["scaler"].transform(positive_magnitude["imputer"].transform(validation_numeric))
    ), positive_magnitude["magnitude_prediction_fallback_value"])
    validation_negative_magnitude, validation_negative_audit = sanitize_magnitude_predictions(negative_ridge.predict(
        negative_magnitude["scaler"].transform(negative_magnitude["imputer"].transform(validation_numeric))
    ), negative_magnitude["magnitude_prediction_fallback_value"])
    validation_positive = np.maximum(validation_positive, 0.0)
    validation_negative = -np.maximum(validation_negative_magnitude, 0.0)
    validation_probability = predict_positive_probability(classifier, X_validation, 1)
    validation_expected = validation_probability * validation_positive + (1.0 - validation_probability) * validation_negative
    validation_regression = safe_regression_metrics(
        pd.to_numeric(validation[target], errors="coerce"), validation_expected
    )
    return {
        "imputer": imputer, "amplitude_scaler": amplitude_scaler,
        "raw_classifier": raw_classifier, "classifier": classifier,
        "classifier_type": selected["classifier_type"],
        "calibration_method": selected["calibration_method"],
        "positive_ridge": positive_ridge, "negative_ridge": negative_ridge,
        "positive_model": positive_ridge, "negative_model": negative_ridge,
        "positive_imputer": positive_magnitude["imputer"],
        "negative_imputer": negative_magnitude["imputer"],
        "positive_scaler": positive_magnitude["scaler"],
        "negative_scaler": negative_magnitude["scaler"],
        "positive_magnitude": positive_magnitude,
        "negative_magnitude": negative_magnitude,
        "large_magnitude_threshold": large_threshold,
        "fit_start_date": fit[DATE_COL].min(), "fit_end_date": fit[DATE_COL].max(),
        "calibration_start_date": calibration[DATE_COL].min(),
        "calibration_end_date": calibration[DATE_COL].max(),
        "validation_start_date": validation[DATE_COL].min(),
        "validation_end_date": validation[DATE_COL].max(),
        "calibration_row_count": int(len(calibration)),
        "calibration_positive_rate": float(y_cal.mean()),
        "train_positive_rate": float(y_fit.mean()),
        "validation_positive_rate": float(y_validation.mean()),
        "raw_probability_mean_train": float(predict_positive_probability(raw_classifier, X_fit, 1).mean()),
        "raw_probability_mean_validation": float(predict_positive_probability(raw_classifier, X_validation, 1).mean()),
        "calibrated_probability_mean_validation": float(predict_positive_probability(classifier, X_validation, 1).mean()),
        "direction_threshold": selected["threshold_selection"]["direction_threshold"],
        "threshold_selection": selected["threshold_selection"],
        "threshold_objective": selected["threshold_selection"]["threshold_objective"],
        "threshold_validation_metric": selected["threshold_selection"]["threshold_validation_metric"],
        "class_weight_mode": selected["class_weight_mode"],
        "raw_scale_pos_weight": raw_weight, "used_scale_pos_weight": used_weight,
        "selected_classifier_params": selected["params"],
        "candidate_comparison": candidate_rows,
        "classifier_selection_status": "selected_eligible_candidate" if eligible_candidate_indices else "diagnostic_fallback_no_eligible_candidate",
        "calibration_comparison": selected["calibration_comparison"],
        "validation_brier_raw": float(brier_score_loss(y_validation, selected_raw_validation_probability)),
        "validation_brier_calibrated": float(brier_score_loss(y_validation, selected_calibrated_validation_probability)),
        "validation_log_loss_raw": float(log_loss(y_validation, selected_raw_validation_probability, labels=[0, 1])),
        "validation_log_loss_calibrated": float(log_loss(y_validation, selected_calibrated_validation_probability, labels=[0, 1])),
        "split_contract": split_contract,
        "group_ablation": _validation_group_ablation(fit, validation, horizon, features),
        "feature_importance": _feature_importance_snapshot(
            raw_classifier, X_validation, y_validation, features, seed
        ),
        "window_selection_metrics": {
            "direction": selected["validation_metrics"],
            "magnitude": {
                "spearman_correlation": validation_regression["spearman"],
                "spearman_status": validation_regression["spearman_status"],
                "spearman_valid_n": validation_regression["spearman_valid_n"],
                "positive_prediction_audit": validation_positive_audit,
                "negative_prediction_audit": validation_negative_audit,
            },
            "selection_data": "inner_validation_only",
            "threshold_selection_status": selected["threshold_selection"]["threshold_selection_status"],
            "threshold_selection_failed": selected["threshold_selection"]["threshold_fallback_used"],
            "calibration_warning": bool(
                abs(float(selected_calibrated_validation_probability.mean()) - float(y_validation.mean())) > 0.15
            ),
        },
        "positive_class_label": 1, "positive_class_index": positive_class_index,
        "negative_class_label": 0,
        "probability_orientation": "P(label_positive_excess=1)",
        "model_classes": model_classes.tolist() if model_classes is not None else None,
    }


def _predict_components(models: dict[str, Any], frame: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    numeric = _numeric_feature_frame(frame, features)
    X = models["imputer"].transform(numeric)
    probability = predict_positive_probability(models["classifier"], X, 1)
    if "positive_imputer" in models:
        X_positive = models["positive_scaler"].transform(models["positive_imputer"].transform(numeric))
        X_negative = models["negative_scaler"].transform(models["negative_imputer"].transform(numeric))
    else:
        X_positive = X_negative = models["amplitude_scaler"].transform(X)
    positive_raw = models.get("positive_model", models["positive_ridge"]).predict(X_positive)
    negative_raw = models.get("negative_model", models["negative_ridge"]).predict(X_negative)
    positive, positive_audit = sanitize_magnitude_predictions(
        positive_raw, models.get("positive_magnitude", {}).get("magnitude_prediction_fallback_value", 0.0)
    )
    negative_magnitude, negative_audit = sanitize_magnitude_predictions(
        negative_raw, models.get("negative_magnitude", {}).get("magnitude_prediction_fallback_value", 0.0)
    )
    positive = np.maximum(positive, 0.0)
    negative = -np.maximum(negative_magnitude, 0.0)
    expected, expected_audit = sanitize_magnitude_predictions(
        probability * positive + (1.0 - probability) * negative, 0.0
    )
    return {
        "probability": probability, "positive": positive, "negative": negative, "expected": expected,
        "positive_prediction_audit": positive_audit,
        "negative_prediction_audit": negative_audit,
        "expected_prediction_audit": expected_audit,
    }


def _logistic_baseline(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(train_scaled, y_train)
    return predict_positive_probability(model, test_scaled, 1)


def _baseline_metrics(
    train: pd.DataFrame, test: pd.DataFrame, horizon: int, features: list[str], seed: int,
) -> dict[str, Any]:
    direction = f"direction_label_{horizon}m"
    train_non_neutral = train[train[direction].isin([-1, 1])]
    test_non_neutral = test[test[direction].isin([-1, 1])]
    if train_non_neutral.empty or test_non_neutral.empty or train_non_neutral[direction].nunique() < 2:
        return {}
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(_numeric_feature_frame(train_non_neutral, features))
    X_test = imputer.transform(_numeric_feature_frame(test_non_neutral, features))
    y_train = train_non_neutral[direction].eq(1).astype(int)
    y_test = test_non_neutral[direction].eq(1).astype(int)
    prevalence = float(y_train.mean())
    majority = np.repeat(1.0 if prevalence >= 0.5 else 0.0, len(y_test))
    constant = np.repeat(prevalence, len(y_test))
    results = {
        "majority_class": _direction_metrics(y_test, majority),
        "constant_probability": _direction_metrics(y_test, constant),
        "logistic_full": _direction_metrics(y_test, _logistic_baseline(X_train, y_train, X_test)),
    }
    for name, layer_names in {
        "market_only": ["market_core", "cross_asset_allocation"],
        "manager_action_only": ["action_deltas", "rolling_deviations", "manager_pti_scores", "holdings_quality"],
    }.items():
        selected = [column for layer in layer_names for column in FEATURE_LAYERS[layer] if column in features]
        if selected:
            indices = [features.index(column) for column in selected]
            results[name] = _direction_metrics(
                y_test, _logistic_baseline(X_train[:, indices], y_train, X_test[:, indices])
            )
    ablations = {}
    for layer in ("manager_pti_scores", "action_deltas", "rolling_deviations", "cross_asset_allocation", "holdings_quality"):
        removed = set(FEATURE_LAYERS[layer])
        indices = [index for index, column in enumerate(features) if column not in removed]
        if indices:
            ablations[f"full_minus_{layer}"] = _direction_metrics(
                y_test, _logistic_baseline(X_train[:, indices], y_train, X_test[:, indices])
            )
    results["ablations"] = ablations
    return results


def _evaluate_variant(
    models: dict[str, Any], test: pd.DataFrame, horizon: int, features: list[str],
) -> dict[str, Any]:
    target = f"future_{horizon}m_excess_return"
    direction = f"direction_label_{horizon}m"
    components = _predict_components(models, test, features)
    non_neutral = test[direction].isin([-1, 1]).to_numpy()
    y_direction = test.loc[non_neutral, direction].eq(1).astype(int)
    direction_metrics = _direction_metrics(
        y_direction, components["probability"][non_neutral], models.get("direction_threshold", 0.5)
    )
    magnitude_metrics = _magnitude_metrics(
        test[target], test[direction], components["positive"], components["negative"], components["expected"]
    )
    predicted_classes = [
        classify_prediction(
            float(probability), float(expected), NEUTRAL_PROBABILITY_LOW,
            NEUTRAL_PROBABILITY_HIGH, NEUTRAL_RETURN_BAND,
            models["large_magnitude_threshold"],
        )[2]
        for probability, expected in zip(components["probability"], components["expected"])
    ]
    actual_classes = [_actual_class(float(value), models["large_magnitude_threshold"]) for value in test[target]]
    raw_test_probability = predict_positive_probability(
        models["raw_classifier"],
        models["imputer"].transform(_numeric_feature_frame(test, features)), 1,
    )
    test_positive_rate = float(y_direction.mean()) if len(y_direction) else None
    test_probability_mean = float(components["probability"][non_neutral].mean()) if non_neutral.any() else None
    validation_metrics = next(
        row["validation_metrics"] for row in models["candidate_comparison"]
        if row["params"] == models["selected_classifier_params"]
        and row["class_weight_mode"] == models["class_weight_mode"]
    )
    warning_reasons = []
    if direction_metrics.get("brier_score") is not None and direction_metrics["brier_score"] > validation_metrics["brier_score"] + 0.05:
        warning_reasons.append("test_brier_exceeds_validation_by_more_than_0.05")
    if test_positive_rate is not None and test_probability_mean is not None and abs(test_probability_mean - test_positive_rate) > 0.15:
        warning_reasons.append("calibrated_probability_mean_differs_from_actual_rate_by_more_than_0.15")
    return {
        "direction": direction_metrics,
        "magnitude": magnitude_metrics,
        "five_class": _five_class_metrics(actual_classes, predicted_classes),
        "test_positive_rate": test_positive_rate,
        "raw_probability_mean_test": float(raw_test_probability[non_neutral].mean()) if non_neutral.any() else None,
        "calibrated_probability_mean_test": test_probability_mean,
        "probability_calibration_warning": bool(warning_reasons),
        "probability_calibration_warning_reason": warning_reasons,
        "magnitude_prediction_audit": {
            "positive": components["positive_prediction_audit"],
            "negative": components["negative_prediction_audit"],
            "expected": components["expected_prediction_audit"],
            "magnitude_test_invalid_row_count": int(
                components["positive_prediction_audit"]["magnitude_invalid_prediction_count"]
                + components["negative_prediction_audit"]["magnitude_invalid_prediction_count"]
                + components["expected_prediction_audit"]["magnitude_invalid_prediction_count"]
            ),
        },
    }


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), nested, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        output[prefix] = float(value)


def summarize_folds(fold_metrics: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    flattened: list[dict[str, float]] = []
    for fold in fold_metrics:
        values: dict[str, float] = {}
        _flatten_numeric("", fold, values)
        flattened.append(values)
    keys = sorted(set().union(*(values.keys() for values in flattened))) if flattened else []
    rng = np.random.default_rng(seed)
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in flattened if key in row], dtype=float)
        if not len(values):
            continue
        bootstrap = np.asarray([
            rng.choice(values, size=len(values), replace=True).mean() for _ in range(1000)
        ])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "fold_count": int(len(values)),
        }
    return summary


def summarize_training_window(
    folds: list[dict[str, Any]], window_name: str = "unknown", required_fold_count: int | None = None,
) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for fold in folds:
            value: Any = fold
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None and np.isfinite(value):
                output.append(float(value))
        return output

    def stats(items: list[float], lower_is_better: bool = False) -> tuple[float | None, float | None, float | None]:
        if not items:
            return None, None, None
        return float(np.mean(items)), float(np.std(items, ddof=1)) if len(items) > 1 else 0.0, (
            float(max(items)) if lower_is_better else float(min(items))
        )

    required_fold_count = required_fold_count if required_fold_count is not None else len(folds)
    auc_mean, auc_std, auc_worst = stats(values(("models", "window_selection_metrics", "direction", "roc_auc")))
    pr_mean, pr_std, pr_worst = stats(values(("models", "window_selection_metrics", "direction", "pr_auc")))
    ba_mean, ba_std, ba_worst = stats(values(("models", "window_selection_metrics", "direction", "balanced_accuracy")))
    brier_mean, brier_std, brier_worst = stats(values(("models", "window_selection_metrics", "direction", "brier_score")), True)
    spearman_mean, spearman_std, spearman_worst = stats(values(("models", "window_selection_metrics", "magnitude", "spearman_correlation")))
    predicted_rates = values(("models", "window_selection_metrics", "direction", "predicted_positive_rate"))
    actual_rates = values(("models", "window_selection_metrics", "direction", "actual_positive_rate"))
    gaps = [predicted - actual for predicted, actual in zip(predicted_rates, actual_rates)]
    mean_abs_rate_gap = float(np.mean(np.abs(gaps))) if gaps else None
    collapse_count = sum(rate < 0.05 or rate > 0.95 for rate in predicted_rates)
    calibration_warning_count = sum(
        bool(fold["models"]["window_selection_metrics"].get("calibration_warning")) for fold in folds
    )
    threshold_selection_failed_count = sum(
        bool(fold["models"]["window_selection_metrics"].get("threshold_selection_failed")) for fold in folds
    )
    rejection_reasons = []
    if auc_worst is not None and auc_worst < 0.48:
        rejection_reasons.append("unstable_below_random_auc")
    if any(rate < 0.02 or rate > 0.98 for rate in predicted_rates):
        rejection_reasons.append("class_collapse")
    if mean_abs_rate_gap is not None and mean_abs_rate_gap > 0.25:
        rejection_reasons.append("probability_rate_drift")
    if threshold_selection_failed_count:
        rejection_reasons.append("threshold_selection_failure")
    if len(folds) < required_fold_count:
        rejection_reasons.append("insufficient_valid_inner_folds")
    if None in (auc_mean, auc_std, auc_worst, brier_mean):
        rejection_reasons.append("missing_required_inner_validation_metrics")

    below_random_penalty = max(0.0, 0.5 - auc_worst) if auc_worst is not None else 0.5
    warning_rate = calibration_warning_count / max(len(folds), 1)
    safe_mean_spearman = spearman_mean if spearman_mean is not None else 0.0
    safe_worst_spearman = spearman_worst if spearman_worst is not None else 0.0
    components = {
        "mean_auc_component": auc_mean or 0.0,
        "auc_std_penalty": -0.50 * (auc_std or 0.0),
        "worst_auc_component": 0.30 * (auc_worst or 0.0),
        "brier_penalty": -0.30 * (brier_mean or 0.0),
        "mean_spearman_component": 0.10 * safe_mean_spearman,
        "worst_spearman_component": 0.10 * safe_worst_spearman,
        "probability_gap_penalty": -0.50 * (mean_abs_rate_gap or 0.0),
        "class_collapse_penalty": -0.15 * collapse_count,
        "calibration_warning_penalty": -0.20 * warning_rate,
        "below_random_auc_penalty": -0.50 * below_random_penalty,
    }
    stability_score = float(sum(components.values()))
    metrics = {
        "window_name": window_name,
        "mean_inner_validation_auc": auc_mean,
        "inner_validation_auc_std": auc_std,
        "worst_inner_validation_auc": auc_worst,
        "mean_inner_validation_pr_auc": pr_mean,
        "inner_validation_pr_auc_std": pr_std,
        "worst_inner_validation_pr_auc": pr_worst,
        "mean_inner_validation_balanced_accuracy": ba_mean,
        "inner_validation_balanced_accuracy_std": ba_std,
        "worst_inner_validation_balanced_accuracy": ba_worst,
        "mean_inner_validation_brier": brier_mean,
        "inner_validation_brier_std": brier_std,
        "mean_inner_validation_spearman": spearman_mean,
        "inner_validation_spearman_std": spearman_std,
        "worst_inner_validation_spearman": spearman_worst,
        "mean_inner_validation_probability_rate_gap": float(np.mean(gaps)) if gaps else None,
        "mean_absolute_inner_validation_probability_rate_gap": mean_abs_rate_gap,
        "inner_validation_fold_count": len(folds),
        "train_row_count_by_fold": [fold["train_rows"] for fold in folds],
        "positive_rate_by_fold": [fold["models"]["train_positive_rate"] for fold in folds],
        "predicted_positive_rate_by_inner_fold": predicted_rates,
        "actual_positive_rate_by_inner_fold": actual_rates,
        "class_collapse_count": collapse_count,
        "calibration_warning_count": calibration_warning_count,
        "threshold_selection_failed_count": threshold_selection_failed_count,
        "eligible": not rejection_reasons,
        "status": "eligible" if not rejection_reasons else "rejected",
        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        "raw_score": stability_score,
        "score_components": components,
        "stability_score": stability_score,
        "selection_rank": None,
        "selection_data": "inner_validation_only",
        "stability_score_formula": "mean_auc - 0.50*auc_std + 0.30*worst_auc - 0.30*mean_brier + 0.10*mean_spearman + 0.10*worst_spearman - 0.50*mean_abs_rate_gap - 0.15*collapse_count - 0.20*calibration_warning_rate - 0.50*below_random_penalty",
    }
    return metrics


def select_window_from_eligible_candidates(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(candidates.values(), key=lambda row: row["stability_score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
    eligible = [row for row in ranked if row["eligible"]]
    selected = eligible[0] if eligible else ranked[0]
    return {
        "selected_window": selected["window_name"],
        "selected_window_score": selected["stability_score"],
        "automatic_best_window": selected["window_name"] if eligible else None,
        "selection_mode": "automatic",
        "selection_reason": "highest_score_among_eligible_candidates" if eligible else "least_bad_candidate_for_diagnostics",
        "automatic_selection_succeeded": bool(eligible),
        "selected_window_role": "production_candidate" if eligible else "diagnostic_fallback",
        "horizon_model_status": "stable_window_selected" if eligible else "no_stable_window_found",
    }


def summarize_outer_test_results(folds: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(section: str, key: str) -> list[float]:
        return [float(fold[section][key]) for fold in folds if fold.get(section, {}).get(key) is not None]
    auc = collect("direction", "roc_auc")
    balanced = collect("direction", "balanced_accuracy")
    brier = collect("direction", "brier_score")
    spearman = collect("magnitude", "spearman_correlation")
    gaps = collect("direction", "probability_rate_gap")
    return {
        "mean_outer_test_auc": float(np.mean(auc)) if auc else None,
        "outer_test_auc_std": float(np.std(auc, ddof=1)) if len(auc) > 1 else (0.0 if auc else None),
        "worst_outer_test_auc": float(min(auc)) if auc else None,
        "mean_outer_test_balanced_accuracy": float(np.mean(balanced)) if balanced else None,
        "worst_outer_test_balanced_accuracy": float(min(balanced)) if balanced else None,
        "mean_outer_test_brier": float(np.mean(brier)) if brier else None,
        "mean_outer_test_spearman": float(np.mean(spearman)) if spearman else None,
        "worst_outer_test_spearman": float(min(spearman)) if spearman else None,
        "mean_outer_test_probability_rate_gap": float(np.mean(gaps)) if gaps else None,
        "outer_test_calibration_warning_count": sum(bool(fold.get("probability_calibration_warning")) for fold in folds),
        "outer_test_fold_count": len(folds),
    }


def summarize_feature_stability(
    fold_rows: list[dict[str, Any]], features: list[str], top_k: int = 15,
) -> dict[str, Any]:
    group_results: dict[str, Any] = {}
    selected_groups, excluded_groups = [], []
    for group in FEATURE_GROUPS:
        rows = [
            fold["models"]["group_ablation"]["groups"].get(group, {}) for fold in fold_rows
            if fold["models"].get("group_ablation")
        ]
        valid = [row for row in rows if row.get("status") == "ok"]
        if not valid:
            group_results[group] = {"status": "missing_group_features"}
            excluded_groups.append(group)
            continue
        auc = [row["delta_auc"] for row in valid if row.get("delta_auc") is not None]
        brier = [row["delta_brier"] for row in valid if row.get("delta_brier") is not None]
        positive_ratio = float(np.mean([value > 0 for value in auc])) if auc else 0.0
        keep = bool(
            auc and positive_ratio >= 2 / 3 and np.mean(auc) > 0
            and min(auc) > -0.02 and brier and np.mean(brier) <= 0
        )
        group_results[group] = {
            "status": "selected" if keep else "excluded_unstable",
            "positive_delta_fold_ratio": positive_ratio,
            "mean_delta_auc": float(np.mean(auc)) if auc else None,
            "worst_delta_auc": float(min(auc)) if auc else None,
            "mean_delta_brier": float(np.mean(brier)) if brier else None,
            "fold_results": valid,
        }
        (selected_groups if keep else excluded_groups).append(group)

    importance_rows = [fold["models"].get("feature_importance", {}) for fold in fold_rows]
    rank_stability: dict[str, Any] = {}
    for feature in features:
        ranks = [row.get("feature_rank", {}).get(feature) for row in importance_rows]
        ranks = [int(rank) for rank in ranks if rank is not None]
        if not ranks:
            continue
        rank_stability[feature] = {
            "mean_rank": float(np.mean(ranks)),
            "rank_std": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
            "rank_min": int(min(ranks)), "rank_max": int(max(ranks)),
            "top_k_frequency": float(np.mean([rank <= top_k for rank in ranks])),
        }
    return {
        "feature_groups": FEATURE_GROUPS, "group_ablation_results": group_results,
        "selected_feature_groups": selected_groups, "excluded_feature_groups": excluded_groups,
        "feature_rank_stability": rank_stability,
        "feature_sign_stability": {
            feature: "unavailable_for_permutation_importance" for feature in rank_stability
        },
        "importance_method_by_fold": [row.get("importance_method") for row in importance_rows],
        "selection_rules": {
            "positive_delta_fold_ratio_min": 2 / 3, "mean_delta_auc": "> 0",
            "worst_delta_auc": "> -0.02", "mean_delta_brier": "<= 0",
        },
    }


def _build_drift_reference(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    reference = {
        "feature_training_min": {}, "feature_training_max": {},
        "feature_training_quantiles": {}, "feature_reference_bins": {},
        "feature_reference_distribution": {}, "feature_training_missing_rate": {},
    }
    for feature in features:
        series = pd.to_numeric(frame[feature], errors="coerce")
        valid = series.dropna()
        reference["feature_training_missing_rate"][feature] = float(series.isna().mean())
        if valid.empty:
            reference["feature_training_min"][feature] = None
            reference["feature_training_max"][feature] = None
            reference["feature_training_quantiles"][feature] = {"p01": None, "p50": None, "p99": None}
            reference["feature_reference_bins"][feature] = []
            reference["feature_reference_distribution"][feature] = []
            continue
        quantiles = valid.quantile([0.01, 0.50, 0.99])
        edges = np.unique(valid.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float))
        if len(edges) >= 2:
            edges[0], edges[-1] = -np.inf, np.inf
            counts, _ = np.histogram(valid.to_numpy(dtype=float), bins=edges)
            distribution = (counts / counts.sum()).tolist()
            serial_edges = [None if not np.isfinite(value) else float(value) for value in edges]
        else:
            serial_edges, distribution = [], []
        reference["feature_training_min"][feature] = float(valid.min())
        reference["feature_training_max"][feature] = float(valid.max())
        reference["feature_training_quantiles"][feature] = {
            "p01": float(quantiles.loc[0.01]), "p50": float(quantiles.loc[0.50]),
            "p99": float(quantiles.loc[0.99]),
        }
        reference["feature_reference_bins"][feature] = serial_edges
        reference["feature_reference_distribution"][feature] = distribution
    return reference


def _regime_level_metrics(
    models: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame,
    horizon: int, features: list[str],
) -> dict[str, Any]:
    available = set(features) & set(FEATURE_GROUPS["regime_features"])
    if not available:
        return {"status": "unavailable_no_point_in_time_regime_features"}
    thresholds = {
        feature: float(pd.to_numeric(train[feature], errors="coerce").median())
        for feature in available
    }
    definitions: dict[str, pd.Series] = {}
    if "lag1_interest_rate_level" in available:
        definitions["high_rate"] = pd.to_numeric(test["lag1_interest_rate_level"], errors="coerce") >= thresholds["lag1_interest_rate_level"]
    if "lag1_interest_rate_change_3m" in available:
        definitions["rising_rate"] = pd.to_numeric(test["lag1_interest_rate_change_3m"], errors="coerce") > 0
    if "lag1_market_volatility_12m" in available:
        definitions["high_volatility"] = pd.to_numeric(test["lag1_market_volatility_12m"], errors="coerce") >= thresholds["lag1_market_volatility_12m"]
    if "lag1_market_drawdown_12m" in available:
        definitions["market_drawdown"] = pd.to_numeric(test["lag1_market_drawdown_12m"], errors="coerce") < thresholds["lag1_market_drawdown_12m"]
    if "lag1_market_return_12m" in available:
        trend = pd.to_numeric(test["lag1_market_return_12m"], errors="coerce")
        definitions["bull_trend"] = trend > 0
        definitions["bear_trend"] = trend < 0
    components = _predict_components(models, test, features)
    direction_column = f"direction_label_{horizon}m"
    target = f"future_{horizon}m_excess_return"
    output: dict[str, Any] = {"status": "ok", "train_derived_thresholds": thresholds, "regimes": {}}
    for name, mask in definitions.items():
        mask = mask.fillna(False).to_numpy()
        subset_direction = test.loc[mask, direction_column]
        non_neutral_local = subset_direction.isin([-1, 1]).to_numpy()
        global_indices = np.where(mask)[0]
        direction_indices = global_indices[non_neutral_local]
        row_count = int(len(direction_indices))
        y = test.iloc[direction_indices][direction_column].eq(1).astype(int)
        if row_count < MIN_CLASS_SUPPORT or y.nunique() < 2:
            output["regimes"][name] = {
                "status": "insufficient_support", "row_count": row_count,
                "positive_count": int(y.sum()), "negative_count": int((1 - y).sum()),
                "positive_rate": float(y.mean()) if row_count else None,
            }
            continue
        direction_metrics = _direction_metrics(y, components["probability"][direction_indices], models["direction_threshold"])
        actual = pd.to_numeric(test.iloc[direction_indices][target], errors="coerce")
        predicted = components["expected"][direction_indices]
        spearman = actual.reset_index(drop=True).corr(pd.Series(predicted), method="spearman")
        output["regimes"][name] = {
            "status": "ok", "row_count": row_count, "positive_count": int(y.sum()),
            "negative_count": int((1 - y).sum()), "positive_rate": float(y.mean()),
            "roc_auc": direction_metrics["roc_auc"], "pr_auc": direction_metrics["pr_auc"],
            "brier_score": direction_metrics["brier_score"],
            "balanced_accuracy": direction_metrics["balanced_accuracy"],
            "expected_excess_mae": float(mean_absolute_error(actual, predicted)),
            "spearman_correlation": None if pd.isna(spearman) else float(spearman),
            "direction_hit_rate": float(np.mean(np.sign(actual) == np.sign(predicted))),
        }
    return output


def _feature_distribution_drift(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str],
) -> dict[str, Any]:
    rows = []
    for feature in features:
        train_values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        validation_values = pd.to_numeric(validation[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_values = pd.to_numeric(test[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_train = train_values.dropna()
        finite_test = test_values.dropna()
        if finite_train.empty:
            continue
        q01, q25, q50, q75, q99 = finite_train.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        train_iqr = float(q75 - q25)
        test_q = finite_test.quantile([0.25, 0.50, 0.75]) if not finite_test.empty else pd.Series({0.25: np.nan, 0.50: np.nan, 0.75: np.nan})
        out_of_range = float(((finite_test < q01) | (finite_test > q99)).mean()) if not finite_test.empty else None
        edges = np.unique(finite_train.quantile(np.linspace(0, 1, 11)).to_numpy(dtype=float))
        psi = None
        if len(edges) >= 2 and not finite_test.empty:
            edges[0], edges[-1] = -np.inf, np.inf
            expected, _ = np.histogram(finite_train, bins=edges)
            actual, _ = np.histogram(finite_test, bins=edges)
            expected = np.clip(expected / max(expected.sum(), 1), 1e-6, None)
            actual = np.clip(actual / max(actual.sum(), 1), 1e-6, None)
            psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        missing_shift = float(test_values.isna().mean() - train_values.isna().mean())
        normalized_median_shift = (
            abs(float(test_q.loc[0.50]) - float(q50)) / train_iqr
            if train_iqr > 0 and pd.notna(test_q.loc[0.50]) else 0.0
        )
        severity = normalized_median_shift + (out_of_range or 0.0) + abs(missing_shift) + (psi or 0.0)
        rows.append({
            "feature": feature, "train_median": float(q50),
            "validation_median": None if validation_values.dropna().empty else float(validation_values.median()),
            "test_median": None if pd.isna(test_q.loc[0.50]) else float(test_q.loc[0.50]),
            "train_iqr": train_iqr,
            "test_iqr": None if pd.isna(test_q.loc[0.75]) else float(test_q.loc[0.75] - test_q.loc[0.25]),
            "psi": psi, "test_outside_train_p01_p99_rate": out_of_range,
            "train_missing_rate": float(train_values.isna().mean()),
            "test_missing_rate": float(test_values.isna().mean()),
            "missing_rate_shift": missing_shift, "severity_score": float(severity),
        })
    rows.sort(key=lambda row: row["severity_score"], reverse=True)
    return {"top_drift_features": rows[:20], "evaluated_feature_count": len(rows)}


def build_outer_fold_diagnostic(
    fold_number: int, models: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame,
    contract: dict[str, str], evaluation: dict[str, Any], horizon: int, features: list[str],
) -> dict[str, Any]:
    validation = train[train[DATE_COL].between(models["validation_start_date"], models["validation_end_date"])]
    direction = f"direction_label_{horizon}m"
    train_non_neutral = train[train[direction].isin([-1, 1])]
    validation_non_neutral = validation[validation[direction].isin([-1, 1])]
    test_non_neutral = test[test[direction].isin([-1, 1])]
    train_rate = float(train_non_neutral[direction].eq(1).mean()) if len(train_non_neutral) else None
    validation_rate = float(validation_non_neutral[direction].eq(1).mean()) if len(validation_non_neutral) else None
    test_rate = float(test_non_neutral[direction].eq(1).mean()) if len(test_non_neutral) else None
    benchmark = {}
    if "lag1_market_return_12m" in features:
        train_market = pd.to_numeric(train["lag1_market_return_12m"], errors="coerce").dropna()
        test_market = pd.to_numeric(test["lag1_market_return_12m"], errors="coerce").dropna()
        if not train_market.empty:
            benchmark = {
                "train_quantiles": {str(q): float(train_market.quantile(q)) for q in (0.1, 0.5, 0.9)},
                "test_median": float(test_market.median()) if not test_market.empty else None,
            }
    return {
        "outer_fold_id": fold_number,
        "train_period": {"start": train[DATE_COL].min().strftime("%Y-%m-%d"), "end": train[DATE_COL].max().strftime("%Y-%m-%d")},
        "validation_period": {"start": models["validation_start_date"].strftime("%Y-%m-%d"), "end": models["validation_end_date"].strftime("%Y-%m-%d")},
        "test_period": {"start": contract["test_start_date"], "end": contract["test_end_date"]},
        "actual_positive_rate": evaluation["direction"].get("actual_positive_rate"),
        "mean_positive_probability": evaluation.get("calibrated_probability_mean_test"),
        "predicted_positive_rate": evaluation["direction"].get("predicted_positive_rate"),
        "probability_rate_gap": evaluation["direction"].get("probability_rate_gap"),
        "auc": evaluation["direction"].get("roc_auc"),
        "balanced_accuracy": evaluation["direction"].get("balanced_accuracy"),
        "brier": evaluation["direction"].get("brier_score"),
        "magnitude_spearman": evaluation["magnitude"].get("spearman_correlation"),
        "magnitude_spearman_status": evaluation["magnitude"].get("spearman_status"),
        "feature_distribution_drift": _feature_distribution_drift(train, validation, test, features),
        "label_distribution_drift": {
            "train_positive_rate": train_rate, "validation_positive_rate": validation_rate,
            "test_positive_rate": test_rate,
            "absolute_prevalence_shift": abs(test_rate - train_rate) if test_rate is not None and train_rate is not None else None,
        },
        "regime_metrics": evaluation.get("regime_metrics", {}),
        "benchmark_return_quantile": benchmark,
        "calibration_warning": evaluation.get("probability_calibration_warning", False),
    }


def summarize_manager_action_contribution(
    primary_folds: list[dict[str, Any]], baseline_folds: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for primary, baselines in zip(primary_folds, baseline_folds):
        market = baselines.get("market_only") or {}
        if primary.get("direction", {}).get("roc_auc") is None or market.get("roc_auc") is None:
            continue
        rows.append({
            "full_model": primary["direction"], "market_only": market,
            "manager_action_only": baselines.get("manager_action_only"),
            "incremental_manager_action_auc": primary["direction"]["roc_auc"] - market["roc_auc"],
            "incremental_manager_action_balanced_accuracy": primary["direction"]["balanced_accuracy"] - market["balanced_accuracy"],
        })
    increments = [row["incremental_manager_action_auc"] for row in rows]
    positive = sum(value > 0 for value in increments)
    negative = sum(value < 0 for value in increments)
    status = "consistent_positive" if rows and positive == len(rows) else "consistent_negative" if rows and negative == len(rows) else "regime_dependent"
    return {
        "fold_results": rows,
        "mean_incremental_manager_action_auc": float(np.mean(increments)) if increments else None,
        "positive_increment_fold_count": positive, "negative_increment_fold_count": negative,
        "manager_action_contribution_status": status,
    }


def validate_feature_artifact_consistency(bundle: dict[str, Any]) -> None:
    features = list(bundle.get("features") or [])
    if bundle.get("feature_count") != len(features):
        raise ValueError(
            f"feature_count mismatch: metadata={bundle.get('feature_count')}, len(features)={len(features)}"
        )
    quantiles = bundle.get("feature_training_quantiles") or {}
    required = list(bundle.get("features_requiring_quantiles", features))
    missing_quantiles = [feature for feature in required if feature not in quantiles]
    if missing_quantiles:
        raise ValueError("Missing feature training quantiles for: " + ", ".join(missing_quantiles[:20]))
    if bundle.get("feature_training_quantile_count") != len(required):
        raise ValueError("feature_training_quantile_count does not match features_requiring_quantiles")
    if bundle.get("probability_orientation") != "P(label_positive_excess=1)":
        raise ValueError("Bundle probability orientation is missing or incorrect.")
    if bundle.get("positive_class_label") != 1:
        raise ValueError("Bundle positive_class_label must be 1.")


def validate_report_invariants(report: dict[str, Any], bundle: dict[str, Any]) -> None:
    if report.get("outer_test_used_for_selection") is not False:
        raise AssertionError("Outer test must never be used for selection.")
    if report.get("feature_count") != len(bundle["features"]):
        raise AssertionError("Report feature_count does not match bundle features.")
    if report.get("positive_probability_definition") != "P(label_positive_excess=1)":
        raise AssertionError("Report positive probability definition is incorrect.")
    summary = report["window_selection_validation_summary"]
    if summary.get("selection_mode", "automatic") == "automatic":
        eligible = [item for item in summary["candidates"].values() if item["eligible"]]
        if eligible:
            expected = max(eligible, key=lambda item: item["stability_score"])["window_name"]
            if summary["selected_window"] != expected:
                raise AssertionError(
                    f"Window selection mismatch: selected={summary['selected_window']}, expected={expected}"
                )
    threshold = float(report["direction_threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise AssertionError("Direction threshold is invalid.")


def train_horizon(df: pd.DataFrame, features: list[str], horizon: int, model_dir: Path, seed: int) -> dict[str, Any]:
    work = _prepare_horizon(df, horizon)
    folds = date_grouped_walk_forward(work, horizon)
    window_folds: dict[str, list[dict[str, Any]]] = {name: [] for name in TRAINING_WINDOWS}
    failed_windows: list[dict[str, Any]] = []
    for fold_number, (outer_train, test, contract) in enumerate(folds, start=1):
        test_start = pd.Timestamp(contract["test_start_date"])
        for window_mode, window_years in TRAINING_WINDOWS.items():
            train = _windowed_train(outer_train, test_start, horizon, window_years)
            try:
                models = _fit_models(train, horizon, features, seed + fold_number, "primary")
            except ValueError as exc:
                failed_windows.append({"fold": fold_number, "window": window_mode, "reason": str(exc)})
                continue
            window_folds[window_mode].append({
                "fold": fold_number, **contract, "train_rows": int(len(train)),
                "test_rows": int(len(test)), "window_years": window_years,
                "train_start_date": train[DATE_COL].min().strftime("%Y-%m-%d"),
                "train_label_end_max": train[f"label_end_date_{horizon}m"].max().strftime("%Y-%m-%d"),
                "models": {
                    key: models[key] for key in (
                        "train_positive_rate", "validation_positive_rate", "calibration_method",
                        "direction_threshold", "class_weight_mode", "used_scale_pos_weight",
                    )
                },
            })
            window_folds[window_mode][-1]["models"]["group_ablation"] = models["group_ablation"]
            window_folds[window_mode][-1]["models"]["feature_importance"] = models["feature_importance"]
            window_folds[window_mode][-1]["models"]["positive_model_type"] = models["positive_magnitude"]["model_type"]
            window_folds[window_mode][-1]["models"]["negative_model_type"] = models["negative_magnitude"]["model_type"]
            window_folds[window_mode][-1]["models"]["window_selection_metrics"] = models["window_selection_metrics"]
    window_comparison = {
        mode: summarize_training_window(rows, mode, len(folds)) for mode, rows in window_folds.items()
    }
    selection = select_window_from_eligible_candidates(window_comparison)
    selected_window = selection["selected_window"]
    eligible = [candidate for candidate in window_comparison.values() if candidate["eligible"]]
    if eligible:
        expected_window = max(eligible, key=lambda item: item["stability_score"])["window_name"]
        assert selected_window == expected_window, (
            f"Window selection metadata mismatch: selected={selected_window}, expected={expected_window}"
        )
    selected_years = TRAINING_WINDOWS[selected_window]
    feature_stability = summarize_feature_stability(window_folds[selected_window], features)
    selected_groups = feature_stability["selected_feature_groups"]
    selected_features = [
        feature for feature in features
        if any(feature in FEATURE_GROUPS[group] for group in selected_groups)
    ]
    if not selected_features:
        selected_features = features
        feature_stability["selection_status"] = "fallback_to_full_feature_set_no_group_passed"
        selected_groups = [group for group, values in FEATURE_GROUPS.items() if any(feature in features for feature in values)]
    else:
        feature_stability["selection_status"] = "selected_from_validation_ablation"
    primary_folds, portable_folds, baseline_folds, fold_contracts = [], [], [], []
    for fold_number, (outer_train, test, contract) in enumerate(folds, start=1):
        train = _windowed_train(
            outer_train, pd.Timestamp(contract["test_start_date"]), horizon, selected_years
        )
        primary = _fit_models(train, horizon, selected_features, seed + fold_number, "primary")
        portable = _fit_models(train, horizon, selected_features, seed + fold_number, "sklearn")
        primary_evaluation = _evaluate_variant(primary, test, horizon, selected_features)
        primary_evaluation["regime_metrics"] = _regime_level_metrics(
            primary, train, test, horizon, selected_features
        )
        portable_evaluation = _evaluate_variant(portable, test, horizon, selected_features)
        primary_folds.append(primary_evaluation)
        portable_folds.append(portable_evaluation)
        # Benchmarks use the full approved feature pool, fit only on this fold's
        # training rows. This keeps market-only and manager-action-only controls
        # available even when feature stability selects a narrower final bundle.
        baselines = _baseline_metrics(train, test, horizon, features, seed + fold_number)
        baseline_folds.append(baselines)
        primary_record = next(row for row in window_folds[selected_window] if row["fold"] == fold_number)
        diagnostic = build_outer_fold_diagnostic(
            fold_number, primary, train, test, contract, primary_evaluation, horizon, selected_features
        )
        fold_contracts.append({
            **primary_record, "primary": primary_evaluation,
            "portable": portable_evaluation, "baselines": baselines,
            "outer_fold_diagnostic": diagnostic,
        })

    final_test_start = pd.Timestamp(folds[-1][2]["test_start_date"])
    eligible_final = work[work[f"label_end_date_{horizon}m"] < final_test_start - pd.DateOffset(months=EMBARGO_MONTHS)].copy()
    final_train = _windowed_train(eligible_final, final_test_start, horizon, selected_years)
    primary_final = _fit_models(final_train, horizon, selected_features, seed, "primary")
    portable_final = _fit_models(final_train, horizon, selected_features, seed, "sklearn")
    feature_schema_hash = _hash_text(json.dumps(selected_features, ensure_ascii=False, separators=(",", ":")))
    training_hash = _training_data_hash(
        final_train, ["event_id", DATE_COL, f"future_{horizon}m_excess_return", *selected_features]
    )
    drift_reference = _build_drift_reference(final_train, selected_features)

    metrics = {
        "walk_forward_folds": fold_contracts,
        "window_comparison": window_comparison,
        "failed_windows": failed_windows,
        "selected_window_by_horizon": selected_window,
        "window_selection_validation_summary": {
            "horizon": f"{horizon}m", "candidates": window_comparison,
            **selection,
        },
        "outer_test_summary": summarize_outer_test_results(primary_folds),
        "outer_fold_diagnostics": [fold["outer_fold_diagnostic"] for fold in fold_contracts],
        "selected_threshold_by_horizon": primary_final["direction_threshold"],
        "selected_calibration_method_by_horizon": primary_final["calibration_method"],
        "selected_class_weight_mode_by_horizon": primary_final["class_weight_mode"],
        "feature_stability": feature_stability,
        "regime_level_metrics": [fold.get("regime_metrics", {}) for fold in primary_folds],
        "primary_summary": summarize_folds(primary_folds, seed),
        "portable_summary": summarize_folds(portable_folds, seed + 1),
        "baseline_summary": summarize_folds(baseline_folds, seed + 2),
        "manager_action_feature_conclusion": summarize_manager_action_contribution(primary_folds, baseline_folds),
    }
    five_class_reasons = sorted({
        reason for fold in primary_folds
        for reason in fold.get("five_class", {}).get("five_class_warning_reasons", [])
    })

    def make_bundle(models: dict[str, Any], variant: str) -> dict[str, Any]:
        return {
            "bundle_version": "v003", "schema_version": 3, "report_schema_version": "2.0",
            "horizon_months": horizon,
            "features": selected_features, "feature_count": len(selected_features),
            "features_requiring_quantiles": selected_features,
            "feature_training_quantile_count": len(selected_features),
            "forbidden_features": sorted(FORBIDDEN_COLUMNS),
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES), "feature_layers": FEATURE_LAYERS,
            "imputer": models["imputer"], "amplitude_scaler": models["amplitude_scaler"],
            "raw_classifier": models["raw_classifier"], "classifier": models["classifier"],
            "classifier_type": models["classifier_type"], "calibration_method": models["calibration_method"],
            "positive_class_label": models["positive_class_label"],
            "positive_class_index": models["positive_class_index"],
            "negative_class_label": models["negative_class_label"],
            "probability_orientation": models["probability_orientation"],
            "model_classes": models["model_classes"],
            "positive_ridge": models["positive_ridge"], "negative_ridge": models["negative_ridge"],
            "positive_model": models["positive_model"], "negative_model": models["negative_model"],
            "positive_imputer": models["positive_imputer"], "negative_imputer": models["negative_imputer"],
            "positive_scaler": models["positive_scaler"], "negative_scaler": models["negative_scaler"],
            "negative_model_predicts_absolute_magnitude": True,
            "neutral_return_band": NEUTRAL_RETURN_BAND,
            "neutral_probability_low": NEUTRAL_PROBABILITY_LOW,
            "neutral_probability_high": NEUTRAL_PROBABILITY_HIGH,
            "large_magnitude_threshold": models["large_magnitude_threshold"],
            "training_window_mode": selected_window,
            "training_window_years": selected_years,
            "window_selection_score": window_comparison[selected_window]["stability_score"],
            "window_selection_metrics": window_comparison[selected_window],
            "compared_training_windows": window_comparison,
            "window_selection_validation_summary": metrics["window_selection_validation_summary"],
            "outer_test_summary": metrics["outer_test_summary"],
            "outer_test_used_for_selection": False,
            "horizon_model_status": selection["horizon_model_status"],
            "selected_window_role": selection["selected_window_role"],
            "five_class_status": "production_ready" if not five_class_reasons else "experimental",
            "five_class_production_ready": not five_class_reasons,
            "five_class_warning_reasons": five_class_reasons,
            "calibration_start_date": models["calibration_start_date"].strftime("%Y-%m-%d"),
            "calibration_end_date": models["calibration_end_date"].strftime("%Y-%m-%d"),
            "calibration_row_count": models["calibration_row_count"],
            "calibration_positive_rate": models["calibration_positive_rate"],
            "train_positive_rate": models["train_positive_rate"],
            "validation_positive_rate": models["validation_positive_rate"],
            "raw_probability_mean_train": models["raw_probability_mean_train"],
            "raw_probability_mean_validation": models["raw_probability_mean_validation"],
            "calibrated_probability_mean_validation": models["calibrated_probability_mean_validation"],
            "calibration_comparison": models["calibration_comparison"],
            "validation_brier_raw": models["validation_brier_raw"],
            "validation_brier_calibrated": models["validation_brier_calibrated"],
            "validation_log_loss_raw": models["validation_log_loss_raw"],
            "validation_log_loss_calibrated": models["validation_log_loss_calibrated"],
            "direction_threshold": models["direction_threshold"],
            "threshold_objective": models["threshold_objective"],
            "threshold_validation_metric": models["threshold_validation_metric"],
            **models["threshold_selection"],
            "class_weight_mode": models["class_weight_mode"],
            "raw_scale_pos_weight": models["raw_scale_pos_weight"],
            "used_scale_pos_weight": models["used_scale_pos_weight"],
            "selected_classifier_params": models["selected_classifier_params"],
            "classifier_candidate_comparison": models["candidate_comparison"],
            "probability_calibration_warning": any(
                fold.get("probability_calibration_warning", False) for fold in primary_folds
            ),
            "probability_calibration_warning_reason": sorted({
                reason for fold in primary_folds
                for reason in fold.get("probability_calibration_warning_reason", [])
            }),
            "calibration_warning_reasons": sorted({
                reason for fold in primary_folds
                for reason in fold.get("probability_calibration_warning_reason", [])
            }),
            "positive_model_type": models["positive_magnitude"]["model_type"],
            "negative_model_type": models["negative_magnitude"]["model_type"],
            "positive_model_params": models["positive_magnitude"]["model_params"],
            "negative_model_params": models["negative_magnitude"]["model_params"],
            "positive_target_winsorized": models["positive_magnitude"]["target_winsorized"],
            "negative_target_winsorized": models["negative_magnitude"]["target_winsorized"],
            "positive_winsor_lower": models["positive_magnitude"]["winsor_lower"],
            "positive_winsor_upper": models["positive_magnitude"]["winsor_upper"],
            "negative_winsor_lower": models["negative_magnitude"]["winsor_lower"],
            "negative_winsor_upper": models["negative_magnitude"]["winsor_upper"],
            "positive_validation_mae": models["positive_magnitude"]["validation_mae"],
            "positive_validation_spearman": models["positive_magnitude"]["validation_spearman"],
            "negative_validation_mae": models["negative_magnitude"]["validation_mae"],
            "negative_validation_spearman": models["negative_magnitude"]["validation_spearman"],
            "positive_magnitude_status": models["positive_magnitude"]["magnitude_status"],
            "negative_magnitude_status": models["negative_magnitude"]["magnitude_status"],
            "positive_magnitude_audit": {
                key: value for key, value in models["positive_magnitude"].items()
                if key.startswith("magnitude_") and key not in {"model"}
            },
            "negative_magnitude_audit": {
                key: value for key, value in models["negative_magnitude"].items()
                if key.startswith("magnitude_") and key not in {"model"}
            },
            "magnitude_model_comparison": {
                "positive": models["positive_magnitude"]["candidate_comparison"],
                "negative": models["negative_magnitude"]["candidate_comparison"],
            },
            "selected_feature_groups": selected_groups,
            "excluded_feature_groups": feature_stability["excluded_feature_groups"],
            **drift_reference,
            "psi_interpretation": {
                "stable": "PSI < 0.10", "moderate_drift": "0.10 <= PSI <= 0.25",
                "high_drift": "PSI > 0.25", "status": "heuristic_thresholds",
            },
            "train_start_date": models["fit_start_date"].strftime("%Y-%m-%d"),
            "train_end_date": models["fit_end_date"].strftime("%Y-%m-%d"),
            "validation_end_date": models["validation_end_date"].strftime("%Y-%m-%d"),
            "validation_start_date": models["validation_start_date"].strftime("%Y-%m-%d"),
            "test_start_date": final_test_start.strftime("%Y-%m-%d"),
            "feature_schema_hash": feature_schema_hash, "training_data_hash": training_hash,
            "model_variant": variant,
            "metrics": metrics["primary_summary" if variant == "primary" else "portable_summary"],
        }

    primary_bundle = make_bundle(primary_final, "primary")
    portable_bundle = make_bundle(portable_final, "sklearn")
    validate_feature_artifact_consistency(primary_bundle)
    validate_feature_artifact_consistency(portable_bundle)
    bundle_path = model_dir / f"dual_stage_model_{horizon}m.pkl"
    fallback_path = model_dir / f"dual_stage_model_{horizon}m_sklearn.pkl"
    joblib.dump(primary_bundle, bundle_path)
    joblib.dump(portable_bundle, fallback_path)
    reloaded_primary = dict(joblib.load(bundle_path))
    reloaded_portable = dict(joblib.load(fallback_path))
    validate_feature_artifact_consistency(reloaded_primary)
    validate_feature_artifact_consistency(reloaded_portable)
    if reloaded_primary["features"] != primary_bundle["features"]:
        raise AssertionError("Primary bundle feature order changed during artifact round trip.")
    if reloaded_portable["features"] != portable_bundle["features"]:
        raise AssertionError("Portable bundle feature order changed during artifact round trip.")
    smoke_row = _numeric_feature_frame(final_train.head(1), selected_features)
    reloaded_primary["imputer"].transform(smoke_row)
    reloaded_portable["imputer"].transform(smoke_row)
    joblib.dump(primary_bundle["raw_classifier"], model_dir / f"direction_model_{horizon}m.pkl")
    joblib.dump(primary_bundle["positive_ridge"], model_dir / f"positive_ridge_{horizon}m.pkl")
    joblib.dump(primary_bundle["negative_ridge"], model_dir / f"negative_ridge_{horizon}m.pkl")
    horizon_report = {
        "report_schema_version": "2.0",
        "evaluation_protocol": "nested_time_series_validation",
        "outer_test_used_for_selection": False,
        "bundle_version": "v003", "horizon": f"{horizon}m",
        "horizon_model_status": selection["horizon_model_status"],
        "feature_count": len(primary_bundle["features"]),
        "features": primary_bundle["features"],
        "positive_probability_definition": "P(label_positive_excess=1)",
        "positive_class_label": primary_bundle["positive_class_label"],
        "positive_class_index": primary_bundle["positive_class_index"],
        "model_classes": primary_bundle["model_classes"],
        "direction_threshold": primary_bundle["direction_threshold"],
        "window_selection_validation_summary": metrics["window_selection_validation_summary"],
        "outer_test_summary": metrics["outer_test_summary"],
        "outer_fold_diagnostics": metrics["outer_fold_diagnostics"],
        "manager_action_feature_conclusion": metrics["manager_action_feature_conclusion"],
        "five_class_status": "production_ready" if not five_class_reasons else "experimental",
        "five_class_production_ready": not five_class_reasons,
        "five_class_warning_reasons": five_class_reasons,
        "deprecated_fields": {
            "window_comparison.worst_fold_auc": {
                "replacement": "window_selection_validation_summary.candidates.*.worst_inner_validation_auc"
            }
        },
    }
    validate_report_invariants(horizon_report, primary_bundle)
    return {
        "bundle": str(bundle_path), "portable_fallback_bundle": str(fallback_path),
        "target": f"future_{horizon}m_excess_return", "feature_count": len(selected_features),
        "feature_schema_hash": feature_schema_hash, "training_data_hash": training_hash,
        "metrics": metrics,
        "feature_stability": feature_stability,
        "magnitude_model_comparison": {
            "positive": primary_final["positive_magnitude"]["candidate_comparison"],
            "negative": primary_final["negative_magnitude"]["candidate_comparison"],
        },
        "drift_reference_summary": {
            "feature_count": len(selected_features),
            "features_with_reference_bins": sum(bool(value) for value in drift_reference["feature_reference_bins"].values()),
        },
        "report": horizon_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v003 3M/6M/9M/12M calibrated dual-stage models.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    dataset = args.data_root / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_multi_horizon.csv"
    if not dataset.exists():
        dataset = args.data_root / "derived" / "prediction" / "part6_prediction_dataset.csv"
    frame = pd.read_csv(dataset, low_memory=False)
    features = infer_features(frame)
    model_dir = args.model_dir or args.data_root.parent / "models" / "action_effectiveness" / "v003"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {str(horizon): train_horizon(frame, features, horizon, model_dir, args.random_state) for horizon in HORIZONS}
    feature_meta = {
        "schema_version": 3, "report_schema_version": "2.0", "bundle_version": "v003",
        "prediction_horizons_months": list(HORIZONS),
        "neutral_return_band": NEUTRAL_RETURN_BAND,
        "numeric_features": features,
        "features_by_horizon": {h: result["report"]["features"] for h, result in results.items()},
        "feature_count_by_horizon": {h: result["report"]["feature_count"] for h, result in results.items()},
        "feature_layers": FEATURE_LAYERS,
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "feature_schema_hash": _hash_text(json.dumps(features, ensure_ascii=False, separators=(",", ":"))),
    }
    (model_dir / "feature_columns.json").write_text(
        json.dumps(_json_safe(feature_meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "bundle_version": "v003", "report_schema_version": "2.0",
        "evaluation_protocol": "nested_time_series_validation",
        "outer_test_used_for_selection": False,
        "dataset": str(dataset), "models": results,
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stability_report = {
        "bundle_version": "v003", "report_schema_version": "2.0",
        "evaluation_protocol": "nested_time_series_validation",
        "outer_test_used_for_selection": False,
        "by_horizon": {h: result["report"] for h, result in results.items()},
        "window_selection_validation_summary": {
            h: result["report"]["window_selection_validation_summary"] for h, result in results.items()
        },
        "outer_test_summary": {
            h: result["report"]["outer_test_summary"] for h, result in results.items()
        },
        "outer_fold_diagnostics": {
            h: result["report"]["outer_fold_diagnostics"] for h, result in results.items()
        },
        "deprecated_fields": {
            "window_comparison": {
                "replacement": "window_selection_validation_summary",
                "reason": "ambiguous_inner_vs_outer_evaluation_level",
            }
        },
    }
    (model_dir / "stability_report.json").write_text(
        json.dumps(_json_safe(stability_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    feature_stability_report = {
        "bundle_version": "v003", "report_schema_version": "2.0",
        "evaluation_protocol": "nested_time_series_validation",
        "outer_test_used_for_selection": False,
        "feature_groups": FEATURE_GROUPS,
        "selection_rules": {
            "positive_delta_fold_ratio_min": 2 / 3, "mean_delta_auc": "> 0",
            "worst_delta_auc": "> -0.02", "mean_delta_brier": "<= 0",
        },
        "by_horizon": {h: result["feature_stability"] for h, result in results.items()},
    }
    (model_dir / "feature_stability_report.json").write_text(
        json.dumps(_json_safe(feature_stability_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    training_config = {
        "bundle_version": "v003", "report_schema_version": "2.0",
        "evaluation_protocol": "nested_time_series_validation",
        "outer_test_used_for_selection": False, "random_state": args.random_state,
        "training_windows": TRAINING_WINDOWS, "embargo_months": EMBARGO_MONTHS,
        "threshold_candidates": THRESHOLD_CANDIDATES.tolist(),
        "threshold_safety": {
            "minimum_threshold": MIN_DIRECTION_THRESHOLD,
            "maximum_threshold": MAX_DIRECTION_THRESHOLD,
            "minimum_predicted_positive_rate": MIN_ALLOWED_PREDICTED_POSITIVE_RATE,
            "maximum_predicted_positive_rate": MAX_ALLOWED_PREDICTED_POSITIVE_RATE,
            "maximum_absolute_positive_rate_gap": MAX_ALLOWED_POSITIVE_RATE_GAP,
        },
        "magnitude_finite_safety": {
            "minimum_train_rows": MIN_MAGNITUDE_TRAIN_ROWS,
            "minimum_validation_rows": MIN_MAGNITUDE_VALID_ROWS,
        },
        "primary_parameter_candidates": PRIMARY_PARAMETER_CANDIDATES,
        "feature_groups": FEATURE_GROUPS,
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(_json_safe(training_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_safe(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
