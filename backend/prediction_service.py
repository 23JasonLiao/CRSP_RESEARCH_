from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

HORIZONS = (3, 6, 9, 12)
FORBIDDEN_PREFIXES = (
    "future_", "direction_label_", "outcome_5class_", "label_positive_excess_",
    "label_start_date_", "label_end_date_", "label_available_",
)
FORBIDDEN_COLUMNS = {
    "manager_reliability_score", "manager_defensive_score", "manager_flow_score",
    "manager_growth_tilt_score", "style_deviation_score",
}


def extract_model_classes(model: Any) -> np.ndarray | None:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for step in reversed(list(named_steps.values())):
            classes = extract_model_classes(step)
            if classes is not None:
                return classes
    estimator = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
    return extract_model_classes(estimator) if estimator is not None else None


def get_positive_class_index(model: Any, positive_label: Any = 1) -> int:
    classes = extract_model_classes(model)
    if classes is None:
        raise ValueError("Cannot determine positive probability orientation: model classes_ metadata is missing.")
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one positive class {positive_label!r}, but classes={classes!r}")
    return int(matches[0])


def validate_probability_array(probability: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(probability, dtype=float).reshape(-1)
    if len(values) != expected_rows or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Positive probability is invalid, non-finite, or row-misaligned.")
    return values


def predict_positive_probability(model: Any, X: np.ndarray, positive_label: Any = 1) -> np.ndarray:
    positive_index = get_positive_class_index(model, positive_label)
    matrix = np.asarray(model.predict_proba(X), dtype=float)
    if matrix.ndim != 2 or positive_index >= matrix.shape[1]:
        raise ValueError(f"Invalid predict_proba shape {matrix.shape} for positive index {positive_index}")
    return validate_probability_array(matrix[:, positive_index], len(X))


def validate_feature_artifact_consistency(bundle: Dict[str, Any]) -> None:
    features = list(bundle.get("features") or [])
    feature_count = bundle.get("feature_count", len(features))  # explicit legacy fallback
    if feature_count != len(features):
        raise ValueError(f"feature_count mismatch: metadata={feature_count}, len(features)={len(features)}")
    if "features_requiring_quantiles" in bundle:
        required = list(bundle["features_requiring_quantiles"])
        quantiles = bundle.get("feature_training_quantiles") or {}
        missing = [feature for feature in required if feature not in quantiles]
        if missing:
            raise ValueError("Missing feature training quantiles for: " + ", ".join(missing[:20]))


def sanitize_magnitude_predictions(prediction: Any, fallback_value: float) -> tuple[np.ndarray, Dict[str, Any]]:
    values = np.asarray(prediction, dtype=float).reshape(-1)
    invalid = ~np.isfinite(values)
    fallback = float(fallback_value) if np.isfinite(fallback_value) else 0.0
    if invalid.any():
        values = values.copy()
        values[invalid] = fallback
    return values, {
        "invalid_prediction_count": int(invalid.sum()), "fallback_value": fallback,
        "fallback_used": bool(invalid.any()),
    }


def _candidate_model_dirs(project_root: Path) -> list[Path]:
    requested = os.environ.get("PART6_MODEL_VERSION", "v003").strip() or "v003"
    names = list(dict.fromkeys([requested, "latest", "v003", "v002"]))
    base = project_root / "models" / "action_effectiveness"
    return [base / name for name in names]


def _find_model_dir(project_root: Path, horizon: int | None = None, model_variant: str = "primary") -> Path:
    filename = None
    if horizon is not None:
        suffix = "_sklearn" if model_variant == "sklearn" else ""
        filename = f"dual_stage_model_{horizon}m{suffix}.pkl"
    for path in _candidate_model_dirs(project_root):
        if path.exists() and (filename is None or (path / filename).exists()):
            return path
    return _candidate_model_dirs(project_root)[0]


def load_bundle(project_root: Path, horizon: int, model_variant: str = "primary") -> Dict[str, Any]:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported Part 6 horizon: {horizon}. Expected one of {HORIZONS}.")
    if os.environ.get("PART6_FORCE_SKLEARN", "").strip() == "1":
        model_variant = "sklearn"
    if model_variant not in {"primary", "sklearn"}:
        raise ValueError("model_variant must be 'primary' or 'sklearn'.")

    load_errors: list[str] = []
    for model_dir in _candidate_model_dirs(project_root):
        suffix = "_sklearn" if model_variant == "sklearn" else ""
        path = model_dir / f"dual_stage_model_{horizon}m{suffix}.pkl"
        if not path.exists():
            continue
        if model_variant == "primary" and importlib.util.find_spec("xgboost") is None:
            load_errors.append(f"{path}: xgboost is unavailable")
            continue
        try:
            bundle = dict(joblib.load(path))
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as exc:
            load_errors.append(f"{path}: {exc}")
            continue
        bundle["_loaded_model_dir"] = str(model_dir)
        bundle["_loaded_bundle_path"] = str(path)
        bundle["_loaded_model_variant"] = model_variant
        validate_feature_artifact_consistency(bundle)
        return bundle

    if model_variant == "primary":
        # Explicit portable fallback, reflected in bundle metadata and API warnings.
        try:
            bundle = load_bundle(project_root, horizon, model_variant="sklearn")
            bundle["_primary_load_errors"] = load_errors
            return bundle
        except FileNotFoundError:
            pass
    searched = ", ".join(str(path) for path in _candidate_model_dirs(project_root))
    details = f" Errors: {'; '.join(load_errors)}" if load_errors else ""
    raise FileNotFoundError(f"Missing Part 6 {model_variant} bundle for {horizon}M. Searched: {searched}.{details}")


def prepare_matrix(frame: pd.DataFrame, bundle: Dict[str, Any]) -> np.ndarray:
    features = list(bundle.get("features") or [])
    if not features:
        raise ValueError("Model bundle has no feature list.")
    forbidden = [
        name for name in features
        if name.startswith(FORBIDDEN_PREFIXES) or name.startswith("current_") or name in FORBIDDEN_COLUMNS
    ]
    if forbidden:
        raise ValueError(f"Target leakage columns found in bundle: {forbidden}")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Required model features are missing: {missing}")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_null = numeric.columns[numeric.isna().all()].tolist()
    if all_null:
        raise ValueError(f"Required model features are entirely null: {all_null}")
    return bundle["imputer"].transform(numeric)


def _prediction_class(
    probability: float, expected_excess: float, bundle: Dict[str, Any],
) -> tuple[str, str, str]:
    # The threshold is validation-derived and used for conflict auditing. The
    # probability-weighted expected excess remains the economic direction.
    float(bundle.get("direction_threshold", 0.5))
    probability_low = float(bundle.get("neutral_probability_low", 0.4))
    probability_high = float(bundle.get("neutral_probability_high", 0.6))
    return_band = float(bundle.get("neutral_return_band", bundle.get("neutral_band", 0.005)))
    if "large_magnitude_threshold" not in bundle:
        raise ValueError("Model bundle is missing large_magnitude_threshold.")
    large_threshold = float(bundle["large_magnitude_threshold"])
    if probability_low <= probability <= probability_high or abs(expected_excess) < return_band:
        return "neutral", "neutral", "neutral"
    direction = "up" if expected_excess > 0 else "down"
    strength = "large" if abs(expected_excess) >= large_threshold else "small"
    prediction_class = (
        "large_win" if direction == "up" and strength == "large" else
        "small_win" if direction == "up" else
        "large_loss" if strength == "large" else "small_loss"
    )
    return direction, strength, prediction_class


def _direction_probability_conflict(
    probability: float, expected_excess: float, bundle: Dict[str, Any],
) -> bool:
    threshold = float(bundle.get("direction_threshold", 0.5))
    return bool(
        (expected_excess > 0 and probability < threshold)
        or (expected_excess < 0 and probability >= threshold)
    )


def _psi(actual: pd.Series, serialized_edges: list[Any], reference: list[float]) -> float | None:
    valid = pd.to_numeric(actual, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(valid) or len(serialized_edges) < 2 or len(reference) != len(serialized_edges) - 1:
        return None
    edges = np.asarray([
        -np.inf if index == 0 and value is None else
        np.inf if index == len(serialized_edges) - 1 and value is None else float(value)
        for index, value in enumerate(serialized_edges)
    ])
    counts, _ = np.histogram(valid, bins=edges)
    actual_distribution = counts / max(counts.sum(), 1)
    expected_distribution = np.asarray(reference, dtype=float)
    actual_distribution = np.clip(actual_distribution, 1e-6, None)
    expected_distribution = np.clip(expected_distribution, 1e-6, None)
    return float(np.sum((actual_distribution - expected_distribution) * np.log(actual_distribution / expected_distribution)))


def _drift_diagnostics(frame: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, Any]:
    features = list(bundle.get("features") or [])
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    quantiles = bundle.get("feature_training_quantiles") or {}
    training_missing = bundle.get("feature_training_missing_rate") or {}
    event_warnings: list[dict[str, Any]] = []
    for row_index in range(len(numeric)):
        drift_features = []
        for feature in features:
            bounds = quantiles.get(feature) or {}
            p01, p99 = bounds.get("p01"), bounds.get("p99")
            value = numeric.iloc[row_index][feature]
            if pd.notna(value) and p01 is not None and p99 is not None and (value < p01 or value > p99):
                drift_features.append({
                    "feature": feature, "event_value": float(value),
                    "training_p01": float(p01), "training_p99": float(p99),
                })
        ratio = len(drift_features) / max(len(features), 1)
        event_warnings.append({
            "warning": "high_feature_range_drift" if ratio > 0.20 else None,
            "out_of_range_ratio": float(ratio), "features": drift_features[:10],
        })

    missing_rate = numeric.isna().mean()
    missing_drift = [
        {"feature": feature, "training_missing_rate": float(training_missing[feature]),
         "inference_missing_rate": float(missing_rate[feature]),
         "missing_rate_delta": float(missing_rate[feature] - training_missing[feature])}
        for feature in features if feature in training_missing
        and float(missing_rate[feature] - training_missing[feature]) > 0.20
    ]
    cohort: Dict[str, Any] = {
        "row_count": int(len(frame)), "missing_rate_drift": missing_drift,
        "psi_heuristic": {"stable": "<0.10", "moderate": "0.10-0.25", "high": ">0.25"},
    }
    if len(frame) < 100:
        cohort.update({"status": "insufficient_cohort_size", "psi": None})
    else:
        bins = bundle.get("feature_reference_bins") or {}
        distributions = bundle.get("feature_reference_distribution") or {}
        psi_values = {
            feature: _psi(numeric[feature], bins.get(feature, []), distributions.get(feature, []))
            for feature in features
        }
        valid_psi = {feature: value for feature, value in psi_values.items() if value is not None}
        cohort.update({
            "status": "high_drift" if any(value > 0.25 for value in valid_psi.values()) else
                      "moderate_drift" if any(value >= 0.10 for value in valid_psi.values()) else "stable",
            "psi": psi_values,
        })
    return {"events": event_warnings, "cohort": cohort}


def predict_events(
    feature_frame: pd.DataFrame,
    metadata: List[Dict[str, Any]],
    project_root: Path,
    model_variant: str = "primary",
    model_name: str | None = None,
) -> Dict[str, Any]:
    # model_name is retained only for the existing API call; it now maps to a real variant.
    if model_name and model_name.lower() in {"sklearn", "random_forest", "fallback"}:
        model_variant = "sklearn"
    if feature_frame.empty:
        return {
            "model_dir": str(_find_model_dir(project_root)),
            "prediction_horizons_months": list(HORIZONS), "models": {}, "predictions": [],
            "warnings": ["No matched events were available for prediction."],
        }
    if len(metadata) != len(feature_frame):
        raise ValueError("Prediction metadata is not aligned with the feature frame.")

    predictions = [dict(metadata[index]) for index in range(len(feature_frame))]
    model_info: Dict[str, Any] = {}
    warnings: list[str] = []
    loaded_dirs: set[str] = set()
    for horizon in HORIZONS:
        bundle = load_bundle(project_root, horizon, model_variant=model_variant)
        X = prepare_matrix(feature_frame, bundle)
        drift = _drift_diagnostics(feature_frame, bundle)
        classifier = bundle["classifier"]
        positive_label = bundle.get("positive_class_label", 1)
        positive_signal = predict_positive_probability(classifier, X, positive_label)
        calibration_method = bundle.get("calibration_method")
        calibrated = calibration_method not in (None, "", "none")
        numeric = feature_frame[bundle["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if all(key in bundle for key in ("positive_imputer", "positive_scaler", "negative_imputer", "negative_scaler")):
            X_positive = bundle["positive_scaler"].transform(bundle["positive_imputer"].transform(numeric))
            X_negative = bundle["negative_scaler"].transform(bundle["negative_imputer"].transform(numeric))
        else:
            X_positive = X_negative = bundle["amplitude_scaler"].transform(X)
        positive_model = bundle.get("positive_model", bundle["positive_ridge"])
        negative_model = bundle.get("negative_model", bundle["negative_ridge"])
        positive_raw, positive_prediction_audit = sanitize_magnitude_predictions(
            positive_model.predict(X_positive),
            (bundle.get("positive_magnitude_audit") or {}).get("magnitude_prediction_fallback_value", 0.0),
        )
        negative_raw, negative_prediction_audit = sanitize_magnitude_predictions(
            negative_model.predict(X_negative),
            (bundle.get("negative_magnitude_audit") or {}).get("magnitude_prediction_fallback_value", 0.0),
        )
        conditional_positive = np.maximum(positive_raw, 0.0)
        conditional_negative = (
            -np.maximum(negative_raw, 0.0)
            if bundle.get("negative_model_predicts_absolute_magnitude", False)
            else np.minimum(negative_raw, 0.0)
        )
        expected, expected_prediction_audit = sanitize_magnitude_predictions(
            positive_signal * conditional_positive + (1.0 - positive_signal) * conditional_negative, 0.0
        )
        for index, (probability, positive, negative, expected_excess) in enumerate(zip(
            positive_signal, conditional_positive, conditional_negative, expected
        )):
            direction, strength, prediction_class = _prediction_class(
                float(probability), float(expected_excess), bundle
            )
            conflict = _direction_probability_conflict(float(probability), float(expected_excess), bundle)
            probability_warning = None
            if bundle.get("probability_calibration_warning"):
                probability_warning = (
                    "The model probability showed temporal calibration drift during validation. "
                    "Interpret it as a relative model score rather than a literal event probability."
                )
            elif not calibrated:
                probability_warning = "This bundle is not calibrated; interpret probability fields as relative model scores."
            predictions[index][f"positive_probability_{horizon}m"] = float(probability)
            predictions[index][f"negative_probability_{horizon}m"] = float(1.0 - probability)
            if not calibrated:
                predictions[index][f"positive_score_{horizon}m"] = float(probability)
                predictions[index][f"negative_score_{horizon}m"] = float(1.0 - probability)
            predictions[index][f"conditional_positive_excess_{horizon}m"] = float(positive)
            predictions[index][f"conditional_negative_excess_{horizon}m"] = float(negative)
            predictions[index][f"predicted_excess_{horizon}m"] = float(expected_excess)
            predictions[index][f"predicted_direction_{horizon}m"] = direction
            predictions[index][f"predicted_strength_{horizon}m"] = strength
            predictions[index][f"predicted_class_{horizon}m"] = prediction_class
            predictions[index][f"direction_threshold_{horizon}m"] = float(bundle.get("direction_threshold", 0.5))
            predictions[index][f"direction_probability_conflict_{horizon}m"] = conflict
            predictions[index][f"probability_warning_{horizon}m"] = probability_warning
            predictions[index][f"drift_warning_{horizon}m"] = drift["events"][index]["warning"]
            predictions[index][f"drift_features_{horizon}m"] = drift["events"][index]["features"]
            predictions[index][f"magnitude_fallback_used_{horizon}m"] = bool(
                positive_prediction_audit["fallback_used"]
                or negative_prediction_audit["fallback_used"]
                or expected_prediction_audit["fallback_used"]
            )
        loaded_dir = str(bundle.get("_loaded_model_dir") or _find_model_dir(project_root, horizon, model_variant))
        loaded_dirs.add(loaded_dir)
        actual_variant = str(bundle.get("_loaded_model_variant") or bundle.get("model_variant") or model_variant)
        if actual_variant != model_variant:
            warnings.append(f"{horizon}M primary bundle was unavailable; loaded {actual_variant}.")
        if not calibrated:
            warnings.append(f"{horizon}M bundle is not calibrated; probability fields are relative scores and score aliases are also returned.")
        if bundle.get("probability_calibration_warning"):
            warnings.append(f"{horizon}M probability calibration showed temporal drift; probabilities should be treated as relative scores.")
        if drift["cohort"].get("status") in {"moderate_drift", "high_drift"}:
            warnings.append(f"{horizon}M request cohort has {drift['cohort']['status']}.")
        if drift["cohort"].get("missing_rate_drift"):
            warnings.append(f"{horizon}M request cohort has feature missing-rate drift.")
        model_info[str(horizon)] = {
            "bundle_version": bundle.get("bundle_version", "legacy"),
            "classifier_type": bundle.get("classifier_type", type(bundle.get("raw_classifier", classifier)).__name__),
            "calibration_method": bundle.get("calibration_method"),
            "calibration_start_date": bundle.get("calibration_start_date"),
            "calibration_end_date": bundle.get("calibration_end_date"),
            "probability_calibration_warning": bool(bundle.get("probability_calibration_warning", False)),
            "probability_calibration_warning_reason": bundle.get("probability_calibration_warning_reason", []),
            "is_calibrated_probability": calibrated,
            "feature_count": len(bundle["features"]),
            "positive_class_label": positive_label,
            "positive_class_index": get_positive_class_index(classifier, positive_label),
            "model_classes": extract_model_classes(classifier).tolist(),
            "probability_orientation": bundle.get("probability_orientation", "P(label_positive_excess=1)"),
            "large_magnitude_threshold": bundle.get("large_magnitude_threshold"),
            "neutral_return_band": bundle.get("neutral_return_band", bundle.get("neutral_band")),
            "training_end_date": bundle.get("train_end_date"),
            "feature_schema_hash": bundle.get("feature_schema_hash"),
            "training_data_hash": bundle.get("training_data_hash"),
            "direction_threshold": float(bundle.get("direction_threshold", 0.5)),
            "threshold_objective": bundle.get("threshold_objective"),
            "threshold_validation_metric": bundle.get("threshold_validation_metric"),
            "training_window_mode": bundle.get("training_window_mode", "legacy_expanding"),
            "training_window_years": bundle.get("training_window_years"),
            "window_selection_score": bundle.get("window_selection_score"),
            "class_weight_mode": bundle.get("class_weight_mode", "legacy_unspecified"),
            "used_scale_pos_weight": bundle.get("used_scale_pos_weight"),
            "positive_model_type": bundle.get("positive_model_type", type(bundle.get("positive_ridge")).__name__),
            "negative_model_type": bundle.get("negative_model_type", type(bundle.get("negative_ridge")).__name__),
            "selected_feature_groups": bundle.get("selected_feature_groups", []),
            "request_cohort_drift": drift["cohort"],
            "horizon_model_status": bundle.get("horizon_model_status", "legacy_unspecified"),
            "five_class_status": bundle.get("five_class_status", "experimental"),
            "five_class_production_ready": bool(bundle.get("five_class_production_ready", False)),
            "magnitude_prediction_audit": {
                "positive": positive_prediction_audit,
                "negative": negative_prediction_audit,
                "expected": expected_prediction_audit,
            },
            "model_variant": actual_variant,
            "bundle_path": bundle.get("_loaded_bundle_path"),
        }
    return {
        "model_dir": next(iter(loaded_dirs)) if len(loaded_dirs) == 1 else sorted(loaded_dirs),
        "prediction_horizons_months": list(HORIZONS),
        "models": model_info, "predictions": predictions, "warnings": warnings,
    }
