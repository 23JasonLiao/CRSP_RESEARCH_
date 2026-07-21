from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import MANAGER_STYLE_DEFINITION, SHAP_PATTERN_DEFINITION
from .prediction_service import (
    HORIZONS,
    get_positive_class_index,
    load_bundle,
    predict_positive_probability,
    prepare_matrix,
)


def _tree_shap(
    model: Any, X: np.ndarray, positive_label: Any,
) -> tuple[np.ndarray, np.ndarray, str, bool, Dict[str, Any]]:
    try:
        if os.environ.get("PART6_DISABLE_SHAP", "").strip() == "1":
            raise ImportError("Part 6 SHAP disabled for runtime verification")
        import shap
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X, check_additivity=True)
        positive_index = get_positive_class_index(model, positive_label)
        if isinstance(raw, list):
            values = np.asarray(raw[positive_index])
        else:
            values = np.asarray(raw)
            if values.ndim == 3:
                values = values[:, :, positive_index]
        expected = np.asarray(explainer.expected_value)
        expected_flat = expected.ravel()
        expected_value = expected_flat[positive_index] if expected_flat.size > 1 else expected_flat[0]
        base = np.repeat(float(expected_value), len(X))
        reconstructed = base + values.sum(axis=1)
        candidate_targets: Dict[str, np.ndarray] = {
            "positive_probability": predict_positive_probability(model, X, positive_label),
        }
        try:
            margin = np.asarray(model.predict(X, output_margin=True), dtype=float)
            if margin.ndim == 2:
                margin = margin[:, positive_index]
            candidate_targets["raw_margin"] = margin.reshape(-1)
        except (AttributeError, TypeError, ValueError):
            pass
        errors = {
            name: float(np.max(np.abs(reconstructed - target)))
            for name, target in candidate_targets.items()
            if target.shape == reconstructed.shape and np.isfinite(target).all()
        }
        output_space = min(errors, key=errors.get) if errors else "unknown"
        max_error = errors.get(output_space)
        fidelity = {
            "status": "passed" if max_error is not None and max_error <= 1e-4 else "warning",
            "passed": bool(max_error is not None and max_error <= 1e-4),
            "max_absolute_reconstruction_error": max_error,
            "reconstructed_output_space": output_space,
            "positive_class_label": positive_label,
            "positive_class_index": positive_index,
        }
        return values.astype(float), base.astype(float), "TreeSHAP", True, fidelity
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        importance = np.asarray(getattr(model, "feature_importances_", np.ones(X.shape[1])), dtype=float)
        if importance.size != X.shape[1]:
            importance = np.resize(importance, X.shape[1])
        center = np.nanmedian(X, axis=0)
        values = (X - center) * importance
        score = predict_positive_probability(model, X, positive_label)
        base = score - values.sum(axis=1)
        return values.astype(float), base.astype(float), "tree_importance_runtime_fallback", False, {
            "status": "not_applicable_runtime_fallback",
            "passed": False,
            "max_absolute_reconstruction_error": None,
            "reconstructed_output_space": "heuristic_positive_score",
            "positive_class_label": positive_label,
            "positive_class_index": get_positive_class_index(model, positive_label),
        }


def _linear_contributions(model: Any, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    if coefficients.size != X_scaled.shape[1]:
        raise ValueError("Ridge coefficient count does not match the model feature matrix.")
    values = X_scaled * coefficients
    base = np.repeat(float(np.asarray(model.intercept_).reshape(-1)[0]), len(X_scaled))
    return values.astype(float), base.astype(float)


def _magnitude_contributions(
    bundle: Dict[str, Any], raw: pd.DataFrame, direction_matrix: np.ndarray, side: str,
) -> tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    model = bundle.get(f"{side}_model", bundle.get(f"{side}_ridge"))
    if model is None:
        raise ValueError(f"Bundle is missing the {side} magnitude model.")
    numeric = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = bundle.get(f"{side}_imputer")
    scaler = bundle.get(f"{side}_scaler")
    if imputer is not None and scaler is not None:
        matrix = scaler.transform(imputer.transform(numeric))
        preprocessing = f"{side}_imputer+{side}_scaler"
    else:
        matrix = bundle["amplitude_scaler"].transform(direction_matrix)
        preprocessing = "legacy_amplitude_scaler"

    model_name = type(model).__name__
    prediction = np.asarray(model.predict(matrix), dtype=float).reshape(-1)
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        values, base = _linear_contributions(model, matrix)
        reconstructed = base + values.sum(axis=1)
        max_error = float(np.max(np.abs(reconstructed - prediction))) if len(prediction) else 0.0
        method = f"{model_name}_linear_contribution"
        exact_for_raw_model = max_error <= 1e-8
    else:
        values = np.zeros((len(matrix), matrix.shape[1]), dtype=float)
        base = prediction.copy()
        max_error = 0.0
        method = f"{model_name}_constant_prediction_decomposition"
        exact_for_raw_model = True
    return values, base, method, {
        "status": "passed" if exact_for_raw_model else "warning",
        "passed": exact_for_raw_model,
        "max_absolute_reconstruction_error": max_error,
        "reconstructed_output_space": "raw_conditional_magnitude_before_sign_or_finite_value_sanitization",
        "model_type": model_name,
        "preprocessing": preprocessing,
    }


def _cluster_name(top_features: list[str]) -> str:
    text = " ".join(top_features).lower()
    if "technology" in text:
        return "Technology Allocation Logic"
    if "turn" in text or "rotation" in text or "action_strength" in text:
        return "High-Rotation Direction Logic"
    if "bond" in text or "cash" in text or "defensive" in text:
        return "Cross-Asset Defensive Direction Logic"
    if "style_deviation" in text:
        return "Style-Drift Direction Logic"
    if "flow" in text:
        return "Flow-Sensitive Direction Logic"
    return "Mixed Direction Logic"


def _cluster_payload(
    direction_values: np.ndarray,
    raw_values: pd.DataFrame,
    features: list[str],
    predictions: list[dict],
    metadata: List[Dict[str, Any]],
    horizon: int,
    contribution_method: str,
) -> Dict[str, Any]:
    count = len(direction_values)
    method_label = (
        "direction-model TreeSHAP" if contribution_method == "TreeSHAP"
        else "direction-model runtime-fallback"
    )
    if count < 2:
        return {
            "horizon_months": horizon,
            "method": f"KMeans on {method_label} vectors; PCA is display-only",
            "clusters": [], "points": [],
        }
    scaled = StandardScaler().fit_transform(direction_values)
    cluster_count = min(5, max(2, int(round(np.sqrt(count / 2)))), count)
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=20).fit_predict(scaled)
    coordinates = (
        PCA(n_components=2, random_state=42).fit_transform(scaled)
        if min(scaled.shape) >= 2 else np.c_[scaled[:, 0], np.zeros(count)]
    )
    cohort_mean = raw_values.mean(numeric_only=True)
    cohort_std = raw_values.std(numeric_only=True).replace(0, np.nan)
    points, clusters = [], []
    for index in range(count):
        prediction = predictions[index] if index < len(predictions) else {}
        meta = metadata[index] if index < len(metadata) else {}
        probability = prediction.get(f"positive_probability_{horizon}m")
        direction_signal = probability if probability is not None else prediction.get(f"positive_score_{horizon}m")
        points.append({
            "event_id": meta.get("event_id"), "manager": meta.get("manager"),
            "report_date": meta.get("report_date"),
            "manager_style_group": meta.get("manager_style_group") or "Unknown style",
            "cluster": int(labels[index]), "x": float(coordinates[index, 0]),
            "y": float(coordinates[index, 1]), "direction_signal": direction_signal,
            "predicted_class": prediction.get(f"predicted_class_{horizon}m"),
        })
    for cluster_id in range(cluster_count):
        indices = np.where(labels == cluster_id)[0]
        mean_direction = direction_values[indices].mean(axis=0)
        order = np.argsort(np.abs(mean_direction))[::-1][:5]
        top_features = [features[index] for index in order]
        classes = [
            predictions[index].get(f"predicted_class_{horizon}m")
            for index in indices if index < len(predictions)
        ]
        styles = [
            str((metadata[index] if index < len(metadata) else {}).get("manager_style_group") or "Unknown style")
            for index in indices
        ]
        style_counts = dict(Counter(styles))
        fidelity = []
        for feature_index in order[:3]:
            series = pd.to_numeric(raw_values.iloc[indices, feature_index], errors="coerce").dropna()
            fidelity.append({
                "feature": features[feature_index],
                "mean_direction_contribution": float(mean_direction[feature_index]),
                "q1": None if series.empty else float(series.quantile(0.25)),
                "median": None if series.empty else float(series.median()),
                "q3": None if series.empty else float(series.quantile(0.75)),
                "analysis_cohort_mean": None if pd.isna(cohort_mean.get(features[feature_index])) else float(cohort_mean[features[feature_index]]),
                "analysis_cohort_std": None if pd.isna(cohort_std.get(features[feature_index])) else float(cohort_std[features[feature_index]]),
                "raw_values": [float(value) for value in series.tolist()],
            })
        clusters.append({
            "cluster": cluster_id, "name": _cluster_name(top_features),
            "event_count": int(len(indices)),
            "predicted_large_win_rate": classes.count("large_win") / len(classes) if classes else 0.0,
            "predicted_large_loss_rate": classes.count("large_loss") / len(classes) if classes else 0.0,
            "class_counts": dict(Counter(classes)), "top_features": top_features,
            "fidelity": fidelity, "manager_style_counts": style_counts,
            "dominant_manager_style": max(style_counts, key=style_counts.get) if style_counts else "Unknown style",
        })
    return {
        "horizon_months": horizon,
        "method": f"KMeans on {method_label} vectors; PCA is display-only",
        "interpretation": SHAP_PATTERN_DEFINITION,
        "clusters": clusters, "points": points,
    }


def _feature_rows(
    features: list[str], raw: pd.DataFrame, contributions: np.ndarray,
    row_index: int, top_k: int,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(contributions[row_index]))[::-1][:top_k]
    return [{
        "feature": features[index],
        "value": None if pd.isna(raw.iloc[row_index, index]) else float(raw.iloc[row_index, index]),
        "contribution": float(contributions[row_index, index]),
    } for index in order]


def explain_events(
    feature_frame: pd.DataFrame,
    metadata: List[Dict[str, Any]],
    project_root: Path,
    model_variant: str = "primary",
    model_name: str | None = None,
    top_k: int = 8,
    selected_horizon: int = 12,
    prediction_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if selected_horizon not in HORIZONS:
        raise ValueError(f"Unsupported selected_horizon {selected_horizon}; expected one of {HORIZONS}.")
    if model_name and model_name.lower() in {"sklearn", "random_forest", "fallback"}:
        model_variant = "sklearn"
    if feature_frame.empty:
        return {
            "method": "none", "additivity_check": False,
            "prediction_horizons_months": list(HORIZONS), "explanations": [],
            "direction_global_importance": [], "positive_magnitude_global_importance": [],
            "negative_magnitude_global_importance": [], "global_importance": [],
            "temporal_clustering": {"horizon_months": selected_horizon, "clusters": [], "points": []},
            "warnings": ["No matched events were available for explanation."],
            "semantic_contract": {
                "shap_pattern_definition": SHAP_PATTERN_DEFINITION,
                "manager_style_definition": MANAGER_STYLE_DEFINITION,
                "causal_interpretation_allowed": False,
                "expected_excess_exact_shap_available": False,
            },
        }
    if len(metadata) != len(feature_frame):
        raise ValueError("Explanation metadata is not aligned with the feature frame.")

    explanations: list[dict[str, Any]] = []
    direction_global, positive_global, negative_global = [], [], []
    methods: set[str] = set()
    additivity_checks: list[bool] = []
    cluster_result: Dict[str, Any] = {}
    predictions = (prediction_result or {}).get("predictions") or []
    warnings: list[str] = []
    for horizon in HORIZONS:
        bundle = load_bundle(project_root, horizon, model_variant=model_variant)
        X = prepare_matrix(feature_frame, bundle)
        features = list(bundle["features"])
        raw = feature_frame[features].apply(pd.to_numeric, errors="coerce")
        direction_model = bundle.get("raw_classifier") or bundle["classifier"]
        positive_label = bundle.get("positive_class_label", 1)
        direction_values, direction_base, direction_method, is_exact, direction_fidelity = _tree_shap(
            direction_model, X, positive_label
        )
        methods.add(direction_method)
        additivity_checks.append(bool(direction_fidelity["passed"]))
        if not is_exact:
            warnings.append(f"{horizon}M direction explanation uses runtime fallback, not TreeSHAP.")
        positive_values, positive_base, positive_method, positive_fidelity = _magnitude_contributions(
            bundle, raw, X, "positive"
        )
        negative_values, negative_base, negative_method, negative_fidelity = _magnitude_contributions(
            bundle, raw, X, "negative"
        )

        for index in np.argsort(np.abs(direction_values).mean(axis=0))[::-1]:
            direction_global.append({
                "horizon_months": horizon, "feature": features[index],
                "mean_abs_direction_contribution": float(np.abs(direction_values[:, index]).mean()),
            })
        for index in np.argsort(np.abs(positive_values).mean(axis=0))[::-1]:
            positive_global.append({
                "horizon_months": horizon, "feature": features[index],
                "mean_abs_positive_magnitude_contribution": float(np.abs(positive_values[:, index]).mean()),
            })
        for index in np.argsort(np.abs(negative_values).mean(axis=0))[::-1]:
            negative_global.append({
                "horizon_months": horizon, "feature": features[index],
                "mean_abs_negative_magnitude_contribution": float(np.abs(negative_values[:, index]).mean()),
            })

        for row_index in range(len(X)):
            meta = metadata[row_index]
            direction_features = _feature_rows(features, raw, direction_values, row_index, top_k)
            explanation = {
                "event_id": meta.get("event_id"), "manager": meta.get("manager"),
                "fund": meta.get("fund"), "report_date": meta.get("report_date"),
                "horizon_months": horizon,
                "direction": {
                    "method": direction_method, "is_exact_shap": is_exact,
                    "additivity_check": bool(direction_fidelity["passed"]),
                    "fidelity": direction_fidelity, "base_value": float(direction_base[row_index]),
                    "features": direction_features,
                    "interpretation": "Drivers of the direction classifier, not drivers of exact predicted excess.",
                },
                "positive_magnitude": {
                    "method": positive_method, "fidelity": positive_fidelity,
                    "base_value": float(positive_base[row_index]),
                    "features": _feature_rows(features, raw, positive_values, row_index, top_k),
                    "contribution_sum_with_base": float(positive_base[row_index] + positive_values[row_index].sum()),
                },
                "negative_magnitude": {
                    "method": negative_method, "fidelity": negative_fidelity,
                    "base_value": float(negative_base[row_index]),
                    "features": _feature_rows(features, raw, negative_values, row_index, top_k),
                    "contribution_sum_with_base": float(negative_base[row_index] + negative_values[row_index].sum()),
                    "predicts_absolute_magnitude": bool(bundle.get("negative_model_predicts_absolute_magnitude", False)),
                },
                "expected_excess_explanation": {
                    "method": "not_computed",
                    "is_exact_shap": False,
                    "reason": "Probability and conditional magnitude components interact; their contributions are not added and called exact SHAP.",
                },
                "part7_attribution_contract": {
                    "shap_pattern_definition": SHAP_PATTERN_DEFINITION,
                    "manager_style_definition": MANAGER_STYLE_DEFINITION,
                    "causal_interpretation_allowed": False,
                    "manager_intent_inference_allowed": False,
                    "direction_and_magnitude_attributions_may_be_added": False,
                    "reliability_context": {
                        "horizon_model_status": bundle.get("horizon_model_status"),
                        "outer_test_summary": bundle.get("outer_test_summary"),
                        "calibration_warning": bool(bundle.get("probability_calibration_warning", False)),
                        "threshold_selection_status": bundle.get("threshold_selection_status"),
                        "five_class_status": bundle.get("five_class_status"),
                    },
                },
                # Compatibility view for existing direction-only renderers.
                "base_value": float(direction_base[row_index]),
                "features": direction_features,
            }
            explanations.append(explanation)
        if horizon == selected_horizon:
            cluster_result = _cluster_payload(
                direction_values, raw, features, predictions, metadata, horizon, direction_method
            )

    compatibility_global = [{
        "horizon_months": row["horizon_months"], "feature": row["feature"],
        "mean_abs_shap": row["mean_abs_direction_contribution"],
    } for row in direction_global]
    return {
        "method": "TreeSHAP" if methods == {"TreeSHAP"} else "+".join(sorted(methods)),
        "direction_method": "TreeSHAP" if methods == {"TreeSHAP"} else "+".join(sorted(methods)),
        "additivity_check": bool(additivity_checks) and all(additivity_checks),
        "prediction_horizons_months": list(HORIZONS), "explanations": explanations,
        "direction_global_importance": direction_global,
        "positive_magnitude_global_importance": positive_global,
        "negative_magnitude_global_importance": negative_global,
        "global_importance": compatibility_global,
        "temporal_clustering": cluster_result, "warnings": warnings,
        "semantic_contract": {
            "shap_pattern_definition": SHAP_PATTERN_DEFINITION,
            "manager_style_definition": MANAGER_STYLE_DEFINITION,
            "causal_interpretation_allowed": False,
            "expected_excess_exact_shap_available": False,
        },
    }
