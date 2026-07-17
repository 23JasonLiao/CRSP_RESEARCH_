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

from .prediction_service import HORIZONS, load_bundle, prepare_matrix


def _tree_shap(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, bool]:
    try:
        if os.environ.get("PART6_DISABLE_SHAP", "").strip() == "1":
            raise ImportError("Part 6 SHAP disabled for portable-runtime verification")
        import shap
    except ImportError:
        # Keep Part 6 visible in lean runtime environments.  This is explicitly
        # labelled as a fallback and is never presented as exact TreeSHAP.
        importance = np.asarray(getattr(model, "feature_importances_", np.ones(X.shape[1])), dtype=float)
        if importance.size != X.shape[1]:
            importance = np.resize(importance, X.shape[1])
        center = np.nanmedian(X, axis=0)
        values = (X - center) * importance
        probability = model.predict_proba(X)[:, 1]
        base = probability - values.sum(axis=1)
        return values.astype(float), base.astype(float), "tree_importance_runtime_fallback", False
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X, check_additivity=True)
    if isinstance(raw, list):
        values = np.asarray(raw[-1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, -1]
    base = np.asarray(explainer.expected_value)
    base = np.repeat(float(base.ravel()[-1]), len(X))
    return values.astype(float), base, "TreeSHAP", True


def _cluster_name(top_features: list[str]) -> str:
    text = " ".join(top_features).lower()
    if "technology" in text: return "Technology Allocation Logic"
    if "turn" in text or "rotation" in text or "action_strength" in text: return "High-Rotation Decision Logic"
    if "bond" in text or "cash" in text or "defensive" in text: return "Cross-Asset Defensive Logic"
    if "style_deviation" in text: return "Style-Drift Decision Logic"
    if "flow" in text: return "Flow-Sensitive Decision Logic"
    return "Mixed Allocation Logic"


def _cluster_payload(
    shap_values: np.ndarray, raw_values: pd.DataFrame, features: list[str], predictions: list[dict],
    metadata: List[Dict[str, Any]], horizon: int, contribution_method: str,
) -> Dict[str, Any]:
    n = len(shap_values)
    if n < 2:
        return {"horizon_months": horizon, "clusters": [], "points": []}
    scaled = StandardScaler().fit_transform(shap_values)
    cluster_count = min(5, max(2, int(round(np.sqrt(n / 2)))))
    cluster_count = min(cluster_count, n)
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=20).fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled) if scaled.shape[1] >= 2 else np.c_[scaled[:, 0], np.zeros(n)]
    global_mean = raw_values.mean(numeric_only=True)
    global_std = raw_values.std(numeric_only=True).replace(0, np.nan)
    points, clusters = [], []
    for i in range(n):
        pred = predictions[i] if i < len(predictions) else {}
        meta = metadata[i] if i < len(metadata) else {}
        points.append({
            "event_id": meta.get("event_id"), "manager": meta.get("manager"), "report_date": meta.get("report_date"),
            "manager_style_group": meta.get("manager_style_group") or "Unknown style",
            "cluster": int(labels[i]), "x": float(coords[i, 0]), "y": float(coords[i, 1]),
            "positive_probability": pred.get(f"positive_probability_{horizon}m"),
            "predicted_class": pred.get(f"predicted_class_{horizon}m"),
        })
    for cluster_id in range(cluster_count):
        idx = np.where(labels == cluster_id)[0]
        mean_shap = shap_values[idx].mean(axis=0)
        order = np.argsort(np.abs(mean_shap))[::-1][:5]
        top = [features[j] for j in order]
        classes = [predictions[i].get(f"predicted_class_{horizon}m") for i in idx if i < len(predictions)]
        styles = [
            str((metadata[i] if i < len(metadata) else {}).get("manager_style_group") or "Unknown style")
            for i in idx
        ]
        style_counts = dict(Counter(styles))
        fidelity = []
        for j in order[:3]:
            series = pd.to_numeric(raw_values.iloc[idx, j], errors="coerce").dropna()
            fidelity.append({
                "feature": features[j], "mean_shap": float(mean_shap[j]),
                "q1": None if series.empty else float(series.quantile(0.25)),
                "median": None if series.empty else float(series.median()),
                "q3": None if series.empty else float(series.quantile(0.75)),
                "market_mean": None if pd.isna(global_mean.get(features[j])) else float(global_mean[features[j]]),
                "market_std": None if pd.isna(global_std.get(features[j])) else float(global_std[features[j]]),
                "raw_values": [float(v) for v in series.tolist()],
            })
        clusters.append({
            "cluster": cluster_id, "name": _cluster_name(top), "event_count": int(len(idx)),
            "large_win_rate": classes.count("large_win") / len(classes) if classes else 0.0,
            "large_loss_rate": classes.count("large_loss") / len(classes) if classes else 0.0,
            "class_counts": dict(Counter(classes)), "top_features": top, "fidelity": fidelity,
            "manager_style_counts": style_counts,
            "dominant_manager_style": max(style_counts, key=style_counts.get) if style_counts else "Unknown style",
        })
    return {
        "horizon_months": horizon,
        "method": f"KMeans on {contribution_method} vectors; PCA is display-only",
        "clusters": clusters, "points": points,
    }


def explain_events(
    feature_frame: pd.DataFrame, metadata: List[Dict[str, Any]], project_root: Path,
    model_name: str = "xgboost", top_k: int = 8, selected_horizon: int = 12,
    prediction_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    explanations, global_rows = [], []
    methods, additivity_checks = set(), []
    cluster_result: Dict[str, Any] = {}
    predictions = (prediction_result or {}).get("predictions") or []
    for horizon in HORIZONS:
        bundle = load_bundle(project_root, horizon)
        X = prepare_matrix(feature_frame, bundle)
        values, base, method, additivity_check = _tree_shap(bundle["classifier"], X)
        methods.add(method)
        additivity_checks.append(additivity_check)
        features = list(bundle["features"])
        raw = feature_frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
        mean_abs = np.abs(values).mean(axis=0)
        global_rows.extend({"horizon_months": horizon, "feature": features[j], "mean_abs_shap": float(mean_abs[j])} for j in np.argsort(mean_abs)[::-1])
        for i in range(len(X)):
            order = np.argsort(np.abs(values[i]))[::-1][:top_k]
            meta = metadata[i] if i < len(metadata) else {}
            explanations.append({
                "event_id": meta.get("event_id"), "manager": meta.get("manager"), "fund": meta.get("fund"),
                "report_date": meta.get("report_date"), "horizon_months": horizon,
                "base_value": float(base[i]),
                "features": [{
                    "feature": features[j], "value": None if pd.isna(raw.iloc[i, j]) else float(raw.iloc[i, j]),
                    "contribution": float(values[i, j]),
                } for j in order],
            })
        if horizon == selected_horizon:
            cluster_result = _cluster_payload(values, raw, features, predictions, metadata, horizon, method)
    return {
        "method": "TreeSHAP" if methods == {"TreeSHAP"} else "+".join(sorted(methods)),
        "additivity_check": bool(additivity_checks) and all(additivity_checks),
        "prediction_horizons_months": list(HORIZONS), "explanations": explanations,
        "global_importance": global_rows, "temporal_clustering": cluster_result,
    }
