from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from .prediction_service import _find_model_dir, _find_model_path, _load_feature_columns


def _pipeline_contributions(model: Any, X: pd.DataFrame, features: List[str]) -> np.ndarray:
    """Return approximate per-feature contributions for sklearn demo pipelines.

    For LogisticRegression pipelines, this computes transformed feature value * coefficient.
    For tree models, it uses feature_importances_ times centered raw values as an approximate SHAP-like ranking.
    This is intentionally lightweight for backend integration; final paper experiments can replace it with TreeSHAP.
    """
    estimator = model
    transformer = None
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
        if len(model.steps) > 1:
            transformer = model[:-1]
    if hasattr(estimator, "coef_"):
        Xt = transformer.transform(X) if transformer is not None else X.to_numpy(dtype=float)
        coef = estimator.coef_[0]
        return Xt * coef
    if hasattr(estimator, "feature_importances_"):
        raw = X.apply(pd.to_numeric, errors="coerce")
        centered = raw.fillna(raw.median(numeric_only=True)) - raw.median(numeric_only=True)
        imp = np.asarray(estimator.feature_importances_)
        if len(imp) != len(features):
            imp = np.resize(imp, len(features))
        return centered.to_numpy(dtype=float) * imp
    return np.zeros((len(X), len(features)))


def explain_events(feature_frame: pd.DataFrame, metadata: List[Dict[str, Any]], project_root: Path, model_name: str = "lightgbm", top_k: int = 8) -> Dict[str, Any]:
    model_dir = _find_model_dir(project_root)
    features = _load_feature_columns(model_dir)
    model_path = _find_model_path(model_dir, model_name)
    model = joblib.load(model_path)
    X = feature_frame.copy()
    for col in features:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[features]
    contrib = _pipeline_contributions(model, X, features)
    explanations = []
    values = X.apply(pd.to_numeric, errors="coerce")
    for i in range(len(X)):
        row_contrib = contrib[i]
        order_pos = np.argsort(row_contrib)[::-1]
        order_neg = np.argsort(row_contrib)
        def pack(indices):
            out = []
            for j in indices:
                c = float(row_contrib[j]) if np.isfinite(row_contrib[j]) else 0.0
                if c == 0:
                    continue
                v = values.iloc[i, j]
                out.append({"feature": features[j], "value": None if pd.isna(v) else float(v), "contribution": c})
                if len(out) >= top_k:
                    break
            return out
        meta = metadata[i] if i < len(metadata) else {}
        explanations.append({
            "event_id": meta.get("event_id"),
            "manager": meta.get("manager"),
            "fund": meta.get("fund"),
            "report_date": meta.get("report_date"),
            "top_positive": pack(order_pos),
            "top_negative": pack(order_neg),
        })
    return {
        "method": "approximate_pipeline_contribution",
        "note": "Demo SHAP-like explanations. Replace with TreeSHAP for final LightGBM/XGBoost experiments.",
        "model_path": str(model_path),
        "explanations": explanations,
    }
