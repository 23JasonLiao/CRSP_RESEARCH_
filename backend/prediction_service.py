from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

HORIZONS = (3, 6, 9, 12)


def _find_model_dir(project_root: Path) -> Path:
    for path in (
        project_root / "models" / "action_effectiveness" / "v002",
        project_root / "models" / "action_effectiveness" / "latest",
    ):
        if path.exists():
            return path
    return project_root / "models" / "action_effectiveness" / "v002"


def load_bundle(project_root: Path, horizon: int) -> Dict[str, Any]:
    model_dir = _find_model_dir(project_root)
    primary = model_dir / f"dual_stage_model_{horizon}m.pkl"
    fallback = model_dir / f"dual_stage_model_{horizon}m_sklearn.pkl"
    if not primary.exists() and not fallback.exists():
        raise FileNotFoundError(f"Missing Part 6 model bundle for {horizon}M under {model_dir}.")
    force_sklearn = os.environ.get("PART6_FORCE_SKLEARN", "").strip() == "1"
    xgboost_available = importlib.util.find_spec("xgboost") is not None
    if primary.exists() and xgboost_available and not force_sklearn:
        try:
            return joblib.load(primary)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            # A running API may use a lean Python environment without xgboost.
            # Keep Part 6 operational with the portable sklearn tree bundle.
            if not fallback.exists():
                raise
    if fallback.exists():
        return joblib.load(fallback)
    return joblib.load(primary)


def prepare_matrix(frame: pd.DataFrame, bundle: Dict[str, Any]) -> np.ndarray:
    numeric = frame.reindex(columns=bundle["features"]).apply(pd.to_numeric, errors="coerce")
    return bundle["imputer"].transform(numeric)


def _five_class(prob: float, amplitude: float, band: float) -> str:
    if 0.4 <= prob <= 0.6 or abs(amplitude) < band:
        return "neutral"
    if amplitude >= 0:
        return "large_win" if amplitude >= 0.03 else "small_win"
    return "large_loss" if amplitude <= -0.03 else "small_loss"


def predict_events(feature_frame: pd.DataFrame, metadata: List[Dict[str, Any]], project_root: Path, model_name: str = "xgboost") -> Dict[str, Any]:
    predictions = [{**(metadata[i] if i < len(metadata) else {})} for i in range(len(feature_frame))]
    model_dir = _find_model_dir(project_root)
    model_info = {}
    for horizon in HORIZONS:
        bundle = load_bundle(project_root, horizon)
        X = prepare_matrix(feature_frame, bundle)
        probability = bundle["classifier"].predict_proba(X)[:, 1]
        X_amp = bundle["amplitude_scaler"].transform(X)
        positive = np.maximum(bundle["positive_ridge"].predict(X_amp), 0.0)
        negative = np.minimum(bundle["negative_ridge"].predict(X_amp), 0.0)
        signed = np.where(probability >= 0.5, positive, negative)
        band = float(bundle.get("neutral_band", 0.005))
        for i, (p, amp) in enumerate(zip(probability, signed)):
            predictions[i][f"positive_probability_{horizon}m"] = float(p)
            predictions[i][f"predicted_excess_{horizon}m"] = float(amp)
            predictions[i][f"predicted_class_{horizon}m"] = _five_class(float(p), float(amp), band)
        model_info[str(horizon)] = {"classifier_type": bundle.get("classifier_type"), "feature_count": len(bundle["features"])}
    return {
        "model_dir": str(model_dir), "prediction_horizons_months": list(HORIZONS),
        "models": model_info, "predictions": predictions,
    }
