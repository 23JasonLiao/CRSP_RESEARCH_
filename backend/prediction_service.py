from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


def _find_model_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "models" / "action_effectiveness" / "v001",
        project_root / "models" / "action_effectiveness" / "latest",
        project_root.parent / "generated_project" / "models" / "action_effectiveness" / "v001",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_feature_columns(model_dir: Path) -> List[str]:
    path = model_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"feature_columns.json not found under {model_dir}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    return meta.get("horizon_specific", {}).get("3", {}).get("numeric_features", [])


def _find_model_path(model_dir: Path, model_name: str = "lightgbm") -> Path:
    candidates = [
        model_dir / f"{model_name}_action_model_trailing3y.pkl",
        model_dir / f"{model_name}_action_model.pkl",
        model_dir / "lightgbm_action_model_trailing3y.pkl",
        model_dir / "lightgbm_action_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No model pickle found under {model_dir}")


def predict_events(feature_frame: pd.DataFrame, metadata: List[Dict[str, Any]], project_root: Path, model_name: str = "lightgbm") -> Dict[str, Any]:
    model_dir = _find_model_dir(project_root)
    features = _load_feature_columns(model_dir)
    model_path = _find_model_path(model_dir, model_name)
    model = joblib.load(model_path)
    X = feature_frame.copy()
    for col in features:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[features]
    probs = model.predict_proba(X)[:, 1]
    predictions = []
    for idx, prob in enumerate(probs):
        meta = metadata[idx] if idx < len(metadata) else {}
        predictions.append({**meta, "prediction_probability": float(prob)})
    return {
        "model_path": str(model_path),
        "model_dir": str(model_dir),
        "feature_count": len(features),
        "features": features,
        "predictions": predictions,
    }
