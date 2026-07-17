#!/usr/bin/env python3
"""Train the four-horizon direction + conditional-amplitude model family.

Stage 1 is a tree classifier trained on non-neutral events (|excess| >= 0.5%).
Stage 2 contains separate positive and negative Ridge regressors.  All numeric
preprocessing is persisted in each bundle so inference and TreeSHAP use exactly
the same transformed matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

HORIZONS = (3, 6, 9, 12)
NEUTRAL_BAND = 0.005
GICS_SECTORS = (
    "energy", "materials", "industrials", "consumer_discretionary", "consumer_staples",
    "health_care", "financials", "information_technology", "communication_services",
    "utilities", "real_estate",
)
DATE_COL = "report_date"
META = {
    "event_id", "training_window_years", "training_window_months", "manager", "fund",
    "crsp_portno", "crsp_fundno", "fund_ticker", "mgmt_name", "report_date", "year",
    "quarter", "month_key", "feature_cutoff_date", "label_start_date", "label_end_date",
}

FEATURE_LAYERS = {
    "layer_1_performance_risk": [
        "fund_trailing_return", "fund_trailing_excess_return", "fund_trailing_max_drawdown",
        "fund_trailing_beta_vs_sp500", "trailing_avg_net_flow", "current_net_flow",
        "current_turn_ratio", "trailing_avg_turn_ratio",
    ],
    "layer_2_allocation_style": [f"sector_{s}_exposure" for s in GICS_SECTORS],
    "layer_3_rotation_drift": [
        "delta_stock", "delta_beta", "delta_technology", "delta_bond_money",
        "delta_indirect_equity", "delta_nonstock_total_exposure", "delta_sector_exposure",
        "action_strength", "top_holding_concentration", "sector_rotation_intensity",
    ] + [f"delta_sector_{s}" for s in GICS_SECTORS],
    "layer_4_rolling_deviation": [
        "style_deviation_score", "rolling_style_deviation_score", "rolling_sector_deviation_score",
        "rolling_cross_asset_deviation_score", "rolling_action_deviation_score",
    ],
    "layer_5_cross_asset_defense": [
        "stock_allocation", "bond_allocation", "cash_allocation", "bond_money_exposure",
        "indirect_equity_exposure", "company_equity_exposure_proxy", "portfolio_beta",
        "manager_defensive_score", "yield10y",
    ],
}


def infer_features(df: pd.DataFrame) -> list[str]:
    target_prefixes = ("future_", "direction_label_", "outcome_5class_", "label_")
    preferred = list(dict.fromkeys(c for cols in FEATURE_LAYERS.values() for c in cols))
    usable = []
    for c in preferred:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 20:
            usable.append(c)
    # Preserve other numeric action features while explicitly excluding all outcomes.
    for c in df.columns:
        if c in META or c.startswith(target_prefixes) or c in usable:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 50:
            usable.append(c)
    return usable


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.assign(_date=pd.to_datetime(df[DATE_COL], errors="coerce")).dropna(subset=["_date"]).sort_values("_date")
    cut = max(1, int(len(ordered) * 0.8))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def make_tree(seed: int):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.035, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=8, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=-1,
        ), "xgboost"
    except ImportError:
        return RandomForestClassifier(
            n_estimators=350, max_depth=9, min_samples_leaf=12, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ), "random_forest_fallback"


def train_horizon(df: pd.DataFrame, features: list[str], horizon: int, model_dir: Path, seed: int) -> dict:
    target = f"future_{horizon}m_excess_return"
    work = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    train, test = time_split(work)
    direction_train = train[train[target].abs() >= NEUTRAL_BAND].copy()
    if direction_train.empty or direction_train[target].gt(0).nunique() < 2:
        raise ValueError(f"{horizon}M has insufficient positive/negative non-neutral observations")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(direction_train[features].apply(pd.to_numeric, errors="coerce"))
    y_train = direction_train[target].gt(0).astype(int)
    classifier, classifier_type = make_tree(seed)
    classifier.fit(X_train, y_train)

    scaler = StandardScaler()
    X_amp = scaler.fit_transform(imputer.transform(train[features].apply(pd.to_numeric, errors="coerce")))
    pos_mask = train[target] >= NEUTRAL_BAND
    neg_mask = train[target] <= -NEUTRAL_BAND
    positive_ridge = Ridge(alpha=4.0).fit(X_amp[pos_mask], train.loc[pos_mask, target])
    negative_ridge = Ridge(alpha=4.0).fit(X_amp[neg_mask], train.loc[neg_mask, target])

    bundle = {
        "schema_version": 2, "horizon_months": horizon, "neutral_band": NEUTRAL_BAND,
        "features": features, "feature_layers": FEATURE_LAYERS, "imputer": imputer,
        "amplitude_scaler": scaler, "classifier": classifier,
        "positive_ridge": positive_ridge, "negative_ridge": negative_ridge,
        "classifier_type": classifier_type,
    }
    bundle_path = model_dir / f"dual_stage_model_{horizon}m.pkl"
    joblib.dump(bundle, bundle_path)
    joblib.dump(classifier, model_dir / f"direction_model_{horizon}m.pkl")
    joblib.dump(positive_ridge, model_dir / f"positive_ridge_{horizon}m.pkl")
    joblib.dump(negative_ridge, model_dir / f"negative_ridge_{horizon}m.pkl")

    # Portable Part 6 fallback.  It deliberately contains no xgboost objects, so
    # an already-running FastAPI environment can still deserialize and render Part 6.
    fallback_classifier = RandomForestClassifier(
        n_estimators=240, max_depth=9, min_samples_leaf=12, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    fallback_classifier.fit(X_train, y_train)
    fallback_bundle = {
        **bundle,
        "classifier": fallback_classifier,
        "classifier_type": "random_forest_portable_fallback",
    }
    fallback_path = model_dir / f"dual_stage_model_{horizon}m_sklearn.pkl"
    joblib.dump(fallback_bundle, fallback_path)

    metrics = {"train_rows": int(len(train)), "direction_train_rows": int(len(direction_train)), "test_rows": int(len(test))}
    if len(test):
        X_test = imputer.transform(test[features].apply(pd.to_numeric, errors="coerce"))
        p = classifier.predict_proba(X_test)[:, 1]
        eligible = test[target].abs() >= NEUTRAL_BAND
        if eligible.any():
            y = test.loc[eligible, target].gt(0).astype(int)
            pe = p[eligible.to_numpy()]
            metrics["direction_accuracy"] = float(accuracy_score(y, pe >= 0.5))
            metrics["direction_auc"] = float(roc_auc_score(y, pe)) if y.nunique() > 1 else None
        X_test_amp = scaler.transform(X_test)
        positive_amp = np.maximum(positive_ridge.predict(X_test_amp), 0.0)
        negative_amp = np.minimum(negative_ridge.predict(X_test_amp), 0.0)
        signed = np.where(p >= 0.5, positive_amp, negative_amp)
        metrics["amplitude_mae"] = float(mean_absolute_error(test[target], signed))
    return {
        "bundle": str(bundle_path), "portable_fallback_bundle": str(fallback_path),
        "classifier_type": classifier_type, "target": target, "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3M/6M/9M/12M dual-stage models.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    dataset = args.data_root / "derived" / "prediction" / "part6_prediction_dataset_trailing3y_multi_horizon.csv"
    if not dataset.exists():
        dataset = args.data_root / "derived" / "prediction" / "part6_prediction_dataset.csv"
    df = pd.read_csv(dataset, low_memory=False)
    features = infer_features(df)
    model_dir = args.model_dir or args.data_root.parent / "models" / "action_effectiveness" / "v002"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {str(h): train_horizon(df, features, h, model_dir, args.random_state) for h in HORIZONS}
    feature_meta = {
        "schema_version": 2, "prediction_horizons_months": list(HORIZONS),
        "neutral_band": NEUTRAL_BAND, "numeric_features": features,
        "feature_layers": FEATURE_LAYERS,
    }
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(json.dumps({"dataset": str(dataset), "models": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
