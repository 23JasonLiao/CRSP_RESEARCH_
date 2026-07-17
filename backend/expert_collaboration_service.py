from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _number(value: Any) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if pd.notna(out) and np.isfinite(out) else np.nan


def _allocation(value: Any) -> float:
    out = _number(value)
    if not np.isfinite(out):
        return np.nan
    return out / 100.0 if abs(out) > 1.5 else out


def _manager_performance(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    records = ((payload.get("part4") or {}).get("manager_records_raw") or [])
    prepared: List[Dict[str, float | str]] = []
    for record in records:
        manager = str(record.get("manager") or "").strip()
        if not manager:
            continue
        raw = record.get("raw") or {}
        scores = record.get("scores") or {}
        volatility = _number(raw.get("annual_volatility"))
        avg_excess = _number(raw.get("avg_excess"))
        information_ratio = avg_excess / volatility if np.isfinite(avg_excess) and np.isfinite(volatility) and volatility > 0 else np.nan
        prepared.append({
            "manager": manager,
            "sharpe": _number(raw.get("sharpe")),
            "information_ratio": information_ratio,
            "alpha": _number(raw.get("alpha")),
            "sharpe_score": _number(raw.get("score_sharpe") or scores.get("Sharpe Ratio")),
            "excess_score": _number(raw.get("score_avg_excess") or scores.get("平均超額報酬")),
            "drawdown_score": _number(raw.get("score_max_drawdown") or scores.get("Max Drawdown")),
            "return_score": _number(raw.get("score_annual_return") or scores.get("年化報酬")),
        })
    if not prepared:
        return {}
    frame = pd.DataFrame(prepared)
    if frame["information_ratio"].notna().any():
        frame["information_ratio_score"] = frame["information_ratio"].rank(pct=True)
    else:
        frame["information_ratio_score"] = np.nan
    if frame["alpha"].notna().any():
        frame["alpha_score"] = frame["alpha"].rank(pct=True)
    else:
        frame["alpha_score"] = np.nan
    score_cols = ["sharpe_score", "excess_score", "drawdown_score", "return_score", "information_ratio_score", "alpha_score"]
    weights = np.asarray([0.30, 0.20, 0.15, 0.10, 0.20, 0.05], dtype=float)
    output: Dict[str, Dict[str, float]] = {}
    for _, row in frame.iterrows():
        values = np.asarray([_number(row.get(c)) for c in score_cols], dtype=float)
        valid = np.isfinite(values)
        composite = float(np.average(values[valid], weights=weights[valid])) if valid.any() else 0.5
        output[str(row["manager"])] = {
            "performance_score": composite,
            "sharpe": _number(row.get("sharpe")),
            "information_ratio": _number(row.get("information_ratio")),
            "alpha": _number(row.get("alpha")),
        }
    return output


def build_expert_collaboration(
    feature_frame: pd.DataFrame,
    metadata: List[Dict[str, Any]],
    prediction_result: Dict[str, Any],
    payload: Dict[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    """Build balanced-fund expert and Human-AI allocation consensus for Part 6 only."""
    performance = _manager_performance(payload)
    predictions = prediction_result.get("predictions") or []
    rows: List[Dict[str, Any]] = []
    work = feature_frame.reset_index(drop=True)
    for index, (_, source) in enumerate(work.iterrows()):
        meta = metadata[index] if index < len(metadata) else {}
        pred = predictions[index] if index < len(predictions) else {}
        manager = str(meta.get("manager") or source.get("manager") or "Unknown manager").strip()
        report_date = pd.to_datetime(meta.get("report_date") or source.get("report_date"), errors="coerce")
        if pd.isna(report_date):
            continue
        manager_perf = performance.get(manager) or {}
        performance_score = _number(manager_perf.get("performance_score"))
        if not np.isfinite(performance_score):
            performance_score = _number(source.get("manager_reliability_score"))
        if not np.isfinite(performance_score):
            performance_score = 0.5
        stock = _allocation(source.get("stock_allocation"))
        bond = _allocation(source.get("bond_allocation"))
        cash = _allocation(source.get("cash_allocation"))
        if not np.isfinite(stock): stock = 0.0
        if not np.isfinite(cash): cash = 0.0
        if not np.isfinite(bond): bond = max(0.0, 1.0 - stock - cash)
        total = stock + bond + cash
        if total > 0:
            stock, bond, cash = stock / total, bond / total, cash / total
        probability = _number(pred.get(f"positive_probability_{horizon}m"))
        if not np.isfinite(probability): probability = 0.5
        rows.append({
            "event_id": meta.get("event_id") or source.get("event_id"),
            "report_date": report_date.strftime("%Y-%m-%d"),
            "manager": manager,
            "fund": meta.get("fund") or source.get("fund"),
            "fund_ticker": meta.get("fund_ticker") or source.get("fund_ticker"),
            "crsp_portno": meta.get("crsp_portno") or source.get("crsp_portno"),
            "style_group": meta.get("manager_style_group") or source.get("manager_style_group") or "Unknown style",
            "stock_allocation": stock, "bond_allocation": bond, "cash_allocation": cash,
            "performance_score": float(np.clip(performance_score, 0.01, 1.0)),
            "sharpe": manager_perf.get("sharpe"),
            "information_ratio": manager_perf.get("information_ratio"),
            "alpha": manager_perf.get("alpha"),
            "ai_positive_probability": probability,
            "ai_predicted_class": pred.get(f"predicted_class_{horizon}m"),
            "future_excess_proxy": meta.get(f"future_{horizon}m_excess_return"),
        })
    if not rows:
        return {"horizon_months": horizon, "recommendations": [], "manager_contributions": [], "report_manager_map": {}}

    frame = pd.DataFrame(rows).drop_duplicates("event_id", keep="first")
    # One expert receives one vote per report date; multiple funds do not mechanically multiply influence.
    manager_date = frame.groupby(["report_date", "manager", "style_group"], as_index=False).agg({
        "fund": "first", "fund_ticker": "first", "crsp_portno": "first", "stock_allocation": "mean", "bond_allocation": "mean",
        "cash_allocation": "mean", "performance_score": "mean", "sharpe": "mean",
        "information_ratio": "mean", "alpha": "mean", "ai_positive_probability": "mean",
        "future_excess_proxy": "mean", "ai_predicted_class": "first",
    })
    manager_date["relative_style_score"] = manager_date.groupby(["report_date", "style_group"])["performance_score"].rank(pct=True)
    singleton = manager_date.groupby(["report_date", "style_group"])["manager"].transform("count").eq(1)
    manager_date.loc[singleton, "relative_style_score"] = manager_date.loc[singleton, "performance_score"]
    manager_date["expert_raw_weight"] = manager_date["relative_style_score"].clip(lower=0.05)
    manager_date["ai_confidence"] = 0.5 + (manager_date["ai_positive_probability"] - 0.5).abs()
    manager_date["ai_raw_weight"] = manager_date["ai_confidence"]
    manager_date["human_ai_raw_weight"] = manager_date["expert_raw_weight"] * manager_date["ai_confidence"]
    manager_date["expert_weight"] = manager_date["expert_raw_weight"] / manager_date.groupby("report_date")["expert_raw_weight"].transform("sum")
    manager_date["ai_weight"] = manager_date["ai_raw_weight"] / manager_date.groupby("report_date")["ai_raw_weight"].transform("sum")
    manager_date["human_ai_weight"] = manager_date["human_ai_raw_weight"] / manager_date.groupby("report_date")["human_ai_raw_weight"].transform("sum")
    for asset in ("stock", "bond", "cash"):
        manager_date[f"expert_{asset}_contribution"] = manager_date["expert_weight"] * manager_date[f"{asset}_allocation"]
        manager_date[f"ai_{asset}_contribution"] = manager_date["ai_weight"] * manager_date[f"{asset}_allocation"]
        manager_date[f"human_ai_{asset}_contribution"] = manager_date["human_ai_weight"] * manager_date[f"{asset}_allocation"]

    recommendations = []
    for report_date, group in manager_date.groupby("report_date", sort=True):
        realized = pd.to_numeric(group["future_excess_proxy"], errors="coerce")
        valid = realized.notna()
        def outcome_proxy(weight_column: str) -> float | None:
            if not valid.any():
                return None
            return float(np.average(realized[valid], weights=group.loc[valid, weight_column]))
        recommendations.append({
            "report_date": report_date, "expert_count": int(group["manager"].nunique()),
            "expert_stock": float(group["expert_stock_contribution"].sum()),
            "expert_bond": float(group["expert_bond_contribution"].sum()),
            "expert_cash": float(group["expert_cash_contribution"].sum()),
            "ai_stock": float(group["ai_stock_contribution"].sum()),
            "ai_bond": float(group["ai_bond_contribution"].sum()),
            "ai_cash": float(group["ai_cash_contribution"].sum()),
            "human_ai_stock": float(group["human_ai_stock_contribution"].sum()),
            "human_ai_bond": float(group["human_ai_bond_contribution"].sum()),
            "human_ai_cash": float(group["human_ai_cash_contribution"].sum()),
            "equal_weight_stock": float(group["stock_allocation"].mean()),
            "equal_weight_bond": float(group["bond_allocation"].mean()),
            "expert_outcome_proxy": outcome_proxy("expert_weight"),
            "ai_outcome_proxy": outcome_proxy("ai_weight"),
            "human_ai_outcome_proxy": outcome_proxy("human_ai_weight"),
            "equal_weight_outcome_proxy": None if not valid.any() else float(realized[valid].mean()),
            "outcome_proxy": outcome_proxy("human_ai_weight"),
            "validation_status": "realized expert-following proxy" if realized.notna().any() else "forward outcome not yet observable",
        })
    contribution_records = manager_date.sort_values(["report_date", "human_ai_weight"], ascending=[False, False]).to_dict("records")
    for record in contribution_records:
        for key, value in list(record.items()):
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
    report_manager_map = {
        f"{str(row.get('crsp_portno') or '').strip()}|{row['report_date']}": row["manager"]
        for row in contribution_records
    }
    return {
        "scope": "balanced_funds_only",
        "horizon_months": horizon,
        "weighting_method": "within-style relative expert performance; Human-AI weight multiplies expert weight by AI confidence",
        "recommendations": recommendations,
        "latest_recommendation": recommendations[-1] if recommendations else None,
        "manager_contributions": contribution_records,
        "report_manager_map": report_manager_map,
        "research_caveat": "Outcome proxy is not a tradable stock-bond backtest. A bond total-return index, disclosure availability lag, transaction costs and walk-forward baselines are required for causal performance claims.",
    }
