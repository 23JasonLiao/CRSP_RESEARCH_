#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_manager_groundtruth_cases.py

Purpose
-------
This script is designed for the current research step suggested by your advisor:
"Before designing the final algorithm, manually inspect the data, choose one or two
fund managers, and find time points where the manager appears to have made an
unusual allocation decision."

Input
-----
manager_action_ground_truth.csv

Output
------
A folder containing:
1. qc_summary.txt
2. manager_case_candidates.csv
3. suspicious_events_top.csv
4. selected_managers_for_manual_review.csv
5. per-manager event tables
6. per-manager timeline figures
7. teacher_case_report.md

Recommended usage
-----------------
python select_manager_groundtruth_cases.py ^
  --groundtruth "C:\\Users\\user\\Desktop\\crsp_research__\\data\\outputs\\manager_action_groundtruth\\manager_action_ground_truth.csv" ^
  --output-dir "C:\\Users\\user\\Desktop\\crsp_research__\\outputs\\teacher_manager_cases" ^
  --top-managers 6

To manually force specific managers:
python select_manager_groundtruth_cases.py --groundtruth manager_action_ground_truth.csv --managers "Manager Name 1" "Manager Name 2"
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "manager",
    "fund",
    "report_date",
    "market_regime",
    "manager_style_group",
    "stock_allocation",
    "bond_allocation",
    "cash_allocation",
    "portfolio_beta",
    "technology_exposure",
    "bond_money_exposure",
    "indirect_equity_exposure",
    "delta_stock",
    "delta_beta",
    "delta_technology",
    "delta_bond_money",
    "style_deviation_score",
    "cross_asset_execution_type",
    "future_12m_return",
    "future_12m_excess_return",
    "future_drawdown",
    "outcome_label",
]

DELTA_COLUMNS = [
    "delta_stock",
    "delta_beta",
    "delta_technology",
    "delta_bond_money",
    "delta_indirect_equity",
]

EXPOSURE_COLUMNS = [
    "stock_allocation",
    "bond_allocation",
    "cash_allocation",
    "technology_exposure",
    "bond_money_exposure",
    "indirect_equity_exposure",
]

CONTEXT_COLUMNS = [
    "market_regime",
    "manager_style_group",
    "cross_asset_execution_type",
    "action_type",
    "outcome_label",
    "data_quality_flags",
]


def safe_slug(text: str, max_len: int = 80) -> str:
    text = str(text or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "unknown"


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in ground truth CSV: " + ", ".join(missing)
        )


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def percentile_rank(series: pd.Series) -> pd.Series:
    """Return percentile ranks in [0, 1], keeping NaN as 0."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index)
    ranked = s.rank(method="average", pct=True)
    return ranked.fillna(0.0).clip(0, 1)


def normalize_outcome_label(label: str) -> str:
    label = str(label or "").strip()
    if not label:
        return "unknown"
    return label


def prepare_groundtruth(path: Path, clean_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    require_columns(df, REQUIRED_COLUMNS)

    numeric_base_cols = [
        "stock_allocation",
        "bond_allocation",
        "cash_allocation",
        "portfolio_beta",
        "technology_exposure",
        "bond_money_exposure",
        "indirect_equity_exposure",
        "delta_stock",
        "delta_beta",
        "delta_technology",
        "delta_bond_money",
        "style_deviation_score",
        "future_12m_return",
        "future_12m_excess_return",
        "future_drawdown",
    ]
    numeric_cols = list(set(
        numeric_base_cols
        + DELTA_COLUMNS
        + EXPOSURE_COLUMNS
        + [
            "action_strength",
            "future_12m_sp500_return",
            "yield10y",
            "sp500_trailing_3y",
            "fund_trailing_3y",
            "fund_trailing_3y_excess",
            "manager_obs_count",
            "manager_reliability_score",
            "manager_defensive_score",
            "manager_flow_score",
            "manager_growth_tilt_score",
            "top_holding_concentration",
            "holding_row_count",
            "beta_matched_holding_count",
            "non_individual_matched_holding_count",
        ]
    ))
    df = to_numeric(df, numeric_cols)

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    if "year" not in df.columns:
        df["year"] = df["report_date"].dt.year
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    for col in ["manager", "fund", "market_regime", "manager_style_group",
                "cross_asset_execution_type", "action_type", "outcome_label",
                "data_quality_flags"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df["outcome_label"] = df["outcome_label"].map(normalize_outcome_label)
    df["has_future_outcome"] = (
        df["future_12m_excess_return"].notna()
        & ~df["outcome_label"].eq("missing_future_outcome")
    )
    df["positive_excess"] = df["future_12m_excess_return"] > 0
    df["is_clean_row"] = df.get("data_quality_flags", "").eq("ok")
    df["is_clear_action"] = ~df.get("action_type", "").isin([
        "",
        "stable_or_no_clear_action",
        "minor_allocation_rotation",
    ])

    if clean_only:
        df = df[df["is_clean_row"]].copy()

    # Ex-ante unusualness score:
    # This uses only action/context features available at report_date, not future outcomes.
    # It is meant to help humans find unusual decisions to inspect, not to prove effectiveness.
    score_parts = []

    if "style_deviation_score" in df.columns:
        score_parts.append(("style_deviation_rank", percentile_rank(df["style_deviation_score"]), 0.30))
    if "action_strength" in df.columns:
        score_parts.append(("action_strength_rank", percentile_rank(df["action_strength"].abs()), 0.20))

    # Allocation/action deltas; each receives smaller weight.
    available_delta_cols = [c for c in DELTA_COLUMNS if c in df.columns]
    delta_weight_total = 0.30
    if available_delta_cols:
        each = delta_weight_total / len(available_delta_cols)
        for col in available_delta_cols:
            score_parts.append((f"abs_{col}_rank", percentile_rank(df[col].abs()), each))

    # Non-stable action bonus.
    clear_action_score = df["is_clear_action"].astype(float)
    score_parts.append(("clear_action_bonus", clear_action_score, 0.15))

    # Data quality bonus: clean rows are easier to defend in a meeting.
    clean_score = df["is_clean_row"].astype(float)
    score_parts.append(("clean_data_bonus", clean_score, 0.05))

    total = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    total_weight = 0.0
    for name, part, weight in score_parts:
        df[name] = part
        total = total + part * weight
        total_weight += weight

    df["ex_ante_unusual_score"] = (total / total_weight if total_weight else total).clip(0, 1)

    # Human-readable reason string.
    df["unusual_reason"] = df.apply(build_unusual_reason, axis=1)
    return df


def build_unusual_reason(row: pd.Series) -> str:
    reasons = []
    if pd.notna(row.get("style_deviation_score")) and row.get("style_deviation_score", 0) >= 0.75:
        reasons.append("high style deviation")
    if pd.notna(row.get("action_strength")) and abs(row.get("action_strength", 0)) >= 0.10:
        reasons.append("large action strength")
    for col, label in [
        ("delta_stock", "large stock change"),
        ("delta_beta", "large beta change"),
        ("delta_technology", "large technology change"),
        ("delta_bond_money", "large bond/money change"),
        ("delta_indirect_equity", "large indirect equity change"),
    ]:
        value = row.get(col)
        if pd.notna(value) and abs(value) >= 0.05:
            reasons.append(label)
    action = str(row.get("action_type", ""))
    if action and action not in {"stable_or_no_clear_action", "minor_allocation_rotation"}:
        reasons.append(f"clear action: {action}")
    if not reasons:
        reasons.append("moderate or stable action")
    return "; ".join(reasons)


def summarize_qc(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Ground-truth CSV quality summary")
    lines.append("")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    if "manager" in df.columns:
        lines.append(f"Unique managers: {df['manager'].replace('', np.nan).nunique():,}")
    if "fund" in df.columns:
        lines.append(f"Unique funds: {df['fund'].replace('', np.nan).nunique():,}")
    if "report_date" in df.columns:
        lines.append(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")
    lines.append("")
    lines.append("Outcome label distribution:")
    lines.append(df["outcome_label"].value_counts(dropna=False).to_string())
    lines.append("")
    if "action_type" in df.columns:
        lines.append("Action type distribution:")
        lines.append(df["action_type"].value_counts(dropna=False).head(20).to_string())
        lines.append("")
    if "data_quality_flags" in df.columns:
        lines.append("Data quality flags:")
        lines.append(df["data_quality_flags"].value_counts(dropna=False).head(20).to_string())
        lines.append("")
    lines.append("Recommended first manual/ML subset:")
    lines.append("- data_quality_flags == 'ok'")
    lines.append("- outcome_label != 'missing_future_outcome'")
    lines.append("- keep future outcomes only for evaluation, not for selecting unusual action points")
    return "\n".join(lines)


def manager_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    usable = df.copy()
    grp = usable.groupby("manager", dropna=False)

    rows = []
    for manager, g in grp:
        if not str(manager).strip():
            continue

        has_future = g[g["has_future_outcome"]]
        clear = g[g["is_clear_action"]]
        years = pd.to_numeric(g["year"], errors="coerce")

        rows.append({
            "manager": manager,
            "row_count": len(g),
            "clean_row_count": int(g["is_clean_row"].sum()),
            "future_outcome_count": int(g["has_future_outcome"].sum()),
            "distinct_fund_count": g["fund"].replace("", np.nan).nunique() if "fund" in g.columns else np.nan,
            "distinct_action_type_count": g["action_type"].replace("", np.nan).nunique() if "action_type" in g.columns else np.nan,
            "clear_action_count": len(clear),
            "clear_action_rate": len(clear) / len(g) if len(g) else np.nan,
            "positive_excess_rate": has_future["positive_excess"].mean() if len(has_future) else np.nan,
            "avg_future_12m_excess_return": has_future["future_12m_excess_return"].mean() if len(has_future) else np.nan,
            "avg_ex_ante_unusual_score": g["ex_ante_unusual_score"].mean(),
            "max_ex_ante_unusual_score": g["ex_ante_unusual_score"].max(),
            "avg_style_deviation_score": g["style_deviation_score"].mean(),
            "max_style_deviation_score": g["style_deviation_score"].max(),
            "first_year": int(years.min()) if years.notna().any() else np.nan,
            "last_year": int(years.max()) if years.notna().any() else np.nan,
            "style_groups": "; ".join(sorted([x for x in g["manager_style_group"].dropna().unique() if str(x).strip()])[:5]),
            "top_action_types": "; ".join(g["action_type"].value_counts().head(5).index.astype(str).tolist()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Recommendation score favors enough data, clear actions, high unusual score, and action diversity.
    out["coverage_score"] = np.log1p(out["future_outcome_count"].fillna(0)) / np.log1p(max(out["future_outcome_count"].max(), 1))
    out["action_diversity_score"] = np.minimum(out["distinct_action_type_count"].fillna(0) / 6.0, 1.0)
    out["recommendation_score"] = (
        0.30 * out["coverage_score"].fillna(0)
        + 0.25 * out["avg_ex_ante_unusual_score"].fillna(0)
        + 0.20 * out["max_ex_ante_unusual_score"].fillna(0)
        + 0.15 * out["clear_action_rate"].fillna(0)
        + 0.10 * out["action_diversity_score"].fillna(0)
    )
    out = out.sort_values(["recommendation_score", "future_outcome_count"], ascending=[False, False])
    return out


def suspicious_events(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    cols = [
        "manager", "fund", "report_date", "year", "market_regime", "manager_style_group",
        "action_type", "cross_asset_execution_type", "unusual_reason",
        "ex_ante_unusual_score", "style_deviation_score", "action_strength",
        "stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta",
        "technology_exposure", "bond_money_exposure", "indirect_equity_exposure",
        "delta_stock", "delta_beta", "delta_technology", "delta_bond_money", "delta_indirect_equity",
        "future_12m_return", "future_12m_sp500_return", "future_12m_excess_return",
        "future_drawdown", "outcome_label", "data_quality_flags",
    ]
    cols = [c for c in cols if c in df.columns]
    ranked = df.sort_values("ex_ante_unusual_score", ascending=False)
    return ranked[cols].head(top_n).copy()


def select_managers(df: pd.DataFrame, candidates: pd.DataFrame, explicit: Optional[List[str]], top_managers: int, include_team_managed: bool = False) -> List[str]:
    if explicit:
        available = set(df["manager"].astype(str))
        chosen = []
        for name in explicit:
            if name in available:
                chosen.append(name)
            else:
                # Try case-insensitive contains.
                matches = [m for m in available if name.lower() in m.lower()]
                if matches:
                    chosen.append(sorted(matches)[0])
                else:
                    print(f"[WARN] Manager not found: {name}")
        return chosen

    # Prefer named individual / team-of-named managers with future outcomes and clear actions.
    # By default, generic labels such as "Team Managed" and "Unknown Manager" are excluded
    # because they are less useful for the advisor's requested manual case study.
    c = candidates.copy()
    if not include_team_managed and "manager" in c.columns:
        bad = c["manager"].astype(str).str.lower().str.contains("unknown manager|team managed", regex=True, na=False)
        c = c[~bad].copy()

    preferred = c[(c["future_outcome_count"] >= 20) & (c["clear_action_count"] >= 5)]
    if preferred.empty:
        preferred = c.copy()
    return preferred["manager"].head(top_managers).astype(str).tolist()


def plot_manager_timelines(manager_df: pd.DataFrame, manager: str, out_dir: Path) -> None:
    mslug = safe_slug(manager)
    g = manager_df.sort_values("report_date").copy()
    if g.empty:
        return

    # If same manager has many funds/reports on same date, aggregate by date for timeline readability.
    agg_dict = {}
    for col in DELTA_COLUMNS + EXPOSURE_COLUMNS + [
        "portfolio_beta", "future_12m_excess_return", "future_drawdown",
        "ex_ante_unusual_score", "style_deviation_score", "action_strength",
    ]:
        if col in g.columns:
            agg_dict[col] = "mean"
    if "report_date" not in g.columns:
        return

    timeline = g.groupby("report_date", as_index=False).agg(agg_dict).sort_values("report_date")

    # 1. Action deltas timeline.
    plt.figure(figsize=(13, 6))
    for col in [c for c in DELTA_COLUMNS if c in timeline.columns]:
        plt.plot(timeline["report_date"], timeline[col], marker="o", linewidth=1, label=col)
    plt.axhline(0, linewidth=1)
    plt.title(f"{manager} - allocation/action deltas over time")
    plt.xlabel("Report date")
    plt.ylabel("Average delta")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mslug}_01_action_deltas_timeline.png", dpi=160)
    plt.close()

    # 2. Allocation and exposure timeline.
    plt.figure(figsize=(13, 6))
    plot_cols = [c for c in EXPOSURE_COLUMNS if c in timeline.columns]
    for col in plot_cols:
        plt.plot(timeline["report_date"], timeline[col], marker="o", linewidth=1, label=col)
    plt.title(f"{manager} - allocation and exposure levels over time")
    plt.xlabel("Report date")
    plt.ylabel("Average level / exposure")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mslug}_02_allocation_exposure_timeline.png", dpi=160)
    plt.close()

    # 3. Unusual score and future excess return.
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(timeline["report_date"], timeline["ex_ante_unusual_score"], marker="o", linewidth=1, label="ex_ante_unusual_score")
    ax1.set_xlabel("Report date")
    ax1.set_ylabel("Unusual action score")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    if "future_12m_excess_return" in timeline.columns:
        ax2.plot(timeline["report_date"], timeline["future_12m_excess_return"], marker="x", linewidth=1, label="future_12m_excess_return")
        ax2.axhline(0, linewidth=1)
        ax2.set_ylabel("Future 12M excess return")
    plt.title(f"{manager} - unusual action score vs future outcome")
    fig.tight_layout()
    plt.savefig(out_dir / f"{mslug}_03_unusual_score_vs_future_outcome.png", dpi=160)
    plt.close()

    # 4. Portfolio beta and technology exposure.
    plt.figure(figsize=(13, 6))
    for col in ["portfolio_beta", "technology_exposure", "bond_money_exposure", "indirect_equity_exposure"]:
        if col in timeline.columns:
            plt.plot(timeline["report_date"], timeline[col], marker="o", linewidth=1, label=col)
    plt.title(f"{manager} - cross-asset / sector exposure timeline")
    plt.xlabel("Report date")
    plt.ylabel("Average exposure")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"{mslug}_04_cross_asset_exposure_timeline.png", dpi=160)
    plt.close()


def write_manager_outputs(df: pd.DataFrame, managers: List[str], out_dir: Path) -> pd.DataFrame:
    selected_rows = []
    for manager in managers:
        g = df[df["manager"].astype(str) == str(manager)].copy()
        if g.empty:
            continue

        g = g.sort_values("ex_ante_unusual_score", ascending=False)
        cols = [
            "manager", "fund", "report_date", "year", "market_regime", "manager_style_group",
            "action_type", "cross_asset_execution_type", "unusual_reason",
            "ex_ante_unusual_score", "style_deviation_score", "action_strength",
            "stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta",
            "technology_exposure", "bond_money_exposure", "indirect_equity_exposure",
            "delta_stock", "delta_beta", "delta_technology", "delta_bond_money", "delta_indirect_equity",
            "future_12m_return", "future_12m_sp500_return", "future_12m_excess_return",
            "future_drawdown", "outcome_label", "data_quality_flags",
        ]
        cols = [c for c in cols if c in g.columns]

        per_path = out_dir / f"{safe_slug(manager)}_manual_review_events.csv"
        g[cols].to_csv(per_path, index=False, encoding="utf-8-sig")

        selected_rows.append(g[cols].head(10))
        plot_manager_timelines(g, manager, out_dir)

    if selected_rows:
        combined = pd.concat(selected_rows, ignore_index=True)
    else:
        combined = pd.DataFrame()
    return combined


def write_teacher_report(
    out_dir: Path,
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_managers: List[str],
    selected_events: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Teacher-facing manager-action manual review report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report supports the advisor's suggestion: first manually inspect the data, choose one or two managers, and identify time points where the manager appears to have made an unusual allocation decision. The script does not prove causality. It only ranks candidate events for human inspection.")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Unique managers: {df['manager'].replace('', np.nan).nunique():,}")
    lines.append(f"- Unique funds: {df['fund'].replace('', np.nan).nunique():,}")
    lines.append(f"- Date range: {df['report_date'].min()} to {df['report_date'].max()}")
    lines.append("")
    lines.append("## Recommended managers for manual inspection")
    lines.append("")
    if selected_managers:
        for i, manager in enumerate(selected_managers, start=1):
            row = candidates[candidates["manager"] == manager].head(1)
            if row.empty:
                lines.append(f"{i}. {manager}")
            else:
                r = row.iloc[0]
                lines.append(
                    f"{i}. {manager} — rows={int(r['row_count'])}, "
                    f"future_outcomes={int(r['future_outcome_count'])}, "
                    f"clear_actions={int(r['clear_action_count'])}, "
                    f"avg_unusual_score={r['avg_ex_ante_unusual_score']:.3f}, "
                    f"positive_excess_rate={r['positive_excess_rate']:.2%}"
                    if pd.notna(r["positive_excess_rate"]) else
                    f"{i}. {manager} — rows={int(r['row_count'])}"
                )
    else:
        lines.append("No manager selected.")
    lines.append("")
    lines.append("## How to read the generated figures")
    lines.append("")
    lines.append("For each selected manager, the script generates four plots:")
    lines.append("")
    lines.append("1. `*_01_action_deltas_timeline.png`: shows changes in stock, beta, technology, bond/money, and indirect equity exposure. Spikes are candidate unusual decisions.")
    lines.append("2. `*_02_allocation_exposure_timeline.png`: shows the manager's allocation/exposure levels over time.")
    lines.append("3. `*_03_unusual_score_vs_future_outcome.png`: compares ex-ante unusual action score with future 12-month excess return. Future outcome is for evaluation, not for defining unusualness.")
    lines.append("4. `*_04_cross_asset_exposure_timeline.png`: focuses on beta, technology, bond/money, and indirect equity exposure.")
    lines.append("")
    lines.append("## Important caveats")
    lines.append("")
    lines.append("- Do not say the action caused the future return. Say the action was followed by, or associated with, the future outcome.")
    lines.append("- The unusual action score uses only ex-ante action/context features, not future return.")
    lines.append("- Use the selected events as starting points for visual inspection, not as final ground truth.")
    lines.append("- After manual review, you can add a new column such as `manual_case_label` or `expert_validated_action`.")
    lines.append("")
    lines.append("## Suggested next manual step")
    lines.append("")
    lines.append("Open the per-manager CSV and choose 1-2 events with high `ex_ante_unusual_score`. Then inspect the corresponding Part5 timeline, holdings details, market regime, and future outcome. Write a short note: what action happened, why it looks unusual for that manager, and what happened afterward.")
    lines.append("")

    report_path = out_dir / "teacher_case_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find manager-action candidate cases for manual ground-truth inspection.")
    parser.add_argument("--groundtruth", type=str, required=True, help="Path to manager_action_ground_truth.csv")
    parser.add_argument("--output-dir", type=str, default="teacher_manager_cases", help="Output directory")
    parser.add_argument("--top-managers", type=int, default=6, help="Number of managers to recommend when --managers is not provided")
    parser.add_argument("--managers", nargs="*", default=None, help="Optional explicit manager names to inspect")
    parser.add_argument("--clean-only", action="store_true", help="Use only rows with data_quality_flags == ok for ranking and plots")
    parser.add_argument("--top-events", type=int, default=300, help="Number of top unusual events to export")
    parser.add_argument("--include-team-managed", action="store_true", help="Allow generic managers such as Team Managed / Unknown Manager in automatic recommendations")
    args = parser.parse_args()

    groundtruth_path = Path(args.groundtruth)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_groundtruth(groundtruth_path, clean_only=args.clean_only)

    qc_text = summarize_qc(df)
    (out_dir / "qc_summary.txt").write_text(qc_text, encoding="utf-8")

    candidates = manager_candidate_table(df)
    candidates.to_csv(out_dir / "manager_case_candidates.csv", index=False, encoding="utf-8-sig")

    top_events = suspicious_events(df, top_n=args.top_events)
    top_events.to_csv(out_dir / "suspicious_events_top.csv", index=False, encoding="utf-8-sig")

    selected_managers = select_managers(df, candidates, args.managers, args.top_managers, include_team_managed=args.include_team_managed)
    selected_df = candidates[candidates["manager"].isin(selected_managers)].copy()
    selected_df.to_csv(out_dir / "selected_managers_for_manual_review.csv", index=False, encoding="utf-8-sig")

    selected_events = write_manager_outputs(df, selected_managers, out_dir)
    if not selected_events.empty:
        selected_events.to_csv(out_dir / "selected_manager_top_events_for_manual_review.csv", index=False, encoding="utf-8-sig")

    write_teacher_report(out_dir, df, candidates, selected_managers, selected_events)

    print("Done.")
    print(f"Input: {groundtruth_path}")
    print(f"Output directory: {out_dir}")
    print(f"Recommended managers: {', '.join(selected_managers) if selected_managers else '(none)'}")
    print("Main files:")
    print(f"  - {out_dir / 'qc_summary.txt'}")
    print(f"  - {out_dir / 'manager_case_candidates.csv'}")
    print(f"  - {out_dir / 'suspicious_events_top.csv'}")
    print(f"  - {out_dir / 'selected_managers_for_manual_review.csv'}")
    print(f"  - {out_dir / 'teacher_case_report.md'}")


if __name__ == "__main__":
    main()
