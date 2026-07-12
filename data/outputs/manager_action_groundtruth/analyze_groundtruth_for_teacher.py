#!/usr/bin/env python3
"""
Analyze manager_action_ground_truth.csv and create teacher-facing QC summaries and figures.

Usage:
  python analyze_groundtruth_for_teacher.py --csv manager_action_ground_truth.csv --audit manager_action_ground_truth_audit.json --out-dir groundtruth_teacher_figures

This script does not modify the ground-truth CSV. It checks whether the CSV has
required fields, summarizes data quality, and produces figures useful for a
teacher/advisor discussion before ML/SHAP/LLM integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = [
    "manager", "fund", "report_date", "market_regime", "manager_style_group",
    "stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta",
    "technology_exposure", "bond_money_exposure", "indirect_equity_exposure",
    "delta_stock", "delta_beta", "delta_technology", "delta_bond_money",
    "style_deviation_score", "cross_asset_execution_type", "future_12m_return",
    "future_12m_excess_return", "future_drawdown", "outcome_label",
]

LEGACY_HOLDINGS_NAMES = [
    "stock berfore 2010_new___.csv",
    "stock between 2010_2014_new___.csv",
    "stock between 2015_2019_new___.csv",
    "stock between 2020_2026_new___.csv",
]

CLEAN_HOLDINGS_NAMES = [
    "stock_before_2010.csv",
    "stock_between_2010_2014.csv",
    "stock_between_2015_2019.csv",
    "stock_between_2020_2026.csv",
]


def safe_rate(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def positive_excess_flag(df: pd.DataFrame) -> pd.Series:
    return df["future_12m_excess_return"].where(df["future_12m_excess_return"].notna()).gt(0)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_holdings_sources(base_dir: Path) -> list[dict]:
    rows = []
    for clean, legacy in zip(CLEAN_HOLDINGS_NAMES, LEGACY_HOLDINGS_NAMES):
        clean_path = base_dir / clean
        legacy_path = base_dir / legacy
        item = {
            "clean_file": clean,
            "legacy_file": legacy,
            "clean_exists": clean_path.exists(),
            "legacy_exists": legacy_path.exists(),
            "same_bytes": None,
            "clean_rows": None,
            "legacy_rows": None,
        }
        if clean_path.exists() and legacy_path.exists():
            item["same_bytes"] = md5_file(clean_path) == md5_file(legacy_path)
            try:
                item["clean_rows"] = sum(1 for _ in clean_path.open("rb")) - 1
                item["legacy_rows"] = sum(1 for _ in legacy_path.open("rb")) - 1
            except Exception:
                pass
        rows.append(item)
    return rows


def plot_outcome_counts(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["outcome_label"].value_counts(dropna=False).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Ground-truth outcome label distribution")
    ax.set_xlabel("Number of manager-action report events")
    ax.set_ylabel("Outcome label")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v):,}", va="center")
    savefig(out_dir / "01_outcome_label_distribution.png")


def plot_action_counts(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["action_type"].value_counts(dropna=False).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Extracted manager action types")
    ax.set_xlabel("Number of report-level action events")
    ax.set_ylabel("Action type")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v):,}", va="center")
    savefig(out_dir / "02_action_type_counts.png")


def plot_action_outcome_summary(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["future_12m_excess_return"].notna()].copy()
    valid["positive_excess"] = valid["future_12m_excess_return"] > 0
    summary = valid.groupby("action_type").agg(
        n=("action_type", "size"),
        positive_rate=("positive_excess", "mean"),
        avg_future_excess=("future_12m_excess_return", "mean"),
    ).reset_index()
    summary = summary[summary["n"] >= 30].sort_values("avg_future_excess")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(summary["action_type"], summary["avg_future_excess"] * 100)
    ax.axvline(0, linewidth=1)
    ax.set_title("Average future 12-month excess return by action type")
    ax.set_xlabel("Average future 12-month excess return (%)")
    ax.set_ylabel("Action type")
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(row["avg_future_excess"] * 100, i, f" n={int(row['n'])}", va="center")
    savefig(out_dir / "03_action_type_avg_future_excess.png")

    summary2 = summary.sort_values("positive_rate")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(summary2["action_type"], summary2["positive_rate"] * 100)
    ax.set_title("Positive future excess rate by action type")
    ax.set_xlabel("Positive future 12-month excess rate (%)")
    ax.set_ylabel("Action type")
    savefig(out_dir / "04_action_type_positive_rate.png")


def plot_style_action_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["future_12m_excess_return"].notna()].copy()
    valid["positive_excess"] = valid["future_12m_excess_return"] > 0
    grouped = valid.groupby(["manager_style_group", "action_type"]).agg(
        n=("positive_excess", "size"),
        positive_rate=("positive_excess", "mean"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= 20]
    pivot = grouped.pivot(index="manager_style_group", columns="action_type", values="positive_rate")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(13, max(4, 0.55 * len(pivot.index))))
    im = ax.imshow(pivot.values * 100, aspect="auto")
    ax.set_title("Style-conditioned positive excess rate by action type")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Positive future 12-month excess rate (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val*100:.0f}", ha="center", va="center", fontsize=8)
    savefig(out_dir / "05_style_conditioned_action_heatmap.png")


def plot_execution_market_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["future_12m_excess_return"].notna()].copy()
    grouped = valid.groupby(["market_regime", "cross_asset_execution_type"]).agg(
        n=("future_12m_excess_return", "size"),
        avg_excess=("future_12m_excess_return", "mean"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= 20]
    pivot = grouped.pivot(index="market_regime", columns="cross_asset_execution_type", values="avg_excess")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(5, 0.55 * len(pivot.index))))
    im = ax.imshow(pivot.values * 100, aspect="auto")
    ax.set_title("Cross-asset execution patterns under market regimes")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average future 12-month excess return (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val*100:.1f}", ha="center", va="center", fontsize=8)
    savefig(out_dir / "06_market_regime_execution_heatmap.png")


def plot_style_deviation_bins(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df[df["future_12m_excess_return"].notna() & df["style_deviation_score"].notna()].copy()
    if len(valid) < 50:
        return
    valid["deviation_bin"] = pd.qcut(valid["style_deviation_score"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"], duplicates="drop")
    summary = valid.groupby("deviation_bin", observed=False).agg(
        n=("future_12m_excess_return", "size"),
        avg_future_excess=("future_12m_excess_return", "mean"),
        positive_rate=("future_12m_excess_return", lambda s: (s > 0).mean()),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["deviation_bin"].astype(str), summary["avg_future_excess"] * 100)
    ax.axhline(0, linewidth=1)
    ax.set_title("Future excess return by style-deviation bucket")
    ax.set_xlabel("Style-deviation score bucket")
    ax.set_ylabel("Average future 12-month excess return (%)")
    for i, row in summary.iterrows():
        ax.text(i, row["avg_future_excess"] * 100, f"n={int(row['n'])}", ha="center", va="bottom")
    savefig(out_dir / "07_style_deviation_bucket_future_excess.png")


def plot_yearly_trend(df: pd.DataFrame, out_dir: Path) -> None:
    valid = df.copy()
    valid["year"] = pd.to_numeric(valid["year"], errors="coerce")
    yearly = valid.groupby("year").agg(
        action_events=("outcome_label", "size"),
        avg_future_excess=("future_12m_excess_return", "mean"),
        positive_rate=("future_12m_excess_return", lambda s: (s.dropna() > 0).mean() if s.dropna().size else np.nan),
    ).dropna(subset=["action_events"]).reset_index()
    yearly = yearly[(yearly["year"] >= 1998) & (yearly["year"] <= 2026)]
    if yearly.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(yearly["year"], yearly["action_events"], marker="o")
    ax.set_title("Number of manager-action events over time")
    ax.set_xlabel("Report year")
    ax.set_ylabel("Number of action events")
    savefig(out_dir / "08_yearly_action_event_count.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(yearly["year"], yearly["avg_future_excess"] * 100, marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_title("Average future 12-month excess return over time")
    ax.set_xlabel("Report year")
    ax.set_ylabel("Average future 12-month excess return (%)")
    savefig(out_dir / "09_yearly_avg_future_excess.png")


def plot_data_quality(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["data_quality_flags"].value_counts().head(12).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Data quality flag distribution")
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Data quality flags")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v):,}", va="center")
    savefig(out_dir / "10_data_quality_flags.png")


def write_summary(df: pd.DataFrame, audit: dict, source_checks: list[dict], out_dir: Path) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    date_min = str(pd.to_datetime(df["report_date"], errors="coerce").min().date())
    date_max = str(pd.to_datetime(df["report_date"], errors="coerce").max().date())
    numeric_cols = [
        "stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta",
        "technology_exposure", "bond_money_exposure", "indirect_equity_exposure",
        "future_12m_return", "future_12m_excess_return", "future_drawdown",
    ]
    null_rates = df[REQUIRED_COLUMNS].isna().mean().sort_values(ascending=False).round(4).to_dict()
    range_summary = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.round(6).to_dict(orient="index")
    output = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_required_columns": missing_cols,
        "report_date_min": date_min,
        "report_date_max": date_max,
        "unique_managers": int(df["manager"].nunique(dropna=True)),
        "unique_funds": int(df["fund"].nunique(dropna=True)),
        "unique_portfolios": int(df["crsp_portno"].nunique(dropna=True)) if "crsp_portno" in df.columns else None,
        "outcome_label_counts": df["outcome_label"].value_counts(dropna=False).to_dict(),
        "action_type_counts": df["action_type"].value_counts(dropna=False).to_dict() if "action_type" in df.columns else {},
        "manager_style_group_counts": df["manager_style_group"].value_counts(dropna=False).to_dict(),
        "data_quality_flag_counts": df["data_quality_flags"].value_counts(dropna=False).to_dict() if "data_quality_flags" in df.columns else {},
        "required_column_null_rates": null_rates,
        "numeric_range_summary": range_summary,
        "holdings_source_equivalence": source_checks,
        "audit_rows_match_csv": int(audit.get("rows", -1)) == int(len(df)),
        "audit_missing_files": audit.get("missing_files", []),
        "audit_warnings": audit.get("warnings", []),
    }
    (out_dir / "groundtruth_qc_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Ground-truth QC summary")
    lines.append("")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {df.shape[1]:,}")
    lines.append(f"Report date range: {date_min} to {date_max}")
    lines.append(f"Unique managers: {df['manager'].nunique(dropna=True):,}")
    lines.append(f"Unique funds: {df['fund'].nunique(dropna=True):,}")
    if missing_cols:
        lines.append(f"Missing required columns: {missing_cols}")
    else:
        lines.append("All required columns are present.")
    lines.append("")
    lines.append("## Main cautions")
    lines.append("- Treat `outcome_label` as historical association, not causal proof.")
    lines.append("- Use `data_quality_flags == ok` as the clean first-pass subset for ML.")
    lines.append("- Rows with missing future outcomes are expected near the end of the sample because there is no complete future 12-month window.")
    lines.append("- Allocation/exposure proxies can exceed 100% when raw CRSP holdings contain leverage, shorts, derivatives, duplicated exposure, or proxy-completed values; review range plots before modeling.")
    lines.append("")
    lines.append("## Figures generated")
    for p in sorted(out_dir.glob("*.png")):
        lines.append(f"- {p.name}")
    (out_dir / "groundtruth_qc_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("manager_action_ground_truth.csv"))
    parser.add_argument("--audit", type=Path, default=Path("manager_action_ground_truth_audit.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("groundtruth_teacher_figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    audit = json.loads(args.audit.read_text(encoding="utf-8")) if args.audit.exists() else {}
    source_checks = check_holdings_sources(args.csv.parent)

    for col in ["report_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.columns:
        if col not in ["manager", "fund", "market_regime", "manager_style_group", "cross_asset_execution_type", "outcome_label", "action_type", "data_quality_flags", "report_date"]:
            # Only convert object columns that look numeric enough.
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.7:
                    df[col] = converted

    plot_outcome_counts(df, args.out_dir)
    plot_action_counts(df, args.out_dir)
    plot_action_outcome_summary(df, args.out_dir)
    plot_style_action_heatmap(df, args.out_dir)
    plot_execution_market_heatmap(df, args.out_dir)
    plot_style_deviation_bins(df, args.out_dir)
    plot_yearly_trend(df, args.out_dir)
    plot_data_quality(df, args.out_dir)
    write_summary(df, audit, source_checks, args.out_dir)
    print(f"Done. Figures and summaries saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
