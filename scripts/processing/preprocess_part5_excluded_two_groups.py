#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Part5 excluded non-company holdings into the two categories requested by the advisor:

1) Bond / Credit / Money-related holdings
   - Treasury / agency / MBS / TBA
   - bond fund / fixed income fund / credit fund / preferred / notes
   - cash / money market / deposit / repo
   - bond ETFs such as AGG, BND, LQD, HYG, TLT, IEF, SHY, TIP

2) Equity Fund / Stock-fund-like holdings
   - equity ETFs / stock index funds / fund-like equity exposure
   - examples: SPY, IVV, VOO, VTI, QQQ, EFA, IEFA, IWM, IJH, IJR

The script intentionally does NOT calculate stock beta for these excluded holdings.
Company individual stocks should remain in the stock-beta pipeline, while this file is used for Part5B
non-individual holdings exposure analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Dict

import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\user\Desktop\crsp_research__")

INPUT_AUDIT_CSV = BASE_DIR / "part5_excluded_non_company_holdings_audit.csv"
INPUT_COMPANY_STOCK_CSV = BASE_DIR / "part5_unique_company_stocks_for_yahoo_beta.csv"

OUTPUT_ENRICHED_CSV = BASE_DIR / "part5_excluded_two_group_enriched.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "part5_excluded_two_group_summary.csv"
OUTPUT_TOP_ITEMS_CSV = BASE_DIR / "part5_excluded_two_group_top_items.csv"
OUTPUT_ACTIVE_YEAR_PANEL_CSV = BASE_DIR / "part5_excluded_two_group_active_year_panel.csv"
OUTPUT_REMOVED_INDIVIDUAL_STOCK_AUDIT_CSV = BASE_DIR / "part5_excluded_individual_stock_like_removed_audit.csv"

# Sandbox fallback: when running outside the user's Windows folder, output beside input.
def resolve_output_path(path: Path, input_path: Path) -> Path:
    return path if path.parent.exists() else input_path.parent / path.name


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def upper_text(value) -> str:
    return clean_text(value).upper()


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if text == "":
        return np.nan
    try:
        return float(text)
    except Exception:
        return np.nan


def parse_date(value) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def looks_like_ticker(ticker: str) -> bool:
    ticker = upper_text(ticker).replace(".", "-")
    return bool(re.match(r"^[A-Z0-9]{1,6}([.-][A-Z0-9]{1,2})?$", ticker))


BOND_ETF_TICKERS = {
    "AGG", "BND", "LQD", "HYG", "JNK", "TLT", "IEF", "SHY", "TIP", "MUB", "BIL", "SHV", "MBB",
    "VCIT", "VCSH", "VCLT", "IGSB", "IGIB", "IGLB", "EMB", "BIV", "BSV", "BLV",
}

EQUITY_FUND_TICKERS = {
    "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "EFA", "IEFA", "EEM", "IEMG", "VEA", "VWO",
    "IJH", "IJR", "IWF", "IWD", "IVE", "IVW", "VUG", "VTV", "VO", "VB", "MDY", "RSP",
    "XLK", "XLF", "XLV", "XLY", "XLI", "XLP", "XLE", "XLB", "XLU", "XLC", "XLRE",
}

FUND_LIKE_PATTERNS = [
    r"\bETF\b", r"\bEXCHANGE TRADED\b", r"\bFUND\b", r"\bFUNDS\b", r"\bPORTFOLIO\b", r"\bPORTFOLIOS\b",
    r"\bTRUST\b", r"\bSPDR\b", r"\bISHARES\b", r"\bVANGUARD\b", r"\bINVESCO\b", r"\bPOWERSHARES\b",
    r"\bBLACKROCK\b", r"\bSTATE STREET\b", r"\bFIDELITY\b", r"\bSCHWAB\b",
]

BOND_MONEY_PATTERNS = [
    # Cash / liquidity
    r"\bCASH\b", r"\bUSD\b", r"\bMONEY MARKET\b", r"\bMMF\b", r"\bDEPOSIT\b", r"\bTIME DEPOSIT\b",
    r"\bCERTIFICATE OF DEPOSIT\b", r"\bREPO\b", r"\bREPURCHASE\b", r"\bSWEEP\b", r"\bRESERVE\b", r"\bLIQUIDITY\b",
    # Government / agency / MBS / TBA
    r"\bTREAS\b", r"\bTREASURY\b", r"\bUNITED STATES\b", r"\bU S\b", r"\bUS GOV\b", r"\bGOVERNMENT\b",
    r"\bFEDERAL\b", r"\bFANNIE MAE\b", r"\bFNMA\b", r"\bFREDDIE MAC\b", r"\bFHLMC\b", r"\bGINNIE MAE\b",
    r"\bGNMA\b", r"\bTBA\b", r"\bMBS\b", r"\bMORTGAGE\b", r"\bAGENCY\b", r"\bNATL MORTGAGE\b",
    # Credit / fixed income
    r"\bBOND\b", r"\bFIXED INCOME\b", r"\bINCOME FUND\b", r"\bCREDIT\b", r"\bHIGH YIELD\b", r"\bNOTE\b", r"\bNOTES\b",
    r"\bDEBENTURE\b", r"\bMTN\b", r"\bMEDIUM TERM\b", r"\bLOAN\b", r"\bPREFERRED\b", r"\bPFD\b",
    # Derivatives/structured financial exposure, kept in money/credit side for advisor's two-bucket view.
    r"\bSWAP\b", r"\bFUTURE\b", r"\bFUTURES\b", r"\bFORWARD\b", r"\bOPTION\b", r"\bWARRANT\b", r"\bCOLLATERAL\b", r"\bSTRUCTURED\b",
]

EQUITY_FUND_PATTERNS = [
    r"\bSTOCK FUND\b", r"\bEQUITY\b", r"\bCOMMON STOCK\b", r"\bTOTAL STOCK MARKET\b", r"\bSTOCK MARKET\b",
    r"\bS&P\b", r"\bS P 500\b", r"\bNASDAQ\b", r"\bRUSSELL\b", r"\bDOW\b",
    r"\bLARGE CAP\b", r"\bMID CAP\b", r"\bSMALL CAP\b", r"\bGROWTH\b", r"\bVALUE\b",
    r"\bINTERNATIONAL\b", r"\bEMERGING MARKETS\b", r"\bDEVELOPED MARKETS\b", r"\bMSCI\b",
]


def fine_subcategory(text: str, ticker: str, teacher_category: str) -> str:
    if teacher_category.startswith("Bond"):
        if contains_any(text, [r"\bCASH\b", r"\bMONEY MARKET\b", r"\bDEPOSIT\b", r"\bREPO\b", r"\bLIQUIDITY\b", r"\bRESERVE\b"]):
            return "Cash / money market / liquidity"
        if contains_any(text, [r"\bTREAS", r"\bGOVERNMENT\b", r"\bUNITED STATES\b", r"\bFEDERAL\b", r"\bFANNIE\b", r"\bFREDDIE\b", r"\bGINNIE\b", r"\bMBS\b", r"\bTBA\b", r"\bMORTGAGE\b"]):
            return "Government / agency / Treasury / MBS"
        if ticker in BOND_ETF_TICKERS or contains_any(text, [r"\bBOND ETF\b", r"\bTOTAL BOND\b", r"\bAGGREGATE BOND\b"]):
            return "Bond ETF / bond index product"
        if contains_any(text, [r"\bBOND\b", r"\bFIXED INCOME\b", r"\bCREDIT\b", r"\bHIGH YIELD\b", r"\bPREFERRED\b", r"\bNOTE\b", r"\bLOAN\b"]):
            return "Corporate credit / fixed income"
        if contains_any(text, [r"\bSWAP\b", r"\bFUTURE\b", r"\bFORWARD\b", r"\bOPTION\b", r"\bCOLLATERAL\b", r"\bSTRUCTURED\b"]):
            return "Derivative / structured financial exposure"
        return "Other money-related / fixed-income-like"
    if ticker in EQUITY_FUND_TICKERS or contains_any(text, [r"\bETF\b", r"\bSPDR\b", r"\bISHARES\b", r"\bVANGUARD\b"]):
        return "Equity ETF / passive index product"
    if contains_any(text, [r"\bSTOCK FUND\b", r"\bEQUITY\b", r"\bTOTAL STOCK\b", r"\bS&P\b", r"\bRUSSELL\b", r"\bNASDAQ\b", r"\bMSCI\b"]):
        return "Stock fund / equity index fund"
    return "Other equity fund-like holding"


def classify_two_group(row: pd.Series, company_ticker_set: set[str]) -> Dict[str, object]:
    ticker = upper_text(row.get("holding_ticker", "")).replace(".", "-")
    name = upper_text(row.get("holding_security_name", ""))
    reason = upper_text(row.get("exclude_reason", ""))
    text = f" {ticker} {name} {reason} "

    is_fund_like = contains_any(text, FUND_LIKE_PATTERNS)
    is_company_universe_ticker = bool(ticker and ticker in company_ticker_set)

    # If something looks like a true individual company equity and is already in company-beta universe,
    # do not use it in this two-group excluded analysis. Keep it in a separate audit.
    individual_stock_like = is_company_universe_ticker and not contains_any(text, BOND_MONEY_PATTERNS) and not is_fund_like

    # Teacher's two buckets.
    if ticker in BOND_ETF_TICKERS or contains_any(text, BOND_MONEY_PATTERNS):
        teacher_category = "Bond / Credit / Money-related"
        exposure_dimension = "Fixed income / credit / liquidity / interest-rate exposure"
        interpretation = "Non-individual-stock holding related to cash, bonds, credit, government/agency/MBS, or financial overlay exposure. Useful with 10-year Treasury yield and defensive allocation discussion."
        confidence = "high"
    elif ticker in EQUITY_FUND_TICKERS or (is_fund_like and contains_any(text, EQUITY_FUND_PATTERNS)):
        teacher_category = "Equity Fund / Stock-fund-like"
        exposure_dimension = "Indirect equity / passive stock index exposure"
        interpretation = "Non-individual-stock holding that provides indirect stock or equity-index exposure, such as an ETF, index fund, or stock fund."
        confidence = "high"
    elif is_fund_like:
        # Fund-like but no clear stock/bond clue. Put it into bond/money by default only if wording implies income/reserve.
        # Otherwise classify as equity-fund-like with review flag? To satisfy two buckets, choose a best-effort bucket and mark review.
        if contains_any(text, [r"\bINCOME\b", r"\bRESERVE\b", r"\bSHORT TERM\b", r"\bULTRA SHORT\b"]):
            teacher_category = "Bond / Credit / Money-related"
            exposure_dimension = "Possible income or liquidity allocation"
        else:
            teacher_category = "Equity Fund / Stock-fund-like"
            exposure_dimension = "Possible indirect fund exposure"
        interpretation = "Fund-like holding without enough text to determine exact asset class; included in the advisor's two-bucket view but should be reviewed."
        confidence = "review"
    else:
        # Non-fund and not clearly equity fund; for advisor's two-bucket view, keep as money/credit side but flag review.
        teacher_category = "Bond / Credit / Money-related"
        exposure_dimension = "Other money-related / manual review"
        interpretation = "Excluded non-company holding not clearly identified as stock-fund-like; assigned to money/credit side for two-bucket analysis and flagged for review."
        confidence = "review"

    subcategory = fine_subcategory(text, ticker, teacher_category)
    return {
        "teacher_category": teacher_category,
        "teacher_subcategory": subcategory,
        "exposure_dimension": exposure_dimension,
        "interpretation": interpretation,
        "classification_confidence": confidence,
        "is_company_universe_ticker": is_company_universe_ticker,
        "is_individual_stock_like_removed": individual_stock_like,
        "use_in_part5b_two_group": not individual_stock_like,
    }


def build_active_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        first_dt = row.get("first_report_dt_parsed")
        last_dt = row.get("last_report_dt_parsed")
        if pd.isna(first_dt) or pd.isna(last_dt):
            continue
        start_year, end_year = int(first_dt.year), int(last_dt.year)
        if end_year < start_year:
            continue
        n_years = max(1, end_year - start_year + 1)
        record_count = row.get("holding_record_count_num", np.nan)
        portfolio_count = row.get("unique_portfolio_count_num", np.nan)
        for year in range(start_year, end_year + 1):
            rows.append({
                "year": year,
                "teacher_category": row.get("teacher_category", ""),
                "teacher_subcategory": row.get("teacher_subcategory", ""),
                "active_item_count": 1,
                "holding_record_count_proxy": record_count / n_years if np.isfinite(record_count) else np.nan,
                "unique_portfolio_count_proxy": portfolio_count / n_years if np.isfinite(portfolio_count) else np.nan,
            })
    if not rows:
        return pd.DataFrame()
    panel = pd.DataFrame(rows)
    return panel.groupby(["year", "teacher_category"], as_index=False).agg(
        active_item_count=("active_item_count", "sum"),
        holding_record_count_proxy=("holding_record_count_proxy", "sum"),
        unique_portfolio_count_proxy=("unique_portfolio_count_proxy", "sum"),
    )


def main() -> None:
    input_path = INPUT_AUDIT_CSV if INPUT_AUDIT_CSV.exists() else Path("/mnt/data/part5_excluded_non_company_holdings_audit.csv")
    company_path = INPUT_COMPANY_STOCK_CSV if INPUT_COMPANY_STOCK_CSV.exists() else Path("/mnt/data/part5_unique_company_stocks_for_yahoo_beta.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input audit CSV not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str)
    required = {"holding_ticker", "holding_security_name", "exclude_reason", "first_report_dt", "last_report_dt", "holding_record_count", "unique_portfolio_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input audit CSV missing columns: {sorted(missing)}")

    company_ticker_set: set[str] = set()
    if company_path.exists():
        company_df = pd.read_csv(company_path, dtype=str)
        if "holding_ticker" in company_df.columns:
            company_ticker_set = set(company_df["holding_ticker"].dropna().astype(str).str.upper().str.replace(".", "-", regex=False))

    print(f"[INFO] Input excluded audit rows: {len(df):,}")
    print(f"[INFO] Company stock universe tickers for removal check: {len(company_ticker_set):,}")

    df["holding_ticker"] = df["holding_ticker"].map(clean_text)
    df["holding_security_name"] = df["holding_security_name"].map(clean_text)
    df["exclude_reason"] = df["exclude_reason"].map(clean_text)
    df["first_report_dt_parsed"] = df["first_report_dt"].map(parse_date)
    df["last_report_dt_parsed"] = df["last_report_dt"].map(parse_date)
    df["holding_record_count_num"] = df["holding_record_count"].map(parse_number)
    df["unique_portfolio_count_num"] = df["unique_portfolio_count"].map(parse_number)
    df["yahoo_ticker"] = df["holding_ticker"].map(lambda x: upper_text(x).replace(".", "-"))

    classifications = df.apply(lambda row: classify_two_group(row, company_ticker_set), axis=1, result_type="expand")
    df = pd.concat([df, classifications], axis=1)

    usable = df[df["use_in_part5b_two_group"]].copy()
    removed = df[df["is_individual_stock_like_removed"]].copy()

    summary = usable.groupby("teacher_category", as_index=False).agg(
        excluded_item_count=("holding_security_name", "count"),
        holding_record_count=("holding_record_count_num", "sum"),
        unique_portfolio_count=("unique_portfolio_count_num", "sum"),
        high_confidence_items=("classification_confidence", lambda s: int((s == "high").sum())),
        review_items=("classification_confidence", lambda s: int((s == "review").sum())),
        first_report_dt=("first_report_dt_parsed", "min"),
        last_report_dt=("last_report_dt_parsed", "max"),
    )
    summary["first_report_dt"] = pd.to_datetime(summary["first_report_dt"]).dt.date.astype(str)
    summary["last_report_dt"] = pd.to_datetime(summary["last_report_dt"]).dt.date.astype(str)
    summary = summary.sort_values("holding_record_count", ascending=False)

    top_items = (usable.sort_values(["teacher_category", "holding_record_count_num"], ascending=[True, False])
                      .groupby("teacher_category", group_keys=False)
                      .head(50)
                      .copy())

    top_cols = [
        "teacher_category", "teacher_subcategory", "exposure_dimension", "holding_ticker", "yahoo_ticker",
        "holding_security_name", "exclude_reason", "holding_record_count", "unique_portfolio_count",
        "first_report_dt", "last_report_dt", "classification_confidence", "interpretation",
    ]

    export_cols = top_cols + ["is_company_universe_ticker", "is_individual_stock_like_removed", "use_in_part5b_two_group"]

    active_panel = build_active_year_panel(usable)

    output_enriched = resolve_output_path(OUTPUT_ENRICHED_CSV, input_path)
    output_summary = resolve_output_path(OUTPUT_SUMMARY_CSV, input_path)
    output_top = resolve_output_path(OUTPUT_TOP_ITEMS_CSV, input_path)
    output_panel = resolve_output_path(OUTPUT_ACTIVE_YEAR_PANEL_CSV, input_path)
    output_removed = resolve_output_path(OUTPUT_REMOVED_INDIVIDUAL_STOCK_AUDIT_CSV, input_path)

    usable[export_cols].to_csv(output_enriched, index=False, encoding="utf-8-sig")
    summary.to_csv(output_summary, index=False, encoding="utf-8-sig")
    top_items[top_cols].to_csv(output_top, index=False, encoding="utf-8-sig")
    active_panel.to_csv(output_panel, index=False, encoding="utf-8-sig")
    removed[export_cols].to_csv(output_removed, index=False, encoding="utf-8-sig")

    print("\n[DONE]")
    print(f"Two-group enriched CSV: {output_enriched}")
    print(f"Two-group summary CSV: {output_summary}")
    print(f"Two-group top items CSV: {output_top}")
    print(f"Two-group active-year panel CSV: {output_panel}")
    print(f"Removed individual-stock-like audit CSV: {output_removed}")
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    print("\n[NOTE]")
    print("The active-year panel is an interval proxy based on first_report_dt and last_report_dt, not true annual holding count.")


if __name__ == "__main__":
    main()
