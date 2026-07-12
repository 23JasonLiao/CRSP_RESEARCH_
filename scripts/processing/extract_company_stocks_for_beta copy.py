#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 Part5 四個 holdings CSV 中整理「可用於 Yahoo Finance 抓報酬 / 算 beta 的個股清單」。

功能：
1. 讀取四個 CSV。
2. 使用 holding_security_name 與 holding_ticker 判斷是否像一般公司個股。
3. 排除債券、存款、money market、ETF、index fund、mutual fund、treasury、swap 等非個股標的。
4. 依 ticker 去重複，整理每個 ticker 的最早 / 最晚 report_dt。
5. 輸出一份乾淨的個股清單 CSV。
6. 另外輸出一份被排除標的的 audit CSV，方便人工檢查規則是否太嚴或太鬆。

注意：
- 這是 rule-based cleaning，不是官方證券分類器。
- 目的為產生「Yahoo Finance beta 計算候選個股」，後續仍建議用 yfinance 測試 ticker 是否可成功下載價格。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

INPUT_FILES = [
    Path(r"C:\Users\user\Desktop\crsp_research__\stock berfore 2010_new___.csv"),
    Path(r"C:\Users\user\Desktop\crsp_research__\stock between 2010_2014_new___.csv"),
    Path(r"C:\Users\user\Desktop\crsp_research__\stock between 2015_2019_new___.csv"),
    Path(r"C:\Users\user\Desktop\crsp_research__\stock between 2020_2026_new___.csv"),
]

OUTPUT_COMPANY_CSV = Path(r"C:\Users\user\Desktop\crsp_research__\part5_unique_company_stocks_for_yahoo_beta.csv")
OUTPUT_EXCLUDED_AUDIT_CSV = Path(r"C:\Users\user\Desktop\crsp_research__\part5_excluded_non_company_holdings_audit.csv")


# ====== 2. 非個股標的排除規則 ======
# 這些關鍵字通常代表：債券、基金、ETF、存款、現金、衍生品、政府債、ABS/MBS 等，不適合直接當個股算 beta。
EXCLUDE_NAME_PATTERNS = [
    # fund / ETF / index products

    # broader fund / investment company words
    r"\bFUND\b",
    r"\bFUNDS\b",
    r"\bSERIES\b",
    r"\bADVISER SERIES\b",
    r"\bINDEXIQ\b",
    r"\bPOWERSHARES\b",
    r"\bDIAMONDS TRUST\b",
    r"\bNASDAQ 100 TRUST\b",
    r"\bE T F\b",
    r"\bET SELF INDEX\b",
    r"\bLIQUIDITY FUNDS\b",
    r"\bCL [A-Z0-9]+ MF\b",
    r"\bMF\b",
    r"\bFD\b",
    r"\bINDEX\b",
    r"\bINDEX FUND\b",
    r"\bBND INDEX FUND\b",
    r"\bMUTUAL FUND\b",
    r"\bEXCHANGE TRADED FUND\b",
    r"\bETF\b",
    r"\bETN\b",
    r"\bISHARES\b",
    r"\bSPDR\b",
    r"\bVANGUARD\b",
    r"\bPROSHARES\b",
    r"\bDIREXION\b",
    r"\bINVESCO QQQ\b",
    r"\bSCHWAB US AGGREGATE\b",
    r"\bPORTFOLIO\b",
    r"\bSEPARATE ACCOUNT\b",
    r"\bMASTER PORTFOLIO\b",

    # bonds / notes / fixed income
    r"\bBOND\b",
    r"\bBONDS\b",
    r"\bNOTE\b",
    r"\bNOTES\b",
    r"\bDEBENTURE\b",
    r"\bDEBENTURES\b",
    r"\bSENIOR\b.*\bDUE\b",
    r"\bSUBORDINATED\b",
    r"\bCONVERTIBLE NOTE\b",
    r"\bCORPORATE NOTE\b",
    r"\bCAPITAL CORP\b",
    r"\bMUNICIPAL\b",
    r"\bMUNI\b",

    r"\bMED TERM\b",
    r"\bMEDIUM TERM\b",
    r"\bMTN\b",
    r"\bCAP\(MED TERM\b",
    r"\bTRUST PFD\b",
    r"\bPFD\b",
    r"\bPREFERRED\b",
    r"\bFIXED INCOME\b",
    r"\bHIGH YIELD\b",
    r"\bFLOATING RATE\b",

    # government / agency / treasury
    r"\bTREASURY\b",
    r"\bT-BILL\b",
    r"\bTBILL\b",
    r"\bBILL\b",
    r"\bBILLS\b",
    r"\bUNITED STATES TREAS\b",
    r"\bU S TREAS\b",
    r"\bUS TREAS\b",
    r"\bGOVERNMENT\b",
    r"\bFEDERAL\b",
    r"\bFNMA\b",
    r"\bFHLMC\b",
    r"\bGNMA\b",
    r"\bFANNIE MAE\b",
    r"\bFREDDIE MAC\b",
    r"\bGINNIE MAE\b",

    # deposits / cash / money market
    r"\bDEPOSIT\b",
    r"\bTIME/TERM DEPOSIT\b",
    r"\bEURODOLLAR\b",
    r"\bMONEY MARKET\b",
    r"\bCASH\b",
    r"\bREPURCHASE\b",
    r"\bCOMMERCIAL PAPER\b",
    r"\bCERTIFICATE OF DEPOSIT\b",
    r"\bCD\b",

    # derivatives / structured products
    r"\bOPTION\b",
    r"\bOPTIONS\b",
    r"\bFUTURE\b",
    r"\bFUTURES\b",
    r"\bSWAP\b",
    r"\bSWAPS\b",
    r"\bWARRANT\b",
    r"\bRIGHTS\b",
    r"\bFORWARD\b",
    r"\bCOLLATERAL\b",
    r"\bMORTGAGE\b",
    r"\bASSET BACKED\b",
    r"\bABS\b",
    r"\bMBS\b",
    r"\bCMBS\b",
]

COMPILED_EXCLUDE_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in EXCLUDE_NAME_PATTERNS]

# 一般美股 ticker 常見格式：AAPL, MSFT, JPM, BRK.B, BRK-B, BF.B 等。
# 這裡刻意排除 LP4021、912828、3135G0 這種基金代碼 / CUSIP-like identifier。
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")


def clean_text(value: object) -> str:
    """把 NaN/None 轉成空字串，並清掉前後空白。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_ticker(value: object) -> str:
    """清理 ticker：轉大寫、去空白。"""
    ticker = clean_text(value).upper()
    ticker = ticker.replace(" ", "")
    return ticker


def is_valid_yahoo_like_ticker(ticker: str) -> bool:
    """判斷 ticker 是否像 Yahoo Finance 可查的公司股票 ticker。"""
    if not ticker:
        return False
    if ticker in {"N/A", "NA", "NULL", "NONE", "-"}:
        return False
    if not VALID_TICKER_RE.match(ticker):
        return False
    return True


def is_safe_equity_trust_name(name: str) -> bool:
    """
    部分個股 / REIT 名稱會含 TRUST，例如 VORNADO REALTY TRUST。
    但很多 ETF / mutual fund 也叫 TRUST。這裡保留較像上市公司或 REIT 的 trust 名稱。
    """
    safe_regexes = [
        r"\bREALTY\b",
        r"\bPROPERTIES\b",
        r"\bPROPERTY\b",
        r"\bREIT\b",
        r"\bBANCORP\b",
        r"\bBANK\b",
        r"\bBANKS\b",
        r"\bCORP\b",
        r"\bINC\b",
        r"\bCO\b",
        r"\bNORTHERN TRUST\b",
        r"\bSUNTRUST\b",
        r"\bSOUTHTRUST\b",
    ]
    return any(re.search(pattern, name) for pattern in safe_regexes)


def exclusion_reason(security_name: str, ticker: str) -> str:
    """回傳排除原因；若不排除，回傳空字串。"""
    if not is_valid_yahoo_like_ticker(ticker):
        return "invalid_or_missing_ticker"

    name = security_name.upper()
    if not name:
        return "missing_security_name"

    # 美國 mutual fund ticker 常常是 5 碼且以 X 結尾，例如 ABCDX。
    # 這類 ticker 即使 Yahoo 可查，也通常不是公司個股，不適合拿來做個股 beta。
    if len(ticker) == 5 and ticker.endswith("X"):
        return "likely_mutual_fund_ticker_5_letters_ending_x"

    # TRUST 很容易混到 ETF / mutual fund trust；只有較像 REIT 或公司名稱時保留。
    if "TRUST" in name and not is_safe_equity_trust_name(name):
        return "excluded_trust_not_likely_company_equity"

    for pattern in COMPILED_EXCLUDE_PATTERNS:
        if pattern.search(name):
            return f"excluded_name_pattern:{pattern.pattern}"

    return ""


def mode_or_first(values: Iterable[str]) -> str:
    """取最常見值；若都空則回傳空字串。"""
    vals = [v for v in values if v]
    if not vals:
        return ""
    return pd.Series(vals).mode().iloc[0]


def top_unique_values(values: Iterable[str], max_items: int = 5) -> str:
    """回傳前幾個 unique 值，方便檢查同 ticker 是否有不同名稱。"""
    seen = []
    for v in values:
        if v and v not in seen:
            seen.append(v)
        if len(seen) >= max_items:
            break
    return " | ".join(seen)


def main() -> None:
    frames = []
    for file_path in INPUT_FILES:
        if not file_path.exists():
            raise FileNotFoundError(f"找不到檔案：{file_path}")

        df = pd.read_csv(file_path, dtype=str)
        required_cols = {"holding_security_name", "holding_ticker", "report_dt", "crsp_portno"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{file_path.name} 缺少必要欄位：{sorted(missing)}")

        df["source_file"] = file_path.name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # 清理基本欄位
    all_df["holding_ticker_clean"] = all_df["holding_ticker"].map(clean_ticker)
    all_df["holding_security_name_clean"] = all_df["holding_security_name"].map(clean_text)
    all_df["report_dt_parsed"] = pd.to_datetime(all_df["report_dt"], errors="coerce")

    # 判斷是否保留
    all_df["exclude_reason"] = all_df.apply(
        lambda r: exclusion_reason(r["holding_security_name_clean"], r["holding_ticker_clean"]),
        axis=1,
    )
    company_df = all_df[all_df["exclude_reason"] == ""].copy()
    excluded_df = all_df[all_df["exclude_reason"] != ""].copy()

    # 依 ticker 去重複；同一 ticker 可能有多個 security name，保留最常見名稱與所有代表名稱
    grouped = []
    for ticker, g in company_df.groupby("holding_ticker_clean", dropna=False):
        first_date = g["report_dt_parsed"].min()
        last_date = g["report_dt_parsed"].max()
        grouped.append(
            {
                "holding_ticker": ticker,
                "primary_security_name": mode_or_first(g["holding_security_name_clean"]),
                "all_observed_security_names_sample": top_unique_values(g["holding_security_name_clean"], max_items=5),
                "first_report_dt": first_date.date().isoformat() if pd.notna(first_date) else "",
                "last_report_dt": last_date.date().isoformat() if pd.notna(last_date) else "",
                "holding_record_count": int(len(g)),
                "unique_portfolio_count": int(g["crsp_portno"].nunique(dropna=True)),
                "source_files": " | ".join(sorted(g["source_file"].dropna().unique())),
            }
        )

    result = pd.DataFrame(grouped)
    if not result.empty:
        result = result.sort_values(
            ["holding_record_count", "holding_ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)

    # 排除清單 audit：幫你看哪些被規則踢掉
    excluded_audit = (
        excluded_df.groupby(
            ["holding_ticker_clean", "holding_security_name_clean", "exclude_reason"],
            dropna=False,
        )
        .agg(
            first_report_dt=("report_dt_parsed", "min"),
            last_report_dt=("report_dt_parsed", "max"),
            holding_record_count=("holding_ticker_clean", "size"),
            unique_portfolio_count=("crsp_portno", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "holding_ticker_clean": "holding_ticker",
                "holding_security_name_clean": "holding_security_name",
            }
        )
    )
    if not excluded_audit.empty:
        excluded_audit["first_report_dt"] = excluded_audit["first_report_dt"].dt.date.astype(str)
        excluded_audit["last_report_dt"] = excluded_audit["last_report_dt"].dt.date.astype(str)
        excluded_audit = excluded_audit.sort_values(
            ["holding_record_count", "holding_ticker"], ascending=[False, True]
        )

    result.to_csv(OUTPUT_COMPANY_CSV, index=False, encoding="utf-8-sig")
    excluded_audit.to_csv(OUTPUT_EXCLUDED_AUDIT_CSV, index=False, encoding="utf-8-sig")

    print("完成！")
    print(f"輸入 holdings rows: {len(all_df):,}")
    print(f"保留 company holding rows: {len(company_df):,}")
    print(f"輸出 unique company tickers: {len(result):,}")
    print(f"公司個股清單: {OUTPUT_COMPANY_CSV}")
    print(f"排除標的 audit 清單: {OUTPUT_EXCLUDED_AUDIT_CSV}")


if __name__ == "__main__":
    main()
