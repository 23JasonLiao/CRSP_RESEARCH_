#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate yearly trailing stock beta for Part5 company stock universe.

Input:
- part5_unique_company_stocks_for_yahoo_beta.csv
  Required columns:
    holding_ticker, primary_security_name, first_report_dt, last_report_dt

Output:
- part5_yearly_trailing_stock_beta.csv
  One row per ticker-year, with trailing 1y / 3y / 5y return and beta.

Logic aligned with your frontend trailing design:
- y1 = 12 monthly returns
- y3 = 36 monthly returns
- y5 = 60 monthly returns
- For each ticker and each calendar year between first_report_dt.year and last_report_dt.year,
  compute trailing metrics ending at that year's December month-end.
- Beta = Cov(stock_monthly_return, market_monthly_return) / Var(market_monthly_return)
- Stock return for each trailing window = compounded monthly return over the window

Important:
- This uses Yahoo Finance via yfinance. Some historical/delisted/renamed tickers may fail.
- This is not investment advice. It is a preprocessing step for research analysis.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# 1. Paths and settings
# =========================

BASE_DIR = Path(r"C:\Users\user\Desktop\crsp_research__")

INPUT_STOCK_UNIVERSE = BASE_DIR / "part5_unique_company_stocks_for_yahoo_beta.csv"
OUTPUT_BETA_CSV = BASE_DIR / "part5_yearly_trailing_stock_beta.csv"
OUTPUT_FAILED_CSV = BASE_DIR / "part5_yfinance_failed_tickers.csv"
OUTPUT_PRICE_CACHE_CSV = BASE_DIR / "part5_yfinance_monthly_close_cache.csv"
OUTPUT_SECTOR_CACHE_CSV = BASE_DIR / "part5_yfinance_sector_cache.csv"

# S&P500 benchmark. You may also use "SPY" if ^GSPC has download issues.
MARKET_TICKER = "^GSPC"

# Match frontend trailing windows: y1=12, y3=36, y5=60 monthly observations.
WINDOWS = {
    "y1": 12,
    "y3": 36,
    "y5": 60,
}

# Like your JS code: require at least 70% of months in the window.
MIN_PERIOD_RATIO = 0.70

# yfinance batch size. Reduce to 25 if your network is unstable.
BATCH_SIZE = 50

# Sleep between batches to be gentle to Yahoo Finance.
SLEEP_SECONDS = 1.0


# =========================
# 2. Helper functions
# =========================

def to_yahoo_ticker(ticker: str) -> str:
    """Convert CRSP-like ticker to Yahoo-compatible ticker."""
    if pd.isna(ticker):
        return ""
    ticker = str(ticker).strip().upper()
    # Yahoo usually uses BRK-B instead of BRK.B
    ticker = ticker.replace(".", "-")
    return ticker


def safe_date(value) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def chunk_list(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]


def extract_monthly_close(downloaded: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Extract monthly adjusted Close matrix from yfinance download result."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    # yfinance with auto_adjust=True usually returns Close, High, Low, Open, Volume.
    # MultiIndex columns: first level price field, second level ticker.
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            return pd.DataFrame()
        close = downloaded["Close"].copy()
    else:
        # Single ticker case.
        if "Close" not in downloaded.columns:
            return pd.DataFrame()
        close = downloaded[["Close"]].copy()
        close.columns = tickers[:1]

    # Ensure all requested tickers exist as columns.
    close = close.rename(columns={c: str(c).upper() for c in close.columns})
    return close


def normalize_monthly_price_index(close: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance monthly index and remove duplicate month labels.

    yfinance can sometimes return multiple rows that map to the same month-end
    after converting dates to Period("M"). If those duplicate index labels are
    not removed, pandas concat/reindex can fail later with:
    ValueError: cannot reindex on an axis with duplicate labels.
    """
    if close is None or close.empty:
        return pd.DataFrame()

    close = close.copy()
    close.index = pd.to_datetime(close.index, errors="coerce").to_period("M").to_timestamp("M")
    close = close[~close.index.isna()]
    close = close.sort_index()

    # Same month may appear more than once. Use the last available monthly close.
    close = close.groupby(level=0).last()

    # Duplicate columns can appear if the same ticker is requested twice.
    close = close.loc[:, ~close.columns.duplicated()]
    return close


def download_monthly_close(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Download monthly adjusted close for all tickers in batches."""
    all_close = []
    failures = []

    for idx, batch in enumerate(chunk_list(tickers, BATCH_SIZE), start=1):
        print(f"[INFO] Download batch {idx}: {len(batch)} tickers")
        try:
            data = yf.download(
                tickers=batch,
                start=start,
                end=end,
                interval="1mo",
                auto_adjust=True,
                group_by="column",
                progress=False,
                threads=True,
            )
            close = extract_monthly_close(data, batch)
            close = normalize_monthly_price_index(close)
            if close.empty:
                for ticker in batch:
                    failures.append({"yahoo_ticker": ticker, "reason": "empty_download_batch"})
            else:
                all_close.append(close)

                # Identify tickers with all missing values in this batch.
                present = set(close.columns)
                for ticker in batch:
                    if ticker not in present:
                        failures.append({"yahoo_ticker": ticker, "reason": "missing_column"})
                    elif close[ticker].dropna().empty:
                        failures.append({"yahoo_ticker": ticker, "reason": "all_missing_prices"})
        except Exception as exc:
            for ticker in batch:
                failures.append({"yahoo_ticker": ticker, "reason": f"download_error:{type(exc).__name__}:{exc}"})

        time.sleep(SLEEP_SECONDS)

    if not all_close:
        return pd.DataFrame(), failures

    # sort=False avoids future pandas default sorting changes and keeps concat deterministic.
    close_all = pd.concat(all_close, axis=1, sort=False)
    close_all = normalize_monthly_price_index(close_all)
    return close_all, failures


def compounded_return(monthly_returns: pd.Series) -> float:
    values = monthly_returns.dropna()
    if values.empty:
        return np.nan
    return float(np.prod(1.0 + values.values) - 1.0)


def annualize_compounded_return(period_return: float, months: int) -> float:
    if not np.isfinite(period_return) or months <= 0 or period_return <= -1:
        return np.nan
    return float((1.0 + period_return) ** (12.0 / months) - 1.0)


def dedupe_return_index(series: pd.Series) -> pd.Series:
    """Ensure a monthly return Series has unique month-end index labels."""
    series = series.copy()
    series.index = pd.to_datetime(series.index, errors="coerce").to_period("M").to_timestamp("M")
    series = series[~series.index.isna()].sort_index()
    return series.groupby(level=0).last()


def beta_from_returns(stock_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, int, float]:
    stock_returns = dedupe_return_index(stock_returns)
    market_returns = dedupe_return_index(market_returns)
    aligned = pd.concat([stock_returns.rename("stock"), market_returns.rename("market")], axis=1, join="inner").dropna()
    n = len(aligned)
    if n < 3:
        return np.nan, n, np.nan

    market_var = aligned["market"].var(ddof=1)
    if not np.isfinite(market_var) or market_var == 0:
        return np.nan, n, np.nan

    cov = aligned["stock"].cov(aligned["market"])
    beta = cov / market_var
    corr = aligned["stock"].corr(aligned["market"])
    return float(beta), int(n), float(corr) if np.isfinite(corr) else np.nan


def window_slice(returns: pd.Series, end_month: pd.Timestamp, months: int) -> pd.Series:
    """Return trailing monthly returns ending at end_month inclusive."""
    returns = dedupe_return_index(returns)
    end_month = pd.Timestamp(end_month).to_period("M").to_timestamp("M")
    start_month = (end_month.to_period("M") - (months - 1)).to_timestamp("M")
    return returns.loc[(returns.index >= start_month) & (returns.index <= end_month)]



def load_existing_sector_cache() -> Dict[str, Dict[str, str]]:
    """Load existing sector cache if available, so repeated runs do not re-query Yahoo."""
    if not OUTPUT_SECTOR_CACHE_CSV.exists():
        return {}
    try:
        cache_df = pd.read_csv(OUTPUT_SECTOR_CACHE_CSV, dtype=str).fillna("")
    except Exception:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for _, row in cache_df.iterrows():
        ticker = str(row.get("yahoo_ticker", "")).strip().upper()
        if ticker:
            out[ticker] = {
                "sector": str(row.get("sector", "")).strip(),
                "industry": str(row.get("industry", "")).strip(),
                "quote_type": str(row.get("quote_type", "")).strip(),
                "sector_source": str(row.get("sector_source", "cache")).strip() or "cache",
            }
    return out


def fetch_sector_for_ticker(yahoo_ticker: str) -> Dict[str, str]:
    """Fetch sector / industry metadata from Yahoo Finance via yfinance.

    Notes:
    - Works best for currently listed companies.
    - Many historical, delisted, renamed, or foreign tickers may return blank sector.
    - This is company metadata, not historical sector classification.
    """
    out = {
        "sector": "",
        "industry": "",
        "quote_type": "",
        "sector_source": "missing",
    }
    if not yahoo_ticker:
        return out

    try:
        ticker_obj = yf.Ticker(yahoo_ticker)
        try:
            info = ticker_obj.get_info()
        except Exception:
            info = ticker_obj.info

        if not isinstance(info, dict) or not info:
            out["sector_source"] = "empty_info"
            return out

        out["sector"] = str(info.get("sector") or "").strip()
        out["industry"] = str(info.get("industry") or "").strip()
        out["quote_type"] = str(info.get("quoteType") or info.get("quote_type") or "").strip()
        out["sector_source"] = "yfinance_info" if out["sector"] or out["industry"] else "no_sector_in_info"
        return out
    except Exception as exc:
        out["sector_source"] = f"sector_error:{type(exc).__name__}"
        return out


def build_sector_map(tickers: List[str]) -> Dict[str, Dict[str, str]]:
    """Build ticker -> sector metadata map and save a cache CSV."""
    sector_map = load_existing_sector_cache()
    rows = []

    for idx, ticker in enumerate(tickers, start=1):
        ticker = str(ticker).strip().upper()
        if not ticker:
            continue

        if ticker not in sector_map:
            print(f"[INFO] Sector {idx:,}/{len(tickers):,}: {ticker}")
            sector_map[ticker] = fetch_sector_for_ticker(ticker)
            time.sleep(0.05)

        row = {"yahoo_ticker": ticker}
        row.update(sector_map[ticker])
        rows.append(row)

    cache_df = pd.DataFrame(rows).drop_duplicates(subset=["yahoo_ticker"])
    cache_df.to_csv(OUTPUT_SECTOR_CACHE_CSV, index=False, encoding="utf-8-sig")
    return sector_map


# =========================
# 3. Main
# =========================

def main() -> None:
    if not INPUT_STOCK_UNIVERSE.exists():
        raise FileNotFoundError(f"Input stock universe not found: {INPUT_STOCK_UNIVERSE}")

    stock_df = pd.read_csv(INPUT_STOCK_UNIVERSE, dtype=str)
    required = {"holding_ticker", "primary_security_name", "first_report_dt", "last_report_dt"}
    missing = required - set(stock_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    stock_df["yahoo_ticker"] = stock_df["holding_ticker"].map(to_yahoo_ticker)
    stock_df["first_report_dt_parsed"] = stock_df["first_report_dt"].map(safe_date)
    stock_df["last_report_dt_parsed"] = stock_df["last_report_dt"].map(safe_date)
    stock_df = stock_df.dropna(subset=["first_report_dt_parsed", "last_report_dt_parsed"])
    stock_df = stock_df[stock_df["yahoo_ticker"].astype(str).str.len() > 0].copy()

    if stock_df.empty:
        raise ValueError("No valid tickers found in input stock universe.")

    # Need 5-year buffer before earliest first_report_dt to calculate trailing y5 at first year.
    earliest = stock_df["first_report_dt_parsed"].min()
    latest = stock_df["last_report_dt_parsed"].max()
    download_start = (earliest - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    download_end = (latest + pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    tickers = sorted(stock_df["yahoo_ticker"].unique().tolist())
    all_tickers_for_download = sorted(set(tickers + [MARKET_TICKER]))

    print(f"[INFO] Input unique tickers: {len(tickers):,}")
    print(f"[INFO] Download date range: {download_start} to {download_end}")
    print(f"[INFO] Market benchmark: {MARKET_TICKER}")

    # Sector / industry metadata from yfinance. This is cached separately because it is slower
    # and many historical or delisted tickers may not have sector metadata.
    sector_map = build_sector_map(tickers)

    close, failures = download_monthly_close(all_tickers_for_download, download_start, download_end)
    if close.empty:
        raise RuntimeError("No price data downloaded. Check internet connection or yfinance installation.")

    # Save raw monthly close cache for inspection/reuse.
    close.to_csv(OUTPUT_PRICE_CACHE_CSV, encoding="utf-8-sig")

    market_col = MARKET_TICKER.upper()
    if market_col not in close.columns:
        # yfinance sometimes preserves ^GSPC instead of uppercase logic already uppercase no issue.
        possible_cols = [c for c in close.columns if str(c).upper() == market_col]
        if possible_cols:
            market_col = possible_cols[0]
        else:
            raise RuntimeError(f"Market benchmark {MARKET_TICKER} was not downloaded successfully.")

    # Monthly return matrix. Index should already be unique after normalization,
    # but normalize again defensively before looping through 1,259 tickers.
    monthly_returns = close.pct_change(fill_method=None)
    monthly_returns.index = pd.to_datetime(monthly_returns.index).to_period("M").to_timestamp("M")
    monthly_returns = monthly_returns.groupby(level=0).last().sort_index()
    market_returns = dedupe_return_index(monthly_returns[market_col])

    output_rows = []

    for _, meta in stock_df.iterrows():
        holding_ticker = str(meta["holding_ticker"]).strip().upper()
        yahoo_ticker = str(meta["yahoo_ticker"]).strip().upper()
        company_name = meta.get("primary_security_name", "")
        first_dt = meta["first_report_dt_parsed"]
        last_dt = meta["last_report_dt_parsed"]
        start_year = int(first_dt.year)
        end_year = int(last_dt.year)

        if yahoo_ticker not in monthly_returns.columns:
            failures.append({"yahoo_ticker": yahoo_ticker, "reason": "not_in_monthly_return_matrix"})
            continue

        stock_returns = dedupe_return_index(monthly_returns[yahoo_ticker])

        for year in range(start_year, end_year + 1):
            end_month = pd.Timestamp(year=year, month=12, day=31)
            row = {
                "holding_ticker": holding_ticker,
                "yahoo_ticker": yahoo_ticker,
                "primary_security_name": company_name,
                "sector": sector_map.get(yahoo_ticker, {}).get("sector", ""),
                "industry": sector_map.get(yahoo_ticker, {}).get("industry", ""),
                "quote_type": sector_map.get(yahoo_ticker, {}).get("quote_type", ""),
                "sector_source": sector_map.get(yahoo_ticker, {}).get("sector_source", ""),
                "year": year,
                "first_report_dt": first_dt.date().isoformat(),
                "last_report_dt": last_dt.date().isoformat(),
                "beta_end_month": end_month.date().isoformat(),
            }

            for label, months in WINDOWS.items():
                min_periods = math.ceil(months * MIN_PERIOD_RATIO)
                s_win = window_slice(stock_returns, end_month, months)
                m_win = window_slice(market_returns, end_month, months)
                # Rename before concat and force unique monthly labels to avoid
                # ValueError: cannot reindex on an axis with duplicate labels.
                s_win = dedupe_return_index(s_win)
                m_win = dedupe_return_index(m_win)
                aligned = pd.concat([s_win.rename("stock"), m_win.rename("market")], axis=1, join="inner").dropna()
                n = len(aligned)

                if n >= min_periods:
                    beta, beta_n, corr = beta_from_returns(aligned["stock"], aligned["market"])
                    stock_period_ret = compounded_return(aligned["stock"])
                    market_period_ret = compounded_return(aligned["market"])
                    stock_ann_ret = annualize_compounded_return(stock_period_ret, n)
                    market_ann_ret = annualize_compounded_return(market_period_ret, n)
                else:
                    beta, beta_n, corr = np.nan, n, np.nan
                    stock_period_ret = np.nan
                    market_period_ret = np.nan
                    stock_ann_ret = np.nan
                    market_ann_ret = np.nan

                row[f"beta_{label}"] = beta
                row[f"corr_sp500_{label}"] = corr
                row[f"n_months_{label}"] = beta_n
                row[f"stock_return_{label}"] = stock_period_ret
                row[f"sp500_return_{label}"] = market_period_ret
                row[f"stock_annual_return_{label}"] = stock_ann_ret
                row[f"sp500_annual_return_{label}"] = market_ann_ret

            output_rows.append(row)

    result = pd.DataFrame(output_rows)
    if not result.empty:
        result = result.sort_values(["holding_ticker", "year"]).reset_index(drop=True)

    result.to_csv(OUTPUT_BETA_CSV, index=False, encoding="utf-8-sig")

    fail_df = pd.DataFrame(failures).drop_duplicates() if failures else pd.DataFrame(columns=["yahoo_ticker", "reason"])
    fail_df.to_csv(OUTPUT_FAILED_CSV, index=False, encoding="utf-8-sig")

    print("\n[DONE]")
    print(f"Output rows: {len(result):,}")
    print(f"Output beta CSV: {OUTPUT_BETA_CSV}")
    print(f"Failed ticker CSV: {OUTPUT_FAILED_CSV}")
    print(f"Monthly close cache: {OUTPUT_PRICE_CACHE_CSV}")
    print(f"Sector cache: {OUTPUT_SECTOR_CACHE_CSV}")


if __name__ == "__main__":
    main()
