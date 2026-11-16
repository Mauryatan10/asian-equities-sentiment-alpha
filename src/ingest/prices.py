"""1-Hour OHLCV price ingestion via yfinance with exact HKEX/KRX session boundaries (Phase 1 Sections 1.2, 1.7, 1.8)."""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ingest.manifest import (
    ManifestStore,
    make_cache_key,
    read_jsonl,
    utc_now_iso,
    write_jsonl_atomic,
)
from src.ingest.schemas import (
    BARS_1H_SCHEMA,
    BARS_PARQUET_PATH,
    RAW_DIR,
    ExchangeCode,
    ExchangeConfig,
    ExchangeWindowBounds,
    ManifestEntry,
    RunMode,
    TickerConfig,
    UniverseConfig,
)

logger = logging.getLogger(__name__)


def _parse_hhmm(hhmm: str) -> time:
    hh, mm = hhmm.split(":")
    return time(int(hh), int(mm))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _download_yf_60m(ticker: str, start_date: date, end_date_exclusive: date) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(
        start=start_date.isoformat(),
        end=end_date_exclusive.isoformat(),
        interval="60m",
        auto_adjust=False,
        prepost=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty 60m DataFrame for {ticker}")
    return df


def _raw_df_to_records(df: pd.DataFrame, ticker: str, exchange: ExchangeCode) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        records.append(
            {
                "ticker": ticker,
                "exchange": exchange,
                "raw_timestamp": ts.isoformat(),
                "open": None if pd.isna(row["Open"]) else float(row["Open"]),
                "high": None if pd.isna(row["High"]) else float(row["High"]),
                "low": None if pd.isna(row["Low"]) else float(row["Low"]),
                "close": None if pd.isna(row["Close"]) else float(row["Close"]),
                "volume": 0 if pd.isna(row["Volume"]) else int(row["Volume"]),
            }
        )
    return records


def _map_raw_records_to_bars(
    raw_records: list[dict[str, Any]],
    ticker_cfg: TickerConfig,
    ex_cfg: ExchangeConfig,
    run_utc: datetime,
    end_date_limit: date | None,
) -> pd.DataFrame:
    """Convert raw yfinance 60m records into exchange-aware UTC bar boundaries."""
    tz = ZoneInfo(ex_cfg.timezone)
    rows: list[dict[str, Any]] = []

    for rec in raw_records:
        raw_dt = datetime.fromisoformat(rec["raw_timestamp"]).astimezone(tz)
        local_date = raw_dt.date()
        if end_date_limit is not None and local_date > end_date_limit:
            continue

        hhmm = raw_dt.strftime("%H:%M")
        slot = ex_cfg.bar_schedule_local.get(hhmm)
        if slot is None:
            raise RuntimeError(
                f"Unmapped 60m bar open time '{hhmm}' for {ticker_cfg.ticker} ({ex_cfg.timezone}). "
                f"Allowed slots: {sorted(ex_cfg.bar_schedule_local.keys())}"
            )

        open_local = datetime.combine(local_date, _parse_hhmm(slot.actual_open), tzinfo=tz)
        close_local = datetime.combine(local_date, _parse_hhmm(slot.close), tzinfo=tz)
        bar_open_utc = open_local.astimezone(timezone.utc)
        bar_close_utc = close_local.astimezone(timezone.utc)

        # Completed-bar rule: strictly drop any live/in-progress bar
        if bar_close_utc > run_utc:
            continue

        rows.append(
            {
                "ticker": ticker_cfg.ticker,
                "exchange": ticker_cfg.exchange,
                "trading_date_local": local_date,
                "bar_open_utc": bar_open_utc,
                "bar_close_utc": bar_close_utc,
                "bar_duration_min": int(slot.duration_min),
                "session_segment": slot.segment,
                "open": rec["open"],
                "high": rec["high"],
                "low": rec["low"],
                "close": rec["close"],
                "volume": int(rec["volume"]),
            }
        )

    if not rows:
        raise RuntimeError(f"Zero completed 60m bars available for {ticker_cfg.ticker} up to {run_utc.isoformat()}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ticker", "bar_open_utc"]).sort_values("bar_open_utc").reset_index(drop=True)
    return df


def _apply_quality_checks_and_targets(
    df: pd.DataFrame,
    ticker_cfg: TickerConfig,
    ex_cfg: ExchangeConfig,
    warmup_dates: set[date],
    active_dates: set[date],
) -> pd.DataFrame:
    """Slice to exact 14 trading days, apply quality flags, verify completeness, and compute next-bar targets."""
    keep_dates = warmup_dates | active_dates
    df = df[df["trading_date_local"].isin(keep_dates)].copy()
    df = df.sort_values("bar_close_utc").reset_index(drop=True)

    distinct_dates = sorted(df["trading_date_local"].unique())
    if len(distinct_dates) < ex_cfg.total_trading_days:
        raise RuntimeError(
            f"Ticker {ticker_cfg.ticker} has only {len(distinct_dates)} trading days in 14-day window "
            f"(required {ex_cfg.total_trading_days})."
        )

    expected_bars = ex_cfg.total_trading_days * ex_cfg.expected_bars_per_day
    missing_ratio = max(0.0, (expected_bars - len(df)) / float(expected_bars))
    if missing_ratio > 0.05:
        raise RuntimeError(
            f"Ticker {ticker_cfg.ticker} missing {missing_ratio * 100:.2f}% of expected bars "
            f"({len(df)}/{expected_bars}, max allowed missing 5.0%)."
        )

    # Flag zero-volume bars (kept intact)
    df["is_zero_volume"] = df["volume"].fillna(0).astype(int) == 0

    # Forward-fill lone NaN price cells only
    price_cols = ["open", "high", "low", "close"]
    nan_mask = df[price_cols].isna().any(axis=1)
    df["is_imputed"] = nan_mask.astype(bool)
    if nan_mask.any():
        # Verify no consecutive multi-bar NaN runs and no leading NaNs
        for col in price_cols:
            is_na = df[col].isna()
            if is_na.iloc[0]:
                raise RuntimeError(f"Leading NaN in {col} for {ticker_cfg.ticker}; cannot forward-fill.")
            if (is_na & is_na.shift(1, fill_value=False)).any():
                raise RuntimeError(f"Consecutive NaNs in {col} for {ticker_cfg.ticker}; only lone NaN cells allowed.")
        df[price_cols] = df[price_cols].ffill()

    df["window_role"] = df["trading_date_local"].map(
        lambda d: "warmup" if d in warmup_dates else "active_10d"
    )

    # Next-bar target columns (strictly within same ticker, nullable on final bar)
    df["next_bar_close_utc"] = df["bar_close_utc"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df["target_log_return_1h"] = np.where(
        df["next_close"].notna() & (df["close"] > 0) & (df["next_close"] > 0),
        np.log(df["next_close"].astype(float) / df["close"].astype(float)),
        np.nan,
    )

    return df


def write_bars_parquet(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Write bars_1h DataFrame strictly matching BARS_1H_SCHEMA with ZSTD compression."""
    target = out_path or BARS_PARQUET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["exchange", "ticker", "bar_close_utc"]).reset_index(drop=True)

    # Ensure exact timestamp precision (us, UTC)
    for ts_col in ["bar_open_utc", "bar_close_utc", "next_bar_close_utc"]:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.as_unit("us")

    table = pa.Table.from_pandas(df[BARS_1H_SCHEMA.names], schema=BARS_1H_SCHEMA, preserve_index=False)
    pq.write_table(table, target, compression="zstd")
    return target


def ingest_prices(
    universe: UniverseConfig,
    tickers: list[TickerConfig],
    manifest_store: ManifestStore,
    end_date_str: str | None = None,
    mode: RunMode = "full",
    force_refresh: bool = False,
    run_utc: datetime | None = None,
) -> tuple[pd.DataFrame, dict[ExchangeCode, ExchangeWindowBounds]]:
    """Fetch 1h OHLCV bars for selected tickers, build independent per-exchange 14-day calendars, and write bars_1h.parquet."""
    now_utc = run_utc or datetime.now(timezone.utc)
    end_date_limit = date.fromisoformat(end_date_str) if end_date_str else None
    cache_end_date = end_date_str or now_utc.strftime("%Y-%m-%d")

    raw_dfs_by_exchange: dict[ExchangeCode, list[tuple[TickerConfig, pd.DataFrame]]] = {}

    for tcfg in tickers:
        ex_cfg = universe.exchanges[tcfg.exchange]
        raw_path = RAW_DIR / "prices" / tcfg.exchange / f"{tcfg.ticker}_{cache_end_date}.jsonl"
        ckey = make_cache_key("prices", "yfinance", tcfg.exchange, tcfg.ticker, cache_end_date, mode)

        if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
            logger.info("[prices] Cache HIT for %s (%s)", tcfg.ticker, raw_path.name)
            raw_records = read_jsonl(raw_path)
        else:
            anchor_date = end_date_limit or now_utc.astimezone(ZoneInfo(ex_cfg.timezone)).date()
            start_d = anchor_date - timedelta(days=ex_cfg.fetch_calendar_buffer_days)
            end_d_excl = anchor_date + timedelta(days=2)  # generous +2d to cover TZ edge; filtered strictly below
            try:
                logger.info("[prices] Fetching yfinance 60m for %s (%s to %s)", tcfg.ticker, start_d, anchor_date)
                raw_df = _download_yf_60m(tcfg.ticker, start_d, end_d_excl)
                raw_records = _raw_df_to_records(raw_df, tcfg.ticker, tcfg.exchange)
                raw_count = write_jsonl_atomic(raw_path, raw_records)
                manifest_store.record(
                    ManifestEntry(
                        cache_key=ckey,
                        stage="prices",
                        source="yfinance",
                        exchange=tcfg.exchange,
                        key=tcfg.ticker,
                        end_date=cache_end_date,
                        mode=mode,
                        status="ok",
                        raw_path=str(raw_path),
                        raw_count=raw_count,
                        kept_count=raw_count,
                        est_cost_usd=0.0,
                        updated_at_utc=utc_now_iso(),
                    )
                )
            except Exception as exc:
                manifest_store.record(
                    ManifestEntry(
                        cache_key=ckey,
                        stage="prices",
                        source="yfinance",
                        exchange=tcfg.exchange,
                        key=tcfg.ticker,
                        end_date=cache_end_date,
                        mode=mode,
                        status="failed",
                        raw_path=str(raw_path),
                        est_cost_usd=0.0,
                        updated_at_utc=utc_now_iso(),
                        error_detail=str(exc),
                    )
                )
                raise

        mapped_df = _map_raw_records_to_bars(raw_records, tcfg, ex_cfg, now_utc, end_date_limit)
        raw_dfs_by_exchange.setdefault(tcfg.exchange, []).append((tcfg, mapped_df))

    # Derive independent 14-trading-day calendar per exchange
    all_ticker_dfs: list[pd.DataFrame] = []
    exchange_windows: dict[ExchangeCode, ExchangeWindowBounds] = {}

    for ex_code, items in raw_dfs_by_exchange.items():
        ex_cfg = universe.exchanges[ex_code]
        # Exchange trading dates = sorted union of completed local dates across its tickers
        union_dates = sorted({d for _, df in items for d in df["trading_date_local"].unique()})
        if len(union_dates) < ex_cfg.total_trading_days:
            raise RuntimeError(
                f"Exchange {ex_code} only has {len(union_dates)} completed trading dates in fetched buffer "
                f"(required {ex_cfg.total_trading_days})."
            )

        selected_14 = union_dates[-ex_cfg.total_trading_days :]
        warmup_list = selected_14[: ex_cfg.warmup_days]
        active_list = selected_14[ex_cfg.warmup_days :]
        warmup_set = set(warmup_list)
        active_set = set(active_list)

        ex_dfs: list[pd.DataFrame] = []
        for tcfg, mapped_df in items:
            clean_df = _apply_quality_checks_and_targets(mapped_df, tcfg, ex_cfg, warmup_set, active_set)
            ex_dfs.append(clean_df)
            all_ticker_dfs.append(clean_df)

        ex_combined = pd.concat(ex_dfs, ignore_index=True)
        first_open_utc = ex_combined["bar_open_utc"].min().to_pydatetime()
        if first_open_utc.tzinfo is None:
            first_open_utc = first_open_utc.replace(tzinfo=timezone.utc)
        window_end_utc = ex_combined["bar_close_utc"].max().to_pydatetime()
        if window_end_utc.tzinfo is None:
            window_end_utc = window_end_utc.replace(tzinfo=timezone.utc)
        warmup_lead_utc = first_open_utc - timedelta(hours=ex_cfg.warmup_lead_hours)

        # Map each local trading date -> session_close_utc (max bar_close_utc on that day)
        session_close_map: dict[str, datetime] = {}
        for d_local, grp in ex_combined.groupby("trading_date_local"):
            dt_close = grp["bar_close_utc"].max().to_pydatetime()
            if dt_close.tzinfo is None:
                dt_close = dt_close.replace(tzinfo=timezone.utc)
            session_close_map[d_local.isoformat()] = dt_close

        bounds = ExchangeWindowBounds(
            exchange=ex_code,
            timezone=ex_cfg.timezone,
            trading_dates_local=selected_14,
            warmup_dates_local=warmup_list,
            active_dates_local=active_list,
            first_warmup_bar_open_utc=first_open_utc,
            exchange_warmup_lead_utc=warmup_lead_utc,
            exchange_window_end_utc=window_end_utc,
            session_close_by_date_utc=session_close_map,
        )
        exchange_windows[ex_code] = bounds
        manifest_store.set_exchange_window(ex_code, bounds.model_dump(mode="json"))

        logger.info(
            "[prices] %s 14-day window: warmup=%s..%s (%dd) | active=%s..%s (%dd) | lead_utc=%s | end_utc=%s",
            ex_code,
            warmup_list[0],
            warmup_list[-1],
            len(warmup_list),
            active_list[0],
            active_list[-1],
            len(active_list),
            warmup_lead_utc.isoformat(),
            window_end_utc.isoformat(),
        )

    final_bars = pd.concat(all_ticker_dfs, ignore_index=True)

    # Sanity assertions (Section 1.8 #1 and #2)
    assert final_bars["bar_open_utc"].notna().all() and final_bars["bar_close_utc"].notna().all()
    assert (final_bars["bar_close_utc"] > final_bars["bar_open_utc"]).all()

    out_path = write_bars_parquet(final_bars)
    logger.info("[prices] Wrote %d bars across %d tickers to %s", len(final_bars), len(tickers), out_path)
    return final_bars, exchange_windows


def load_exchange_windows_from_bars(bars_path: Path | None = None) -> dict[ExchangeCode, ExchangeWindowBounds]:
    """Reconstruct ExchangeWindowBounds deterministically from bars_1h.parquet."""
    path = bars_path or BARS_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"bars_1h.parquet not found at {path}; run --stage prices first.")

    df = pd.read_parquet(path)
    from src.ingest.schemas import load_universe

    universe = load_universe()
    windows: dict[ExchangeCode, ExchangeWindowBounds] = {}
    for ex_code, grp in df.groupby("exchange"):
        ex_typed: ExchangeCode = ex_code  # type: ignore[assignment]
        ex_cfg = universe.exchanges[ex_typed]
        all_dates = sorted(grp["trading_date_local"].unique())
        warmup_dates = sorted(grp[grp["window_role"] == "warmup"]["trading_date_local"].unique())
        active_dates = sorted(grp[grp["window_role"] == "active_10d"]["trading_date_local"].unique())
        first_open = pd.Timestamp(grp["bar_open_utc"].min()).tz_convert("UTC").to_pydatetime()
        end_utc = pd.Timestamp(grp["bar_close_utc"].max()).tz_convert("UTC").to_pydatetime()
        lead_utc = first_open - timedelta(hours=ex_cfg.warmup_lead_hours)
        close_map = {
            d.isoformat(): pd.Timestamp(sub["bar_close_utc"].max()).tz_convert("UTC").to_pydatetime()
            for d, sub in grp.groupby("trading_date_local")
        }
        windows[ex_typed] = ExchangeWindowBounds(
            exchange=ex_typed,
            timezone=ex_cfg.timezone,
            trading_dates_local=list(all_dates),
            warmup_dates_local=list(warmup_dates),
            active_dates_local=list(active_dates),
            first_warmup_bar_open_utc=first_open,
            exchange_warmup_lead_utc=lead_utc,
            exchange_window_end_utc=end_utc,
            session_close_by_date_utc=close_map,
        )
    return windows
