"""Zero-look-ahead temporal alignment and verification assertions (Phase 1 Section 1.8)."""

from __future__ import annotations

import bisect
import logging
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingest.schemas import (
    BARS_PARQUET_PATH,
    EVENTS_ALIGNED_SCHEMA,
    EVENTS_PARQUET_PATH,
    ExchangeCode,
    TickerConfig,
    UniverseConfig,
)

logger = logging.getLogger(__name__)


def _classify_bucket(
    effective_utc: datetime,
    bar_open_utc: datetime,
    timestamp_precision: str,
    ex_tz: ZoneInfo,
    lunch_bounds: tuple[time, time] | None,
) -> str:
    if timestamp_precision == "date_only":
        return "date_only_eod"
    if effective_utc >= bar_open_utc:
        return "intraday"
    if lunch_bounds is not None:
        ev_local = effective_utc.astimezone(ex_tz)
        bar_local = bar_open_utc.astimezone(ex_tz)
        if ev_local.date() == bar_local.date() and lunch_bounds[0] <= ev_local.time() < lunch_bounds[1]:
            return "lunch_break"
    return "overnight_premarket"


def align_events_to_bars(
    events: list[dict[str, Any]],
    bars_df: pd.DataFrame,
    universe: UniverseConfig,
) -> pd.DataFrame:
    """Strictly align normalized events to the earliest bar t with bar_close_utc >= effective_event_utc.

    Event assigned to bar t informs prediction for bar t+1 ONLY (zero look-ahead).
    """
    # Pre-index sorted bar arrays per ticker for O(log N) binary search
    bar_index_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker, grp in bars_df.groupby("ticker"):
        g = grp.sort_values("bar_close_utc").reset_index(drop=True)
        close_ts = pd.to_datetime(g["bar_close_utc"], utc=True)
        open_ts = pd.to_datetime(g["bar_open_utc"], utc=True)
        bar_index_by_ticker[str(ticker)] = {
            "close_py": [ts.to_pydatetime() for ts in close_ts],
            "open_py": [ts.to_pydatetime() for ts in open_ts],
            "window_role": g["window_role"].astype(str).tolist(),
        }

    # Precompute timezone & lunch break objects per exchange
    tz_by_ex: dict[ExchangeCode, ZoneInfo] = {}
    lunch_by_ex: dict[ExchangeCode, tuple[time, time] | None] = {}
    for ex_code, ex_cfg in universe.exchanges.items():
        tz_by_ex[ex_code] = ZoneInfo(ex_cfg.timezone)
        if ex_cfg.lunch_break_local and len(ex_cfg.lunch_break_local) == 2:
            sh, sm = ex_cfg.lunch_break_local[0].split(":")
            eh, em = ex_cfg.lunch_break_local[1].split(":")
            lunch_by_ex[ex_code] = (time(int(sh), int(sm)), time(int(eh), int(em)))
        else:
            lunch_by_ex[ex_code] = None

    aligned_rows: list[dict[str, Any]] = []
    for ev in events:
        ticker = ev["ticker"]
        binfo = bar_index_by_ticker.get(ticker)
        if binfo is None:
            continue

        eff_utc: datetime = ev["effective_event_utc"]
        close_py: list[datetime] = binfo["close_py"]

        # Earliest bar t with bar_close_utc >= effective_event_utc (exact UTC datetime binary search)
        idx = bisect.bisect_left(close_py, eff_utc)

        row = dict(ev)
        if idx < len(close_py):
            assigned_close: datetime = binfo["close_py"][idx]
            assigned_open: datetime = binfo["open_py"][idx]
            lag_min = (assigned_close - eff_utc).total_seconds() / 60.0
            ex_code: ExchangeCode = ev["exchange"]
            row["assigned_bar_close_utc"] = assigned_close
            row["lag_minutes_to_bar_close"] = float(lag_min)
            row["alignment_bucket"] = _classify_bucket(
                effective_utc=eff_utc,
                bar_open_utc=assigned_open,
                timestamp_precision=ev["timestamp_precision"],
                ex_tz=tz_by_ex[ex_code],
                lunch_bounds=lunch_by_ex[ex_code],
            )
            row["window_role"] = binfo["window_role"][idx]
        else:
            row["assigned_bar_close_utc"] = None
            row["lag_minutes_to_bar_close"] = None
            row["alignment_bucket"] = "post_window_live"
            row["window_role"] = "post_window_live"

        aligned_rows.append(row)

    if not aligned_rows:
        empty_cols = {name: [] for name in EVENTS_ALIGNED_SCHEMA.names}
        return pd.DataFrame(empty_cols)

    df = pd.DataFrame(aligned_rows)
    df = df.sort_values(["exchange", "ticker", "effective_event_utc", "event_id"]).reset_index(drop=True)
    return df


def verify_phase1_alignment(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
    active_tickers: list[TickerConfig],
    universe: UniverseConfig,
) -> dict[str, Any]:
    """Execute all 5 mandatory Phase 1 alignment verification assertions (Section 1.8)."""
    # Assertion 1: Bar boundaries non-null
    assert bars_df["bar_open_utc"].notna().all() and bars_df["bar_close_utc"].notna().all(), (
        "Assertion 1 failed: null bar_open_utc or bar_close_utc detected."
    )

    # Assertion 2: Strictly positive bar duration
    assert (bars_df["bar_close_utc"] > bars_df["bar_open_utc"]).all(), (
        "Assertion 2 failed: bar_close_utc <= bar_open_utc detected."
    )

    # Assertions 3 & 4: Zero look-ahead & non-negative lag for all historical aligned events
    aligned_hist = events_df[events_df["assigned_bar_close_utc"].notna()].copy()
    if len(aligned_hist) > 0:
        eff_ts = pd.to_datetime(aligned_hist["effective_event_utc"], utc=True)
        asgn_ts = pd.to_datetime(aligned_hist["assigned_bar_close_utc"], utc=True)
        assert (eff_ts <= asgn_ts).all(), (
            "Assertion 3 FAILED (LOOK-AHEAD VIOLATION): effective_event_utc > assigned_bar_close_utc!"
        )
        assert (aligned_hist["lag_minutes_to_bar_close"] >= 0.0).all(), (
            "Assertion 4 FAILED: negative lag_minutes_to_bar_close detected!"
        )

    # Assertion 5: Per-ticker completeness (exact 4 warmup + 10 active_10d trading days, >=2 social sources, >=1 news)
    summary_per_ticker: dict[str, Any] = {}
    failures: list[str] = []

    for tcfg in active_tickers:
        ex_cfg = universe.exchanges[tcfg.exchange]
        t_bars = bars_df[bars_df["ticker"] == tcfg.ticker]
        n_warmup_days = int(t_bars[t_bars["window_role"] == "warmup"]["trading_date_local"].nunique())
        n_active_days = int(t_bars[t_bars["window_role"] == "active_10d"]["trading_date_local"].nunique())

        t_events = events_df[events_df["ticker"] == tcfg.ticker]
        t_social = t_events[t_events["event_type"] == "social"]
        t_news = t_events[t_events["event_type"] == "news"]
        social_sources = sorted(t_social["source"].unique().tolist())

        summary_per_ticker[tcfg.ticker] = {
            "exchange": tcfg.exchange,
            "bars": int(len(t_bars)),
            "warmup_days": n_warmup_days,
            "active_days": n_active_days,
            "social_events": int(len(t_social)),
            "social_sources": social_sources,
            "news_events": int(len(t_news)),
        }

        if n_warmup_days != ex_cfg.warmup_days:
            failures.append(f"{tcfg.ticker}: expected {ex_cfg.warmup_days} warmup days, got {n_warmup_days}")
        if n_active_days != ex_cfg.active_days:
            failures.append(f"{tcfg.ticker}: expected {ex_cfg.active_days} active_10d days, got {n_active_days}")
        if len(social_sources) < 2:
            failures.append(
                f"{tcfg.ticker}: expected >= 2 distinct social sources, got {len(social_sources)} ({social_sources})"
            )
        if len(t_news) < 1:
            failures.append(f"{tcfg.ticker}: expected >= 1 news article, got 0")

    if failures:
        raise AssertionError("Assertion 5 failed:\n  - " + "\n  - ".join(failures))

    logger.info(
        "[align] All 5 mandatory assertions PASSED across %d tickers (%d bars, %d aligned events)",
        len(active_tickers),
        len(bars_df),
        len(events_df),
    )
    return {
        "assertions_passed": True,
        "total_bars": int(len(bars_df)),
        "total_events": int(len(events_df)),
        "historical_aligned_events": int(len(aligned_hist)),
        "per_ticker": summary_per_ticker,
    }


def write_events_parquet(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Write events_aligned DataFrame strictly matching EVENTS_ALIGNED_SCHEMA with ZSTD compression."""
    target = out_path or EVENTS_PARQUET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    for ts_col in ["event_utc", "effective_event_utc", "assigned_bar_close_utc"]:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.as_unit("us")

    # Enforce nullable integer/bool/float32 column types cleanly for PyArrow
    df["author_followers"] = pd.array(df["author_followers"], dtype="Int64")
    df["is_holder_verified"] = pd.array(df["is_holder_verified"], dtype="boolean")
    df["source_credibility_weight"] = df["source_credibility_weight"].astype("float32")

    table = pa.Table.from_pandas(df[EVENTS_ALIGNED_SCHEMA.names], schema=EVENTS_ALIGNED_SCHEMA, preserve_index=False)
    pq.write_table(table, target, compression="zstd")
    return target
