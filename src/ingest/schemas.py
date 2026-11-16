"""Pydantic models for configuration, raw ingestion records, manifest entries, and canonical PyArrow Parquet schemas."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
MANIFEST_PATH = MANIFESTS_DIR / "phase1_manifest.json"
BARS_PARQUET_PATH = PROCESSED_DIR / "bars_1h.parquet"
EVENTS_PARQUET_PATH = PROCESSED_DIR / "events_aligned.parquet"

ExchangeCode = Literal["HKEX", "KRX"]
RunMode = Literal["full", "smoke"]
StageName = Literal["preflight", "prices", "social", "news", "align"]


# =============================================================================
# Configuration Models (config/universe.yaml & config/sources.yaml)
# =============================================================================


class BarScheduleSlot(BaseModel):
    actual_open: str
    close: str
    duration_min: int
    segment: Literal["morning", "afternoon", "continuous"]


class ExchangeConfig(BaseModel):
    timezone: str
    utc_offset_hours: int
    currency: str
    warmup_days: int = 4
    active_days: int = 10
    total_trading_days: int = 14
    fetch_calendar_buffer_days: int = 28
    warmup_lead_hours: int = 18
    lunch_break_local: list[str] | None = None
    session_open_local: str
    session_close_local: str
    expected_bars_per_day: int
    bar_schedule_local: dict[str, BarScheduleSlot]


class TickerConfig(BaseModel):
    ticker: str
    exchange: ExchangeCode
    exchange_code: str
    code_short: str
    company_en: str
    company_local: str
    sector: str
    enabled: bool = True
    role: Literal["primary", "standby"] = "primary"
    cashtags: list[str] = Field(default_factory=list)
    ascii_aliases: list[str] = Field(default_factory=list)
    cjk_aliases: list[str] = Field(default_factory=list)
    telegram_q_terms: list[str] = Field(default_factory=list)
    lihkg_keywords: list[str] = Field(default_factory=list)


class UniverseConfig(BaseModel):
    exchanges: dict[ExchangeCode, ExchangeConfig]
    tickers: list[TickerConfig]

    def get_tickers(
        self,
        exchange: str = "all",
        explicit_tickers: list[str] | None = None,
        smoke_test: bool = False,
        smoke_tickers_by_exchange: dict[str, list[str]] | None = None,
    ) -> list[TickerConfig]:
        ex_norm = exchange.upper()
        if explicit_tickers:
            wanted = {t.strip().upper() for t in explicit_tickers if t.strip()}
            matched = [t for t in self.tickers if t.ticker.upper() in wanted]
            missing = wanted - {t.ticker.upper() for t in matched}
            if missing:
                raise ValueError(f"Unknown ticker(s) not in config/universe.yaml: {sorted(missing)}")
            if ex_norm != "ALL":
                matched = [t for t in matched if t.exchange == ex_norm]
            return matched

        active = [t for t in self.tickers if t.enabled]
        if ex_norm != "ALL":
            active = [t for t in active if t.exchange == ex_norm]

        if smoke_test and smoke_tickers_by_exchange:
            allowed: set[str] = set()
            for ex_k, t_list in smoke_tickers_by_exchange.items():
                if ex_norm == "ALL" or ex_k.upper() == ex_norm:
                    allowed.update(x.upper() for x in t_list)
            active = [t for t in active if t.ticker.upper() in allowed]

        return active


class TelegramChannelConfig(BaseModel):
    exchange: ExchangeCode
    handle: str
    name: str
    reach_label: str
    mode: Literal["search_q", "full_feed"]
    max_pages: int
    enabled: bool = True


class ExchangeWindowBounds(BaseModel):
    """Per-exchange independent 14-trading-day calendar bounds derived from yfinance 60m bars."""

    exchange: ExchangeCode
    timezone: str
    trading_dates_local: list[date]
    warmup_dates_local: list[date]
    active_dates_local: list[date]
    first_warmup_bar_open_utc: datetime
    exchange_warmup_lead_utc: datetime
    exchange_window_end_utc: datetime
    session_close_by_date_utc: dict[str, datetime]  # YYYY-MM-DD -> UTC datetime


# =============================================================================
# Manifest Schema (data/manifests/phase1_manifest.json)
# =============================================================================


class ManifestEntry(BaseModel):
    cache_key: str
    stage: str
    source: str
    exchange: str
    key: str
    end_date: str
    mode: RunMode
    status: Literal["ok", "failed", "skipped"]
    actor_id: str | None = None
    apify_run_id: str | None = None
    raw_path: str | None = None
    raw_count: int = 0
    kept_count: int = 0
    dropped_empty_count: int = 0
    dropped_cleanbot_count: int = 0
    dropped_pre_warmup_count: int = 0
    dropped_dedup_count: int = 0
    est_cost_usd: float = 0.0
    updated_at_utc: str
    error_detail: str | None = None


class Phase1Manifest(BaseModel):
    schema_version: str = "1.0"
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)
    exchange_windows: dict[str, dict[str, Any]] = Field(default_factory=dict)
    preflight: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Canonical PyArrow Parquet Schemas (Section 1.8)
# =============================================================================

BARS_1H_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("trading_date_local", pa.date32(), nullable=False),
        pa.field("bar_open_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("bar_close_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("bar_duration_min", pa.int16(), nullable=False),
        pa.field("session_segment", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("is_zero_volume", pa.bool_(), nullable=False),
        pa.field("is_imputed", pa.bool_(), nullable=False),
        pa.field("window_role", pa.string(), nullable=False),
        pa.field("next_bar_close_utc", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("next_close", pa.float64(), nullable=True),
        pa.field("target_log_return_1h", pa.float64(), nullable=True),
    ]
)

EVENTS_ALIGNED_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("channel_or_domain", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("event_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("effective_event_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("timestamp_precision", pa.string(), nullable=False),
        pa.field("likes", pa.int64(), nullable=False),
        pa.field("dislikes", pa.int64(), nullable=False),
        pa.field("reposts", pa.int64(), nullable=False),
        pa.field("views", pa.int64(), nullable=False),
        pa.field("comments", pa.int64(), nullable=False),
        pa.field("author_followers", pa.int64(), nullable=True),
        pa.field("author_followers_is_proxy", pa.bool_(), nullable=False),
        pa.field("effective_reach", pa.int64(), nullable=False),
        pa.field("is_holder_verified", pa.bool_(), nullable=True),
        pa.field("is_cross_channel_repost", pa.bool_(), nullable=False),
        pa.field("is_short_text", pa.bool_(), nullable=False),
        pa.field("source_tier", pa.string(), nullable=True),
        pa.field("source_credibility_weight", pa.float32(), nullable=False),
        pa.field("matched_aliases", pa.list_(pa.string()), nullable=False),
        pa.field("is_multi_ticker", pa.bool_(), nullable=False),
        pa.field("has_full_body", pa.bool_(), nullable=False),
        pa.field("assigned_bar_close_utc", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("lag_minutes_to_bar_close", pa.float64(), nullable=True),
        pa.field("alignment_bucket", pa.string(), nullable=False),
        pa.field("window_role", pa.string(), nullable=False),
    ]
)


# =============================================================================
# Config Loaders
# =============================================================================


def load_universe(path: Path | None = None) -> UniverseConfig:
    cfg_path = path or (CONFIG_DIR / "universe.yaml")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return UniverseConfig.model_validate(raw)


def load_sources(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (CONFIG_DIR / "sources.yaml")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
