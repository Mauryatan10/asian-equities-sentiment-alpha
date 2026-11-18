"""Pydantic models, LangGraph state TypedDict, config loader, and canonical PyArrow schemas for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
import pyarrow as pa
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.ingest.schemas import (
    CONFIG_DIR,
    DATA_DIR,
    EVENTS_ALIGNED_SCHEMA,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)

PHASE2_CONFIG_PATH = CONFIG_DIR / "phase2.yaml"
CACHE_DIR = DATA_DIR / "cache"
LLM_CACHE_PATH = CACHE_DIR / "phase2_llm_scores.jsonl"
EVENTS_SCORED_PATH = PROCESSED_DIR / "events_scored.parquet"
FEATURES_1H_PATH = PROCESSED_DIR / "features_1h.parquet"
LIVE_SNAPSHOT_PATH = PROCESSED_DIR / "live_snapshot.parquet"
PHASE2_MANIFEST_PATH = MANIFESTS_DIR / "phase2_manifest.json"
MERMAID_GRAPH_PATH = MANIFESTS_DIR / "phase2_graph.mmd"

LLMProviderName = Literal["nvidia", "gemini", "groq"]
DirectionLabel = Literal["bullish", "bearish", "neutral"]
CatalystType = Literal[
    "earnings",
    "guidance",
    "analyst_rating",
    "macro_policy",
    "product_tech",
    "flow_squeeze",
    "retail_rumor",
    "other",
]
RoutingBucket = Literal[
    "non_naver",
    "naver_verified",
    "naver_fresh",
    "naver_priority",
    "naver_tail_unscored",
    "smoke_unscored",
]

VALID_CATALYST_TYPES: tuple[str, ...] = (
    "earnings",
    "guidance",
    "analyst_rating",
    "macro_policy",
    "product_tech",
    "flow_squeeze",
    "retail_rumor",
    "other",
)


# =============================================================================
# Configuration Models (config/phase2.yaml)
# =============================================================================


class LLMProviderConfig(BaseModel):
    base_url: str
    api_key_env: str
    cultural_nlp_model: str
    portfolio_manager_model: str
    alt_models: list[str] = Field(default_factory=list)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    use_response_format_json: bool = True


class LLMConfig(BaseModel):
    default_provider: LLMProviderName = "nvidia"
    temperature: float = 0.1
    max_tokens: int = 4096
    batch_size: int = 25
    rpm_limit: int = 25
    max_concurrency: int = 2
    request_timeout_seconds: float = 120.0
    retry_max_attempts: int = 5
    retry_backoff_seconds: list[int] = Field(default_factory=lambda: [3, 6, 12, 24, 48])
    bisect_max_retries: int = 2
    social_title_max_chars: int = 200
    social_text_max_chars: int = 600
    news_title_max_chars: int = 200
    news_text_max_chars: int = 1200
    providers: dict[LLMProviderName, LLMProviderConfig]


class SmokeTestConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["0700.HK", "005930.KS"])
    max_llm_items_per_ticker: int = 10
    batch_size: int = 10


class PriorityWeightsConfig(BaseModel):
    views_log: float = 1.0
    engagement_log: float = 2.0
    holder_verified_bonus: float = 1.5
    min_char_len_bonus: float = 0.5
    min_char_len_threshold: int = 30


class RouterConfig(BaseModel):
    sqrt_multiplier: float = 2.0
    min_quota: int = 6
    max_quota: int = 50
    verified_reserve_ratio: float = 0.30
    fresh_reserve_ratio: float = 0.15
    fresh_window_minutes: float = 60.0
    priority_weights: PriorityWeightsConfig = Field(default_factory=PriorityWeightsConfig)


class QuantTechnicalConfig(BaseModel):
    min_bars: int = 18
    max_window: int = 48
    atr_period: int = 14
    rsi_period: int = 14
    vwap_window: int = 24
    realized_vol_window: int = 12
    realized_vol_min_periods: int = 6
    volume_zscore_window: int = 24
    volume_zscore_min_periods: int = 6
    zscore_clip: float = 5.0
    vpvr_points_per_bar: int = 5
    vpvr_weights: list[float] = Field(default_factory=lambda: [0.15, 0.20, 0.30, 0.20, 0.15])
    kde_bw_atr_factor: float = 0.75
    kde_bw_min: float = 0.15
    kde_bw_max: float = 0.80
    kde_grid_points: int = 250
    kde_grid_pad_atr: float = 0.5
    peak_prominence_frac: float = 0.05
    peak_min_dist_atr: float = 0.5
    value_area_pct: float = 0.70
    sr_rr_denom_atr_floor: float = 0.1


class AlphaFactorsConfig(BaseModel):
    social_fast_half_life_hours: float = 6.0
    social_slow_half_life_hours: float = 24.0
    news_half_life_hours: float = 24.0
    decay_epsilon: float = 1.0e-6
    source_blend_weights: dict[str, dict[str, float]]
    composite_retail_weight: float = 0.55
    composite_news_weight: float = 0.45
    event_direction_threshold: float = 0.15
    composite_direction_threshold: float = 0.10
    price_target_min_ratio: float = 0.5
    price_target_max_ratio: float = 2.0
    velocity_window: int = 24
    velocity_min_periods: int = 6
    zscore_clip: float = 5.0
    holder_verified_weight_boost: float = 0.5
    repost_weight_factor: float = 0.5
    catalyst_lookback_hours: float = 24.0
    max_top_catalysts_per_type: int = 5


class SlangEntry(BaseModel):
    term: str
    meaning: str


class Phase2Config(BaseModel):
    llm: LLMConfig
    smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    quant_technical: QuantTechnicalConfig = Field(default_factory=QuantTechnicalConfig)
    alpha_factors: AlphaFactorsConfig
    slang_glossaries: dict[str, list[SlangEntry]] = Field(default_factory=dict)


def load_phase2_config(path: Path | None = None) -> Phase2Config:
    cfg_path = path or PHASE2_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Phase2Config.model_validate(raw)


# =============================================================================
# Per-Event Structured LLM Output & Live Catalyst Summary Models
# =============================================================================


def deterministic_direction(score: float, threshold: float = 0.15) -> DirectionLabel:
    if score > threshold:
        return "bullish"
    if score < -threshold:
        return "bearish"
    return "neutral"


class ScoredEventItem(BaseModel):
    """8-field structured output (+ event_id) produced by CulturalNLPAgent."""

    event_id: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    direction: DirectionLabel = "neutral"
    confidence: float = Field(ge=0.0, le=1.0)
    is_sarcastic: bool = False
    hype_score: float = Field(ge=0.0, le=1.0)
    catalyst_type: CatalystType = "other"
    price_target_mentioned: float | None = None
    english_summary: str

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def _clamp_sentiment(cls, v: Any) -> float:
        val = float(v)
        return max(-1.0, min(1.0, val))

    @field_validator("confidence", "hype_score", mode="before")
    @classmethod
    def _clamp_unit(cls, v: Any) -> float:
        val = float(v)
        return max(0.0, min(1.0, val))

    @field_validator("catalyst_type", mode="before")
    @classmethod
    def _normalize_catalyst_type(cls, v: Any) -> str:
        if v is None:
            return "other"
        s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        return s if s in VALID_CATALYST_TYPES else "other"

    @field_validator("price_target_mentioned", mode="before")
    @classmethod
    def _parse_price_target(cls, v: Any) -> float | None:
        if v is None or v == "" or str(v).strip().lower() in ("null", "none", "nan"):
            return None
        try:
            val = float(v)
        except (TypeError, ValueError):
            return None
        if val != val or val <= 0.0:  # NaN or non-positive
            return None
        return val

    @field_validator("english_summary", mode="before")
    @classmethod
    def _clean_summary(cls, v: Any) -> str:
        s = " ".join(str(v or "").split()).strip()
        if not s:
            s = "No summary provided."
        if len(s) > 120:
            s = s[:117].rstrip() + "..."
        return s

    @model_validator(mode="after")
    def _enforce_deterministic_direction(self) -> "ScoredEventItem":
        self.direction = deterministic_direction(self.sentiment_score, threshold=0.15)
        return self


class CatalystSummary(BaseModel):
    """Exact 13-key catalyst JSON contract for live_snapshot.parquet."""

    event_id: str
    event_type: str
    source: str
    channel_or_domain: str
    effective_event_utc: str
    sentiment_score: float
    direction: DirectionLabel
    is_sarcastic: bool
    hype_score: float
    catalyst_type: CatalystType
    english_summary: str
    url: str
    impact_weight: float


# =============================================================================
# Phase 2 Manifest Schema (data/manifests/phase2_manifest.json)
# =============================================================================


class TickerPhase2Stats(BaseModel):
    ticker: str
    exchange: str
    total_events: int
    llm_routed_events: int
    unique_deduped_texts_sent: int
    dedup_calls_saved: int
    llm_calls: int
    bisect_retries: int
    cache_hits: int
    mean_bar_impact_weight_coverage_pct: float
    verified_holder_llm_count: int
    fresh_bucket_llm_count: int
    sarcasm_rate: float
    live_event_count: int
    active_10d_bars: int


class Phase2Manifest(BaseModel):
    run_started_utc: str
    run_completed_utc: str
    llm_provider: str
    llm_model: str
    batch_size: int
    rpm_limit: int
    total_events: int
    llm_routed_events: int
    unique_deduped_texts_sent: int
    dedup_calls_saved: int
    total_llm_calls: int
    bisect_retries: int
    cache_hits: int
    per_ticker_stats: dict[str, TickerPhase2Stats] = Field(default_factory=dict)
    assertions_passed: bool = False
    mermaid_graph_path: str = ""


# =============================================================================
# LangGraph Per-Ticker State TypedDict
# =============================================================================


class TickerPhase2State(TypedDict, total=False):
    ticker: str
    exchange: str
    company_en: str
    company_local: str
    bars_df: pd.DataFrame
    events_df: pd.DataFrame
    routed_events_df: pd.DataFrame
    dedup_map: dict[str, list[str]]
    representative_events_df: pd.DataFrame
    scored_events_df: pd.DataFrame
    tech_features_df: pd.DataFrame
    full_corpus_stats_df: pd.DataFrame
    features_1h_df: pd.DataFrame
    live_snapshot_row: dict[str, Any]
    llm_stats: dict[str, Any]
    ticker_stats: dict[str, Any]
    assertions_ok: bool


# =============================================================================
# Canonical PyArrow Schemas (Section 2.6)
# =============================================================================

EVENTS_SCORED_SCHEMA = pa.schema(
    list(EVENTS_ALIGNED_SCHEMA)
    + [
        pa.field("priority_score", pa.float64(), nullable=True),
        pa.field("routing_bucket", pa.string(), nullable=False),
        pa.field("is_llm_scored", pa.bool_(), nullable=False),
        pa.field("sentiment_source", pa.string(), nullable=True),
        pa.field("sentiment_score", pa.float64(), nullable=True),
        pa.field("direction", pa.string(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("is_sarcastic", pa.bool_(), nullable=True),
        pa.field("hype_score", pa.float64(), nullable=True),
        pa.field("catalyst_type", pa.string(), nullable=True),
        pa.field("price_target_mentioned", pa.float64(), nullable=True),
        pa.field("english_summary", pa.string(), nullable=True),
        pa.field("impact_weight", pa.float64(), nullable=True),
    ]
)

# Warmup bars 0..16 may have nulls in rolling/KDE columns; active_10d non-null invariants
# are strictly enforced in verification assertions (Section 2.6).
FEATURES_1H_SCHEMA = pa.schema(
    [
        # Keys & Targets (13)
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("trading_date_local", pa.date32(), nullable=False),
        pa.field("bar_open_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("bar_close_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("window_role", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("next_close", pa.float64(), nullable=True),
        pa.field("target_log_return_1h", pa.float64(), nullable=True),
        # Causal Technicals (12)
        pa.field("log_ret_1h", pa.float64(), nullable=True),
        pa.field("log_ret_3h", pa.float64(), nullable=True),
        pa.field("log_ret_6h", pa.float64(), nullable=True),
        pa.field("realized_vol_12h", pa.float64(), nullable=True),
        pa.field("atr_14", pa.float64(), nullable=True),
        pa.field("atr_pct", pa.float64(), nullable=True),
        pa.field("rsi_14", pa.float64(), nullable=True),
        pa.field("vwap_24h", pa.float64(), nullable=True),
        pa.field("price_vs_vwap_pct", pa.float64(), nullable=True),
        pa.field("volume_zscore_24h", pa.float64(), nullable=True),
        pa.field("bar_range_pct", pa.float64(), nullable=True),
        pa.field("close_pos_in_bar", pa.float64(), nullable=True),
        # Causal KDE S/R (14)
        pa.field("poc_price", pa.float64(), nullable=True),
        pa.field("val_price", pa.float64(), nullable=True),
        pa.field("vah_price", pa.float64(), nullable=True),
        pa.field("support_1", pa.float64(), nullable=True),
        pa.field("resistance_1", pa.float64(), nullable=True),
        pa.field("dist_to_support_1_atr", pa.float64(), nullable=True),
        pa.field("dist_to_resistance_1_atr", pa.float64(), nullable=True),
        pa.field("sr_risk_reward_ratio", pa.float64(), nullable=True),
        pa.field("dist_to_poc_atr", pa.float64(), nullable=True),
        pa.field("in_value_area", pa.bool_(), nullable=True),
        pa.field("support_2", pa.float64(), nullable=True),
        pa.field("support_3", pa.float64(), nullable=True),
        pa.field("resistance_2", pa.float64(), nullable=True),
        pa.field("resistance_3", pa.float64(), nullable=True),
        # Core Decayed Sentiment & Divergence (15)
        pa.field("retail_sent_6h", pa.float64(), nullable=True),
        pa.field("retail_sent_24h", pa.float64(), nullable=True),
        pa.field("news_sent_24h", pa.float64(), nullable=True),
        pa.field("composite_sentiment_score", pa.float64(), nullable=True),
        pa.field("divergence_retail_vs_news_24h", pa.float64(), nullable=True),
        pa.field("divergence_retail_vs_news_fast", pa.float64(), nullable=True),
        pa.field("hours_since_last_social", pa.float64(), nullable=True),
        pa.field("hours_since_last_news", pa.float64(), nullable=True),
        pa.field("sarcasm_rate_24h", pa.float64(), nullable=True),
        pa.field("hype_intensity_24h", pa.float64(), nullable=True),
        pa.field("sentiment_dispersion_24h", pa.float64(), nullable=True),
        pa.field("net_crowd_approval_24h", pa.float64(), nullable=True),
        pa.field("prevailing_direction", pa.string(), nullable=False),
        pa.field("has_social_history", pa.bool_(), nullable=False),
        pa.field("has_news_history", pa.bool_(), nullable=False),
        # Full-Corpus Activity & Velocity (6)
        pa.field("social_mention_count", pa.int32(), nullable=False),
        pa.field("news_mention_count", pa.int32(), nullable=False),
        pa.field("bar_engagement_sum", pa.float64(), nullable=False),
        pa.field("mention_velocity_zscore_24h", pa.float64(), nullable=True),
        pa.field("engagement_velocity_zscore_24h", pa.float64(), nullable=True),
        pa.field("mention_acceleration", pa.float64(), nullable=False),
        # Optional Sub-Streams & Community PT (10)
        pa.field("pure_forum_sent_24h", pa.float64(), nullable=True),
        pa.field("divergence_forum_vs_news_24h", pa.float64(), nullable=True),
        pa.field("holder_vs_nonholder_spread_24h", pa.float64(), nullable=True),
        pa.field("verified_holder_share_24h", pa.float64(), nullable=True),
        pa.field("community_pt_median_24h", pa.float64(), nullable=True),
        pa.field("community_pt_implied_upside", pa.float64(), nullable=True),
        pa.field("community_pt_vs_resistance_1_atr", pa.float64(), nullable=True),
        pa.field("community_pt_count_24h", pa.int32(), nullable=False),
        pa.field("has_forum_history", pa.bool_(), nullable=False),
        pa.field("has_verified_holder_signal", pa.bool_(), nullable=False),
    ]
)

ACTIVE_10D_NON_NULL_COLUMNS: tuple[str, ...] = (
    # Keys (non-target)
    "ticker",
    "exchange",
    "trading_date_local",
    "bar_open_utc",
    "bar_close_utc",
    "window_role",
    "open",
    "high",
    "low",
    "close",
    "volume",
    # Causal Technicals (12)
    "log_ret_1h",
    "log_ret_3h",
    "log_ret_6h",
    "realized_vol_12h",
    "atr_14",
    "atr_pct",
    "rsi_14",
    "vwap_24h",
    "price_vs_vwap_pct",
    "volume_zscore_24h",
    "bar_range_pct",
    "close_pos_in_bar",
    # Core Causal KDE S/R (10)
    "poc_price",
    "val_price",
    "vah_price",
    "support_1",
    "resistance_1",
    "dist_to_support_1_atr",
    "dist_to_resistance_1_atr",
    "sr_risk_reward_ratio",
    "dist_to_poc_atr",
    "in_value_area",
    # Core Decayed Sentiment & Divergence (15)
    "retail_sent_6h",
    "retail_sent_24h",
    "news_sent_24h",
    "composite_sentiment_score",
    "divergence_retail_vs_news_24h",
    "divergence_retail_vs_news_fast",
    "hours_since_last_social",
    "hours_since_last_news",
    "sarcasm_rate_24h",
    "hype_intensity_24h",
    "sentiment_dispersion_24h",
    "net_crowd_approval_24h",
    "prevailing_direction",
    "has_social_history",
    "has_news_history",
    # Full-Corpus Activity & Velocity (6)
    "social_mention_count",
    "news_mention_count",
    "bar_engagement_sum",
    "mention_velocity_zscore_24h",
    "engagement_velocity_zscore_24h",
    "mention_acceleration",
    # Non-null flags/counts in optional group (3)
    "community_pt_count_24h",
    "has_forum_history",
    "has_verified_holder_signal",
)

LIVE_SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=False),
        pa.field("as_of_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("last_bar_close_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("current_price", pa.float64(), nullable=False),
        pa.field("live_event_count", pa.int32(), nullable=False),
        pa.field("total_active_10d_mentions", pa.int32(), nullable=False),
        pa.field("poc_price", pa.float64(), nullable=False),
        pa.field("val_price", pa.float64(), nullable=False),
        pa.field("vah_price", pa.float64(), nullable=False),
        pa.field("support_1", pa.float64(), nullable=False),
        pa.field("support_2", pa.float64(), nullable=True),
        pa.field("support_3", pa.float64(), nullable=True),
        pa.field("resistance_1", pa.float64(), nullable=False),
        pa.field("resistance_2", pa.float64(), nullable=True),
        pa.field("resistance_3", pa.float64(), nullable=True),
        pa.field("sr_risk_reward_ratio", pa.float64(), nullable=False),
        pa.field("atr_14", pa.float64(), nullable=False),
        pa.field("rsi_14", pa.float64(), nullable=False),
        pa.field("retail_sent_6h", pa.float64(), nullable=False),
        pa.field("retail_sent_24h", pa.float64(), nullable=False),
        pa.field("news_sent_24h", pa.float64(), nullable=False),
        pa.field("composite_sentiment_score", pa.float64(), nullable=False),
        pa.field("divergence_retail_vs_news_24h", pa.float64(), nullable=False),
        pa.field("holder_vs_nonholder_spread_24h", pa.float64(), nullable=True),
        pa.field("mention_velocity_zscore_24h", pa.float64(), nullable=False),
        pa.field("sarcasm_rate_24h", pa.float64(), nullable=False),
        pa.field("hype_intensity_24h", pa.float64(), nullable=False),
        pa.field("community_pt_median_24h", pa.float64(), nullable=True),
        pa.field("prevailing_direction", pa.string(), nullable=False),
        pa.field("top_social_catalysts_json", pa.string(), nullable=False),
        pa.field("top_news_catalysts_json", pa.string(), nullable=False),
    ]
)

__all__ = [
    "PROJECT_ROOT",
    "PHASE2_CONFIG_PATH",
    "CACHE_DIR",
    "LLM_CACHE_PATH",
    "EVENTS_SCORED_PATH",
    "FEATURES_1H_PATH",
    "LIVE_SNAPSHOT_PATH",
    "PHASE2_MANIFEST_PATH",
    "MERMAID_GRAPH_PATH",
    "LLMProviderName",
    "DirectionLabel",
    "CatalystType",
    "RoutingBucket",
    "VALID_CATALYST_TYPES",
    "Phase2Config",
    "load_phase2_config",
    "deterministic_direction",
    "ScoredEventItem",
    "CatalystSummary",
    "TickerPhase2Stats",
    "Phase2Manifest",
    "TickerPhase2State",
    "EVENTS_SCORED_SCHEMA",
    "FEATURES_1H_SCHEMA",
    "ACTIVE_10D_NON_NULL_COLUMNS",
    "LIVE_SNAPSHOT_SCHEMA",
]
