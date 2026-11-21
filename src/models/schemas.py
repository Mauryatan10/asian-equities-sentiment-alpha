"""Pydantic models, configuration loader, and canonical PyArrow schemas for Phase 3."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
import pyarrow as pa
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.ingest.schemas import (
    CONFIG_DIR,
    DATA_DIR,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)

PHASE3_CONFIG_PATH = CONFIG_DIR / "phase3.yaml"
CACHE_DIR = DATA_DIR / "cache"
PREDICTIONS_OOS_PATH = PROCESSED_DIR / "predictions_oos.parquet"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "feature_importance.parquet"
TEARSHEET_METRICS_PATH = PROCESSED_DIR / "tearsheet_metrics.json"
REPORTS_DIR = DATA_DIR / "reports"
PHASE3_TEARSHEET_REPORT_PATH = REPORTS_DIR / "phase3_tearsheet.md"
PHASE3_MANIFEST_PATH = MANIFESTS_DIR / "phase3_manifest.json"
MERMAID_GRAPH_PATH = MANIFESTS_DIR / "phase3_graph.mmd"

PositionStance = Literal["Long", "Short", "Hold"]
PrevailingDirection = Literal["bullish", "bearish", "neutral"]


# =============================================================================
# Configuration Models (config/phase3.yaml)
# =============================================================================


class Phase3FeaturesConfig(BaseModel):
    group_a_price_vol_clock: list[str]
    group_b_kde_sr: list[str]
    group_c_sentiment_alpha: list[str]

    @property
    def tech_features(self) -> list[str]:
        return self.group_a_price_vol_clock + self.group_b_kde_sr

    @property
    def full_features(self) -> list[str]:
        return (
            self.group_a_price_vol_clock
            + self.group_b_kde_sr
            + self.group_c_sentiment_alpha
        )


class TargetConfig(BaseModel):
    vol_floor: float = 0.002
    y_clip: float = 4.0


class WarmupConfig(BaseModel):
    min_bar_idx: int = 17
    staleness_cap_hours: float = 72.0


class HuberModelConfig(BaseModel):
    epsilon: float = 1.35
    alpha: float = 10.0
    max_iter: int = 500


class LGBMModelConfig(BaseModel):
    objective: str = "huber"
    alpha: float = 1.35
    max_depth: int = 2
    num_leaves: int = 4
    min_child_samples: int = 15
    learning_rate: float = 0.03
    n_estimators: int = 60
    colsample_bytree: float = 0.7
    subsample: float = 0.8
    subsample_freq: int = 1
    reg_alpha: float = 1.0
    reg_lambda: float = 10.0
    random_state: int = 42
    verbosity: int = -1


class BlendConfig(BaseModel):
    huber_weight: float = 0.60
    lgbm_weight: float = 0.40


class MoEGatingConfig(BaseModel):
    enabled: bool = True
    state_features: list[str] = Field(
        default_factory=lambda: ["realized_vol_12h", "atr_pct", "sentiment_dispersion_24h"]
    )
    l2_reg: float = 1.0
    cv_folds: int = 5


class ModelsConfig(BaseModel):
    huber: HuberModelConfig
    lgbm: LGBMModelConfig
    blend: BlendConfig
    moe_gating: MoEGatingConfig = Field(default_factory=MoEGatingConfig)


class ExecutionGateConfig(BaseModel):
    mode: str = "snr"
    snr_threshold: float = 1.28
    aleatoric_weight: float = 0.5


class RiskSizingConfig(BaseModel):
    mode: str = "volatility_targeted"
    annual_target_vol: float = 0.15


class CooldownConfig(BaseModel):
    enabled: bool = True
    max_consecutive_losses: int = 2
    cooldown_bars: int = 2


class BacktestConfig(BaseModel):
    hurdle_bps: float = 15.0
    slippage_bps: float = 5.0
    sr_buffer_atr: float = 0.25
    hkex_bars_per_day: int = 7
    krx_bars_per_day: int = 6
    execution_gate: ExecutionGateConfig = Field(default_factory=ExecutionGateConfig)
    risk_sizing: RiskSizingConfig = Field(default_factory=RiskSizingConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)


class SmokeTestConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["0700.HK", "005930.KS"])


class Phase3Config(BaseModel):
    features: Phase3FeaturesConfig
    target: TargetConfig = Field(default_factory=TargetConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    models: ModelsConfig
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)


def load_phase3_config(path: Path | None = None) -> Phase3Config:
    cfg_path = path or PHASE3_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Phase 3 config not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Phase3Config.model_validate(raw)


# =============================================================================
# Canonical PyArrow Schemas
# =============================================================================

PREDICTIONS_OOS_SCHEMA = pa.schema([
    ("ticker", pa.string()),
    ("exchange", pa.string()),
    ("trading_date_local", pa.date32()),
    ("bar_close_utc", pa.timestamp("us", tz="UTC")),
    ("next_bar_close_utc", pa.timestamp("us", tz="UTC")),
    ("close", pa.float64()),
    ("actual_next_close", pa.float64()),
    ("actual_log_return_1h", pa.float64()),
    ("atr_pct", pa.float64()),
    ("full_pred_huber_ret", pa.float64()),
    ("full_pred_lgbm_ret", pa.float64()),
    ("full_pred_ret", pa.float64()),
    ("full_pred_price", pa.float64()),
    ("tech_pred_ret", pa.float64()),
    ("tech_pred_price", pa.float64()),
    ("rw_pred_price", pa.float64()),
    ("full_pos_ls", pa.int8()),
    ("full_pos_lo", pa.int8()),
    ("tech_pos_ls", pa.int8()),
    ("full_net_ret_ls", pa.float64()),
    ("full_net_ret_lo", pa.float64()),
    ("full_slippage_ls", pa.float64()),
    ("tech_net_ret_ls", pa.float64()),
])

FEATURE_IMPORTANCE_SCHEMA = pa.schema([
    ("feature_name", pa.string()),
    ("feature_group", pa.string()),
    ("huber_coef_signed", pa.float64()),
    ("huber_coef_abs", pa.float64()),
    ("lgbm_gain_importance", pa.float64()),
    ("combined_score", pa.float64()),
    ("combined_rank", pa.int32()),
])
