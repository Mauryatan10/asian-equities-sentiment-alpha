"""Dataset loading, 36-feature transform, zero-lookahead UTC-embargoed splitting,
and live snapshot overlay."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.alpha_factors import compute_rolling_copula_features
from src.features.schemas import (
    FEATURES_1H_PATH,
    LIVE_SNAPSHOT_PATH,
)
from src.ingest.schemas import BARS_PARQUET_PATH
from src.models.schemas import Phase3Config, load_phase3_config

logger = logging.getLogger(__name__)


@dataclass
class ModelDataset:
    train_split: pd.DataFrame
    test_oos: pd.DataFrame
    train_live: pd.DataFrame
    live: pd.DataFrame
    test_start_utc: pd.Timestamp
    feature_cols: list[str]
    tech_cols: list[str]
    vol_scale_train: np.ndarray
    y_train: np.ndarray
    vol_scale_test: np.ndarray
    actual_log_ret_test: np.ndarray
    vol_scale_train_live: np.ndarray
    y_train_live: np.ndarray


def prepare_features_df(
    df: pd.DataFrame,
    cfg: Phase3Config,
) -> pd.DataFrame:
    """Apply deterministic imputation, clock flags, and non-linear transforms to features df."""
    df = df.copy()

    # Fill warmup missing values deterministically
    df["hours_since_last_social"] = (
        df["hours_since_last_social"]
        .fillna(cfg.warmup.staleness_cap_hours)
        .clip(upper=cfg.warmup.staleness_cap_hours)
    )
    df["hours_since_last_news"] = (
        df["hours_since_last_news"]
        .fillna(cfg.warmup.staleness_cap_hours)
        .clip(upper=cfg.warmup.staleness_cap_hours)
    )

    zero_fill_cols = [
        "retail_sent_6h",
        "retail_sent_24h",
        "news_sent_24h",
        "divergence_retail_vs_news_24h",
        "divergence_retail_vs_news_fast",
        "sarcasm_rate_24h",
        "hype_intensity_24h",
        "sentiment_dispersion_24h",
        "net_crowd_approval_24h",
        "holder_vs_nonholder_spread_24h",
        "verified_holder_share_24h",
        "copula_div_trend",
        "copula_div_vol",
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Cast boolean indicator flags to float before boolean masks
    if "has_verified_holder_signal" in df.columns:
        df["has_verified_holder_signal"] = df["has_verified_holder_signal"].astype(float)
    if "in_value_area" in df.columns:
        df["in_value_area"] = df["in_value_area"].astype(float)

    # 100% causal session close bar clock flag
    hk_mask = df["exchange"] == "HKEX"
    kr_mask = df["exchange"] == "KRX"

    hk_close = (
        df.loc[hk_mask, "bar_close_utc"]
        .dt.tz_convert("Asia/Hong_Kong")
        .dt.time
        >= pd.to_datetime("16:00").time()
    )
    kr_close = (
        df.loc[kr_mask, "bar_close_utc"]
        .dt.tz_convert("Asia/Seoul")
        .dt.time
        >= pd.to_datetime("15:00").time()
    )

    df["is_session_close_bar"] = 0.0
    df.loc[hk_mask, "is_session_close_bar"] = hk_close.astype(float)
    df.loc[kr_mask, "is_session_close_bar"] = kr_close.astype(float)

    if "bar_duration_min" in df.columns:
        df["bar_duration_min_is_30"] = (df["bar_duration_min"] == 30).astype(float)
    else:
        df["bar_duration_min_is_30"] = 0.0

    df["is_hkex"] = (df["exchange"] == "HKEX").astype(float)

    # Bounded R/R and momentum deltas
    df["sr_risk_reward_ratio"] = df["sr_risk_reward_ratio"].clip(0.0, 10.0)
    df["retail_sent_momentum"] = df["retail_sent_6h"] - df["retail_sent_24h"]
    df["divergence_retail_vs_news_fast"] = (
        df["news_sent_24h"] - df["retail_sent_6h"]
    )

    # Signed log-compression on mention acceleration to tame KRX vs HKEX scale explosion
    df["mention_acceleration"] = np.sign(df["mention_acceleration"]) * np.log1p(
        np.abs(df["mention_acceleration"])
    )

    # Empirical copula transform and bilinear interaction terms
    df = compute_rolling_copula_features(df, window=42)

    return df


def load_and_split_dataset(
    cfg: Phase3Config | None = None,
    features_path: Path | None = None,
    bars_path: Path | None = None,
    live_snapshot_path: Path | None = None,
) -> ModelDataset:
    """Inner-join features with bars, apply 36-feature transforms, build strict UTC-embargoed
    OOS split and live prediction table with overlay."""
    cfg = cfg or load_phase3_config()
    feat_path = features_path or FEATURES_1H_PATH
    b_path = bars_path or BARS_PARQUET_PATH
    live_path = live_snapshot_path or LIVE_SNAPSHOT_PATH

    if not feat_path.exists():
        raise FileNotFoundError(f"features_1h.parquet not found at: {feat_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"bars_1h.parquet not found at: {b_path}")
    if not live_path.exists():
        raise FileNotFoundError(f"live_snapshot.parquet not found at: {live_path}")

    feats = pd.read_parquet(feat_path)
    bars = pd.read_parquet(b_path)
    live_snap = pd.read_parquet(live_path)

    # Inner join bars metadata needed for duration and embargo alignment
    join_cols = ["ticker", "bar_close_utc", "bar_duration_min", "next_bar_close_utc"]
    df = feats.merge(bars[join_cols], on=["ticker", "bar_close_utc"])
    if len(df) not in (905, 910):
        raise ValueError(f"Expected 905 or 910 joined rows, got {len(df)}")

    df = df.sort_values(["ticker", "bar_close_utc"]).reset_index(drop=True)
    df["bar_idx"] = df.groupby("ticker").cumcount()

    # Apply feature transforms
    df = prepare_features_df(df, cfg)

    feature_cols = cfg.features.full_features
    tech_cols = cfg.features.tech_features
    if len(feature_cols) != len(cfg.features.full_features):
        raise ValueError(f"Expected {len(cfg.features.full_features)} features, got {len(feature_cols)}")
    if len(tech_cols) != len(cfg.features.tech_features):
        raise ValueError(f"Expected {len(cfg.features.tech_features)} tech features, got {len(tech_cols)}")

    # Verify zero NaNs in features for bar_idx >= min_bar_idx
    panel_df = df[df["bar_idx"] >= cfg.warmup.min_bar_idx].copy()
    nan_counts = panel_df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        bad_cols = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"Found NaNs in panel feature columns: {bad_cols}")

    # Build OOS Test split: Days 8-10 for each ticker
    test_rows = []
    for ticker, grp in df.groupby("ticker"):
        active_dates = (
            grp[grp["window_role"] == "active_10d"]["trading_date_local"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if len(active_dates) != 10:
            raise ValueError(f"Ticker {ticker} does not have exactly 10 active dates, got {len(active_dates)}")
        d8_10 = active_dates[7:]
        oos = grp[
            (grp["trading_date_local"].isin(d8_10))
            & (grp["target_log_return_1h"].notna())
        ]
        test_rows.append(oos)

    test_oos = pd.concat(test_rows, ignore_index=True)
    if len(test_oos) not in (180, 185):
        raise ValueError(f"Expected 180 or 185 test_oos rows, got {len(test_oos)}")

    test_start_utc = test_oos["bar_close_utc"].min()

    # Build Train split: bar_idx >= min_bar_idx AND target notna AND next_bar_close_utc <= test_start_utc
    train_split = df[
        (df["bar_idx"] >= cfg.warmup.min_bar_idx)
        & (df["target_log_return_1h"].notna())
        & (df["next_bar_close_utc"] <= test_start_utc)
    ].copy()

    if len(train_split) != 540:
        raise ValueError(f"Expected 540 train_split rows, got {len(train_split)}")

    # Zero-lookahead assertion
    assert (
        train_split["next_bar_close_utc"].max() <= test_start_utc
    ), f"Leakage violation: train next_bar_close {train_split['next_bar_close_utc'].max()} > test_start {test_start_utc}"

    # Build train_live (all completed labeled bars)
    train_live = df[
        (df["bar_idx"] >= cfg.warmup.min_bar_idx)
        & (df["target_log_return_1h"].notna())
    ].copy()

    if len(train_live) not in (725, 730):
        raise ValueError(f"Expected 725 or 730 train_live rows, got {len(train_live)}")

    # Build live prediction dataset (10 rows, 1 per ticker) with overlay
    live_df = df[df["target_log_return_1h"].isna()].copy()
    if len(live_df) != 10:
        raise ValueError(f"Expected 10 live rows, got {len(live_df)}")

    overlay_cols = [
        "retail_sent_6h",
        "retail_sent_24h",
        "news_sent_24h",
        "divergence_retail_vs_news_24h",
        "holder_vs_nonholder_spread_24h",
        "mention_velocity_zscore_24h",
        "sarcasm_rate_24h",
        "hype_intensity_24h",
    ]
    for _, snap_row in live_snap.iterrows():
        t = snap_row["ticker"]
        idx = live_df[live_df["ticker"] == t].index
        for c in overlay_cols:
            live_df.loc[idx, c] = snap_row[c]
        has_vh = 1.0 if pd.notna(snap_row["holder_vs_nonholder_spread_24h"]) else 0.0
        live_df.loc[idx, "has_verified_holder_signal"] = has_vh

    # Fill NaNs from live snapshot
    live_df["holder_vs_nonholder_spread_24h"] = live_df[
        "holder_vs_nonholder_spread_24h"
    ].fillna(0.0)
    live_df["verified_holder_share_24h"] = live_df[
        "verified_holder_share_24h"
    ].fillna(0.0)

    # Recompute derived sentiment features for live
    live_df["retail_sent_momentum"] = (
        live_df["retail_sent_6h"] - live_df["retail_sent_24h"]
    )
    live_df["divergence_retail_vs_news_fast"] = (
        live_df["news_sent_24h"] - live_df["retail_sent_6h"]
    )

    if live_df[feature_cols].isna().sum().sum() > 0:
        bad_live = live_df[feature_cols].isna().sum()
        raise ValueError(f"Found NaNs in live feature columns: {bad_live[bad_live > 0].to_dict()}")

    # Volatility scales and targets
    vol_floor = cfg.target.vol_floor
    y_clip = cfg.target.y_clip

    vol_scale_train = np.maximum(train_split["atr_pct"].values, vol_floor)
    y_train = np.clip(
        train_split["target_log_return_1h"].values / vol_scale_train,
        -y_clip,
        y_clip,
    )

    vol_scale_test = np.maximum(test_oos["atr_pct"].values, vol_floor)
    actual_log_ret_test = test_oos["target_log_return_1h"].values

    vol_scale_train_live = np.maximum(train_live["atr_pct"].values, vol_floor)
    y_train_live = np.clip(
        train_live["target_log_return_1h"].values / vol_scale_train_live,
        -y_clip,
        y_clip,
    )

    return ModelDataset(
        train_split=train_split,
        test_oos=test_oos,
        train_live=train_live,
        live=live_df,
        test_start_utc=test_start_utc,
        feature_cols=feature_cols,
        tech_cols=tech_cols,
        vol_scale_train=vol_scale_train,
        y_train=y_train,
        vol_scale_test=vol_scale_test,
        actual_log_ret_test=actual_log_ret_test,
        vol_scale_train_live=vol_scale_train_live,
        y_train_live=y_train_live,
    )
