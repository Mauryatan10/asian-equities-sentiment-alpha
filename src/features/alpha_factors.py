"""Engagement-weighted dual-horizon continuous UTC decay, full-corpus activity stats,
5 custom alpha factor families, live_snapshot builder, and per-ticker zero-leakage verification."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.features.schemas import (
    ACTIVE_10D_NON_NULL_COLUMNS,
    CatalystSummary,
    Phase2Config,
    deterministic_direction,
)

_LN2 = math.log(2.0)
_FORUM_SOURCES = {"naver_finance", "lihkg", "aastocks", "blind"}
_EPOCH_UTC = pd.Timestamp("1970-01-01", tz="UTC")


def _series_to_unix_seconds(series: pd.Series) -> np.ndarray:
    """Resolution-independent conversion of a UTC timestamp Series to float64 Unix seconds."""
    return (pd.to_datetime(series, utc=True) - _EPOCH_UTC).dt.total_seconds().to_numpy(dtype=np.float64)


def _ts_to_unix_seconds(ts: pd.Timestamp) -> float:
    return float(pd.Timestamp(ts).tz_convert("UTC").timestamp())


def _decayed_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    dh_hours: np.ndarray,
    half_life: float,
    eps: float,
) -> float:
    if len(values) == 0:
        return 0.0
    decay = np.exp(-(_LN2 / half_life) * dh_hours)
    eff_w = weights * decay
    denom = float(np.sum(eff_w)) + eps
    return float(np.sum(eff_w * values) / denom)


def _decayed_weighted_std(
    values: np.ndarray,
    weights: np.ndarray,
    dh_hours: np.ndarray,
    half_life: float,
    eps: float,
) -> float:
    if len(values) <= 1:
        return 0.0
    decay = np.exp(-(_LN2 / half_life) * dh_hours)
    eff_w = weights * decay
    w_sum = float(np.sum(eff_w))
    if w_sum <= 0.0:
        return 0.0
    mu = float(np.sum(eff_w * values) / (w_sum + eps))
    var = float(np.sum(eff_w * ((values - mu) ** 2)) / (w_sum + eps))
    return float(math.sqrt(max(0.0, var)))


def _rolling_causal_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
    clip_val: float,
) -> np.ndarray:
    logged = np.log1p(np.maximum(series.to_numpy(dtype=np.float64), 0.0))
    s = pd.Series(logged, index=series.index, dtype="float64")
    mean = s.rolling(window=window, min_periods=min_periods).mean()
    std = s.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = np.where(
        std.isna(),
        np.nan,
        np.where(
            std < 1e-8,
            0.0,
            np.clip((s - mean) / std, -clip_val, clip_val),
        ),
    )
    return z.astype(np.float64)


def compute_full_corpus_stats(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cfg: Phase2Config,
) -> pd.DataFrame:
    """Compute strictly causal full-corpus activity, velocity z-scores, decayed approval,
    and KRX verified-holder share over 100% of events (including unscored Naver tail)."""
    af = cfg.alpha_factors
    bars = bars_df.sort_values("bar_close_utc").reset_index(drop=True)
    n_bars = len(bars)
    bar_closes_sec = _series_to_unix_seconds(bars["bar_close_utc"])
    exchange = str(bars["exchange"].iloc[0]).upper() if n_bars > 0 else ""

    social_counts = np.zeros(n_bars, dtype=np.int32)
    news_counts = np.zeros(n_bars, dtype=np.int32)
    bar_eng_sum = np.zeros(n_bars, dtype=np.float64)

    # Map bar_close_utc -> bar index
    bar_idx_map = {pd.Timestamp(ts): idx for idx, ts in enumerate(bars["bar_close_utc"])}

    ev = events_df.copy()
    if not ev.empty:
        assigned = ev["assigned_bar_close_utc"]
        hist_ev = ev[assigned.notna()].copy()
        if not hist_ev.empty:
            hist_ev["_bar_idx"] = [bar_idx_map.get(pd.Timestamp(ts), -1) for ts in hist_ev["assigned_bar_close_utc"]]
            hist_ev = hist_ev[hist_ev["_bar_idx"] >= 0]

            is_social = hist_ev["event_type"].astype(str) == "social"
            soc_hist = hist_ev[is_social]
            news_hist = hist_ev[~is_social]

            if not soc_hist.empty:
                sc_by_bar = soc_hist.groupby("_bar_idx").size()
                for b_idx, cnt in sc_by_bar.items():
                    social_counts[int(b_idx)] = int(cnt)

                eng_vals = (
                    np.maximum(soc_hist["likes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
                    + np.maximum(soc_hist["dislikes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
                    + np.maximum(soc_hist["reposts"].fillna(0).to_numpy(dtype=np.float64), 0.0)
                    + np.maximum(soc_hist["comments"].fillna(0).to_numpy(dtype=np.float64), 0.0)
                    + np.maximum(soc_hist["views"].fillna(0).to_numpy(dtype=np.float64), 0.0)
                )
                eng_series = pd.Series(eng_vals).groupby(soc_hist["_bar_idx"].to_numpy()).sum()
                for b_idx, val in eng_series.items():
                    bar_eng_sum[int(b_idx)] = float(val)

            if not news_hist.empty:
                nc_by_bar = news_hist.groupby("_bar_idx").size()
                for b_idx, cnt in nc_by_bar.items():
                    news_counts[int(b_idx)] = int(cnt)

    mention_z = _rolling_causal_zscore(
        pd.Series(social_counts, dtype="float64"),
        window=af.velocity_window,
        min_periods=af.velocity_min_periods,
        clip_val=af.zscore_clip,
    )
    eng_z = _rolling_causal_zscore(
        pd.Series(bar_eng_sum, dtype="float64"),
        window=af.velocity_window,
        min_periods=af.velocity_min_periods,
        clip_val=af.zscore_clip,
    )

    # Continuous UTC decayed full-corpus quantities (strictly effective_event_utc <= T_t)
    mention_accel = np.zeros(n_bars, dtype=np.float64)
    net_approval = np.zeros(n_bars, dtype=np.float64)
    verified_share = np.full(n_bars, np.nan, dtype=np.float64)

    soc_all = ev[ev["event_type"].astype(str) == "social"].copy() if not ev.empty else ev
    if not soc_all.empty:
        soc_all = soc_all.sort_values("effective_event_utc")
        soc_sec = _series_to_unix_seconds(soc_all["effective_event_utc"])
        soc_likes = np.maximum(soc_all["likes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        soc_dislikes = np.maximum(soc_all["dislikes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        is_naver = (soc_all["source"].astype(str) == "naver_finance").to_numpy()
        is_verified = soc_all["is_holder_verified"].fillna(False).astype(bool).to_numpy(dtype=np.float64)

        hl_fast = af.social_fast_half_life_hours
        hl_slow = af.social_slow_half_life_hours

        # Pointer scan using searchsorted (strictly <= T_t)
        end_indices = np.searchsorted(soc_sec, bar_closes_sec, side="right")
        for t in range(n_bars):
            k_end = int(end_indices[t])
            if k_end <= 0:
                continue
            dh = (bar_closes_sec[t] - soc_sec[:k_end]) / 3600.0
            d_fast = np.exp(-(_LN2 / hl_fast) * dh)
            d_slow = np.exp(-(_LN2 / hl_slow) * dh)

            c_fast = float(np.sum(d_fast))
            c_slow = float(np.sum(d_slow))
            mention_accel[t] = (c_fast / hl_fast) - (c_slow / hl_slow)

            dl = float(np.sum(soc_likes[:k_end] * d_slow))
            dd = float(np.sum(soc_dislikes[:k_end] * d_slow))
            net_approval[t] = (dl - dd) / (dl + dd + 1.0)

            if exchange == "KRX":
                nv_mask = is_naver[:k_end]
                if nv_mask.any():
                    nv_d = d_slow[nv_mask]
                    nv_denom = float(np.sum(nv_d))
                    if nv_denom > 0.0:
                        verified_share[t] = float(np.sum(is_verified[:k_end][nv_mask] * nv_d) / nv_denom)

    return pd.DataFrame(
        {
            "bar_close_utc": bars["bar_close_utc"],
            "social_mention_count": social_counts,
            "news_mention_count": news_counts,
            "bar_engagement_sum": bar_eng_sum,
            "mention_velocity_zscore_24h": mention_z,
            "engagement_velocity_zscore_24h": eng_z,
            "mention_acceleration": mention_accel,
            "net_crowd_approval_24h": net_approval,
            "verified_holder_share_24h": verified_share,
        }
    )


def _rolling_ecdf_copula(arr: np.ndarray, window: int = 42) -> np.ndarray:
    """Calculate causal rolling empirical cumulative distribution function (ECDF) rank
    mapped to scale-invariant uniform variable u in [-1.0, +1.0] over lookback window."""
    n = len(arr)
    u = np.zeros(n, dtype=np.float64)
    for t in range(n):
        start = max(0, t - window + 1)
        w = arr[start : t + 1]
        val = arr[t]
        if np.isnan(val):
            u[t] = 0.0
            continue
        valid_w = w[~np.isnan(w)]
        k = len(valid_w)
        if k <= 1:
            u[t] = 0.0
            continue
        rank = float(np.sum(valid_w < val) + 0.5 * np.sum(valid_w == val) + 0.5)
        v = (rank - 0.5) / float(k)
        u[t] = float(np.clip(2.0 * v - 1.0, -1.0, 1.0))
    return u


def compute_rolling_copula_features(df: pd.DataFrame, window: int = 42) -> pd.DataFrame:
    """Compute rolling rank-transformed copula variables in [-1, +1] and bilinear interactions per ticker."""
    res_df = df.copy()
    if "copula_div_trend" not in res_df.columns:
        res_df["copula_div_trend"] = 0.0
    if "copula_div_vol" not in res_df.columns:
        res_df["copula_div_vol"] = 0.0

    div_col = "divergence_retail_vs_news_24h"
    trend_col = "price_vs_vwap_pct"
    vol_col = "realized_vol_12h"

    if div_col not in res_df.columns or trend_col not in res_df.columns or vol_col not in res_df.columns:
        return res_df

    if "ticker" in res_df.columns:
        groups = res_df.groupby("ticker", sort=False)
    else:
        groups = [(None, res_df)]

    for _, grp in groups:
        if "bar_close_utc" in grp.columns:
            grp_sorted = grp.sort_values("bar_close_utc")
        else:
            grp_sorted = grp
        idx = grp_sorted.index

        div_vals = grp_sorted[div_col].fillna(0.0).to_numpy(dtype=np.float64)
        trend_vals = grp_sorted[trend_col].fillna(0.0).to_numpy(dtype=np.float64)
        vol_vals = grp_sorted[vol_col].fillna(0.0).to_numpy(dtype=np.float64)

        u_div = _rolling_ecdf_copula(div_vals, window=window)
        u_trend = _rolling_ecdf_copula(trend_vals, window=window)
        u_vol = _rolling_ecdf_copula(vol_vals, window=window)

        res_df.loc[idx, "copula_div_trend"] = u_div * u_trend
        res_df.loc[idx, "copula_div_vol"] = u_div * u_vol

    return res_df


def _evaluate_decayed_sentiment_at_time(
    target_sec: float,
    exchange: str,
    close_price: float,
    resistance_1: float,
    atr_14: float,
    scored_social: dict[str, np.ndarray],
    scored_news: dict[str, np.ndarray],
    cfg: Phase2Config,
) -> dict[str, Any]:
    """Evaluate all dual-horizon decayed sentiment & alpha factors at exact UTC timestamp `target_sec`
    using ONLY events with effective_event_utc <= target_sec."""
    af = cfg.alpha_factors
    hl_fast = af.social_fast_half_life_hours
    hl_slow = af.social_slow_half_life_hours
    hl_news = af.news_half_life_hours
    eps = af.decay_epsilon

    # Social slice strictly <= target_sec
    s_sec = scored_social["sec"]
    s_end = int(np.searchsorted(s_sec, target_sec, side="right"))
    has_social = s_end > 0

    # News slice strictly <= target_sec
    n_sec = scored_news["sec"]
    n_end = int(np.searchsorted(n_sec, target_sec, side="right"))
    has_news = n_end > 0

    # --- Source-balanced retail sentiment (6h & 24h) ---
    retail_6h = 0.0
    retail_24h = 0.0
    hours_since_social = np.nan
    sarcasm_24h = 0.0
    hype_24h = 0.0
    disp_24h = 0.0
    has_forum = False
    pure_forum_24h = np.nan
    has_holder_sig = False
    holder_spread_24h = np.nan

    if has_social:
        dh_s = (target_sec - s_sec[:s_end]) / 3600.0
        hours_since_social = float(np.min(dh_s))
        sent_s = scored_social["sentiment"][:s_end]
        w_s = scored_social["weight"][:s_end]
        src_s = scored_social["source"][:s_end]
        sarc_s = scored_social["sarcastic"][:s_end]
        hype_s = scored_social["hype"][:s_end]
        ver_s = scored_social["verified"][:s_end]

        blend_cfg = af.source_blend_weights.get(exchange, {})
        # Compute per-source decayed sentiment first so Naver never drowns out Telegram
        src_6h: dict[str, float] = {}
        src_24h: dict[str, float] = {}
        src_weights: dict[str, float] = {}

        unique_sources = set(src_s.tolist())
        for src_name in unique_sources:
            mask = src_s == src_name
            src_6h[src_name] = _decayed_weighted_mean(sent_s[mask], w_s[mask], dh_s[mask], hl_fast, eps)
            src_24h[src_name] = _decayed_weighted_mean(sent_s[mask], w_s[mask], dh_s[mask], hl_slow, eps)
            # Map standby backup forum sources to primary forum weight if present
            if src_name in blend_cfg:
                src_weights[src_name] = float(blend_cfg[src_name])
            elif src_name in _FORUM_SOURCES:
                primary_forum = "naver_finance" if exchange == "KRX" else "lihkg"
                src_weights[src_name] = float(blend_cfg.get(primary_forum, 0.60))
            else:
                src_weights[src_name] = 0.35

        total_bw = sum(src_weights.values())
        if total_bw > 0.0:
            retail_6h = sum((src_weights[k] / total_bw) * src_6h[k] for k in src_weights)
            retail_24h = sum((src_weights[k] / total_bw) * src_24h[k] for k in src_weights)

        # Sarcasm, hype, dispersion over scored social events (24h decay)
        sarcasm_24h = _decayed_weighted_mean(sarc_s, w_s, dh_s, hl_slow, eps)
        hype_24h = _decayed_weighted_mean(hype_s, w_s, dh_s, hl_slow, eps)
        disp_24h = _decayed_weighted_std(sent_s, w_s, dh_s, hl_slow, eps)

        # Pure-forum sub-stream
        forum_mask = np.isin(src_s, list(_FORUM_SOURCES))
        if forum_mask.any():
            has_forum = True
            pure_forum_24h = _decayed_weighted_mean(
                sent_s[forum_mask], w_s[forum_mask], dh_s[forum_mask], hl_slow, eps
            )

        # KRX Verified-Shareholder Conviction Alpha
        if exchange == "KRX":
            nv_mask = src_s == "naver_finance"
            ver_mask = nv_mask & (ver_s > 0.5)
            nonver_mask = nv_mask & (ver_s <= 0.5)
            if ver_mask.any() and nonver_mask.any():
                has_holder_sig = True
                ver_sent = _decayed_weighted_mean(
                    sent_s[ver_mask], w_s[ver_mask], dh_s[ver_mask], hl_slow, eps
                )
                nonver_sent = _decayed_weighted_mean(
                    sent_s[nonver_mask], w_s[nonver_mask], dh_s[nonver_mask], hl_slow, eps
                )
                holder_spread_24h = ver_sent - nonver_sent

    # --- Institutional news sentiment (24h) ---
    news_24h = 0.0
    hours_since_news = np.nan
    if has_news:
        dh_n = (target_sec - n_sec[:n_end]) / 3600.0
        hours_since_news = float(np.min(dh_n))
        news_24h = _decayed_weighted_mean(
            scored_news["sentiment"][:n_end],
            scored_news["weight"][:n_end],
            dh_n,
            hl_news,
            eps,
        )

    if has_social and has_news:
        composite = af.composite_retail_weight * retail_24h + af.composite_news_weight * news_24h
    elif has_social:
        composite = retail_24h
    elif has_news:
        composite = news_24h
    else:
        composite = 0.0
    prev_dir = deterministic_direction(composite, threshold=af.composite_direction_threshold)

    div_24h = news_24h - retail_24h
    div_fast = news_24h - retail_6h
    div_forum = (news_24h - pure_forum_24h) if (has_forum and has_news) else np.nan

    # --- Community Price Target vs. KDE S/R Alignment (past 24h causal window) ---
    pts_list: list[float] = []
    if has_social:
        dh_s = (target_sec - s_sec[:s_end]) / 3600.0
        pt_s = scored_social["price_target"][:s_end]
        mask_pt_s = (dh_s <= 24.0) & np.isfinite(pt_s)
        if mask_pt_s.any():
            pts_list.extend(pt_s[mask_pt_s].tolist())
    if has_news:
        dh_n = (target_sec - n_sec[:n_end]) / 3600.0
        pt_n = scored_news["price_target"][:n_end]
        mask_pt_n = (dh_n <= 24.0) & np.isfinite(pt_n)
        if mask_pt_n.any():
            pts_list.extend(pt_n[mask_pt_n].tolist())

    pt_count = len(pts_list)
    if pt_count > 0:
        pt_median = float(np.median(pts_list))
        pt_upside = (pt_median - close_price) / close_price if close_price > 0.0 else np.nan
        pt_vs_res1_atr = (
            (pt_median - resistance_1) / atr_14
            if (np.isfinite(resistance_1) and np.isfinite(atr_14) and atr_14 > 0.0)
            else np.nan
        )
    else:
        pt_median = np.nan
        pt_upside = np.nan
        pt_vs_res1_atr = np.nan

    return {
        "retail_sent_6h": float(retail_6h),
        "retail_sent_24h": float(retail_24h),
        "news_sent_24h": float(news_24h),
        "composite_sentiment_score": float(composite),
        "divergence_retail_vs_news_24h": float(div_24h),
        "divergence_retail_vs_news_fast": float(div_fast),
        "hours_since_last_social": float(hours_since_social) if np.isfinite(hours_since_social) else np.nan,
        "hours_since_last_news": float(hours_since_news) if np.isfinite(hours_since_news) else np.nan,
        "sarcasm_rate_24h": float(sarcasm_24h),
        "hype_intensity_24h": float(hype_24h),
        "sentiment_dispersion_24h": float(disp_24h),
        "prevailing_direction": prev_dir,
        "has_social_history": bool(has_social),
        "has_news_history": bool(has_news),
        "pure_forum_sent_24h": float(pure_forum_24h) if np.isfinite(pure_forum_24h) else np.nan,
        "divergence_forum_vs_news_24h": float(div_forum) if np.isfinite(div_forum) else np.nan,
        "holder_vs_nonholder_spread_24h": float(holder_spread_24h) if np.isfinite(holder_spread_24h) else np.nan,
        "community_pt_median_24h": float(pt_median) if np.isfinite(pt_median) else np.nan,
        "community_pt_implied_upside": float(pt_upside) if np.isfinite(pt_upside) else np.nan,
        "community_pt_vs_resistance_1_atr": float(pt_vs_res1_atr) if np.isfinite(pt_vs_res1_atr) else np.nan,
        "community_pt_count_24h": int(pt_count),
        "has_forum_history": bool(has_forum),
        "has_verified_holder_signal": bool(has_holder_sig),
    }


def _prepare_scored_arrays(scored_events_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scored = scored_events_df[scored_events_df["is_llm_scored"]].copy()
    if scored.empty:
        empty = {
            "sec": np.array([], dtype=np.float64),
            "sentiment": np.array([], dtype=np.float64),
            "weight": np.array([], dtype=np.float64),
            "source": np.array([], dtype=object),
            "sarcastic": np.array([], dtype=np.float64),
            "hype": np.array([], dtype=np.float64),
            "verified": np.array([], dtype=np.float64),
            "price_target": np.array([], dtype=np.float64),
        }
        return empty, dict(empty)

    scored = scored.sort_values("effective_event_utc")

    def _pack(sub: pd.DataFrame) -> dict[str, np.ndarray]:
        if sub.empty:
            return {
                "sec": np.array([], dtype=np.float64),
                "sentiment": np.array([], dtype=np.float64),
                "weight": np.array([], dtype=np.float64),
                "source": np.array([], dtype=object),
                "sarcastic": np.array([], dtype=np.float64),
                "hype": np.array([], dtype=np.float64),
                "verified": np.array([], dtype=np.float64),
                "price_target": np.array([], dtype=np.float64),
            }
        return {
            "sec": _series_to_unix_seconds(sub["effective_event_utc"]),
            "sentiment": sub["sentiment_score"].to_numpy(dtype=np.float64),
            "weight": sub["impact_weight"].to_numpy(dtype=np.float64),
            "source": sub["source"].astype(str).to_numpy(dtype=object),
            "sarcastic": sub["is_sarcastic"].fillna(False).astype(bool).to_numpy(dtype=np.float64),
            "hype": sub["hype_score"].fillna(0.0).to_numpy(dtype=np.float64),
            "verified": sub["is_holder_verified"].fillna(False).astype(bool).to_numpy(dtype=np.float64),
            "price_target": sub["price_target_mentioned"].to_numpy(dtype=np.float64),
        }

    is_news = scored["event_type"].astype(str) == "news"
    return _pack(scored[~is_news]), _pack(scored[is_news])


def _build_top_catalysts_json(
    scored_events_df: pd.DataFrame,
    event_type: str,
    as_of_sec: float,
    cutoff_sec: float,
    half_life_hours: float,
    max_items: int,
) -> str:
    """Build exact 13-key CatalystSummary JSON array sorted by decayed impact_weight DESC."""
    sub = scored_events_df[
        scored_events_df["is_llm_scored"] & (scored_events_df["event_type"].astype(str) == event_type)
    ].copy()
    if sub.empty:
        return "[]"

    ev_sec = _series_to_unix_seconds(sub["effective_event_utc"])
    in_window = (ev_sec >= cutoff_sec) & (ev_sec <= as_of_sec)
    sub = sub.iloc[np.flatnonzero(in_window)].copy()
    if sub.empty:
        return "[]"

    ev_sec_win = ev_sec[in_window]
    dh_hours = np.maximum((as_of_sec - ev_sec_win) / 3600.0, 0.0)
    decayed_w = sub["impact_weight"].to_numpy(dtype=np.float64) * np.exp(-(_LN2 / half_life_hours) * dh_hours)
    sub["_decayed_w"] = decayed_w
    sub["_ev_sec"] = ev_sec_win

    sub = sub.sort_values(
        by=["_decayed_w", "_ev_sec", "event_id"],
        ascending=[False, False, True],
    ).head(max_items)

    catalysts: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        eff_iso = pd.Timestamp(row["effective_event_utc"]).tz_convert("UTC").isoformat()
        item = CatalystSummary(
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            source=str(row["source"]),
            channel_or_domain=str(row["channel_or_domain"]),
            effective_event_utc=eff_iso,
            sentiment_score=float(row["sentiment_score"]),
            direction=str(row["direction"]),  # type: ignore[arg-type]
            is_sarcastic=bool(row["is_sarcastic"]),
            hype_score=float(row["hype_score"]),
            catalyst_type=str(row["catalyst_type"]),  # type: ignore[arg-type]
            english_summary=str(row["english_summary"]),
            url=str(row["url"]),
            impact_weight=float(row["_decayed_w"]),
        )
        catalysts.append(item.model_dump())

    return json.dumps(catalysts, ensure_ascii=False)


def compute_alpha_factors_and_snapshot(
    ticker: str,
    exchange: str,
    company_en: str,
    company_local: str,
    bars_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    full_corpus_df: pd.DataFrame,
    scored_events_df: pd.DataFrame,
    cfg: Phase2Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine causal technicals, full-corpus stats, and dual-horizon decayed alpha factors
    into the ticker's canonical `features_1h` DataFrame and `live_snapshot` dict."""
    bars = bars_df.sort_values("bar_close_utc").reset_index(drop=True)
    tech = tech_df.sort_values("bar_close_utc").reset_index(drop=True)
    fc = full_corpus_df.sort_values("bar_close_utc").reset_index(drop=True)

    scored_social, scored_news = _prepare_scored_arrays(scored_events_df)

    bar_closes_sec = _series_to_unix_seconds(bars["bar_close_utc"])
    closes = bars["close"].to_numpy(dtype=np.float64)
    res1_arr = tech["resistance_1"].to_numpy(dtype=np.float64)
    atr_arr = tech["atr_14"].to_numpy(dtype=np.float64)

    n_bars = len(bars)
    alpha_rows: list[dict[str, Any]] = []
    for t in range(n_bars):
        row_vals = _evaluate_decayed_sentiment_at_time(
            target_sec=float(bar_closes_sec[t]),
            exchange=exchange,
            close_price=float(closes[t]),
            resistance_1=float(res1_arr[t]),
            atr_14=float(atr_arr[t]),
            scored_social=scored_social,
            scored_news=scored_news,
            cfg=cfg,
        )
        alpha_rows.append(row_vals)

    alpha_df = pd.DataFrame(alpha_rows)

    # Assemble canonical features_1h columns in exact schema order
    features_1h = pd.DataFrame(
        {
            # Keys & Targets (13)
            "ticker": bars["ticker"].astype("string"),
            "exchange": bars["exchange"].astype("string"),
            "trading_date_local": bars["trading_date_local"],
            "bar_open_utc": bars["bar_open_utc"],
            "bar_close_utc": bars["bar_close_utc"],
            "window_role": bars["window_role"].astype("string"),
            "open": bars["open"].astype("float64"),
            "high": bars["high"].astype("float64"),
            "low": bars["low"].astype("float64"),
            "close": bars["close"].astype("float64"),
            "volume": bars["volume"].astype("int64"),
            "next_close": bars["next_close"].astype("float64"),
            "target_log_return_1h": bars["target_log_return_1h"].astype("float64"),
            # Causal Technicals (12)
            "log_ret_1h": tech["log_ret_1h"].astype("float64"),
            "log_ret_3h": tech["log_ret_3h"].astype("float64"),
            "log_ret_6h": tech["log_ret_6h"].astype("float64"),
            "realized_vol_12h": tech["realized_vol_12h"].astype("float64"),
            "atr_14": tech["atr_14"].astype("float64"),
            "atr_pct": tech["atr_pct"].astype("float64"),
            "rsi_14": tech["rsi_14"].astype("float64"),
            "vwap_24h": tech["vwap_24h"].astype("float64"),
            "price_vs_vwap_pct": tech["price_vs_vwap_pct"].astype("float64"),
            "volume_zscore_24h": tech["volume_zscore_24h"].astype("float64"),
            "bar_range_pct": tech["bar_range_pct"].astype("float64"),
            "close_pos_in_bar": tech["close_pos_in_bar"].astype("float64"),
            # Causal KDE S/R (14)
            "poc_price": tech["poc_price"].astype("float64"),
            "val_price": tech["val_price"].astype("float64"),
            "vah_price": tech["vah_price"].astype("float64"),
            "support_1": tech["support_1"].astype("float64"),
            "resistance_1": tech["resistance_1"].astype("float64"),
            "dist_to_support_1_atr": tech["dist_to_support_1_atr"].astype("float64"),
            "dist_to_resistance_1_atr": tech["dist_to_resistance_1_atr"].astype("float64"),
            "sr_risk_reward_ratio": tech["sr_risk_reward_ratio"].astype("float64"),
            "dist_to_poc_atr": tech["dist_to_poc_atr"].astype("float64"),
            "in_value_area": tech["in_value_area"].astype("boolean"),
            "support_2": tech["support_2"].astype("float64"),
            "support_3": tech["support_3"].astype("float64"),
            "resistance_2": tech["resistance_2"].astype("float64"),
            "resistance_3": tech["resistance_3"].astype("float64"),
            # Core Decayed Sentiment & Divergence (15)
            "retail_sent_6h": alpha_df["retail_sent_6h"].astype("float64"),
            "retail_sent_24h": alpha_df["retail_sent_24h"].astype("float64"),
            "news_sent_24h": alpha_df["news_sent_24h"].astype("float64"),
            "composite_sentiment_score": alpha_df["composite_sentiment_score"].astype("float64"),
            "divergence_retail_vs_news_24h": alpha_df["divergence_retail_vs_news_24h"].astype("float64"),
            "divergence_retail_vs_news_fast": alpha_df["divergence_retail_vs_news_fast"].astype("float64"),
            "hours_since_last_social": alpha_df["hours_since_last_social"].astype("float64"),
            "hours_since_last_news": alpha_df["hours_since_last_news"].astype("float64"),
            "sarcasm_rate_24h": alpha_df["sarcasm_rate_24h"].astype("float64"),
            "hype_intensity_24h": alpha_df["hype_intensity_24h"].astype("float64"),
            "sentiment_dispersion_24h": alpha_df["sentiment_dispersion_24h"].astype("float64"),
            "net_crowd_approval_24h": fc["net_crowd_approval_24h"].astype("float64"),
            "prevailing_direction": alpha_df["prevailing_direction"].astype("string"),
            "has_social_history": alpha_df["has_social_history"].astype("bool"),
            "has_news_history": alpha_df["has_news_history"].astype("bool"),
            # Full-Corpus Activity & Velocity (6)
            "social_mention_count": fc["social_mention_count"].astype("int32"),
            "news_mention_count": fc["news_mention_count"].astype("int32"),
            "bar_engagement_sum": fc["bar_engagement_sum"].astype("float64"),
            "mention_velocity_zscore_24h": fc["mention_velocity_zscore_24h"].astype("float64"),
            "engagement_velocity_zscore_24h": fc["engagement_velocity_zscore_24h"].astype("float64"),
            "mention_acceleration": fc["mention_acceleration"].astype("float64"),
            # Optional Sub-Streams & Community PT (10)
            "pure_forum_sent_24h": alpha_df["pure_forum_sent_24h"].astype("float64"),
            "divergence_forum_vs_news_24h": alpha_df["divergence_forum_vs_news_24h"].astype("float64"),
            "holder_vs_nonholder_spread_24h": alpha_df["holder_vs_nonholder_spread_24h"].astype("float64"),
            "verified_holder_share_24h": fc["verified_holder_share_24h"].astype("float64"),
            "community_pt_median_24h": alpha_df["community_pt_median_24h"].astype("float64"),
            "community_pt_implied_upside": alpha_df["community_pt_implied_upside"].astype("float64"),
            "community_pt_vs_resistance_1_atr": alpha_df["community_pt_vs_resistance_1_atr"].astype("float64"),
            "community_pt_count_24h": alpha_df["community_pt_count_24h"].astype("int32"),
            "has_forum_history": alpha_df["has_forum_history"].astype("bool"),
            "has_verified_holder_signal": alpha_df["has_verified_holder_signal"].astype("bool"),
        }
    )

    # --- Build live_snapshot row (incorporating post_window_live events) ---
    last_bar = features_1h.iloc[-1]
    last_bar_close_utc = pd.Timestamp(last_bar["bar_close_utc"]).tz_convert("UTC")
    live_mask = scored_events_df["window_role"].astype(str) == "post_window_live"
    live_event_count = int(live_mask.sum())
    active_10d_mentions = int((scored_events_df["window_role"].astype(str) == "active_10d").sum())

    if live_event_count > 0:
        max_live_utc = pd.to_datetime(
            scored_events_df.loc[live_mask, "effective_event_utc"], utc=True
        ).max()
        as_of_utc = max(last_bar_close_utc, pd.Timestamp(max_live_utc).tz_convert("UTC"))
    else:
        as_of_utc = last_bar_close_utc

    as_of_sec = _ts_to_unix_seconds(as_of_utc)
    live_alpha = _evaluate_decayed_sentiment_at_time(
        target_sec=as_of_sec,
        exchange=exchange,
        close_price=float(last_bar["close"]),
        resistance_1=float(last_bar["resistance_1"]),
        atr_14=float(last_bar["atr_14"]),
        scored_social=scored_social,
        scored_news=scored_news,
        cfg=cfg,
    )

    # Past 24h before last_bar_close_utc + live window
    af = cfg.alpha_factors
    cutoff_sec = _ts_to_unix_seconds(last_bar_close_utc) - (af.catalyst_lookback_hours * 3600.0)
    top_social_json = _build_top_catalysts_json(
        scored_events_df=scored_events_df,
        event_type="social",
        as_of_sec=as_of_sec,
        cutoff_sec=cutoff_sec,
        half_life_hours=af.social_slow_half_life_hours,
        max_items=af.max_top_catalysts_per_type,
    )
    top_news_json = _build_top_catalysts_json(
        scored_events_df=scored_events_df,
        event_type="news",
        as_of_sec=as_of_sec,
        cutoff_sec=cutoff_sec,
        half_life_hours=af.news_half_life_hours,
        max_items=af.max_top_catalysts_per_type,
    )

    def _opt_float(val: Any) -> float | None:
        if val is None or pd.isna(val):
            return None
        return float(val)

    live_snapshot_row: dict[str, Any] = {
        "ticker": ticker,
        "exchange": exchange,
        "company_name": f"{company_en} ({company_local})",
        "as_of_utc": as_of_utc,
        "last_bar_close_utc": last_bar_close_utc,
        "current_price": float(last_bar["close"]),
        "live_event_count": live_event_count,
        "total_active_10d_mentions": active_10d_mentions,
        "poc_price": float(last_bar["poc_price"]),
        "val_price": float(last_bar["val_price"]),
        "vah_price": float(last_bar["vah_price"]),
        "support_1": float(last_bar["support_1"]),
        "support_2": _opt_float(last_bar["support_2"]),
        "support_3": _opt_float(last_bar["support_3"]),
        "resistance_1": float(last_bar["resistance_1"]),
        "resistance_2": _opt_float(last_bar["resistance_2"]),
        "resistance_3": _opt_float(last_bar["resistance_3"]),
        "sr_risk_reward_ratio": float(last_bar["sr_risk_reward_ratio"]),
        "atr_14": float(last_bar["atr_14"]),
        "rsi_14": float(last_bar["rsi_14"]),
        "retail_sent_6h": float(live_alpha["retail_sent_6h"]),
        "retail_sent_24h": float(live_alpha["retail_sent_24h"]),
        "news_sent_24h": float(live_alpha["news_sent_24h"]),
        "composite_sentiment_score": float(live_alpha["composite_sentiment_score"]),
        "divergence_retail_vs_news_24h": float(live_alpha["divergence_retail_vs_news_24h"]),
        "holder_vs_nonholder_spread_24h": _opt_float(live_alpha["holder_vs_nonholder_spread_24h"]),
        "mention_velocity_zscore_24h": float(last_bar["mention_velocity_zscore_24h"]),
        "sarcasm_rate_24h": float(live_alpha["sarcasm_rate_24h"]),
        "hype_intensity_24h": float(live_alpha["hype_intensity_24h"]),
        "community_pt_median_24h": _opt_float(live_alpha["community_pt_median_24h"]),
        "prevailing_direction": str(live_alpha["prevailing_direction"]),
        "top_social_catalysts_json": top_social_json,
        "top_news_catalysts_json": top_news_json,
    }

    return features_1h, live_snapshot_row


def compute_impact_weight_coverage_pct(scored_events_df: pd.DataFrame, cfg: Phase2Config) -> float:
    """Compute mean per-bar proxy impact-weight coverage % of routed posts vs all posts."""
    if scored_events_df.empty:
        return 100.0
    af = cfg.alpha_factors
    df = scored_events_df.copy()
    is_news = (df["event_type"].astype(str) == "news").to_numpy()
    reach = np.maximum(df["effective_reach"].fillna(1).to_numpy(dtype=np.float64), 0.0)
    eng = (
        np.maximum(df["likes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        + np.maximum(df["dislikes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        + np.maximum(df["reposts"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        + np.maximum(df["comments"].fillna(0).to_numpy(dtype=np.float64), 0.0)
    )
    verified = df["is_holder_verified"].fillna(False).astype(bool).to_numpy(dtype=np.float64)
    is_repost = df["is_cross_channel_repost"].fillna(False).astype(bool).to_numpy()
    repost_mult = np.where(is_repost, af.repost_weight_factor, 1.0)
    cred = df["source_credibility_weight"].fillna(1.0).to_numpy(dtype=np.float64)

    proxy_w = np.where(
        is_news,
        cred,
        np.log1p(reach) * (1.0 + np.log1p(eng)) * (1.0 + af.holder_verified_weight_boost * verified) * repost_mult,
    )
    df["_proxy_w"] = proxy_w
    df["_bar_key"] = df["assigned_bar_close_utc"].astype(str)

    coverages: list[float] = []
    for _, grp in df.groupby("_bar_key"):
        tot = float(grp["_proxy_w"].sum())
        if tot <= 0.0:
            continue
        scored_sum = float(grp.loc[grp["is_llm_scored"], "_proxy_w"].sum())
        coverages.append(100.0 * scored_sum / tot)

    return round(float(np.mean(coverages)), 2) if coverages else 100.0


def verify_ticker_phase2_outputs(
    ticker: str,
    bars_df: pd.DataFrame,
    raw_events_df: pd.DataFrame,
    scored_events_df: pd.DataFrame,
    features_1h_df: pd.DataFrame,
    live_snapshot_row: dict[str, Any],
    smoke_mode: bool = False,
) -> bool:
    """Enforce mandatory per-ticker Phase 2 contract & zero-leakage assertions."""
    assert len(scored_events_df) == len(raw_events_df), (
        f"[{ticker}] events_scored row count {len(scored_events_df)} != events_aligned {len(raw_events_df)}"
    )
    assert len(features_1h_df) == len(bars_df), (
        f"[{ticker}] features_1h row count {len(features_1h_df)} != bars_1h {len(bars_df)}"
    )

    # Scored rows invariants
    scored_mask = scored_events_df["is_llm_scored"].to_numpy(dtype=bool)
    scored_sub = scored_events_df[scored_mask]
    unscored_sub = scored_events_df[~scored_mask]

    if not scored_sub.empty:
        s_vals = scored_sub["sentiment_score"].to_numpy(dtype=np.float64)
        w_vals = scored_sub["impact_weight"].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(s_vals)), f"[{ticker}] Non-finite sentiment_score in scored rows"
        assert np.all((s_vals >= -1.0) & (s_vals <= 1.0)), f"[{ticker}] sentiment_score out of [-1, 1]"
        assert scored_sub["english_summary"].notna().all(), f"[{ticker}] Null english_summary in scored rows"
        assert np.all(np.isfinite(w_vals)), f"[{ticker}] Non-finite impact_weight in scored rows"

    if not unscored_sub.empty:
        assert unscored_sub["sentiment_score"].isna().all(), f"[{ticker}] Unscored row has non-null sentiment_score"
        assert unscored_sub["impact_weight"].isna().all(), f"[{ticker}] Unscored row has non-null impact_weight"

    # Active 10d zero-NaN invariants (relaxed for sentiment history only in smoke mode if tiny sample had no warmup)
    active_df = features_1h_df[features_1h_df["window_role"] == "active_10d"]
    assert len(active_df) > 0, f"[{ticker}] Zero active_10d bars in features_1h"

    for col in ACTIVE_10D_NON_NULL_COLUMNS:
        if smoke_mode and col in ("hours_since_last_social", "hours_since_last_news"):
            continue
        null_cnt = int(active_df[col].isna().sum())
        assert null_cnt == 0, f"[{ticker}] Column '{col}' has {null_cnt} NaNs on active_10d bars"

    # Optional sparse sub-stream companion flag validation
    no_forum = ~features_1h_df["has_forum_history"]
    yes_forum = features_1h_df["has_forum_history"]
    yes_forum_and_news = yes_forum & features_1h_df["has_news_history"]
    assert features_1h_df.loc[no_forum, "pure_forum_sent_24h"].isna().all(), (
        f"[{ticker}] pure_forum_sent_24h non-null when has_forum_history=False"
    )
    assert features_1h_df.loc[~yes_forum_and_news, "divergence_forum_vs_news_24h"].isna().all(), (
        f"[{ticker}] divergence_forum_vs_news_24h non-null when forum or news history is absent"
    )
    assert features_1h_df.loc[yes_forum, "pure_forum_sent_24h"].notna().all(), (
        f"[{ticker}] pure_forum_sent_24h null when has_forum_history=True"
    )
    assert features_1h_df.loc[yes_forum_and_news, "divergence_forum_vs_news_24h"].notna().all(), (
        f"[{ticker}] divergence_forum_vs_news_24h null when both forum and news history exist"
    )

    # Value area, technical bounds, and verified_holder_share_24h invariants
    assert (active_df["val_price"] <= active_df["poc_price"] + 1e-9).all(), (
        f"[{ticker}] val_price > poc_price on active_10d"
    )
    assert (active_df["poc_price"] <= active_df["vah_price"] + 1e-9).all(), (
        f"[{ticker}] poc_price > vah_price on active_10d"
    )
    assert ((active_df["rsi_14"] >= 0.0) & (active_df["rsi_14"] <= 100.0)).all(), (
        f"[{ticker}] rsi_14 out of [0, 100] on active_10d"
    )
    assert (active_df["atr_14"] > 0.0).all(), f"[{ticker}] atr_14 <= 0 on active_10d"
    assert ((active_df["sarcasm_rate_24h"] >= 0.0) & (active_df["sarcasm_rate_24h"] <= 1.0 + 1e-9)).all(), (
        f"[{ticker}] sarcasm_rate_24h out of [0, 1] on active_10d"
    )
    assert ((active_df["hype_intensity_24h"] >= 0.0) & (active_df["hype_intensity_24h"] <= 1.0 + 1e-9)).all(), (
        f"[{ticker}] hype_intensity_24h out of [0, 1] on active_10d"
    )
    assert (active_df["sentiment_dispersion_24h"] >= 0.0).all(), (
        f"[{ticker}] sentiment_dispersion_24h < 0 on active_10d"
    )

    exchange = str(features_1h_df["exchange"].iloc[0]).upper()
    vhs = features_1h_df["verified_holder_share_24h"]
    if exchange == "HKEX":
        assert vhs.isna().all(), f"[{ticker}] HKEX must have all-NaN verified_holder_share_24h"
    else:
        vhs_valid = vhs.dropna()
        if not vhs_valid.empty:
            assert ((vhs_valid >= 0.0) & (vhs_valid <= 1.0 + 1e-9)).all(), (
                f"[{ticker}] KRX verified_holder_share_24h out of [0, 1]"
            )

    no_vh = ~features_1h_df["has_verified_holder_signal"]
    yes_vh = features_1h_df["has_verified_holder_signal"]
    assert features_1h_df.loc[no_vh, "holder_vs_nonholder_spread_24h"].isna().all(), (
        f"[{ticker}] holder_vs_nonholder_spread_24h non-null when has_verified_holder_signal=False"
    )
    assert features_1h_df.loc[yes_vh, "holder_vs_nonholder_spread_24h"].notna().all(), (
        f"[{ticker}] holder_vs_nonholder_spread_24h null when has_verified_holder_signal=True"
    )

    no_pt = features_1h_df["community_pt_count_24h"] == 0
    yes_pt = features_1h_df["community_pt_count_24h"] > 0
    assert features_1h_df.loc[no_pt, "community_pt_median_24h"].isna().all(), (
        f"[{ticker}] community_pt_median_24h non-null when community_pt_count_24h == 0"
    )
    assert features_1h_df.loc[yes_pt, "community_pt_median_24h"].notna().all(), (
        f"[{ticker}] community_pt_median_24h null when community_pt_count_24h > 0"
    )

    # Live snapshot as_of_utc >= last_bar_close_utc
    assert live_snapshot_row["as_of_utc"] >= live_snapshot_row["last_bar_close_utc"], (
        f"[{ticker}] live_snapshot as_of_utc < last_bar_close_utc"
    )
    # Validate catalyst JSON arrays parse cleanly
    for key in ("top_social_catalysts_json", "top_news_catalysts_json"):
        parsed = json.loads(live_snapshot_row[key])
        assert isinstance(parsed, list), f"[{ticker}] {key} did not parse to list"
        for item in parsed:
            CatalystSummary.model_validate(item)

    return True
