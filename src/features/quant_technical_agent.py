"""QuantTechnicalAgent: Strictly causal Wilder technical indicators and ATR-scaled VPVR / Gaussian KDE S&R."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from src.features.schemas import Phase2Config, QuantTechnicalConfig


def _compute_wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rsi = np.full(len(close), np.nan, dtype=np.float64)
    ag = avg_gain.to_numpy(dtype=np.float64)
    al = avg_loss.to_numpy(dtype=np.float64)
    valid = np.isfinite(ag) & np.isfinite(al)
    for i in np.flatnonzero(valid):
        if ag[i] == 0.0 and al[i] == 0.0:
            rsi[i] = 50.0
        elif al[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = ag[i] / al[i]
            rsi[i] = np.clip(100.0 - (100.0 / (1.0 + rs)), 0.0, 100.0)
    return pd.Series(rsi, index=close.index, dtype="float64")


def _compute_causal_kde_for_window(
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    current_close: float,
    atr_val: float,
    qcfg: QuantTechnicalConfig,
) -> dict[str, float | bool | None]:
    """Compute ATR-scaled VPVR / Gaussian KDE POC, 70% Value Area (VAL/VAH),
    and top-3 Support/Resistance levels using strictly the provided causal window."""
    n_pts = qcfg.vpvr_points_per_bar
    tri_weights = np.asarray(qcfg.vpvr_weights, dtype=np.float64)
    if len(tri_weights) != n_pts:
        raise ValueError(f"vpvr_weights length {len(tri_weights)} != vpvr_points_per_bar {n_pts}")

    price_samples: list[float] = []
    weight_samples: list[float] = []
    for lo, hi, cl, vol in zip(lows, highs, closes, volumes):
        if vol <= 0.0 or hi <= lo:
            price_samples.append(float(cl))
            weight_samples.append(1e-6)
        else:
            pts = np.linspace(lo, hi, n_pts)
            ws = tri_weights * float(vol)
            price_samples.extend(pts.tolist())
            weight_samples.extend(ws.tolist())

    prices_arr = np.asarray(price_samples, dtype=np.float64)
    weights_arr = np.asarray(weight_samples, dtype=np.float64)

    win_low = float(np.min(lows))
    win_high = float(np.max(highs))
    eff_atr = max(float(atr_val), 1e-8)

    grid_lo = win_low - qcfg.kde_grid_pad_atr * eff_atr
    grid_hi = win_high + qcfg.kde_grid_pad_atr * eff_atr
    if grid_hi <= grid_lo:
        grid_hi = grid_lo + max(eff_atr, 1e-4)
    grid = np.linspace(grid_lo, grid_hi, qcfg.kde_grid_points)
    grid_step = float(grid[1] - grid[0])

    # Adaptive ATR-scaled bandwidth using volume-weighted std (matches gaussian_kde's weighted covariance)
    w_sum = float(np.sum(weights_arr))
    if w_sum > 0.0:
        w_mean = float(np.sum(weights_arr * prices_arr) / w_sum)
        w_var = float(np.sum(weights_arr * ((prices_arr - w_mean) ** 2)) / w_sum)
        price_std = math.sqrt(max(0.0, w_var))
    else:
        price_std = float(np.std(prices_arr))
    bw = float(np.clip((qcfg.kde_bw_atr_factor * eff_atr) / max(price_std, 1e-8), qcfg.kde_bw_min, qcfg.kde_bw_max))

    # Guard degenerate constant-price window (only possible in synthetic edge tests)
    if price_std < 1e-10:
        sigma = max(eff_atr * 0.5, 1e-4)
        density = np.exp(-0.5 * ((grid - current_close) / sigma) ** 2)
    else:
        kde = gaussian_kde(prices_arr, weights=weights_arr, bw_method=bw)
        density = kde(grid)

    density = np.maximum(density, 0.0)
    total_mass = float(np.sum(density))
    if total_mass <= 0.0:
        density = np.ones_like(grid)
        total_mass = float(len(grid))

    # POC (Point of Control)
    poc_idx = int(np.argmax(density))
    poc_price = float(grid[poc_idx])

    # Deterministic 70% Value Area expansion from POC
    left = poc_idx
    right = poc_idx
    cum_mass = float(density[poc_idx])
    target_mass = qcfg.value_area_pct * total_mass
    n_grid = len(grid)
    while cum_mass < target_mass and (left > 0 or right < n_grid - 1):
        left_val = float(density[left - 1]) if left > 0 else -1.0
        right_val = float(density[right + 1]) if right < n_grid - 1 else -1.0
        if left_val >= right_val and left > 0:
            left -= 1
            cum_mass += float(density[left])
        elif right < n_grid - 1:
            right += 1
            cum_mass += float(density[right])
        else:
            left -= 1
            cum_mass += float(density[left])

    val_price = float(grid[left])
    vah_price = float(grid[right])
    in_value_area = bool(val_price <= current_close <= vah_price)

    # Institutional volume node peak detection
    min_dist_pts = max(3, int(qcfg.peak_min_dist_atr * eff_atr / max(grid_step, 1e-12)))
    prominence = qcfg.peak_prominence_frac * float(np.max(density))
    peak_indices, _ = find_peaks(density, prominence=prominence, distance=min_dist_pts)
    peak_prices = grid[peak_indices] if len(peak_indices) > 0 else np.array([], dtype=np.float64)

    # Supports strictly < close (nearest first => descending)
    supports = np.sort(peak_prices[peak_prices < current_close])[::-1]
    if len(supports) >= 1:
        support_1 = float(supports[0])
        support_2 = float(supports[1]) if len(supports) >= 2 else np.nan
        support_3 = float(supports[2]) if len(supports) >= 3 else np.nan
    else:
        support_1 = win_low
        support_2 = np.nan
        support_3 = np.nan

    # Resistances strictly > close (nearest first => ascending)
    resistances = np.sort(peak_prices[peak_prices > current_close])
    if len(resistances) >= 1:
        resistance_1 = float(resistances[0])
        resistance_2 = float(resistances[1]) if len(resistances) >= 2 else np.nan
        resistance_3 = float(resistances[2]) if len(resistances) >= 3 else np.nan
    else:
        resistance_1 = win_high
        resistance_2 = np.nan
        resistance_3 = np.nan

    dist_to_support_1_atr = (current_close - support_1) / eff_atr
    dist_to_resistance_1_atr = (resistance_1 - current_close) / eff_atr
    sr_rr = (resistance_1 - current_close) / max(
        current_close - support_1,
        qcfg.sr_rr_denom_atr_floor * eff_atr,
    )
    dist_to_poc_atr = (current_close - poc_price) / eff_atr

    return {
        "poc_price": poc_price,
        "val_price": val_price,
        "vah_price": vah_price,
        "support_1": support_1,
        "resistance_1": resistance_1,
        "dist_to_support_1_atr": float(dist_to_support_1_atr),
        "dist_to_resistance_1_atr": float(dist_to_resistance_1_atr),
        "sr_risk_reward_ratio": float(sr_rr),
        "dist_to_poc_atr": float(dist_to_poc_atr),
        "in_value_area": in_value_area,
        "support_2": support_2,
        "support_3": support_3,
        "resistance_2": resistance_2,
        "resistance_3": resistance_3,
    }


def compute_quant_technical_features(
    bars_df: pd.DataFrame,
    cfg: Phase2Config,
) -> pd.DataFrame:
    """Compute all 12 causal technical indicators + 14 causal KDE S/R features for one ticker.

    Strict Causality Rule:
      Every indicator and KDE level at bar t uses ONLY bars [max(0, t - max_window + 1) .. t].
    """
    qcfg = cfg.quant_technical
    df = bars_df.sort_values("bar_close_utc").reset_index(drop=True).copy()
    n = len(df)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # 1. Causal companion technicals (12 cols)
    log_close = np.log(close)
    log_ret_1h = log_close - log_close.shift(1)
    log_ret_3h = log_close - log_close.shift(3)
    log_ret_6h = log_close - log_close.shift(6)
    realized_vol_12h = log_ret_1h.rolling(
        window=qcfg.realized_vol_window,
        min_periods=qcfg.realized_vol_min_periods,
    ).std(ddof=1)

    atr_14 = _compute_wilder_atr(high, low, close, period=qcfg.atr_period)
    atr_pct = atr_14 / close
    rsi_14 = _compute_wilder_rsi(close, period=qcfg.rsi_period)

    typical_price = (high + low + close) / 3.0
    tp_vol = (typical_price * volume).rolling(window=qcfg.vwap_window, min_periods=1).sum()
    vol_sum = volume.rolling(window=qcfg.vwap_window, min_periods=1).sum()
    vwap_24h = np.where(vol_sum > 0.0, tp_vol / vol_sum, close)
    vwap_24h_s = pd.Series(vwap_24h, index=df.index, dtype="float64")
    price_vs_vwap_pct = (close - vwap_24h_s) / vwap_24h_s

    log_vol = np.log1p(np.maximum(volume, 0.0))
    vol_mean = log_vol.rolling(
        window=qcfg.volume_zscore_window,
        min_periods=qcfg.volume_zscore_min_periods,
    ).mean()
    vol_std = log_vol.rolling(
        window=qcfg.volume_zscore_window,
        min_periods=qcfg.volume_zscore_min_periods,
    ).std(ddof=1)
    vol_z = np.where(
        vol_std.isna(),
        np.nan,
        np.where(
            vol_std < 1e-8,
            0.0,
            np.clip((log_vol - vol_mean) / vol_std, -qcfg.zscore_clip, qcfg.zscore_clip),
        ),
    )

    bar_range_pct = (high - low) / close
    close_pos_in_bar = (close - low) / np.maximum((high - low).to_numpy(dtype=np.float64), 1e-8)

    out = pd.DataFrame(
        {
            "bar_close_utc": df["bar_close_utc"],
            "log_ret_1h": log_ret_1h.to_numpy(dtype=np.float64),
            "log_ret_3h": log_ret_3h.to_numpy(dtype=np.float64),
            "log_ret_6h": log_ret_6h.to_numpy(dtype=np.float64),
            "realized_vol_12h": realized_vol_12h.to_numpy(dtype=np.float64),
            "atr_14": atr_14.to_numpy(dtype=np.float64),
            "atr_pct": atr_pct.to_numpy(dtype=np.float64),
            "rsi_14": rsi_14.to_numpy(dtype=np.float64),
            "vwap_24h": vwap_24h_s.to_numpy(dtype=np.float64),
            "price_vs_vwap_pct": price_vs_vwap_pct.to_numpy(dtype=np.float64),
            "volume_zscore_24h": vol_z.astype(np.float64),
            "bar_range_pct": bar_range_pct.to_numpy(dtype=np.float64),
            "close_pos_in_bar": close_pos_in_bar.astype(np.float64),
        }
    )

    # 2. Causal ATR-scaled Gaussian KDE VPVR S&R (14 cols)
    kde_float_cols = [
        "poc_price",
        "val_price",
        "vah_price",
        "support_1",
        "resistance_1",
        "dist_to_support_1_atr",
        "dist_to_resistance_1_atr",
        "sr_risk_reward_ratio",
        "dist_to_poc_atr",
        "support_2",
        "support_3",
        "resistance_2",
        "resistance_3",
    ]
    kde_arrays: dict[str, np.ndarray] = {c: np.full(n, np.nan, dtype=np.float64) for c in kde_float_cols}
    in_va_list: list[bool | None] = [None] * n

    lows_np = low.to_numpy(dtype=np.float64)
    highs_np = high.to_numpy(dtype=np.float64)
    closes_np = close.to_numpy(dtype=np.float64)
    vols_np = volume.to_numpy(dtype=np.float64)
    atr_np = atr_14.to_numpy(dtype=np.float64)

    for t in range(n):
        start_idx = max(0, t - qcfg.max_window + 1)
        win_len = t - start_idx + 1
        if win_len < qcfg.min_bars or not np.isfinite(atr_np[t]):
            continue
        res = _compute_causal_kde_for_window(
            lows=lows_np[start_idx : t + 1],
            highs=highs_np[start_idx : t + 1],
            closes=closes_np[start_idx : t + 1],
            volumes=vols_np[start_idx : t + 1],
            current_close=float(closes_np[t]),
            atr_val=float(atr_np[t]),
            qcfg=qcfg,
        )
        for col in kde_float_cols:
            kde_arrays[col][t] = float(res[col]) if res[col] is not None else np.nan
        in_va_list[t] = bool(res["in_value_area"])

    for col in kde_float_cols:
        out[col] = kde_arrays[col]
    out["in_value_area"] = pd.Series(in_va_list, dtype="boolean")

    return out
