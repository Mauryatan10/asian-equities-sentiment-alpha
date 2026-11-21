"""Vectorized backtesting engine, discrete trade gates, turnover slippage modeling,
and multi-exchange tear sheet metrics generator."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.schemas import (
    PHASE3_TEARSHEET_REPORT_PATH,
    PREDICTIONS_OOS_PATH,
    TEARSHEET_METRICS_PATH,
    BacktestConfig,
    Phase3Config,
)

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    rmse_price: float
    rmse_bps: float
    mae_price: float
    mae_bps: float
    mape_pct: float
    theils_u: float
    directional_accuracy_all_pct: float
    directional_hit_rate_active_pct: float | None
    spearman_ic: float


@dataclass
class StrategyMetrics:
    total_net_return_pct: float
    buy_and_hold_return_pct: float
    excess_return_pct: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown_pct: float
    profit_factor: float
    active_trade_bars: int
    position_changes: int
    total_slippage_drag_bps: float


def compute_forecast_metrics(
    actual_prices: np.ndarray,
    pred_prices: np.ndarray,
    actual_returns: np.ndarray,
    pred_returns: np.ndarray,
    prev_prices: np.ndarray,
    positions: np.ndarray | None = None,
) -> ForecastMetrics:
    """Compute scale-invariant price and return forecast metrics + Theil's U and Spearman IC."""
    price_err = pred_prices - actual_prices
    rmse_price = float(np.sqrt(np.mean(price_err**2)))
    mae_price = float(np.mean(np.abs(price_err)))
    mape_pct = float(np.mean(np.abs(price_err) / np.maximum(actual_prices, 1e-8)) * 100.0)

    ret_err = pred_returns - actual_returns
    rmse_bps = float(np.sqrt(np.mean(ret_err**2)) * 10000.0)
    mae_bps = float(np.mean(np.abs(ret_err)) * 10000.0)

    rw_err = prev_prices - actual_prices
    rmse_rw = float(np.sqrt(np.mean(rw_err**2)))
    theils_u = float(rmse_price / max(rmse_rw, 1e-12))

    non_zero = actual_returns != 0.0
    if np.sum(non_zero) > 0:
        dir_acc = float(
            np.mean(np.sign(pred_returns[non_zero]) == np.sign(actual_returns[non_zero]))
            * 100.0
        )
    else:
        dir_acc = 50.0

    hit_rate: float | None = None
    if positions is not None:
        active_mask = (positions != 0) & non_zero
        if np.sum(active_mask) > 0:
            hit_rate = float(
                np.mean(np.sign(positions[active_mask]) == np.sign(actual_returns[active_mask]))
                * 100.0
            )

    if np.all(pred_returns == pred_returns[0]) or np.all(actual_returns == actual_returns[0]):
        spearman_ic = 0.0
    else:
        ic, _ = spearmanr(pred_returns, actual_returns)
        spearman_ic = float(ic) if not np.isnan(ic) else 0.0

    return ForecastMetrics(
        rmse_price=rmse_price,
        rmse_bps=rmse_bps,
        mae_price=mae_price,
        mae_bps=mae_bps,
        mape_pct=mape_pct,
        theils_u=theils_u,
        directional_accuracy_all_pct=dir_acc,
        directional_hit_rate_active_pct=hit_rate,
        spearman_ic=spearman_ic,
    )


def compute_strategy_metrics(
    net_returns: np.ndarray,
    actual_simple_returns: np.ndarray,
    slippage_drags: np.ndarray,
    positions: np.ndarray,
    prev_positions: np.ndarray,
    bars_per_day: float,
) -> StrategyMetrics:
    """Compute financial tear sheet metrics with zero-trade safe defaults."""
    active_bars = int(np.sum(positions != 0))
    pos_changes = int(np.sum(positions != prev_positions))
    slippage_bps = float(np.sum(slippage_drags) * 10000.0)

    # Compound total net return and benchmark return
    total_net = float((np.prod(1.0 + net_returns) - 1.0) * 100.0)
    bh_ret = float((np.prod(1.0 + actual_simple_returns) - 1.0) * 100.0)
    excess_ret = total_net - bh_ret

    # Zero-trade safe defaults
    if active_bars == 0 or np.std(net_returns) < 1e-12:
        sharpe = 0.0
        sortino = 0.0
        profit_factor = 0.0
        max_dd = 0.0
    else:
        ann_factor = np.sqrt(252.0 * bars_per_day)
        mean_r = float(np.mean(net_returns))
        std_r = float(np.std(net_returns))
        sharpe = float((mean_r / max(std_r, 1e-12)) * ann_factor)

        downside = np.minimum(net_returns, 0.0)
        downside_std = float(np.sqrt(np.mean(downside**2)))
        sortino = (
            float((mean_r / max(downside_std, 1e-12)) * ann_factor)
            if downside_std > 1e-12
            else 0.0
        )

        pos_gains = float(np.sum(np.maximum(net_returns, 0.0)))
        neg_losses = float(np.abs(np.sum(np.minimum(net_returns, 0.0))))
        profit_factor = float(
            min(pos_gains / max(neg_losses, 1e-12), 99.99)
            if neg_losses > 0
            else (99.99 if pos_gains > 0 else 0.0)
        )

        # Max drawdown
        wealth = np.cumprod(1.0 + net_returns)
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / np.maximum(peak, 1e-12)
        max_dd = float(np.min(dd) * 100.0)

    return StrategyMetrics(
        total_net_return_pct=total_net,
        buy_and_hold_return_pct=bh_ret,
        excess_return_pct=excess_ret,
        annualized_sharpe=sharpe,
        annualized_sortino=sortino,
        max_drawdown_pct=max_dd,
        profit_factor=profit_factor,
        active_trade_bars=active_bars,
        position_changes=pos_changes,
        total_slippage_drag_bps=slippage_bps,
    )


def generate_discrete_positions(
    pred_ret: np.ndarray,
    pred_huber_ret: np.ndarray,
    pred_lgbm_ret: np.ndarray,
    dist_to_resistance_atr: np.ndarray,
    dist_to_support_atr: np.ndarray,
    hurdle_rate: float,
    sr_buffer: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply legacy discrete position rule {-1, 0, +1} and derive long-only {0, +1}."""
    n = len(pred_ret)
    pos_ls = np.zeros(n, dtype=np.int8)

    long_cond = (
        (pred_ret >= hurdle_rate)
        & (pred_huber_ret > 0.0)
        & (pred_lgbm_ret > 0.0)
        & (dist_to_resistance_atr >= sr_buffer)
    )
    short_cond = (
        (pred_ret <= -hurdle_rate)
        & (pred_huber_ret < 0.0)
        & (pred_lgbm_ret < 0.0)
        & (dist_to_support_atr >= sr_buffer)
    )

    pos_ls[long_cond] = 1
    pos_ls[short_cond] = -1
    pos_lo = np.maximum(pos_ls, 0).astype(np.int8)

    return pos_ls, pos_lo


def compute_snr_positions(
    pred_ret: np.ndarray,
    pred_huber_ret: np.ndarray,
    pred_lgbm_ret: np.ndarray,
    atr_pct: np.ndarray,
    dist_to_resistance_atr: np.ndarray,
    dist_to_support_atr: np.ndarray,
    snr_threshold: float = 1.28,
    sr_buffer: float = 0.25,
    aleatoric_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Epistemic uncertainty & scale-invariant SNR execution gate."""
    n = len(pred_ret)
    pos_ls = np.zeros(n, dtype=np.int8)

    sigma_epistemic = np.abs(pred_lgbm_ret - pred_huber_ret)
    sigma_aleatoric = atr_pct
    se = np.sqrt(sigma_epistemic**2 + (aleatoric_weight * sigma_aleatoric) ** 2)
    snr = pred_ret / np.maximum(se, 1e-8)

    long_cond = (
        (snr >= snr_threshold)
        & (pred_huber_ret > 0.0)
        & (pred_lgbm_ret > 0.0)
        & (dist_to_resistance_atr >= sr_buffer)
    )
    short_cond = (
        (snr <= -snr_threshold)
        & (pred_huber_ret < 0.0)
        & (pred_lgbm_ret < 0.0)
        & (dist_to_support_atr >= sr_buffer)
    )

    pos_ls[long_cond] = 1
    pos_ls[short_cond] = -1
    pos_lo = np.maximum(pos_ls, 0).astype(np.int8)

    return pos_ls, pos_lo, snr, se


def compute_single_bar_quant_stance(
    pred_ret: float,
    pred_huber: float,
    pred_lgbm: float,
    atr_pct: float,
    dist_to_res: float,
    dist_to_sup: float,
    cfg: Phase3Config,
) -> tuple[str, float]:
    """Deterministically compute Layer 1 Quant Stance for a single live bar using active config."""
    b_cfg = cfg.backtest
    use_snr = (
        getattr(b_cfg.execution_gate, "mode", "snr") == "snr"
        if hasattr(b_cfg, "execution_gate")
        else False
    )
    sr_buf = b_cfg.sr_buffer_atr

    if use_snr:
        snr_thresh = getattr(b_cfg.execution_gate, "snr_threshold", 0.20)
        aleatoric_w = getattr(b_cfg.execution_gate, "aleatoric_weight", 0.5)
        sigma_epistemic = abs(pred_lgbm - pred_huber)
        sigma_aleatoric = atr_pct
        se = float(np.sqrt(sigma_epistemic**2 + (aleatoric_w * sigma_aleatoric) ** 2))
        snr = float(pred_ret / max(se, 1e-8))

        if snr >= snr_thresh and pred_huber > 0.0 and pred_lgbm > 0.0 and dist_to_res >= sr_buf:
            return "Long", snr
        elif snr <= -snr_thresh and pred_huber < 0.0 and pred_lgbm < 0.0 and dist_to_sup >= sr_buf:
            return "Short", snr
        else:
            return "Hold", snr
    else:
        hurdle_rate = b_cfg.hurdle_bps / 10000.0
        if pred_ret >= hurdle_rate and pred_huber > 0.0 and pred_lgbm > 0.0 and dist_to_res >= sr_buf:
            return "Long", 0.0
        elif pred_ret <= -hurdle_rate and pred_huber < 0.0 and pred_lgbm < 0.0 and dist_to_sup >= sr_buf:
            return "Short", 0.0
        else:
            return "Hold", 0.0


def apply_deterministic_sequential_cooldown(
    positions: np.ndarray,
    actual_returns: np.ndarray,
    max_consecutive_losses: int = 2,
    cooldown_bars: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically suppress trades after consecutive losses in the same direction."""
    n = len(positions)
    new_pos = np.zeros(n, dtype=np.int8)
    cooldown_active = np.zeros(n, dtype=bool)

    consec_losses = 0
    cooldown_rem = 0
    prev_stance = 0

    for t in range(n):
        if cooldown_rem > 0:
            new_pos[t] = 0
            cooldown_active[t] = True
            cooldown_rem -= 1
            continue

        stance = positions[t]
        if stance == 0:
            new_pos[t] = 0
            continue

        new_pos[t] = stance
        act_r = actual_returns[t]
        is_win = (stance > 0 and act_r > 0) or (stance < 0 and act_r < 0)
        is_loss = (stance > 0 and act_r < 0) or (stance < 0 and act_r > 0)

        if is_loss:
            if stance == prev_stance:
                consec_losses += 1
            else:
                consec_losses = 1
            prev_stance = stance
            if consec_losses >= max_consecutive_losses:
                cooldown_rem = cooldown_bars
                consec_losses = 0
        elif is_win:
            consec_losses = 0
            prev_stance = stance

    return new_pos, cooldown_active


def compute_volatility_targeted_weights(
    stances: np.ndarray,
    realized_vol_12h: np.ndarray,
    bars_per_day: float,
    annual_target_vol: float = 0.15,
) -> np.ndarray:
    """Continuous volatility-targeted position sizing:
    Weight_{i,t} = Stance_{i,t} * min(1.0, sigma_target_hourly / max(realized_vol_12h, 1e-4))
    where sigma_target_hourly = annual_target_vol / sqrt(252 * bars_per_day)
    """
    sigma_target_hourly = annual_target_vol / np.sqrt(252.0 * bars_per_day)
    vol_safe = np.maximum(realized_vol_12h, 1e-4)
    scale = np.minimum(1.0, sigma_target_hourly / vol_safe)
    weights = stances.astype(np.float64) * scale
    return weights


def run_full_backtest(
    test_oos_df: pd.DataFrame,
    cfg: Phase3Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run discrete position simulation, slippage drag, and exchange-native vectorized tear sheet."""
    df = test_oos_df.copy()
    b_cfg = cfg.backtest
    hurdle_rate = b_cfg.hurdle_bps / 10000.0
    slippage_rate = b_cfg.slippage_bps / 10000.0
    sr_buffer = b_cfg.sr_buffer_atr

    use_snr = (
        getattr(b_cfg.execution_gate, "mode", "snr") == "snr"
        if hasattr(b_cfg, "execution_gate")
        else False
    )
    snr_thresh = getattr(b_cfg.execution_gate, "snr_threshold", 1.28) if hasattr(b_cfg, "execution_gate") else 1.28
    aleatoric_w = getattr(b_cfg.execution_gate, "aleatoric_weight", 0.5) if hasattr(b_cfg, "execution_gate") else 0.5

    # Generate Full Ensemble positions
    if use_snr:
        full_pos_ls, full_pos_lo, snr_full, se_full = compute_snr_positions(
            pred_ret=df["full_pred_ret"].values,
            pred_huber_ret=df["full_pred_huber_ret"].values,
            pred_lgbm_ret=df["full_pred_lgbm_ret"].values,
            atr_pct=df["atr_pct"].values,
            dist_to_resistance_atr=df["dist_to_resistance_1_atr"].values,
            dist_to_support_atr=df["dist_to_support_1_atr"].values,
            snr_threshold=snr_thresh,
            sr_buffer=sr_buffer,
            aleatoric_weight=aleatoric_w,
        )
    else:
        full_pos_ls, full_pos_lo = generate_discrete_positions(
            pred_ret=df["full_pred_ret"].values,
            pred_huber_ret=df["full_pred_huber_ret"].values,
            pred_lgbm_ret=df["full_pred_lgbm_ret"].values,
            dist_to_resistance_atr=df["dist_to_resistance_1_atr"].values,
            dist_to_support_atr=df["dist_to_support_1_atr"].values,
            hurdle_rate=hurdle_rate,
            sr_buffer=sr_buffer,
        )
    df["full_pos_ls"] = full_pos_ls
    df["full_pos_lo"] = full_pos_lo

    # Generate Tech-Only Ablation positions
    tech_huber = (
        df["tech_pred_huber_ret"].values
        if "tech_pred_huber_ret" in df.columns
        else df["tech_pred_ret"].values
    )
    tech_lgbm = (
        df["tech_pred_lgbm_ret"].values
        if "tech_pred_lgbm_ret" in df.columns
        else df["tech_pred_ret"].values
    )
    if use_snr:
        tech_pos_ls, tech_pos_lo, _, _ = compute_snr_positions(
            pred_ret=df["tech_pred_ret"].values,
            pred_huber_ret=tech_huber,
            pred_lgbm_ret=tech_lgbm,
            atr_pct=df["atr_pct"].values,
            dist_to_resistance_atr=df["dist_to_resistance_1_atr"].values,
            dist_to_support_atr=df["dist_to_support_1_atr"].values,
            snr_threshold=snr_thresh,
            sr_buffer=sr_buffer,
            aleatoric_weight=aleatoric_w,
        )
    else:
        tech_pos_ls, tech_pos_lo = generate_discrete_positions(
            pred_ret=df["tech_pred_ret"].values,
            pred_huber_ret=tech_huber,
            pred_lgbm_ret=tech_lgbm,
            dist_to_resistance_atr=df["dist_to_resistance_1_atr"].values,
            dist_to_support_atr=df["dist_to_support_1_atr"].values,
            hurdle_rate=hurdle_rate,
            sr_buffer=sr_buffer,
        )
    df["tech_pos_ls"] = tech_pos_ls

    use_vol_sizing = (
        getattr(b_cfg.risk_sizing, "mode", "volatility_targeted") == "volatility_targeted"
        if hasattr(b_cfg, "risk_sizing")
        else False
    )
    target_annual_vol = getattr(b_cfg.risk_sizing, "annual_target_vol", 0.15) if hasattr(b_cfg, "risk_sizing") else 0.15

    # Initialize return columns
    df["full_net_ret_ls"] = 0.0
    df["full_net_ret_lo"] = 0.0
    df["full_slippage_ls"] = 0.0
    df["tech_net_ret_ls"] = 0.0

    per_ticker_metrics: dict[str, Any] = {}

    for ticker, grp in df.groupby("ticker"):
        idx = grp.index
        actual_log_ret = grp["actual_log_return_1h"].values
        actual_simple = np.exp(actual_log_ret) - 1.0

        exchange = grp.iloc[0]["exchange"]
        bars_per_day = (
            float(b_cfg.hkex_bars_per_day)
            if exchange == "HKEX"
            else float(b_cfg.krx_bars_per_day)
        )

        p_ls = grp["full_pos_ls"].values
        p_lo = grp["full_pos_lo"].values
        p_tech = grp["tech_pos_ls"].values

        cd_cfg = getattr(b_cfg, "cooldown", None)
        use_cooldown = getattr(cd_cfg, "enabled", True) if cd_cfg else True
        max_losses = getattr(cd_cfg, "max_consecutive_losses", 2) if cd_cfg else 2
        cd_bars = getattr(cd_cfg, "cooldown_bars", 2) if cd_cfg else 2

        if use_cooldown:
            p_ls, _ = apply_deterministic_sequential_cooldown(p_ls, actual_simple, max_losses, cd_bars)
            p_lo = np.maximum(p_ls, 0).astype(np.int8)
            p_tech, _ = apply_deterministic_sequential_cooldown(p_tech, actual_simple, max_losses, cd_bars)
            df.loc[idx, "full_pos_ls"] = p_ls
            df.loc[idx, "full_pos_lo"] = p_lo
            df.loc[idx, "tech_pos_ls"] = p_tech

        prev_ls = np.concatenate(([0], p_ls[:-1]))
        prev_lo = np.concatenate(([0], p_lo[:-1]))
        prev_tech = np.concatenate(([0], p_tech[:-1]))

        if use_vol_sizing:
            rvol = grp["realized_vol_12h"].values
            w_ls = compute_volatility_targeted_weights(p_ls, rvol, bars_per_day, target_annual_vol)
            w_lo = compute_volatility_targeted_weights(p_lo, rvol, bars_per_day, target_annual_vol)
            w_tech = compute_volatility_targeted_weights(p_tech, rvol, bars_per_day, target_annual_vol)
        else:
            w_ls = p_ls.astype(np.float64)
            w_lo = p_lo.astype(np.float64)
            w_tech = p_tech.astype(np.float64)

        prev_w_ls = np.concatenate(([0.0], w_ls[:-1]))
        prev_w_lo = np.concatenate(([0.0], w_lo[:-1]))
        prev_w_tech = np.concatenate(([0.0], w_tech[:-1]))

        slip_ls = slippage_rate * np.abs(w_ls - prev_w_ls)
        slip_lo = slippage_rate * np.abs(w_lo - prev_w_lo)
        slip_tech = slippage_rate * np.abs(w_tech - prev_w_tech)

        net_ls = w_ls * actual_simple - slip_ls
        net_lo = w_lo * actual_simple - slip_lo
        net_tech = w_tech * actual_simple - slip_tech

        df.loc[idx, "full_net_ret_ls"] = net_ls
        df.loc[idx, "full_net_ret_lo"] = net_lo
        df.loc[idx, "full_slippage_ls"] = slip_ls
        df.loc[idx, "tech_net_ret_ls"] = net_tech

        exchange = grp.iloc[0]["exchange"]
        bars_per_day = (
            float(b_cfg.hkex_bars_per_day)
            if exchange == "HKEX"
            else float(b_cfg.krx_bars_per_day)
        )

        # Metrics for this ticker
        f_fc = compute_forecast_metrics(
            actual_prices=grp["actual_next_close"].values,
            pred_prices=grp["full_pred_price"].values,
            actual_returns=actual_log_ret,
            pred_returns=grp["full_pred_ret"].values,
            prev_prices=grp["close"].values,
            positions=p_ls,
        )
        s_ls = compute_strategy_metrics(
            net_returns=net_ls,
            actual_simple_returns=actual_simple,
            slippage_drags=slip_ls,
            positions=p_ls,
            prev_positions=prev_ls,
            bars_per_day=bars_per_day,
        )
        s_lo = compute_strategy_metrics(
            net_returns=net_lo,
            actual_simple_returns=actual_simple,
            slippage_drags=slip_lo,
            positions=p_lo,
            prev_positions=prev_lo,
            bars_per_day=bars_per_day,
        )
        s_tech = compute_strategy_metrics(
            net_returns=net_tech,
            actual_simple_returns=actual_simple,
            slippage_drags=slip_tech,
            positions=p_tech,
            prev_positions=prev_tech,
            bars_per_day=bars_per_day,
        )

        per_ticker_metrics[ticker] = {
            "exchange": exchange,
            "forecast": asdict(f_fc),
            "strategy_full_ls": asdict(s_ls),
            "strategy_full_lo": asdict(s_lo),
            "strategy_tech_ls": asdict(s_tech),
        }

    # =========================================================================
    # Portfolio Aggregations
    # =========================================================================

    # 1. HKEX 5-Ticker Portfolio (equal weight 0.20 across 20 shared timestamps)
    hk_df = df[df["exchange"] == "HKEX"].copy()
    hk_piv_ls = hk_df.pivot(index="bar_close_utc", columns="ticker", values="full_net_ret_ls").fillna(0.0)
    hk_piv_lo = hk_df.pivot(index="bar_close_utc", columns="ticker", values="full_net_ret_lo").fillna(0.0)
    hk_piv_tech = hk_df.pivot(index="bar_close_utc", columns="ticker", values="tech_net_ret_ls").fillna(0.0)
    hk_piv_bh = hk_df.pivot(index="bar_close_utc", columns="ticker", values="actual_log_return_1h").fillna(0.0)
    hk_piv_bh_simple = np.exp(hk_piv_bh) - 1.0

    hk_port_ls = hk_piv_ls.mean(axis=1).values
    hk_port_lo = hk_piv_lo.mean(axis=1).values
    hk_port_tech = hk_piv_tech.mean(axis=1).values
    hk_port_bh = hk_piv_bh_simple.mean(axis=1).values

    hk_metrics_ls = compute_strategy_metrics(
        net_returns=hk_port_ls,
        actual_simple_returns=hk_port_bh,
        slippage_drags=np.zeros_like(hk_port_ls),
        positions=np.ones_like(hk_port_ls),
        prev_positions=np.ones_like(hk_port_ls),
        bars_per_day=float(b_cfg.hkex_bars_per_day),
    )
    hk_metrics_lo = compute_strategy_metrics(
        net_returns=hk_port_lo,
        actual_simple_returns=hk_port_bh,
        slippage_drags=np.zeros_like(hk_port_lo),
        positions=np.ones_like(hk_port_lo),
        prev_positions=np.ones_like(hk_port_lo),
        bars_per_day=float(b_cfg.hkex_bars_per_day),
    )
    hk_metrics_tech = compute_strategy_metrics(
        net_returns=hk_port_tech,
        actual_simple_returns=hk_port_bh,
        slippage_drags=np.zeros_like(hk_port_tech),
        positions=np.ones_like(hk_port_tech),
        prev_positions=np.ones_like(hk_port_tech),
        bars_per_day=float(b_cfg.hkex_bars_per_day),
    )

    # 2. KRX 5-Ticker Portfolio (equal weight 0.20 across 17 shared timestamps)
    kr_df = df[df["exchange"] == "KRX"].copy()
    kr_piv_ls = kr_df.pivot(index="bar_close_utc", columns="ticker", values="full_net_ret_ls").fillna(0.0)
    kr_piv_lo = kr_df.pivot(index="bar_close_utc", columns="ticker", values="full_net_ret_lo").fillna(0.0)
    kr_piv_tech = kr_df.pivot(index="bar_close_utc", columns="ticker", values="tech_net_ret_ls").fillna(0.0)
    kr_piv_bh = kr_df.pivot(index="bar_close_utc", columns="ticker", values="actual_log_return_1h").fillna(0.0)
    kr_piv_bh_simple = np.exp(kr_piv_bh) - 1.0

    kr_port_ls = kr_piv_ls.mean(axis=1).values
    kr_port_lo = kr_piv_lo.mean(axis=1).values
    kr_port_tech = kr_piv_tech.mean(axis=1).values
    kr_port_bh = kr_piv_bh_simple.mean(axis=1).values

    kr_metrics_ls = compute_strategy_metrics(
        net_returns=kr_port_ls,
        actual_simple_returns=kr_port_bh,
        slippage_drags=np.zeros_like(kr_port_ls),
        positions=np.ones_like(kr_port_ls),
        prev_positions=np.ones_like(kr_port_ls),
        bars_per_day=float(b_cfg.krx_bars_per_day),
    )
    kr_metrics_lo = compute_strategy_metrics(
        net_returns=kr_port_lo,
        actual_simple_returns=kr_port_bh,
        slippage_drags=np.zeros_like(kr_port_lo),
        positions=np.ones_like(kr_port_lo),
        prev_positions=np.ones_like(kr_port_lo),
        bars_per_day=float(b_cfg.krx_bars_per_day),
    )
    kr_metrics_tech = compute_strategy_metrics(
        net_returns=kr_port_tech,
        actual_simple_returns=kr_port_bh,
        slippage_drags=np.zeros_like(kr_port_tech),
        positions=np.ones_like(kr_port_tech),
        prev_positions=np.ones_like(kr_port_tech),
        bars_per_day=float(b_cfg.krx_bars_per_day),
    )

    # 3. Overall 10-Ticker Union Portfolio (fixed 0.10 capital weight per ticker on union UTC grid)
    union_piv_ls = df.pivot(index="bar_close_utc", columns="ticker", values="full_net_ret_ls").fillna(0.0)
    union_piv_lo = df.pivot(index="bar_close_utc", columns="ticker", values="full_net_ret_lo").fillna(0.0)
    union_piv_tech = df.pivot(index="bar_close_utc", columns="ticker", values="tech_net_ret_ls").fillna(0.0)
    union_piv_bh = df.pivot(index="bar_close_utc", columns="ticker", values="actual_log_return_1h").fillna(0.0)
    union_piv_bh_simple = np.exp(union_piv_bh) - 1.0

    union_port_ls = (union_piv_ls * 0.10).sum(axis=1).values
    union_port_lo = (union_piv_lo * 0.10).sum(axis=1).values
    union_port_tech = (union_piv_tech * 0.10).sum(axis=1).values
    union_port_bh = (union_piv_bh_simple * 0.10).sum(axis=1).values

    n_oos_dates = len(df["trading_date_local"].unique())
    bars_per_day_union = len(union_piv_ls) / max(n_oos_dates, 1)

    overall_metrics_ls = compute_strategy_metrics(
        net_returns=union_port_ls,
        actual_simple_returns=union_port_bh,
        slippage_drags=np.zeros_like(union_port_ls),
        positions=np.ones_like(union_port_ls),
        prev_positions=np.ones_like(union_port_ls),
        bars_per_day=bars_per_day_union,
    )
    overall_metrics_lo = compute_strategy_metrics(
        net_returns=union_port_lo,
        actual_simple_returns=union_port_bh,
        slippage_drags=np.zeros_like(union_port_lo),
        positions=np.ones_like(union_port_lo),
        prev_positions=np.ones_like(union_port_lo),
        bars_per_day=bars_per_day_union,
    )
    overall_metrics_tech = compute_strategy_metrics(
        net_returns=union_port_tech,
        actual_simple_returns=union_port_bh,
        slippage_drags=np.zeros_like(union_port_tech),
        positions=np.ones_like(union_port_tech),
        prev_positions=np.ones_like(union_port_tech),
        bars_per_day=bars_per_day_union,
    )

    # 4. Overall Forecast Comparison (185 pooled OOS rows)
    overall_fc_full = compute_forecast_metrics(
        actual_prices=df["actual_next_close"].values,
        pred_prices=df["full_pred_price"].values,
        actual_returns=df["actual_log_return_1h"].values,
        pred_returns=df["full_pred_ret"].values,
        prev_prices=df["close"].values,
        positions=df["full_pos_ls"].values,
    )
    overall_fc_tech = compute_forecast_metrics(
        actual_prices=df["actual_next_close"].values,
        pred_prices=df["tech_pred_price"].values,
        actual_returns=df["actual_log_return_1h"].values,
        pred_returns=df["tech_pred_ret"].values,
        prev_prices=df["close"].values,
        positions=df["tech_pos_ls"].values,
    )
    overall_fc_rw = compute_forecast_metrics(
        actual_prices=df["actual_next_close"].values,
        pred_prices=df["rw_pred_price"].values,
        actual_returns=df["actual_log_return_1h"].values,
        pred_returns=np.zeros(len(df)),
        prev_prices=df["close"].values,
        positions=np.zeros(len(df), dtype=int),
    )

    tearsheet = {
        "overall": {
            "forecast_full": asdict(overall_fc_full),
            "forecast_tech": asdict(overall_fc_tech),
            "forecast_rw": asdict(overall_fc_rw),
            "strategy_full_ls": asdict(overall_metrics_ls),
            "strategy_full_lo": asdict(overall_metrics_lo),
            "strategy_tech_ls": asdict(overall_metrics_tech),
            "exchange_50_50_summary": {
                "total_net_return_pct": 0.5 * (hk_metrics_ls.total_net_return_pct + kr_metrics_ls.total_net_return_pct),
                "buy_and_hold_return_pct": 0.5 * (hk_metrics_ls.buy_and_hold_return_pct + kr_metrics_ls.buy_and_hold_return_pct),
                "excess_return_pct": 0.5 * (hk_metrics_ls.excess_return_pct + kr_metrics_ls.excess_return_pct),
                "annualized_sharpe": 0.5 * (hk_metrics_ls.annualized_sharpe + kr_metrics_ls.annualized_sharpe),
            },
        },
        "hkex_portfolio": {
            "strategy_full_ls": asdict(hk_metrics_ls),
            "strategy_full_lo": asdict(hk_metrics_lo),
            "strategy_tech_ls": asdict(hk_metrics_tech),
        },
        "krx_portfolio": {
            "strategy_full_ls": asdict(kr_metrics_ls),
            "strategy_full_lo": asdict(kr_metrics_lo),
            "strategy_tech_ls": asdict(kr_metrics_tech),
        },
        "per_ticker": per_ticker_metrics,
    }

    return df, tearsheet


def render_markdown_tearsheet(tearsheet: dict[str, Any]) -> str:
    """Render a clean GitHub Flavored Markdown tear sheet report with side-by-side ablations (strictly zero Buy & Hold)."""
    ov = tearsheet["overall"]
    hk = tearsheet["hkex_portfolio"]
    kr = tearsheet["krx_portfolio"]
    pt = tearsheet["per_ticker"]

    hk_full_ret = hk["strategy_full_ls"]["total_net_return_pct"]
    hk_tech_ret = hk["strategy_tech_ls"]["total_net_return_pct"]
    hk_delta_bps = (hk_full_ret - hk_tech_ret) * 100.0

    kr_full_ret = kr["strategy_full_ls"]["total_net_return_pct"]
    kr_tech_ret = kr["strategy_tech_ls"]["total_net_return_pct"]
    kr_delta_bps = (kr_full_ret - kr_tech_ret) * 100.0

    ov_full_ret = ov["strategy_full_ls"]["total_net_return_pct"]
    ov_tech_ret = ov["strategy_tech_ls"]["total_net_return_pct"]
    ov_delta_bps = (ov_full_ret - ov_tech_ret) * 100.0

    md = f"""# Phase 3 Walk-Forward Out-of-Sample (OOS) Validation Tear Sheet
**Evaluation Window:** Days 8–10 (`180` hourly bars across 10 assets)
**Execution Assumptions:** `15 bps` hurdle rate (3× slippage), `5 bps` one-way turnover slippage, mathematical KDE risk/reward buffer.
**Institutional Benchmark:** Tech-Only Baseline (strictly zero Buy & Hold).

---

## 1. Executive Summary & Model Ablation

| Evaluation Dimension | Full MoE Alpha Strategy (38 Features) | Tech-Only Baseline (18 Features) | Random Walk Baseline | Alpha Spread Delta ($\\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Price RMSE** | `{ov['forecast_full']['rmse_price']:.4f}` | `{ov['forecast_tech']['rmse_price']:.4f}` | `{ov['forecast_rw']['rmse_price']:.4f}` | `{(ov['forecast_full']['rmse_price'] - ov['forecast_tech']['rmse_price']):.4f}` |
| **Return RMSE (bps)** | `{ov['forecast_full']['rmse_bps']:.1f}` | `{ov['forecast_tech']['rmse_bps']:.1f}` | `{ov['forecast_rw']['rmse_bps']:.1f}` | `{(ov['forecast_full']['rmse_bps'] - ov['forecast_tech']['rmse_bps']):.1f}` |
| **Price MAE (bps)** | `{ov['forecast_full']['mae_bps']:.1f}` | `{ov['forecast_tech']['mae_bps']:.1f}` | `{ov['forecast_rw']['mae_bps']:.1f}` | `{(ov['forecast_full']['mae_bps'] - ov['forecast_tech']['mae_bps']):.1f}` |
| **MAPE (%)** | `{ov['forecast_full']['mape_pct']:.2f}%` | `{ov['forecast_tech']['mape_pct']:.2f}%` | `{ov['forecast_rw']['mape_pct']:.2f}%` | `{(ov['forecast_full']['mape_pct'] - ov['forecast_tech']['mape_pct']):.2f}%` |
| **Theil's U Score** | `{ov['forecast_full']['theils_u']:.4f}` | `{ov['forecast_tech']['theils_u']:.4f}` | `1.0000` | `{(ov['forecast_full']['theils_u'] - ov['forecast_tech']['theils_u']):.4f}` |
| **Spearman Rank IC** | `{ov['forecast_full']['spearman_ic']:.4f}` | `{ov['forecast_tech']['spearman_ic']:.4f}` | `0.0000` | `+{(ov['forecast_full']['spearman_ic'] - ov['forecast_tech']['spearman_ic']):.4f}` |
| **Overall Net Return (L/S)** | `{ov_full_ret:.2f}% ({ov_full_ret*100.0:+.1f} bps)` | `{ov_tech_ret:.2f}% ({ov_tech_ret*100.0:+.1f} bps)` | — | `+{ov_delta_bps:.1f} bps` |
| **Annualized Sharpe** | `{ov['strategy_full_ls']['annualized_sharpe']:.2f}` | `{ov['strategy_tech_ls']['annualized_sharpe']:.2f}` | — | `{(ov['strategy_full_ls']['annualized_sharpe'] - ov['strategy_tech_ls']['annualized_sharpe']):.2f}` |
| **Max Drawdown** | `{ov['strategy_full_ls']['max_drawdown_pct']:.2f}%` | `{ov['strategy_tech_ls']['max_drawdown_pct']:.2f}%` | — | `{(ov['strategy_full_ls']['max_drawdown_pct'] - ov['strategy_tech_ls']['max_drawdown_pct']):.2f}%` |
| **Profit Factor** | `{ov['strategy_full_ls']['profit_factor']:.2f}` | `{ov['strategy_tech_ls']['profit_factor']:.2f}` | — | `{(ov['strategy_full_ls']['profit_factor'] - ov['strategy_tech_ls']['profit_factor']):.2f}` |

---

## 2. Portfolio Tear Sheet by Exchange (Equal-Weight 5-Ticker Portfolios)

### HKEX 5-Ticker Portfolio (`20` shared hourly bars, 7 bars/day)
- **Strategy Net Return (Long/Short):** `{hk_full_ret:.2f}% ({hk_full_ret*100.0:+.1f} bps)`
- **Tech Baseline Net Return:** `{hk_tech_ret:.2f}% ({hk_tech_ret*100.0:+.1f} bps)`
- **Net Alpha Spread ($\\Delta$):** `+{hk_delta_bps:.1f} bps`
- **Annualized Sharpe Ratio:** `{hk['strategy_full_ls']['annualized_sharpe']:.2f}` (Tech Baseline: `{hk['strategy_tech_ls']['annualized_sharpe']:.2f}`)
- **Max Drawdown:** `{hk['strategy_full_ls']['max_drawdown_pct']:.2f}%`
- **Profit Factor:** `{hk['strategy_full_ls']['profit_factor']:.2f}`

### KRX 5-Ticker Portfolio (`17` shared hourly bars, 6 bars/day)
- **Strategy Net Return (Long/Short):** `{kr_full_ret:.2f}% ({kr_full_ret*100.0:+.1f} bps)`
- **Tech Baseline Net Return:** `{kr_tech_ret:.2f}% ({kr_tech_ret*100.0:+.1f} bps)`
- **Net Alpha Spread ($\\Delta$):** `+{kr_delta_bps:.1f} bps`
- **Annualized Sharpe Ratio:** `{kr['strategy_full_ls']['annualized_sharpe']:.2f}` (Tech Baseline: `{kr['strategy_tech_ls']['annualized_sharpe']:.2f}`)
- **Max Drawdown:** `{kr['strategy_full_ls']['max_drawdown_pct']:.2f}%`
- **Profit Factor:** `{kr['strategy_full_ls']['profit_factor']:.2f}`

---

## 3. Per-Asset OOS Performance Metrics

| Ticker | Exchange | Forecast MAPE | Directional Acc. | Strategy Net Ret | Tech Baseline Ret | Alpha Spread ($\\Delta$) | Ann. Sharpe | Active Bars | Turnover Drag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for ticker, data in sorted(pt.items()):
        fc = data["forecast"]
        st = data["strategy_full_ls"]
        st_tech = data.get("strategy_tech_ls", {})
        tech_ret = st_tech.get("total_net_return_pct", 0.0)
        delta_bps = (st["total_net_return_pct"] - tech_ret) * 100.0
        delta_str = f"+{delta_bps:.1f} bps" if delta_bps >= 0 else f"{delta_bps:.1f} bps"
        exch = data["exchange"]
        md += f"| `{ticker}` | {exch} | `{fc['mape_pct']:.2f}%` | `{fc['directional_accuracy_all_pct']:.1f}%` | `{st['total_net_return_pct']:.2f}%` | `{tech_ret:.2f}%` | `{delta_str}` | `{st['annualized_sharpe']:.2f}` | `{st['active_trade_bars']}` | `{st['total_slippage_drag_bps']:.1f} bps` |\n"

    md += """
---
*Generated automatically by `src.models.backtest` during Phase 3 pipeline execution (Strictly Zero Buy & Hold).*
"""
    return md
    return md
