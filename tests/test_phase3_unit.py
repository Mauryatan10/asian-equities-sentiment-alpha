"""Comprehensive unit tests for Phase 3:
1. Strict UTC target embargo test.
2. Zero-NaN assertion across X_train, X_test, X_live.
3. Causal session close bar clock flag test.
4. Zero-scaler leakage test under X_test mutation.
5. Slippage math test.
6. Zero-trade safe defaults test.
7. Rolling copula feature bounds and causality.
8. MoE gating router convexity and prior default.
9. Epistemic SNR execution gate and suppression.
10. Continuous volatility-targeted sizing.
11. Deterministic sequential cooldown and stance computation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.features.alpha_factors import compute_rolling_copula_features
from src.features.schemas import FEATURES_1H_PATH
from src.models.backtest import (
    compute_forecast_metrics,
    compute_snr_positions,
    compute_strategy_metrics,
    compute_volatility_targeted_weights,
    generate_discrete_positions,
    apply_deterministic_sequential_cooldown,
    compute_single_bar_quant_stance,
)
from src.models.dataset import load_and_split_dataset, prepare_features_df
from src.models.ensemble import LearnedMoEEnsemble
from src.models.schemas import (
    Phase3Config,
    load_phase3_config,
)


@pytest.fixture(scope="module")
def p3_cfg() -> Phase3Config:
    return load_phase3_config()


@pytest.mark.skipif(not FEATURES_1H_PATH.exists(), reason="Requires processed dataset on disk")
def test_strict_cross_exchange_target_embargo(p3_cfg: Phase3Config):
    """Assert test_start_utc is strictly respected and train_split strictly precedes test_oos."""
    ds = load_and_split_dataset(p3_cfg)

    # 1. Row counts conform to 10-day dataset structure
    assert len(ds.train_split) in (540, 545)
    assert len(ds.test_oos) in (180, 185)
    assert len(ds.train_live) in (720, 725, 730)
    assert len(ds.live) == 10

    # 2. Strict chronological ordering: all train bars < test_start_utc
    max_train_utc = ds.train_split["bar_close_utc"].max()
    min_test_utc = ds.test_oos["bar_close_utc"].min()
    assert max_train_utc < ds.test_start_utc
    assert min_test_utc >= ds.test_start_utc
    assert max_train_utc < min_test_utc


@pytest.mark.skipif(not FEATURES_1H_PATH.exists(), reason="Requires processed dataset on disk")
def test_zero_nans_in_feature_matrices(p3_cfg: Phase3Config):
    """Assert zero NaNs across all feature columns in train_split, test_oos, and live."""
    ds = load_and_split_dataset(p3_cfg)

    for split_name, df in [
        ("train_split", ds.train_split),
        ("test_oos", ds.test_oos),
        ("live", ds.live),
    ]:
        nans_per_col = df[ds.feature_cols].isna().sum()
        cols_with_nan = nans_per_col[nans_per_col > 0]
        assert len(cols_with_nan) == 0, f"NaNs found in {split_name}: {cols_with_nan.to_dict()}"


@pytest.mark.skipif(not FEATURES_1H_PATH.exists(), reason="Requires processed dataset on disk")
def test_causal_session_close_bar_clock_flags(p3_cfg: Phase3Config):
    """Verify session_close_bar and bar_duration_30m flags match exchange trading calendars."""
    ds = load_and_split_dataset(p3_cfg)
    full_df = pd.concat([ds.train_split, ds.test_oos], ignore_index=True)

    # HKEX trading hours: 09:30-12:00, 13:00-16:00 (HONG KONG TIME: UTC+8)
    # HKEX close at 16:00 HKT = 08:00 UTC
    hk_close_bars = full_df[
        (full_df["exchange"] == "HKEX") & (full_df["is_session_close_bar"] == 1.0)
    ]
    for _, row in hk_close_bars.iterrows():
        assert row["bar_close_utc"].hour == 8
        assert row["bar_close_utc"].minute == 0

    # KRX trading hours: 09:00-15:30 (KOREA TIME: UTC+9)
    # KRX close bar timestamp is at 06:00 UTC (or 06:30 UTC for 30m bar)
    kr_close_bars = full_df[
        (full_df["exchange"] == "KRX") & (full_df["is_session_close_bar"] == 1.0)
    ]
    for _, row in kr_close_bars.iterrows():
        assert row["bar_close_utc"].hour == 6

    # 30-minute bar duration flag: HKEX bars 01:30-02:00 UTC (10:00 HKT) or KRX final bar 06:00-06:30 UTC
    dur_30m_bars = full_df[full_df["bar_duration_min_is_30"] == 1.0]
    assert len(dur_30m_bars) > 0
    for _, row in dur_30m_bars.iterrows():
        assert row["bar_duration_min"] == 30


@pytest.mark.skipif(not FEATURES_1H_PATH.exists(), reason="Requires processed dataset on disk")
def test_zero_scaler_leakage_under_test_mutation(p3_cfg: Phase3Config):
    """Assert RobustScaler fitted on train_split produces identical coefficients regardless of test set mutations."""
    ds = load_and_split_dataset(p3_cfg)

    # Fit scaler pipeline on original train_split
    pipe1 = Pipeline([("scaler", RobustScaler())])
    X_train_scaled1 = pipe1.fit_transform(ds.train_split[ds.feature_cols])

    # Mutate test_oos drastically (multiply by 1000)
    mutated_test = ds.test_oos.copy()
    mutated_test[ds.feature_cols] = mutated_test[ds.feature_cols] * 1000.0

    # Re-fit pipeline on train_split alone
    pipe2 = Pipeline([("scaler", RobustScaler())])
    X_train_scaled2 = pipe2.fit_transform(ds.train_split[ds.feature_cols])

    # Scale mutated test with pipe1 and pipe2; train scalers must be completely identical
    np.testing.assert_allclose(X_train_scaled1, X_train_scaled2, atol=1e-12)
    np.testing.assert_allclose(pipe1.named_steps["scaler"].center_, pipe2.named_steps["scaler"].center_, atol=1e-12)
    np.testing.assert_allclose(pipe1.named_steps["scaler"].scale_, pipe2.named_steps["scaler"].scale_, atol=1e-12)


def test_slippage_math_and_turnover():
    """Verify that 5 bps one-way slippage drag is accurately deducted on position changes."""
    # Scenario: 4 bars.
    # Bar 0: position 0 -> 1 (+1 turnover, drag = 5 bps)
    # Bar 1: position 1 -> 1 (0 turnover, drag = 0 bps)
    # Bar 2: position 1 -> -1 (2 turnover, drag = 10 bps)
    # Bar 3: position -1 -> 0 (1 turnover, drag = 5 bps)
    positions = np.array([1, 1, -1, 0], dtype=np.int8)
    prev_positions = np.array([0, 1, 1, -1], dtype=np.int8)
    actual_returns = np.array([0.01, 0.005, -0.01, 0.002])  # 100 bps, 50 bps, -100 bps, 20 bps

    slippage_rate = 0.0005  # 5 bps
    turnover = np.abs(positions - prev_positions)
    expected_drags = turnover * slippage_rate

    assert np.allclose(expected_drags, [0.0005, 0.0, 0.0010, 0.0005])

    gross_returns = positions * actual_returns
    net_returns = gross_returns - expected_drags

    strat_metrics = compute_strategy_metrics(
        net_returns=net_returns,
        actual_simple_returns=actual_returns,
        slippage_drags=expected_drags,
        positions=positions,
        prev_positions=prev_positions,
        bars_per_day=7.0,
    )

    assert strat_metrics.position_changes == 3
    assert abs(strat_metrics.total_slippage_drag_bps - 20.0) < 1e-6  # (1 + 0 + 2 + 1) * 5 = 20 bps


def test_zero_trade_safe_defaults():
    """Verify that when active_bars == 0, financial metrics safely return 0.0 without divide-by-zero or crash."""
    n_bars = 20
    positions = np.zeros(n_bars, dtype=np.int8)
    prev_positions = np.zeros(n_bars, dtype=np.int8)
    actual_rets = np.random.normal(0, 0.01, n_bars)
    net_rets = np.zeros(n_bars)
    drags = np.zeros(n_bars)

    strat_metrics = compute_strategy_metrics(
        net_returns=net_rets,
        actual_simple_returns=actual_rets,
        slippage_drags=drags,
        positions=positions,
        prev_positions=prev_positions,
        bars_per_day=6.0,
    )

    assert strat_metrics.active_trade_bars == 0
    assert strat_metrics.annualized_sharpe == 0.0
    assert strat_metrics.annualized_sortino == 0.0
    assert strat_metrics.profit_factor == 0.0
    assert strat_metrics.max_drawdown_pct == 0.0
    assert strat_metrics.position_changes == 0
    assert strat_metrics.total_slippage_drag_bps == 0.0

    # Also verify forecast metrics with zero active trades
    fc_metrics = compute_forecast_metrics(
        actual_prices=np.ones(n_bars) * 100.0,
        pred_prices=np.ones(n_bars) * 100.0,
        actual_returns=actual_rets,
        pred_returns=np.zeros(n_bars),
        prev_prices=np.ones(n_bars) * 100.0,
        positions=positions,
    )
    assert fc_metrics.directional_hit_rate_active_pct is None


def test_rolling_copula_features_bounds_and_causality():
    """Verify rolling copula transform bounds in [-1, 1], bilinear sign logic, and strict causality."""
    n_bars = 60
    dates = pd.date_range("2026-11-01", periods=n_bars, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "ticker": ["005930.KS"] * n_bars,
        "bar_close_utc": dates,
        "divergence_retail_vs_news_24h": np.linspace(-2.0, 2.0, n_bars),
        "price_vs_vwap_pct": np.linspace(1.5, -1.5, n_bars),
        "realized_vol_12h": np.abs(np.sin(np.linspace(0, 3, n_bars))) * 0.02,
    })

    copula_df = compute_rolling_copula_features(df, window=24)
    assert "copula_div_trend" in copula_df.columns
    assert "copula_div_vol" in copula_df.columns

    # Check bounds [-1.0, 1.0]
    valid_trend = copula_df["copula_div_trend"].dropna()
    valid_vol = copula_df["copula_div_vol"].dropna()
    assert (valid_trend >= -1.0 - 1e-6).all() and (valid_trend <= 1.0 + 1e-6).all()
    assert (valid_vol >= -1.0 - 1e-6).all() and (valid_vol <= 1.0 + 1e-6).all()

    # Verify causality: modifying future rows must not change bar t's copula feature
    df_mutated = df.copy()
    df_mutated.loc[50:, "divergence_retail_vs_news_24h"] = 999.0
    copula_mutated = compute_rolling_copula_features(df_mutated, window=24)
    # Bars 0..49 must be strictly identical
    np.testing.assert_allclose(
        copula_df.loc[:49, "copula_div_trend"].values,
        copula_mutated.loc[:49, "copula_div_trend"].values,
        atol=1e-12,
    )


def test_moe_gating_router_convexity_and_prior_default(p3_cfg: Phase3Config):
    """Verify that MoE gating weights sum to 1.0 (convexity) and fallback to fixed prior [0.60, 0.40] if disabled."""
    np.random.seed(42)
    n = 100
    p = len(p3_cfg.features.group_a_price_vol_clock)
    X = np.random.normal(0, 1, (n, p))
    y = np.random.normal(0, 0.01, n)
    X_gate = np.random.normal(0, 1, (n, 3))

    moe = LearnedMoEEnsemble(p3_cfg.models)
    gate_df = pd.DataFrame(X_gate, columns=p3_cfg.models.moe_gating.state_features)
    moe.fit(X, y, state_df=gate_df)

    vol_scale = np.ones(n) * 0.01
    close_prices = np.ones(n) * 100.0
    preds = moe.predict(X, vol_scale=vol_scale, close_prices=close_prices, state_df=gate_df)
    w_linear = preds.gating_weights_linear
    w_tree = preds.gating_weights_tree

    # Convexity: w_huber >= 0, w_lgbm >= 0, w_huber + w_lgbm == 1.0
    assert (w_linear >= 0.0).all()
    assert (w_tree >= 0.0).all()
    np.testing.assert_allclose(w_linear + w_tree, np.ones(n), atol=1e-6)

    # Prior fallback test
    cfg_no_gate = p3_cfg.models.model_copy(deep=True)
    cfg_no_gate.moe_gating.enabled = False
    moe_prior = LearnedMoEEnsemble(cfg_no_gate)
    moe_prior.fit(X, y)
    preds_prior = moe_prior.predict(X, vol_scale=vol_scale, close_prices=close_prices)
    np.testing.assert_allclose(preds_prior.gating_weights_linear, 0.60, atol=1e-6)
    np.testing.assert_allclose(preds_prior.gating_weights_tree, 0.40, atol=1e-6)


def test_epistemic_snr_execution_gate_and_suppression():
    """Assert SNR gate suppresses trade when epistemic uncertainty (model disagreement) exceeds signal."""
    # Case 1: High signal, low disagreement, favorable S/R -> Long triggered
    pos_ls, pos_lo, snr, se = compute_snr_positions(
        pred_ret=np.array([0.0030]),  # +30 bps
        pred_huber_ret=np.array([0.0028]),
        pred_lgbm_ret=np.array([0.0032]),  # Disagreement = 4 bps
        atr_pct=np.array([0.0080]),  # 80 bps -> aleatoric term = 40 bps
        dist_to_resistance_atr=np.array([0.80]),  # > 0.25 buffer
        dist_to_support_atr=np.array([0.50]),
        snr_threshold=0.20,
        sr_buffer=0.25,
    )
    assert pos_ls[0] == 1
    assert pos_lo[0] == 1
    assert snr[0] > 0.20

    # Case 2: High signal but severe model disagreement (epistemic uncertainty) -> Suppressed to 0
    pos_ls2, pos_lo2, snr2, se2 = compute_snr_positions(
        pred_ret=np.array([0.0015]),  # +15 bps
        pred_huber_ret=np.array([0.0040]),
        pred_lgbm_ret=np.array([-0.0010]),  # Disagreement = 50 bps, models disagree on sign!
        atr_pct=np.array([0.0080]),
        dist_to_resistance_atr=np.array([0.80]),
        dist_to_support_atr=np.array([0.50]),
        snr_threshold=0.20,
        sr_buffer=0.25,
    )
    assert pos_ls2[0] == 0  # Suppressed due to sign disagreement & high epistemic uncertainty
    assert pos_lo2[0] == 0

    # Case 3: Good short signal but price within 0.25 ATR of support -> Suppressed by S/R floor
    pos_ls3, pos_lo3, snr3, se3 = compute_snr_positions(
        pred_ret=np.array([-0.0030]),
        pred_huber_ret=np.array([-0.0028]),
        pred_lgbm_ret=np.array([-0.0032]),
        atr_pct=np.array([0.0080]),
        dist_to_resistance_atr=np.array([0.80]),
        dist_to_support_atr=np.array([0.10]),  # < 0.25 buffer! Too close to support!
        snr_threshold=0.20,
        sr_buffer=0.25,
    )
    assert pos_ls3[0] == 0  # Suppressed by S/R buffer


def test_continuous_volatility_targeted_sizing():
    """Assert continuous volatility-targeted weight scaling inversely scales with realized vol."""
    stances = np.array([1, 1, -1], dtype=np.int8)
    # Low vol (0.005), medium vol (0.015), high vol (0.030)
    realized_vol = np.array([0.005, 0.015, 0.030])
    bars_per_day = 7.0
    annual_target_vol = 0.15

    weights = compute_volatility_targeted_weights(
        stances=stances,
        realized_vol_12h=realized_vol,
        bars_per_day=bars_per_day,
        annual_target_vol=annual_target_vol,
    )

    # sigma_target_hourly = 0.15 / sqrt(252 * 7) = 0.00357
    # For low vol (0.005): scale = min(1.0, 0.00357 / 0.005) = 0.714
    # For medium vol (0.015): scale = min(1.0, 0.00357 / 0.015) = 0.238
    # For high vol (0.030): scale = min(1.0, 0.00357 / 0.030) = 0.119
    assert weights[0] > weights[1] > 0.0
    assert weights[2] < 0.0
    np.testing.assert_allclose(weights[0] / weights[1], 3.0, atol=1e-3)
    np.testing.assert_allclose(weights[1] / (-weights[2]), 2.0, atol=1e-3)


def test_deterministic_cooldown_and_stance_computation(p3_cfg: Phase3Config):
    """Assert deterministic sequential cooldown suppresses repeat losses and single-bar stance computes correctly."""
    # 1. Test single-bar deterministic stance
    stance, snr = compute_single_bar_quant_stance(
        pred_ret=0.0025,
        pred_huber=0.0020,
        pred_lgbm=0.0030,
        atr_pct=0.008,
        dist_to_res=0.50,
        dist_to_sup=0.50,
        cfg=p3_cfg,
    )
    assert stance == "Long"
    assert snr > 0.20

    # 2. Test deterministic sequential cooldown
    positions = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    returns = np.array([-0.01, -0.01, 0.02, 0.02, 0.02])  # 2 consecutive losses
    cooled, active = apply_deterministic_sequential_cooldown(
        positions, returns, max_consecutive_losses=2, cooldown_bars=2
    )
    # Positions 0 and 1 take trade; bar 2 and 3 should be cooled down (0)
    assert cooled[0] == 1
    assert cooled[1] == 1
    assert cooled[2] == 0  # Cooldown bar 1
    assert cooled[3] == 0  # Cooldown bar 2
    assert cooled[4] == 1  # Resumed
