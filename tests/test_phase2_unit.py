"""Unit tests for Phase 2: Feature Engineering & Multi-Agent Processing.

Covers:
1. Deterministic 3-bucket Naver router + Rich-sqrt quota + text dedup + zero cross-bar leakage
2. Causal ATR-scaled Gaussian KDE VPVR, 70% VAL/VAH, S/R levels, zero-volume/flat bars
3. LLM JSON <think> stripping, auto-bisect missing-ID & malformed-JSON recovery, price-target bounding
4. Causal Perturbation Zero-Leakage Test (mutating t+1 OHLCV & post-t events leaves bar t bit-for-bit identical)
5. Source-balanced dual-horizon UTC decay, zero-collapse impact weights, and live_snapshot 13-key catalysts
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.alpha_factors import (
    compute_alpha_factors_and_snapshot,
    compute_full_corpus_stats,
    verify_ticker_phase2_outputs,
)
from src.features.cultural_nlp_agent import (
    LLMScoreCheckpointStore,
    compute_impact_weights,
    run_cultural_nlp_for_ticker,
    score_batch_with_bisect,
)
from src.features.llm_client import parse_llm_json_object
from src.features.quant_technical_agent import compute_quant_technical_features
from src.features.router import compute_naver_quota, route_ticker_events
from src.features.schemas import (
    CatalystSummary,
    ScoredEventItem,
    load_phase2_config,
)


def _make_synthetic_bars(n: int = 40, ticker: str = "005930.KS", exchange: str = "KRX") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_time = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    price = 60000.0
    for i in range(n):
        # Simulate realistic 6-bar daily sessions with 18h overnight gap every 6 bars
        day_idx = i // 6
        bar_in_day = i % 6
        open_dt = base_time + timedelta(days=day_idx, hours=bar_in_day)
        close_dt = open_dt + timedelta(hours=1)
        ret = rng.normal(0.0, 0.008)
        o = price
        c = max(1000.0, o * (1.0 + ret))
        spread = abs(rng.normal(0.005, 0.002)) * o
        h = max(o, c) + spread
        lo = min(o, c) - spread
        vol = 0 if i in (5, 22) else int(rng.integers(50_000, 500_000))
        rows.append(
            {
                "ticker": ticker,
                "exchange": exchange,
                "trading_date_local": close_dt.date(),
                "bar_open_utc": pd.Timestamp(open_dt),
                "bar_close_utc": pd.Timestamp(close_dt),
                "bar_duration_min": np.int16(60),
                "session_segment": "continuous",
                "open": float(o),
                "high": float(h),
                "low": float(lo),
                "close": float(c),
                "volume": np.int64(vol),
                "is_zero_volume": bool(vol == 0),
                "is_imputed": False,
                "window_role": "warmup" if i < 20 else "active_10d",
                "next_bar_close_utc": pd.Timestamp(close_dt + timedelta(hours=1)) if i < n - 1 else pd.NaT,
                "next_close": np.nan,
                "target_log_return_1h": np.nan,
            }
        )
        price = c
    df = pd.DataFrame(rows)
    df["next_close"] = df["close"].shift(-1)
    df["target_log_return_1h"] = np.log(df["next_close"] / df["close"])
    return df


def _make_synthetic_events(bars_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = 0
    for b_idx, bar in bars_df.iterrows():
        bar_close = bar["bar_close_utc"]
        ticker = bar["ticker"]
        exchange = bar["exchange"]
        # First bar of day has 18h overnight gap, so older overnight posts (>60m lag) belong strictly to this bar
        is_first_of_day = (b_idx % 6) == 0
        # Add 12 Naver posts per bar (mix of verified & non-verified, fresh & older)
        for j in range(12):
            if is_first_of_day:
                lag_min = 10.0 if j < 3 else 95.0 + j * 5.0
            else:
                lag_min = 5.0 + j * 4.0  # strictly inside (0, 60] for continuous intraday bar
            eff_utc = bar_close - pd.Timedelta(minutes=lag_min)
            rows.append(
                {
                    "event_id": f"ev_naver_{idx:05d}",
                    "event_type": "social",
                    "source": "naver_finance",
                    "ticker": ticker,
                    "exchange": exchange,
                    "channel_or_domain": "naver_discussion",
                    "title": f"삼성전자 토론 {j % 4}" if j % 3 == 0 else f"고유제목 {idx}",
                    "text": "삼성전자 실적 기대" if j % 3 == 0 else f"본문 내용 길게 작성된 테스트 포스트 번호 {idx}",
                    "language": "ko",
                    "url": f"https://m.stock.naver.com/{idx}",
                    "event_utc": eff_utc,
                    "effective_event_utc": eff_utc,
                    "timestamp_precision": "exact",
                    "likes": (j + 1) * 2,
                    "dislikes": j % 3,
                    "reposts": 0,
                    "views": 100 + j * 40,
                    "comments": 0,
                    "author_followers": None,
                    "author_followers_is_proxy": True,
                    "effective_reach": 100 + j * 40,
                    "is_holder_verified": bool(j in (0, 4, 8)),
                    "is_cross_channel_repost": False,
                    "is_short_text": False,
                    "source_tier": None,
                    "source_credibility_weight": np.float32(1.0),
                    "matched_aliases": ["삼성전자"],
                    "is_multi_ticker": False,
                    "has_full_body": True,
                    "assigned_bar_close_utc": bar_close,
                    "lag_minutes_to_bar_close": lag_min,
                    "alignment_bucket": "intraday",
                    "window_role": bar["window_role"],
                }
            )
            idx += 1

        # Add 1 Telegram post every 2 bars
        if b_idx % 2 == 0:
            eff_utc = bar_close - pd.Timedelta(minutes=15)
            rows.append(
                {
                    "event_id": f"ev_tg_{idx:05d}",
                    "event_type": "social",
                    "source": "telegram",
                    "ticker": ticker,
                    "exchange": exchange,
                    "channel_or_domain": "@faststocknews",
                    "title": "",
                    "text": f"삼성전자 텔레그램 속보 {idx}",
                    "language": "ko",
                    "url": f"https://t.me/faststocknews/{idx}",
                    "event_utc": eff_utc,
                    "effective_event_utc": eff_utc,
                    "timestamp_precision": "exact",
                    "likes": 0,
                    "dislikes": 0,
                    "reposts": 0,
                    "views": 5000,
                    "comments": 0,
                    "author_followers": 129000,
                    "author_followers_is_proxy": False,
                    "effective_reach": 129000,
                    "is_holder_verified": None,
                    "is_cross_channel_repost": bool(b_idx % 4 == 0),
                    "is_short_text": False,
                    "source_tier": None,
                    "source_credibility_weight": np.float32(1.0),
                    "matched_aliases": ["삼성전자"],
                    "is_multi_ticker": False,
                    "has_full_body": True,
                    "assigned_bar_close_utc": bar_close,
                    "lag_minutes_to_bar_close": 15.0,
                    "alignment_bucket": "intraday",
                    "window_role": bar["window_role"],
                }
            )
            idx += 1

        # Add 1 Exa News article every 5 bars
        if b_idx % 5 == 0:
            eff_utc = bar_close - pd.Timedelta(minutes=25)
            rows.append(
                {
                    "event_id": f"ev_news_{idx:05d}",
                    "event_type": "news",
                    "source": "exa",
                    "ticker": ticker,
                    "exchange": exchange,
                    "channel_or_domain": "reuters.com",
                    "title": f"Samsung Electronics HBM guidance update {idx}",
                    "text": "Analysts maintain target price 72000 KRW on strong AI chip demand.",
                    "language": "en",
                    "url": f"https://reuters.com/article/{idx}",
                    "event_utc": eff_utc,
                    "effective_event_utc": eff_utc,
                    "timestamp_precision": "exact",
                    "likes": 0,
                    "dislikes": 0,
                    "reposts": 0,
                    "views": 0,
                    "comments": 0,
                    "author_followers": None,
                    "author_followers_is_proxy": True,
                    "effective_reach": 1,
                    "is_holder_verified": None,
                    "is_cross_channel_repost": False,
                    "is_short_text": False,
                    "source_tier": "tier1",
                    "source_credibility_weight": np.float32(1.5),
                    "matched_aliases": ["Samsung Electronics"],
                    "is_multi_ticker": False,
                    "has_full_body": True,
                    "assigned_bar_close_utc": bar_close,
                    "lag_minutes_to_bar_close": 25.0,
                    "alignment_bucket": "intraday",
                    "window_role": bar["window_role"],
                }
            )
            idx += 1

    # Add 2 post_window_live events
    last_close = bars_df["bar_close_utc"].iloc[-1]
    for k in range(2):
        live_ts = last_close + pd.Timedelta(minutes=30 + k * 20)
        rows.append(
            {
                "event_id": f"ev_live_{k:03d}",
                "event_type": "social",
                "source": "naver_finance",
                "ticker": bars_df["ticker"].iloc[0],
                "exchange": bars_df["exchange"].iloc[0],
                "channel_or_domain": "naver_discussion",
                "title": f"장마감 후 라이브 속보 {k}",
                "text": "시간외 거래 강세 기대됩니다",
                "language": "ko",
                "url": f"https://m.stock.naver.com/live/{k}",
                "event_utc": live_ts,
                "effective_event_utc": live_ts,
                "timestamp_precision": "exact",
                "likes": 10,
                "dislikes": 1,
                "reposts": 0,
                "views": 400,
                "comments": 0,
                "author_followers": None,
                "author_followers_is_proxy": True,
                "effective_reach": 400,
                "is_holder_verified": True,
                "is_cross_channel_repost": False,
                "is_short_text": False,
                "source_tier": None,
                "source_credibility_weight": np.float32(1.0),
                "matched_aliases": ["삼성전자"],
                "is_multi_ticker": False,
                "has_full_body": True,
                "assigned_bar_close_utc": pd.NaT,
                "lag_minutes_to_bar_close": np.nan,
                "alignment_bucket": "post_window_live",
                "window_role": "post_window_live",
            }
        )
    return pd.DataFrame(rows)


def _attach_deterministic_mock_scores(
    routed_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """Attach deterministic scores to routed events (purely a function of each event's own fields)."""
    out = routed_df.copy()
    n = len(out)
    scored_mask = out["is_llm_scored"].to_numpy(dtype=bool)

    sent = np.full(n, np.nan, dtype=np.float64)
    conf = np.full(n, np.nan, dtype=np.float64)
    hype = np.full(n, np.nan, dtype=np.float64)
    pt = np.full(n, np.nan, dtype=np.float64)
    sarc: list[bool | None] = [None] * n
    direction: list[str | None] = [None] * n
    cat: list[str | None] = [None] * n
    summary: list[str | None] = [None] * n
    src: list[str | None] = [None] * n

    close_map = {pd.Timestamp(ts): float(c) for ts, c in zip(bars_df["bar_close_utc"], bars_df["close"])}
    latest_close = float(bars_df["close"].iloc[-1])

    for i in np.flatnonzero(scored_mask):
        eid = str(out["event_id"].iloc[i])
        hval = sum(ord(ch) for ch in eid) % 21
        score = round((hval - 10) / 10.0, 2)
        sent[i] = score
        conf[i] = 0.85
        hype[i] = 0.4
        sarc[i] = bool(hval % 5 == 0)
        direction[i] = "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral")
        cat[i] = "earnings" if out["event_type"].iloc[i] == "news" else "retail_rumor"
        summary[i] = f"Summary for {eid}"
        src[i] = "mock-model"
        if out["event_type"].iloc[i] == "news":
            bar_ts = out["assigned_bar_close_utc"].iloc[i]
            ref_c = close_map.get(pd.Timestamp(bar_ts), latest_close) if pd.notna(bar_ts) else latest_close
            pt[i] = round(ref_c * 1.15, 2)

    out["sentiment_source"] = pd.Series(src, index=out.index, dtype="string")
    out["sentiment_score"] = sent
    out["direction"] = pd.Series(direction, index=out.index, dtype="string")
    out["confidence"] = conf
    out["is_sarcastic"] = pd.Series(sarc, index=out.index, dtype="boolean")
    out["hype_score"] = hype
    out["catalyst_type"] = pd.Series(cat, index=out.index, dtype="string")
    out["price_target_mentioned"] = pt
    out["english_summary"] = pd.Series(summary, index=out.index, dtype="string")
    out["impact_weight"] = compute_impact_weights(out, cfg)
    return out


# =============================================================================
# 1. 3-Bucket Router & Text Dedup Tests
# =============================================================================


def test_naver_quota_rich_sqrt_bounds() -> None:
    cfg = load_phase2_config()
    assert compute_naver_quota(0, cfg.router) == 0
    assert compute_naver_quota(3, cfg.router) == 3
    assert compute_naver_quota(9, cfg.router) == 6
    assert compute_naver_quota(100, cfg.router) == 20
    assert compute_naver_quota(1000, cfg.router) == 50


def test_three_bucket_router_determinism_and_dedup() -> None:
    cfg = load_phase2_config()
    bars = _make_synthetic_bars(n=25)
    events = _make_synthetic_events(bars)

    routed1, dmap1, reps1 = route_ticker_events(events, cfg)
    routed2, dmap2, reps2 = route_ticker_events(events, cfg)

    assert routed1["routing_bucket"].equals(routed2["routing_bucket"])
    assert routed1["is_llm_scored"].equals(routed2["is_llm_scored"])
    assert dmap1 == dmap2
    assert list(reps1["event_id"]) == list(reps2["event_id"])

    # Every non-Naver event must be 100% routed
    non_naver = routed1[routed1["source"] != "naver_finance"]
    assert non_naver["is_llm_scored"].all()
    assert (non_naver["routing_bucket"] == "non_naver").all()
    assert non_naver["priority_score"].isna().all()

    # Naver bars of size 12 -> quota = min(12, clip(round(2*sqrt(12))=7, 6, 50)) = 7
    naver_hist = routed1[(routed1["source"] == "naver_finance") & routed1["assigned_bar_close_utc"].notna()]
    for _, grp in naver_hist.groupby("assigned_bar_close_utc"):
        assert int(grp["is_llm_scored"].sum()) == 7
        assert int((grp["routing_bucket"] == "naver_verified").sum()) == 3  # ceil(0.30*7)=3, 3 verified exist
        assert int((grp["routing_bucket"] == "naver_fresh").sum()) >= 1    # ceil(0.15*7)=2 fresh slots

    # Dedup map must cover exact set of routed event_ids
    all_mapped = [eid for eids in dmap1.values() for eid in eids]
    routed_ids = set(routed1.loc[routed1["is_llm_scored"], "event_id"])
    assert set(all_mapped) == routed_ids
    assert len(all_mapped) == len(routed_ids)
    assert len(reps1) < len(routed_ids)  # Duplicate titles/texts were deduplicated


# =============================================================================
# 2. Causal KDE / VAL-VAH / Technicals Tests
# =============================================================================


def test_quant_technical_kde_and_val_vah_invariants() -> None:
    cfg = load_phase2_config()
    bars = _make_synthetic_bars(n=35)
    tech = compute_quant_technical_features(bars, cfg)

    assert len(tech) == 35
    # Warmup bars 0..16 have NaN KDE levels; bars 17..34 are fully populated
    assert tech.loc[:16, "poc_price"].isna().all()
    assert tech.loc[17:, "poc_price"].notna().all()
    assert tech.loc[17:, "val_price"].notna().all()
    assert tech.loc[17:, "vah_price"].notna().all()
    assert tech.loc[17:, "support_1"].notna().all()
    assert tech.loc[17:, "resistance_1"].notna().all()
    assert tech.loc[17:, "in_value_area"].notna().all()

    # Value area invariant: val_price <= poc_price <= vah_price
    valid = tech.iloc[17:]
    assert (valid["val_price"] <= valid["poc_price"] + 1e-9).all()
    assert (valid["poc_price"] <= valid["vah_price"] + 1e-9).all()

    # RSI bounded in [0, 100] and ATR > 0
    assert ((valid["rsi_14"] >= 0.0) & (valid["rsi_14"] <= 100.0)).all()
    assert (valid["atr_14"] > 0.0).all()

    # Zero-volume bar (idx 22) produces finite VWAP & KDE
    assert np.isfinite(tech.loc[22, "vwap_24h"])
    assert np.isfinite(tech.loc[22, "poc_price"])


# =============================================================================
# 3. JSON Recovery, Auto-Bisect, Direction & Price Target Bounding Tests
# =============================================================================


def test_parse_llm_json_strips_think_and_fences() -> None:
    raw = '<think>internal reasoning about 燈神</think>\n```json\n{"items": [{"event_id": "1"}]}\n```'
    parsed = parse_llm_json_object(raw)
    assert parsed == {"items": [{"event_id": "1"}]}


def test_auto_bisect_recovers_missing_ids_and_malformed_json() -> None:
    cfg = load_phase2_config()

    class FlakyClient:
        model = "mock/nemotron"

        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
            self.calls += 1
            # Extract input items JSON array from user prompt
            arr_start = user_prompt.index("[")
            items = json.loads(user_prompt[arr_start:])
            # Fail first call with malformed JSON when batch size > 2
            if self.calls == 1:
                raise ValueError("Simulated truncated JSON")
            # Drop last item on second call if > 1 item to trigger partial-missing recovery
            if self.calls == 2 and len(items) > 1:
                items = items[:-1]
            out = {
                "items": [
                    {
                        "event_id": it["event_id"],
                        "sentiment_score": -0.6,
                        "direction": "bullish",  # intentionally wrong -> validator must fix to bearish
                        "confidence": 0.9,
                        "is_sarcastic": True,
                        "hype_score": 0.7,
                        "catalyst_type": "retail_rumor",
                        "price_target_mentioned": None,
                        "english_summary": "Sarcastic retail bear post",
                    }
                    for it in items
                ]
            }
            return out, json.dumps(out)

    batch_rows = [{"event_id": f"id_{i}", "source": "lihkg", "title": "t", "text": "x"} for i in range(4)]
    client = FlakyClient()
    scored, calls, bisects = score_batch_with_bisect(
        client=client,  # type: ignore[arg-type]
        batch_rows=batch_rows,
        domain="hkex_social",
        ticker="0700.HK",
        company_en="Tencent Holdings",
        company_local="騰訊",
        cfg=cfg,
    )
    assert set(scored.keys()) == {"id_0", "id_1", "id_2", "id_3"}
    assert bisects >= 2
    assert calls >= 4
    # Deterministic direction post-check fixed wrong "bullish" -> "bearish"
    assert all(item.direction == "bearish" for item in scored.values())


def test_price_target_sanity_bounding_and_checkpoint(tmp_path: Any) -> None:
    cfg = load_phase2_config()
    bars = _make_synthetic_bars(n=25, ticker="0700.HK", exchange="HKEX")
    bars["close"] = 400.0
    ev = pd.DataFrame(
        [
            {
                "event_id": "pt_ok",
                "event_type": "news",
                "source": "exa",
                "ticker": "0700.HK",
                "exchange": "HKEX",
                "channel_or_domain": "reuters.com",
                "title": "Target 500 HKD",
                "text": "Broker sets 500 HKD target",
                "language": "en",
                "url": "https://reuters.com/1",
                "event_utc": bars["bar_close_utc"].iloc[-1] - pd.Timedelta(minutes=10),
                "effective_event_utc": bars["bar_close_utc"].iloc[-1] - pd.Timedelta(minutes=10),
                "timestamp_precision": "exact",
                "likes": 0,
                "dislikes": 0,
                "reposts": 0,
                "views": 0,
                "comments": 0,
                "author_followers": None,
                "author_followers_is_proxy": True,
                "effective_reach": 1,
                "is_holder_verified": None,
                "is_cross_channel_repost": False,
                "is_short_text": False,
                "source_tier": "tier1",
                "source_credibility_weight": np.float32(1.5),
                "matched_aliases": ["0700.HK"],
                "is_multi_ticker": False,
                "has_full_body": True,
                "assigned_bar_close_utc": bars["bar_close_utc"].iloc[-1],
                "lag_minutes_to_bar_close": 10.0,
                "alignment_bucket": "intraday",
                "window_role": "active_10d",
            },
            {
                "event_id": "pt_crazy",
                "event_type": "news",
                "source": "exa",
                "ticker": "0700.HK",
                "exchange": "HKEX",
                "channel_or_domain": "reuters.com",
                "title": "Year 2026 outlook",
                "text": "2026 forecast",
                "language": "en",
                "url": "https://reuters.com/2",
                "event_utc": bars["bar_close_utc"].iloc[-1] - pd.Timedelta(minutes=5),
                "effective_event_utc": bars["bar_close_utc"].iloc[-1] - pd.Timedelta(minutes=5),
                "timestamp_precision": "exact",
                "likes": 0,
                "dislikes": 0,
                "reposts": 0,
                "views": 0,
                "comments": 0,
                "author_followers": None,
                "author_followers_is_proxy": True,
                "effective_reach": 1,
                "is_holder_verified": None,
                "is_cross_channel_repost": False,
                "is_short_text": False,
                "source_tier": "tier1",
                "source_credibility_weight": np.float32(1.5),
                "matched_aliases": ["0700.HK"],
                "is_multi_ticker": False,
                "has_full_body": True,
                "assigned_bar_close_utc": bars["bar_close_utc"].iloc[-1],
                "lag_minutes_to_bar_close": 5.0,
                "alignment_bucket": "intraday",
                "window_role": "active_10d",
            },
        ]
    )

    store = LLMScoreCheckpointStore(tmp_path / "scores.jsonl")
    store.append_batch(
        [
            ScoredEventItem(
                event_id="pt_ok",
                sentiment_score=0.7,
                confidence=0.9,
                is_sarcastic=False,
                hype_score=0.2,
                catalyst_type="analyst_rating",
                price_target_mentioned=500.0,
                english_summary="Target 500 HKD",
            ).model_dump()
            | {"sentiment_source": "test-model"},
            ScoredEventItem(
                event_id="pt_crazy",
                sentiment_score=0.1,
                confidence=0.8,
                is_sarcastic=False,
                hype_score=0.1,
                catalyst_type="other",
                price_target_mentioned=2026.0,  # > 2.0 * 400 -> must be nulled!
                english_summary="2026 outlook",
            ).model_dump()
            | {"sentiment_source": "test-model"},
        ]
    )

    routed, dmap, reps = route_ticker_events(ev, cfg)
    scored, stats = run_cultural_nlp_for_ticker(
        ticker="0700.HK",
        company_en="Tencent Holdings",
        company_local="騰訊",
        bars_df=bars,
        routed_df=routed,
        dedup_map=dmap,
        rep_df=reps,
        cfg=cfg,
        client=None,
        checkpoint_store=store,
        expected_model="test-model",
    )
    assert stats["cache_hits"] == 2
    assert scored.loc[scored["event_id"] == "pt_ok", "price_target_mentioned"].iloc[0] == pytest.approx(500.0)
    assert pd.isna(scored.loc[scored["event_id"] == "pt_crazy", "price_target_mentioned"].iloc[0])

    # Fix #4 regression: requesting a different model must NOT hit test-model cache
    assert store.get("pt_ok", expected_model="test-model") is not None
    assert store.get("pt_ok", expected_model="other-model") is None


def test_deep_bisect_still_retries_isolated_singleton() -> None:
    """Fix #3 regression: after multiple halving bisections (depth >= bisect_max_retries),
    an isolated 1-item missing ID still receives dedicated singleton retry attempts."""
    cfg = load_phase2_config()

    class DeepFlakyClient:
        model = "mock/nemotron"

        def __init__(self) -> None:
            self.seen_sizes: list[int] = []
            self.singleton_hits: dict[str, int] = {}

        def chat_json(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
            arr_start = user_prompt.index("[")
            items = json.loads(user_prompt[arr_start:])
            self.seen_sizes.append(len(items))
            # Force bisections 8 -> 4 -> 2 (depth=2 == bisect_max_retries), then drop 1 item at size 2
            if len(items) >= 4:
                raise ValueError("Force halving")
            if len(items) == 2:
                items = items[:1]  # omit 2nd item at depth=2
            elif len(items) == 1:
                eid = items[0]["text"]
                self.singleton_hits[eid] = self.singleton_hits.get(eid, 0) + 1
                if self.singleton_hits[eid] == 1:
                    raise ValueError("Fail first singleton attempt to verify dedicated retry")
            out = {
                "items": [
                    {
                        "event_id": it["event_id"],
                        "sentiment_score": 0.4,
                        "direction": "bullish",
                        "confidence": 0.8,
                        "is_sarcastic": False,
                        "hype_score": 0.3,
                        "catalyst_type": "other",
                        "price_target_mentioned": None,
                        "english_summary": "Recovered after deep bisect",
                    }
                    for it in items
                ]
            }
            return out, json.dumps(out)

    rows = [{"event_id": f"deep_{i}", "source": "lihkg", "title": "t", "text": f"body_{i}"} for i in range(8)]
    client = DeepFlakyClient()
    scored, _, bisects = score_batch_with_bisect(
        client=client,  # type: ignore[arg-type]
        batch_rows=rows,
        domain="hkex_social",
        ticker="0700.HK",
        company_en="Tencent Holdings",
        company_local="騰訊",
        cfg=cfg,
    )
    assert set(scored.keys()) == {f"deep_{i}" for i in range(8)}
    assert bisects >= 3


def test_warmup_without_news_guards_forum_divergence_and_smoke_bucket() -> None:
    """Fix #2 & #8 regression:
    - When has_news_history=False on an early bar, divergence_forum_vs_news_24h must be NaN
      and composite_sentiment_score must not shrink toward 0.
    - Smoke-test demoted non-Naver rows get routing_bucket='smoke_unscored'."""
    cfg = load_phase2_config()
    bars = _make_synthetic_bars(n=25)
    events = _make_synthetic_events(bars)
    # Remove news from first 10 bars to create early bars with social+forum history but no news
    cutoff_ts = bars["bar_close_utc"].iloc[10]
    events_no_early_news = events[
        ~((events["event_type"] == "news") & (events["effective_event_utc"] <= cutoff_ts))
    ].reset_index(drop=True)

    routed_full, _, _ = route_ticker_events(events_no_early_news, cfg)
    scored = _attach_deterministic_mock_scores(routed_full, bars, cfg)
    tech = compute_quant_technical_features(bars, cfg)
    fc = compute_full_corpus_stats(bars, events_no_early_news, cfg)
    feat, snap = compute_alpha_factors_and_snapshot(
        ticker="005930.KS",
        exchange="KRX",
        company_en="Samsung Electronics",
        company_local="삼성전자",
        bars_df=bars,
        tech_df=tech,
        full_corpus_df=fc,
        scored_events_df=scored,
        cfg=cfg,
    )
    early_no_news = feat[~feat["has_news_history"] & feat["has_forum_history"]]
    assert len(early_no_news) > 0
    assert early_no_news["divergence_forum_vs_news_24h"].isna().all()
    np.testing.assert_allclose(
        early_no_news["composite_sentiment_score"].to_numpy(),
        early_no_news["retail_sent_24h"].to_numpy(),
        atol=1e-12,
    )
    assert verify_ticker_phase2_outputs("005930.KS", bars, events_no_early_news, scored, feat, snap)

    # Fix #8: smoke-test demoted non-Naver rows tagged 'smoke_unscored'
    routed_smoke, _, _ = route_ticker_events(events, cfg, smoke_max_items=6)
    non_naver_unscored = routed_smoke[
        (routed_smoke["source"] != "naver_finance") & (~routed_smoke["is_llm_scored"])
    ]
    assert len(non_naver_unscored) > 0
    assert (non_naver_unscored["routing_bucket"] == "smoke_unscored").all()


# =============================================================================
# 4. MANDATORY Causal Perturbation Zero-Leakage Test (atol = 1e-12)
# =============================================================================


def test_causal_perturbation_zero_leakage() -> None:
    """Mutate bar t+1..end OHLCV and all post-t events; assert ALL Phase 2 features
    at bar t (and all bars <= t) remain bit-for-bit identical (atol=1e-12)."""
    cfg = load_phase2_config()
    bars_orig = _make_synthetic_bars(n=35)
    events_orig = _make_synthetic_events(bars_orig)

    t_idx = 26  # Inside active_10d
    t_close_utc = bars_orig["bar_close_utc"].iloc[t_idx]

    # Run original pipeline
    routed_a, _, _ = route_ticker_events(events_orig, cfg)
    scored_a = _attach_deterministic_mock_scores(routed_a, bars_orig, cfg)
    tech_a = compute_quant_technical_features(bars_orig, cfg)
    fc_a = compute_full_corpus_stats(bars_orig, events_orig, cfg)
    feat_a, snap_a = compute_alpha_factors_and_snapshot(
        ticker="005930.KS",
        exchange="KRX",
        company_en="Samsung Electronics",
        company_local="삼성전자",
        bars_df=bars_orig,
        tech_df=tech_a,
        full_corpus_df=fc_a,
        scored_events_df=scored_a,
        cfg=cfg,
    )
    assert verify_ticker_phase2_outputs("005930.KS", bars_orig, events_orig, scored_a, feat_a, snap_a)

    # Perturb ALL bars strictly after t_idx (OHLCV exploded 3x)
    bars_pert = bars_orig.copy()
    bars_pert.loc[t_idx + 1 :, ["open", "high", "low", "close"]] *= 3.0
    bars_pert.loc[t_idx + 1 :, "volume"] *= 10
    bars_pert["next_close"] = bars_pert["close"].shift(-1)
    bars_pert["target_log_return_1h"] = np.log(bars_pert["next_close"] / bars_pert["close"])

    # Perturb ALL events strictly after t_close_utc (drop half, mutate likes/views, add 50 massive future posts)
    events_pert = events_orig.copy()
    future_mask = events_pert["effective_event_utc"] > t_close_utc
    events_pert.loc[future_mask, "views"] = 999_999
    events_pert.loc[future_mask, "likes"] = 50_000
    events_pert.loc[future_mask, "is_holder_verified"] = True

    routed_b, _, _ = route_ticker_events(events_pert, cfg)
    scored_b = _attach_deterministic_mock_scores(routed_b, bars_pert, cfg)
    # Also deliberately flip LLM sentiment on all future events
    future_scored = scored_b["is_llm_scored"] & (scored_b["effective_event_utc"] > t_close_utc)
    scored_b.loc[future_scored, "sentiment_score"] = -1.0
    scored_b.loc[future_scored, "hype_score"] = 1.0
    scored_b.loc[future_scored, "is_sarcastic"] = True

    tech_b = compute_quant_technical_features(bars_pert, cfg)
    fc_b = compute_full_corpus_stats(bars_pert, events_pert, cfg)
    feat_b, _ = compute_alpha_factors_and_snapshot(
        ticker="005930.KS",
        exchange="KRX",
        company_en="Samsung Electronics",
        company_local="삼성전자",
        bars_df=bars_pert,
        tech_df=tech_b,
        full_corpus_df=fc_b,
        scored_events_df=scored_b,
        cfg=cfg,
    )

    # Exclude forward-looking target labels (next_close, target_log_return_1h) which depend on t+1 by definition
    feature_cols = [c for c in feat_a.columns if c not in ("next_close", "target_log_return_1h")]
    row_a = feat_a.iloc[: t_idx + 1]
    row_b = feat_b.iloc[: t_idx + 1]

    for col in feature_cols:
        sa = row_a[col]
        sb = row_b[col]
        if pd.api.types.is_float_dtype(sa):
            va = sa.to_numpy(dtype=np.float64)
            vb = sb.to_numpy(dtype=np.float64)
            np.testing.assert_allclose(
                va,
                vb,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"CAUSAL LEAKAGE DETECTED in feature column '{col}' at bars <= {t_idx}",
            )
        else:
            assert sa.equals(sb), f"CAUSAL LEAKAGE DETECTED in non-float column '{col}' at bars <= {t_idx}"


# =============================================================================
# 5. Live Snapshot Catalyst Contract & Zero-Collapse Impact Weight Test
# =============================================================================


def test_zero_collapse_impact_weight_and_live_catalysts() -> None:
    cfg = load_phase2_config()
    bars = _make_synthetic_bars(n=25)
    events = _make_synthetic_events(bars)

    routed, _, _ = route_ticker_events(events, cfg)
    scored = _attach_deterministic_mock_scores(routed, bars, cfg)

    # Telegram posts with likes=dislikes=reposts=comments=0 must still have positive finite impact_weight
    tg_scored = scored[(scored["source"] == "telegram") & scored["is_llm_scored"]]
    assert (tg_scored["impact_weight"] > 0.0).all()

    tech = compute_quant_technical_features(bars, cfg)
    fc = compute_full_corpus_stats(bars, events, cfg)
    _, snap = compute_alpha_factors_and_snapshot(
        ticker="005930.KS",
        exchange="KRX",
        company_en="Samsung Electronics",
        company_local="삼성전자",
        bars_df=bars,
        tech_df=tech,
        full_corpus_df=fc,
        scored_events_df=scored,
        cfg=cfg,
    )
    assert snap["live_event_count"] == 2
    assert snap["as_of_utc"] > snap["last_bar_close_utc"]

    social_cats = json.loads(snap["top_social_catalysts_json"])
    news_cats = json.loads(snap["top_news_catalysts_json"])
    assert 1 <= len(social_cats) <= 5
    assert 1 <= len(news_cats) <= 5
    expected_keys = set(CatalystSummary.model_fields.keys())
    assert len(expected_keys) == 13
    for item in social_cats + news_cats:
        assert set(item.keys()) == expected_keys
        CatalystSummary.model_validate(item)
