"""$0.00 Unit Tests for Phase 1: Config, Prices, Attribution, Hygiene, Dedup, Alignment & Parquet Schemas."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingest.align import align_events_to_bars, verify_phase1_alignment, write_events_parquet
from src.ingest.news_exa import canonicalize_url, deduplicate_news_records, parse_news_timestamp
from src.ingest.normalize import (
    build_ticker_matchers,
    compute_effective_event_utc,
    detect_language,
    match_tickers_in_text,
    normalize_events,
)
from src.ingest.prices import _apply_quality_checks_and_targets, _map_raw_records_to_bars, write_bars_parquet
from src.ingest.schemas import (
    BARS_1H_SCHEMA,
    EVENTS_ALIGNED_SCHEMA,
    ExchangeWindowBounds,
    load_sources,
    load_universe,
)


def _make_synthetic_14d_bars(exchange: str, ticker: str) -> tuple[pd.DataFrame, ExchangeWindowBounds]:
    universe = load_universe()
    ex_cfg = universe.exchanges[exchange]  # type: ignore[index]
    tcfg = next(t for t in universe.tickers if t.ticker == ticker)
    tz = ZoneInfo(ex_cfg.timezone)

    # Generate 14 weekday dates
    dates: list[date] = []
    cur = date(2026, 10, 20)
    while len(dates) < 14:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)

    open_slots = (
        ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
        if exchange == "HKEX"
        else ["09:00", "10:00", "11:00", "12:00", "13:00", "14:00"]
    )

    raw_records = []
    px = 100.0
    for d in dates:
        for slot in open_slots:
            hh, mm = map(int, slot.split(":"))
            ts = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)
            px += 0.5
            raw_records.append(
                {
                    "ticker": ticker,
                    "exchange": exchange,
                    "raw_timestamp": ts.isoformat(),
                    "open": px - 0.2,
                    "high": px + 0.3,
                    "low": px - 0.4,
                    "close": px,
                    "volume": 100000,
                }
            )

    run_utc = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
    mapped = _map_raw_records_to_bars(raw_records, tcfg, ex_cfg, run_utc=run_utc, end_date_limit=None)
    warmup = dates[:4]
    active = dates[4:]
    bars = _apply_quality_checks_and_targets(mapped, tcfg, ex_cfg, set(warmup), set(active))

    first_open = bars["bar_open_utc"].min().to_pydatetime()
    window_end = bars["bar_close_utc"].max().to_pydatetime()
    close_map = {
        d.isoformat(): grp["bar_close_utc"].max().to_pydatetime()
        for d, grp in bars.groupby("trading_date_local")
    }
    bounds = ExchangeWindowBounds(
        exchange=exchange,  # type: ignore[arg-type]
        timezone=ex_cfg.timezone,
        trading_dates_local=dates,
        warmup_dates_local=warmup,
        active_dates_local=active,
        first_warmup_bar_open_utc=first_open,
        exchange_warmup_lead_utc=first_open - timedelta(hours=18),
        exchange_window_end_utc=window_end,
        session_close_by_date_utc=close_map,
    )
    return bars, bounds


def test_config_loading_and_ticker_selection() -> None:
    universe = load_universe()
    sources = load_sources()
    assert len(universe.get_tickers("all")) == 10
    assert len(universe.get_tickers("hkex")) == 5
    assert len(universe.get_tickers("krx")) == 5

    smoke_tickers = universe.get_tickers(
        "all",
        smoke_test=True,
        smoke_tickers_by_exchange=sources["smoke_test"]["tickers_by_exchange"],
    )
    assert [t.ticker for t in smoke_tickers] == ["0700.HK", "005930.KS"]


def test_hkex_session_mapping_and_completed_bar_rule() -> None:
    bars, _ = _make_synthetic_14d_bars("HKEX", "0700.HK")
    # 14 days * 7 bars/day = 98 bars
    assert len(bars) == 98
    assert bars["window_role"].value_counts().to_dict() == {"active_10d": 70, "warmup": 28}
    assert pd.isna(bars.iloc[-1]["next_bar_close_utc"])
    assert pd.isna(bars.iloc[-1]["target_log_return_1h"])
    assert bars.iloc[:-1]["target_log_return_1h"].notna().all()

    # Check afternoon open (mapped from 12:30 Yahoo slot) has exact 13:00->13:30 HKT = 05:00->05:30 UTC
    first_day = bars[bars["trading_date_local"] == bars["trading_date_local"].iloc[0]]
    bar_1300 = first_day.iloc[3]
    assert bar_1300["bar_open_utc"].strftime("%H:%M") == "05:00"
    assert bar_1300["bar_close_utc"].strftime("%H:%M") == "05:30"
    assert int(bar_1300["bar_duration_min"]) == 30
    assert bar_1300["session_segment"] == "afternoon"


def test_cjk_and_cashtag_multi_ticker_attribution() -> None:
    universe = load_universe()
    tickers = universe.get_tickers("all")
    matchers = build_ticker_matchers(tickers)

    # Multi-ticker HKEX post
    hits_hk = match_tickers_in_text("今日 $0700.HK 騰訊同阿里巴巴齊升!", matchers["HKEX"])
    assert {t.ticker for t, _ in hits_hk} == {"0700.HK", "9988.HK"}

    # Boundary check: 07001 must NOT match 0700
    assert match_tickers_in_text("Order ID 07001 confirmed", matchers["HKEX"]) == []

    # KRX Hangul slang attribution
    hits_kr = match_tickers_in_text("삼전 실적 호조, 닉스 HBM 수혜!", matchers["KRX"])
    assert {t.ticker for t, _ in hits_kr} == {"005930.KS", "000660.KS"}


def test_news_dedup_and_timestamp_precision() -> None:
    assert canonicalize_url("https://www.reuters.com/a/b/?utm_source=x&id=1#top") == "https://www.reuters.com/a/b?id=1"

    dt1, p1 = parse_news_timestamp("2026-11-10")
    assert dt1 is not None and p1 == "date_only"
    dt2, p2 = parse_news_timestamp("2026-11-10T00:00:00.000Z")
    assert dt2 is not None and p2 == "date_only"
    dt3, p3 = parse_news_timestamp("2026-11-10T04:15:00Z")
    assert dt3 is not None and p3 == "exact"

    recs = [
        # Tier-2 date_only
        {
            "url": "https://blog.example.com/tencent-q3?utm_medium=rss",
            "title": "Tencent Q3 profit beats estimates on gaming rebound",
            "published_date": "2026-11-10",
            "source_tier": "tier2",
        },
        # Tier-1 exact syndicated headline (>0.85 similarity within 24h) -> must win!
        {
            "url": "https://reuters.com/tencent-q3-profit",
            "title": "Tencent Q3 profit beats estimates on gaming rebound!",
            "published_date": "2026-11-10T06:00:00Z",
            "source_tier": "tier1",
        },
        # Missing date -> dropped
        {
            "url": "https://reuters.com/undated",
            "title": "Undated article",
            "published_date": None,
            "source_tier": "tier1",
        },
    ]
    kept, n_dedup, n_invalid = deduplicate_news_records(recs)
    assert len(kept) == 1
    assert kept[0]["source_tier"] == "tier1"
    assert kept[0]["timestamp_precision"] == "exact"
    assert n_dedup == 1
    assert n_invalid == 1


def test_normalize_align_and_parquet_roundtrip(tmp_path: Path) -> None:
    universe = load_universe()
    sources = load_sources()
    bars_hk, win_hk = _make_synthetic_14d_bars("HKEX", "0700.HK")
    bars_kr, win_kr = _make_synthetic_14d_bars("KRX", "005930.KS")
    bars_all = pd.concat([bars_hk, bars_kr], ignore_index=True)

    active_tickers = [t for t in universe.tickers if t.ticker in ("0700.HK", "005930.KS")]
    windows = {"HKEX": win_hk, "KRX": win_kr}

    d0_hk = win_hk.trading_dates_local[0]
    d5_hk = win_hk.trading_dates_local[5]
    d5_kr = win_kr.trading_dates_local[5]

    raw_social = [
        # Pre-warmup HKEX Telegram -> strictly dropped
        {
            "source": "telegram",
            "exchange": "HKEX",
            "channel_handle": "Financial_Express",
            "channel_subscribers": 37000,
            "message_id": "1",
            "text": "騰訊 舊消息",
            "published_utc": (win_hk.exchange_warmup_lead_utc - timedelta(hours=1)).isoformat(),
            "views": 100,
            "reactions_total": 2,
            "url": "https://t.me/Financial_Express/1",
        },
        # HKEX lunch-break Telegram (12:20 HKT = 04:20 UTC on d5_hk) -> aligns to 13:30 HKT (05:30 UTC), bucket=lunch_break
        {
            "source": "telegram",
            "exchange": "HKEX",
            "channel_handle": "Financial_Express",
            "channel_subscribers": 37000,
            "message_id": "2",
            "text": "騰訊 0700 午間消息 🚀",
            "published_utc": datetime(d5_hk.year, d5_hk.month, d5_hk.day, 4, 20, tzinfo=timezone.utc).isoformat(),
            "views": 500,
            "reactions_total": 10,
            "url": "https://t.me/Financial_Express/2",
        },
        # Cross-channel Telegram repost 30 mins later on @jin10data -> flagged is_cross_channel_repost=True
        {
            "source": "telegram",
            "exchange": "HKEX",
            "channel_handle": "jin10data",
            "channel_subscribers": 11500,
            "message_id": "9",
            "text": "騰訊  0700 午間消息 🚀",
            "published_utc": datetime(d5_hk.year, d5_hk.month, d5_hk.day, 4, 50, tzinfo=timezone.utc).isoformat(),
            "views": 200,
            "reactions_total": 3,
            "url": "https://t.me/jin10data/9",
        },
        # HKEX LIHKG post (exact bar close at 10:30 HKT = 02:30 UTC -> lag=0.0, intraday)
        {
            "source": "lihkg",
            "exchange": "HKEX",
            "thread_id": "999001",
            "reply_id": None,
            "title": "0700 騰訊升爆",
            "text": "🚀",
            "published_utc": datetime(d5_hk.year, d5_hk.month, d5_hk.day, 2, 30, tzinfo=timezone.utc).isoformat(),
            "likes": 40,
            "dislikes": 5,
            "reply_count": 20,
            "category": "財經台",
            "url": "https://lihkg.com/thread/999001/page/1",
            "has_full_body": True,
        },
        # KRX Telegram
        {
            "source": "telegram",
            "exchange": "KRX",
            "channel_handle": "faststocknews",
            "channel_subscribers": 129000,
            "message_id": "55",
            "text": "삼성전자 외국인 순매수",
            "published_utc": datetime(d5_kr.year, d5_kr.month, d5_kr.day, 1, 15, tzinfo=timezone.utc).isoformat(),
            "views": 1000,
            "reactions_total": 5,
            "url": "https://t.me/faststocknews/55",
        },
        # KRX Naver valid short slang post (kept, is_short_text=True)
        {
            "source": "naver_finance",
            "exchange": "KRX",
            "ticker": "005930.KS",
            "krx_code": "005930",
            "source_item_id": "777",
            "title": "가즈아",
            "text": "",
            "written_at_kst": f"{d5_kr.isoformat()}T10:30:00",
            "views": 120,
            "likes": 4,
            "dislikes": 1,
            "is_holder_verified": True,
            "is_cleanbot_passed": True,
            "has_full_body": True,
            "url": "https://m.stock.naver.com/domestic/stock/005930/discussion/777",
        },
        # KRX Naver cleanbot spam -> dropped
        {
            "source": "naver_finance",
            "exchange": "KRX",
            "ticker": "005930.KS",
            "krx_code": "005930",
            "source_item_id": "778",
            "title": "spam",
            "text": "spam",
            "written_at_kst": f"{d5_kr.isoformat()}T10:35:00",
            "views": 1,
            "likes": 0,
            "dislikes": 0,
            "is_holder_verified": False,
            "is_cleanbot_passed": False,
            "has_full_body": True,
            "url": "https://m.stock.naver.com/domestic/stock/005930/discussion/778",
        },
        # Post-window live event -> assigned_bar_close_utc is None, post_window_live
        {
            "source": "telegram",
            "exchange": "KRX",
            "channel_handle": "faststocknews",
            "channel_subscribers": 129000,
            "message_id": "999",
            "text": "삼성전자 장 마감 후 속보",
            "published_utc": (win_kr.exchange_window_end_utc + timedelta(hours=2)).isoformat(),
            "views": 50,
            "reactions_total": 1,
            "url": "https://t.me/faststocknews/999",
        },
    ]

    raw_news = [
        # Date-only news on d0_hk -> snaps to d0_hk session_close_utc (16:00 HKT = 08:00 UTC), bucket=date_only_eod
        {
            "source": "exa",
            "exchange": "HKEX",
            "ticker": "0700.HK",
            "source_item_id": "n1",
            "domain": "reuters.com",
            "title": "Tencent expands cloud AI offerings",
            "text": "Tencent Holdings announced new AI models.",
            "url": "https://reuters.com/tencent-ai",
            "published_date": d0_hk.isoformat(),
            "language_hint": "en",
            "source_tier": "tier1",
            "source_credibility_weight": 1.5,
            "has_full_body": True,
        },
        # Exact KRX news
        {
            "source": "exa",
            "exchange": "KRX",
            "ticker": "005930.KS",
            "source_item_id": "n2",
            "domain": "kedglobal.com",
            "title": "Samsung Electronics HBM4 ramp accelerates",
            "text": "Samsung Electronics reported strong foundry demand.",
            "url": "https://kedglobal.com/samsung-hbm4",
            "published_date": datetime(d5_kr.year, d5_kr.month, d5_kr.day, 2, 10, tzinfo=timezone.utc).isoformat(),
            "language_hint": "en",
            "source_tier": "tier1",
            "source_credibility_weight": 1.5,
            "has_full_body": True,
        },
    ]

    norm_events, stats = normalize_events(
        raw_social_records=raw_social,
        raw_news_records=raw_news,
        universe=universe,
        active_tickers=active_tickers,
        exchange_windows=windows,
        sources_cfg=sources,
    )
    assert stats["telegram"].dropped_pre_warmup_count == 1
    assert stats["telegram"].cross_channel_repost_count == 1
    assert stats["naver_finance"].dropped_cleanbot_count == 1

    aligned = align_events_to_bars(norm_events, bars_all, universe)
    summary = verify_phase1_alignment(bars_all, aligned, active_tickers, universe)
    assert summary["assertions_passed"] is True

    # Verify specific alignment buckets
    lunch_row = aligned[aligned["url"] == "https://t.me/Financial_Express/2"].iloc[0]
    assert lunch_row["alignment_bucket"] == "lunch_break"
    assert lunch_row["is_cross_channel_repost"] == False  # noqa: E712

    repost_row = aligned[aligned["url"] == "https://t.me/jin10data/9"].iloc[0]
    assert repost_row["is_cross_channel_repost"] == True  # noqa: E712

    eod_row = aligned[aligned["url"] == "https://reuters.com/tencent-ai"].iloc[0]
    assert eod_row["alignment_bucket"] == "date_only_eod"
    assert eod_row["effective_event_utc"] == eod_row["assigned_bar_close_utc"]
    assert eod_row["lag_minutes_to_bar_close"] == 0.0

    live_row = aligned[aligned["url"] == "https://t.me/faststocknews/999"].iloc[0]
    assert pd.isna(live_row["assigned_bar_close_utc"])
    assert live_row["alignment_bucket"] == "post_window_live"
    assert live_row["window_role"] == "post_window_live"

    short_row = aligned[aligned["url"] == "https://m.stock.naver.com/domestic/stock/005930/discussion/777"].iloc[0]
    assert short_row["is_short_text"] == True  # noqa: E712
    assert short_row["is_holder_verified"] == True  # noqa: E712

    # Parquet exact schema round-trip test
    bars_pq = write_bars_parquet(bars_all, out_path=tmp_path / "bars_1h.parquet")
    events_pq = write_events_parquet(aligned, out_path=tmp_path / "events_aligned.parquet")
    assert pq.read_schema(bars_pq).equals(BARS_1H_SCHEMA)
    assert pq.read_schema(events_pq).equals(EVENTS_ALIGNED_SCHEMA)


def test_language_detector() -> None:
    assert detect_language("騰訊控股業績超預期", "") == "zh"
    assert detect_language("삼성전자 실적 발표", "") == "ko"
    assert detect_language("Tencent Q3 earnings beat expectations", "") == "en"
    assert detect_language("Samsung Electronics 삼성전자 HBM4 실적", "") == "mixed"
