"""Normalization, CJK-safe multi-ticker attribution, hygiene, repost flagging, and reach normalization (Phase 1 Section 1.5 & 1.8)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.ingest.news_exa import deduplicate_news_records, parse_news_timestamp
from src.ingest.schemas import ExchangeCode, ExchangeConfig, ExchangeWindowBounds, TickerConfig, UniverseConfig

logger = logging.getLogger(__name__)

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u3130-\u318f]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass
class NormalizationStats:
    raw_count: int = 0
    kept_count: int = 0
    dropped_empty_count: int = 0
    dropped_cleanbot_count: int = 0
    dropped_pre_warmup_count: int = 0
    dropped_dedup_count: int = 0
    dropped_unmatched_count: int = 0
    dropped_invalid_ts_count: int = 0
    cross_channel_repost_count: int = 0


@dataclass
class CompiledTickerMatcher:
    ticker_cfg: TickerConfig
    patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)


def build_ticker_matchers(tickers: list[TickerConfig]) -> dict[ExchangeCode, list[CompiledTickerMatcher]]:
    """Compile CJK-safe regex matchers per exchange (cashtags + ASCII boundary + CJK/Hangul substring)."""
    by_exchange: dict[ExchangeCode, list[CompiledTickerMatcher]] = {"HKEX": [], "KRX": []}
    for tcfg in tickers:
        pats: list[tuple[str, re.Pattern[str]]] = []
        for tag in tcfg.cashtags:
            pats.append((tag, re.compile(rf"(?i){re.escape(tag)}(?![A-Za-z0-9])")))
        for alias in tcfg.ascii_aliases:
            pats.append((alias, re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])")))
        for cjk in tcfg.cjk_aliases:
            pats.append((cjk, re.compile(re.escape(cjk), re.IGNORECASE)))
        by_exchange[tcfg.exchange].append(CompiledTickerMatcher(ticker_cfg=tcfg, patterns=pats))
    return by_exchange


def match_tickers_in_text(
    text: str,
    matchers: list[CompiledTickerMatcher],
) -> list[tuple[TickerConfig, list[str]]]:
    """Return all matched (TickerConfig, matched_aliases) pairs for a post."""
    if not text:
        return []
    hits: list[tuple[TickerConfig, list[str]]] = []
    for matcher in matchers:
        matched_aliases: list[str] = []
        seen: set[str] = set()
        for label, pat in matcher.patterns:
            if label not in seen and pat.search(text):
                seen.add(label)
                matched_aliases.append(label)
        if matched_aliases:
            hits.append((matcher.ticker_cfg, matched_aliases))
    return hits


def detect_language(title: str, text: str, fallback: str = "en") -> str:
    """Fast deterministic script-based language detector returning 'zh' | 'ko' | 'en' | 'mixed'."""
    combined = f"{title} {text}"
    n_han = len(_HAN_RE.findall(combined))
    n_ko = len(_HANGUL_RE.findall(combined))
    n_en = len(_ASCII_LETTER_RE.findall(combined))
    total = n_han + n_ko + n_en
    if total == 0:
        return fallback

    # Count distinct major scripts (weighted: 1 CJK/Hangul char ~ 2.5 ASCII chars of information)
    w_han = n_han * 2.5
    w_ko = n_ko * 2.5
    w_en = float(n_en)
    w_tot = w_han + w_ko + w_en

    present = [
        lang
        for lang, weight in (("zh", w_han), ("ko", w_ko), ("en", w_en))
        if (weight / w_tot) >= 0.25 and weight >= 5.0
    ]
    if len(present) >= 2:
        return "mixed"
    if w_ko >= w_han and w_ko >= w_en and n_ko > 0:
        return "ko"
    if w_han >= w_ko and w_han >= w_en and n_han > 0:
        return "zh"
    if n_en > 0:
        return "en"
    return fallback


def compute_event_id(event_type: str, source: str, source_item_id: str, ticker: str) -> str:
    raw = f"{event_type}|{source}|{source_item_id}|{ticker}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_social_timestamp(raw_ts: str | None, default_tz_name: str) -> datetime | None:
    if not raw_ts or not isinstance(raw_ts, str):
        return None
    s = raw_ts.strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(default_tz_name))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def compute_effective_event_utc(
    event_utc: datetime,
    timestamp_precision: str,
    window: ExchangeWindowBounds,
    ex_cfg: ExchangeConfig,
) -> datetime:
    """Compute zero-look-ahead effective_event_utc.

    - exact timestamps remain unchanged.
    - date_only news snaps to that local calendar day's session_close_utc (or exchange
      session_close_local in UTC if weekend/holiday, guaranteeing assignment strictly
      to the next open session with zero look-ahead).
    """
    if timestamp_precision != "date_only":
        return event_utc

    day_str = event_utc.date().isoformat()
    if day_str in window.session_close_by_date_utc:
        return window.session_close_by_date_utc[day_str]

    tz = ZoneInfo(ex_cfg.timezone)
    hh, mm = ex_cfg.session_close_local.split(":")
    close_local = datetime.combine(date.fromisoformat(day_str), time(int(hh), int(mm)), tzinfo=tz)
    return close_local.astimezone(timezone.utc)


def _record_richness_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        1 if row["has_full_body"] else 0,
        len(row["text"]) + len(row["title"]),
        int(row["views"]),
        int(row["likes"]) + int(row["comments"]),
    )


def normalize_events(
    raw_social_records: list[dict[str, Any]],
    raw_news_records: list[dict[str, Any]],
    universe: UniverseConfig,
    active_tickers: list[TickerConfig],
    exchange_windows: dict[ExchangeCode, ExchangeWindowBounds],
    sources_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, NormalizationStats]]:
    """Normalize raw social & news records into canonical event dicts ready for temporal alignment."""
    matchers_by_ex = build_ticker_matchers(active_tickers)
    active_ticker_map = {t.ticker: t for t in active_tickers}
    stats_by_source: dict[str, NormalizationStats] = {}

    def get_stats(src: str) -> NormalizationStats:
        return stats_by_source.setdefault(src, NormalizationStats())

    normalized: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Social records (telegram, naver_finance, lihkg, aastocks, blind)
    # -------------------------------------------------------------------------
    for rec in raw_social_records:
        source = str(rec.get("source") or "unknown")
        st = get_stats(source)
        st.raw_count += 1

        exchange_str = str(rec.get("exchange") or "").upper()
        if exchange_str not in exchange_windows:
            continue
        exchange: ExchangeCode = exchange_str  # type: ignore[assignment]
        window = exchange_windows[exchange]
        ex_cfg = universe.exchanges[exchange]

        # Hygiene check #1: Naver Cleanbot spam filter
        if source == "naver_finance" and rec.get("is_cleanbot_passed") is False:
            st.dropped_cleanbot_count += 1
            continue

        title = str(rec.get("title") or "").strip()
        text = str(rec.get("text") or "").strip()

        # Hygiene check #2: Drop ONLY empty/whitespace-only posts
        if not title and not text:
            st.dropped_empty_count += 1
            continue

        # Timestamp parsing
        if source == "telegram":
            event_utc = _parse_social_timestamp(rec.get("published_utc"), "UTC")
        elif source == "naver_finance":
            event_utc = _parse_social_timestamp(rec.get("written_at_kst"), "Asia/Seoul")
        elif source == "lihkg":
            event_utc = _parse_social_timestamp(rec.get("published_utc"), "Asia/Hong_Kong")
        else:
            event_utc = _parse_social_timestamp(
                rec.get("published_utc") or rec.get("published_raw") or rec.get("created_at"),
                ex_cfg.timezone,
            )

        if event_utc is None:
            st.dropped_invalid_ts_count += 1
            continue

        effective_utc = compute_effective_event_utc(event_utc, "exact", window, ex_cfg)

        # Pre-warmup cutoff guard
        if effective_utc < window.exchange_warmup_lead_utc:
            st.dropped_pre_warmup_count += 1
            continue

        # Ticker attribution
        if source in ("naver_finance", "aastocks"):
            t_sym = str(rec.get("ticker") or "")
            if t_sym not in active_ticker_map:
                st.dropped_unmatched_count += 1
                continue
            attributed: list[tuple[TickerConfig, list[str]]] = [(active_ticker_map[t_sym], [t_sym])]
        else:
            combined_for_match = f"{title}\n{text}"
            attributed = match_tickers_in_text(combined_for_match, matchers_by_ex[exchange])
            if not attributed:
                st.dropped_unmatched_count += 1
                continue

        is_multi = len(attributed) > 1
        combined_stripped = f"{title} {text}".strip()
        is_short = len(combined_stripped) < 5
        default_lang = "zh" if exchange == "HKEX" else "ko"
        lang = detect_language(title, text, fallback=default_lang)

        # Source-specific engagement & reach normalization
        if source == "telegram":
            ch_handle = str(rec.get("channel_handle") or "unknown")
            channel_or_domain = f"@{ch_handle.lstrip('@')}"
            source_item_id = f"{ch_handle}/{rec.get('message_id')}"
            subs = int(rec.get("channel_subscribers") or 0)
            views = int(rec.get("views") or 0)
            likes = int(rec.get("reactions_total") or 0)
            dislikes = 0
            reposts = 0
            comments = 0
            author_followers: int | None = subs if subs > 0 else 0
            author_followers_is_proxy = False
            effective_reach = max(subs, 1) if subs > 0 else max(views, 1)
            is_holder_verified: bool | None = None
            has_full_body = True
            url = str(rec.get("url") or f"https://t.me/{source_item_id}")
        elif source == "naver_finance":
            channel_or_domain = f"naver_board_{rec.get('krx_code', '')}"
            source_item_id = str(rec.get("source_item_id") or "")
            views = int(rec.get("views") or 0)
            likes = int(rec.get("likes") or 0)
            dislikes = int(rec.get("dislikes") or 0)
            reposts = 0
            comments = 0
            author_followers = None
            author_followers_is_proxy = True
            effective_reach = max(views, 1)
            is_holder_verified = bool(rec.get("is_holder_verified", False))
            has_full_body = bool(rec.get("has_full_body", True))
            url = str(rec.get("url") or "")
        elif source == "lihkg":
            channel_or_domain = str(rec.get("category") or "LIHKG_財經台")
            tid = str(rec.get("thread_id") or "")
            rid = rec.get("reply_id")
            source_item_id = f"{tid}_r{rid}" if rid else f"{tid}_op"
            views = 0
            likes = int(rec.get("likes") or 0)
            dislikes = int(rec.get("dislikes") or 0)
            reposts = 0
            comments = int(rec.get("reply_count") or 0)
            author_followers = None
            author_followers_is_proxy = True
            effective_reach = max(comments + likes + dislikes, 1)
            is_holder_verified = None
            has_full_body = bool(rec.get("has_full_body", False))
            url = str(rec.get("url") or f"https://lihkg.com/thread/{tid}/page/1")
        else:
            channel_or_domain = source
            source_item_id = str(rec.get("source_item_id") or rec.get("url") or "")
            views = int(rec.get("views") or 0)
            likes = int(rec.get("likes") or 0)
            dislikes = int(rec.get("dislikes") or 0)
            reposts = 0
            comments = int(rec.get("comments") or 0)
            author_followers = None
            author_followers_is_proxy = True
            effective_reach = max(views, likes + comments, 1)
            is_holder_verified = None
            has_full_body = bool(rec.get("has_full_body", False))
            url = str(rec.get("url") or "")

        for tcfg, matched_aliases in attributed:
            normalized.append(
                {
                    "event_id": compute_event_id("social", source, source_item_id, tcfg.ticker),
                    "event_type": "social",
                    "source": source,
                    "source_item_id": source_item_id,
                    "ticker": tcfg.ticker,
                    "exchange": exchange,
                    "channel_or_domain": channel_or_domain,
                    "title": title,
                    "text": text,
                    "language": lang,
                    "url": url,
                    "event_utc": event_utc,
                    "effective_event_utc": effective_utc,
                    "timestamp_precision": "exact",
                    "likes": likes,
                    "dislikes": dislikes,
                    "reposts": reposts,
                    "views": views,
                    "comments": comments,
                    "author_followers": author_followers,
                    "author_followers_is_proxy": author_followers_is_proxy,
                    "effective_reach": effective_reach,
                    "is_holder_verified": is_holder_verified,
                    "is_cross_channel_repost": False,
                    "is_short_text": is_short,
                    "source_tier": None,
                    "source_credibility_weight": 1.0,
                    "matched_aliases": matched_aliases,
                    "is_multi_ticker": is_multi,
                    "has_full_body": has_full_body,
                }
            )

    # -------------------------------------------------------------------------
    # 2. News records (exa, tavily) — grouped by (source, ticker) for syndication dedup
    # -------------------------------------------------------------------------
    sim_thresh = float(sources_cfg["news"]["exa"]["syndication_title_similarity_threshold"])
    sim_hours = int(sources_cfg["news"]["exa"]["syndication_window_hours"])

    news_by_src_ticker: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in raw_news_records:
        src = str(rec.get("source") or "exa")
        t_sym = str(rec.get("ticker") or "")
        if t_sym not in active_ticker_map:
            continue
        news_by_src_ticker.setdefault((src, t_sym), []).append(rec)

    for (src, t_sym), recs in news_by_src_ticker.items():
        st = get_stats(src)
        st.raw_count += len(recs)
        tcfg = active_ticker_map[t_sym]
        window = exchange_windows[tcfg.exchange]
        ex_cfg = universe.exchanges[tcfg.exchange]

        deduped, n_dedup, n_invalid = deduplicate_news_records(
            recs,
            similarity_threshold=sim_thresh,
            window_hours=sim_hours,
        )
        st.dropped_dedup_count += n_dedup
        st.dropped_invalid_ts_count += n_invalid

        for rec in deduped:
            title = str(rec.get("title") or "").strip()
            text = str(rec.get("text") or "").strip()
            if not title and not text:
                st.dropped_empty_count += 1
                continue

            event_utc, prec = parse_news_timestamp(rec.get("published_date"))
            if event_utc is None:
                st.dropped_invalid_ts_count += 1
                continue

            effective_utc = compute_effective_event_utc(event_utc, prec, window, ex_cfg)
            if effective_utc < window.exchange_warmup_lead_utc:
                st.dropped_pre_warmup_count += 1
                continue

            source_item_id = str(rec.get("source_item_id") or rec.get("url") or "")
            lang_hint = str(rec.get("language_hint") or "en")
            lang = detect_language(title, text, fallback=lang_hint)
            combined_stripped = f"{title} {text}".strip()

            normalized.append(
                {
                    "event_id": compute_event_id("news", src, source_item_id, tcfg.ticker),
                    "event_type": "news",
                    "source": src,
                    "source_item_id": source_item_id,
                    "ticker": tcfg.ticker,
                    "exchange": tcfg.exchange,
                    "channel_or_domain": str(rec.get("domain") or src),
                    "title": title,
                    "text": text,
                    "language": lang,
                    "url": str(rec.get("url") or ""),
                    "event_utc": event_utc,
                    "effective_event_utc": effective_utc,
                    "timestamp_precision": prec,
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
                    "is_short_text": len(combined_stripped) < 5,
                    "source_tier": str(rec.get("source_tier") or "tier2"),
                    "source_credibility_weight": float(rec.get("source_credibility_weight") or 1.0),
                    "matched_aliases": [tcfg.ticker],
                    "is_multi_ticker": False,
                    "has_full_body": bool(rec.get("has_full_body", False)),
                }
            )

    # -------------------------------------------------------------------------
    # 3. Within-source deduplication on (event_type, source, source_item_id, ticker)
    # -------------------------------------------------------------------------
    dedup_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in normalized:
        key = (row["event_type"], row["source"], row["source_item_id"], row["ticker"])
        prev = dedup_map.get(key)
        if prev is None:
            dedup_map[key] = row
        else:
            get_stats(row["source"]).dropped_dedup_count += 1
            if _record_richness_key(row) > _record_richness_key(prev):
                dedup_map[key] = row

    final_rows = list(dedup_map.values())

    # -------------------------------------------------------------------------
    # 4. Cross-channel repost flagging for Telegram (same ticker + normalized text SHA-256 within 24h)
    # -------------------------------------------------------------------------
    tg_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in final_rows:
        if row["source"] != "telegram":
            continue
        norm_txt = re.sub(r"\s+", " ", row["text"].strip().lower())
        if not norm_txt:
            continue
        txt_hash = hashlib.sha256(norm_txt.encode("utf-8")).hexdigest()
        tg_groups.setdefault((row["ticker"], txt_hash), []).append(row)

    for grp in tg_groups.values():
        if len(grp) < 2:
            continue
        grp.sort(key=lambda r: r["event_utc"])
        seen_earlier: list[tuple[datetime, str]] = []
        for row in grp:
            ts = row["event_utc"]
            ch = row["channel_or_domain"]
            is_repost = any(
                prev_ch != ch and 0.0 <= (ts - prev_ts).total_seconds() <= 86400.0
                for prev_ts, prev_ch in seen_earlier
            )
            if is_repost:
                row["is_cross_channel_repost"] = True
                get_stats("telegram").cross_channel_repost_count += 1
            seen_earlier.append((ts, ch))

    # Update kept counts & drop temporary helper key
    for row in final_rows:
        row.pop("source_item_id", None)
        get_stats(row["source"]).kept_count += 1

    return final_rows, stats_by_source
