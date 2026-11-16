"""Institutional financial news retrieval via Exa.com (exa-py) (Phase 1 Section 1.6).

Implements:
- Bilingual 2-Query Matrix per ticker (English Tier-1 + Local-Language Tier-1)
- Automatic Tier-2 Exa backfill if < 10 deduplicated articles (< 2 in smoke test)
- Canonical URL tracking-param stripping & 24h SequenceMatcher (> 0.85) syndication dedup
- Timestamp precision guard (exact vs date_only)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from exa_py import Exa
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ingest.manifest import (
    ManifestStore,
    make_cache_key,
    read_jsonl,
    utc_now_iso,
    write_jsonl_atomic,
)
from src.ingest.schemas import (
    RAW_DIR,
    ExchangeWindowBounds,
    ManifestEntry,
    RunMode,
    TickerConfig,
)

logger = logging.getLogger(__name__)

_TRACKING_PARAM_PREFIXES = ("utm_", "ref", "fbclid", "gclid", "mc_cid", "mc_eid", "ocid", "guccounter")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MIDNIGHT_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]00:00:00(?:\.0+)?(?:Z|\+00:00)?$")


def canonicalize_url(url: str) -> str:
    """Strip tracking query parameters and URL fragments for canonical URL deduplication."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    kept_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not any(k.lower().startswith(pref) for pref in _TRACKING_PARAM_PREFIXES)
    ]
    clean_query = urlencode(sorted(kept_params))
    clean_path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), clean_path, "", clean_query, ""))


def extract_domain(url: str) -> str:
    netloc = urlparse(url or "").netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def parse_news_timestamp(raw_date: str | None) -> tuple[datetime | None, str]:
    """Parse Exa/Tavily published_date into UTC datetime and precision ('exact' | 'date_only').

    Returns (None, 'missing') if unparseable or empty.
    """
    if not raw_date or not isinstance(raw_date, str):
        return None, "missing"
    s = raw_date.strip()
    if not s:
        return None, "missing"

    is_date_only = bool(_DATE_ONLY_RE.match(s) or _MIDNIGHT_UTC_RE.match(s))

    try:
        if _DATE_ONLY_RE.match(s):
            dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        return dt, ("date_only" if is_date_only else "exact")
    except ValueError:
        return None, "missing"


def normalize_title_for_dedup(title: str) -> str:
    t = (title or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\u4e00-\u9fff\uac00-\ud7af]", "", t)
    return t


def deduplicate_news_records(
    records: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
    window_hours: int = 24,
) -> tuple[list[dict[str, Any]], int, int]:
    """Deduplicate news articles by canonical URL and 24h title similarity (> 0.85).

    Preference order when collapsing duplicates:
      1. tier1 over tier2
      2. exact timestamp over date_only
      3. earliest published_utc
    Returns: (kept_records, dropped_dedup_count, dropped_invalid_date_count).
    """
    valid: list[dict[str, Any]] = []
    dropped_invalid = 0
    for rec in records:
        dt, prec = parse_news_timestamp(rec.get("published_date"))
        if dt is None:
            dropped_invalid += 1
            continue
        item = dict(rec)
        item["_parsed_utc"] = dt
        item["timestamp_precision"] = prec
        item["_canon_url"] = canonicalize_url(rec.get("url", ""))
        item["_norm_title"] = normalize_title_for_dedup(rec.get("title", ""))
        valid.append(item)

    # Sort best candidates first: tier1 (0) < tier2 (1), exact (0) < date_only (1), earliest UTC
    valid.sort(
        key=lambda r: (
            0 if r.get("source_tier") == "tier1" else 1,
            0 if r["timestamp_precision"] == "exact" else 1,
            r["_parsed_utc"],
        )
    )

    kept: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    dropped_dedup = 0
    window_secs = window_hours * 3600

    for cand in valid:
        curl = cand["_canon_url"]
        if curl and curl in seen_urls:
            dropped_dedup += 1
            continue

        is_syndicated = False
        cand_title = cand["_norm_title"]
        if cand_title:
            for existing in kept:
                if abs((cand["_parsed_utc"] - existing["_parsed_utc"]).total_seconds()) <= window_secs:
                    ex_title = existing["_norm_title"]
                    if ex_title and SequenceMatcher(None, cand_title, ex_title).ratio() > similarity_threshold:
                        is_syndicated = True
                        break

        if is_syndicated:
            dropped_dedup += 1
            continue

        if curl:
            seen_urls.add(curl)
        kept.append(cand)

    # Strip internal helper keys
    cleaned: list[dict[str, Any]] = []
    for rec in kept:
        rec = {k: v for k, v in rec.items() if not k.startswith("_")}
        cleaned.append(rec)

    return cleaned, dropped_dedup, dropped_invalid


def _extract_exa_cost(resp: Any) -> float:
    cost = getattr(resp, "cost_dollars", None)
    if cost is None:
        return 0.0
    if isinstance(cost, (int, float)):
        return float(cost)
    total = getattr(cost, "total", None)
    if total is not None:
        return float(total)
    if isinstance(cost, dict) and "total" in cost:
        return float(cost["total"])
    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _run_exa_search(
    exa: Exa,
    query: str,
    num_results: int,
    start_iso: str,
    end_iso: str,
    contents: dict[str, Any],
    search_type: str,
    category: str,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> tuple[list[Any], float]:
    kwargs: dict[str, Any] = {
        "query": query,
        "type": search_type,
        "category": category,
        "num_results": num_results,
        "start_published_date": start_iso,
        "end_published_date": end_iso,
        "contents": contents,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains

    resp = exa.search(**kwargs)
    return list(getattr(resp, "results", []) or []), _extract_exa_cost(resp)


def _exa_results_to_raw_records(
    results: list[Any],
    ticker_cfg: TickerConfig,
    query_label: str,
    query_text: str,
    source_tier: str,
    credibility_weight: float,
    lang_hint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for res in results:
        url = getattr(res, "url", "") or ""
        title = getattr(res, "title", "") or ""
        highlights = getattr(res, "highlights", None) or []
        text_body = getattr(res, "text", "") or ""
        combined_text = text_body.strip() or "\n".join(h.strip() for h in highlights if h)
        pub_date = getattr(res, "published_date", None)
        res_id = getattr(res, "id", None) or canonicalize_url(url) or url

        rows.append(
            {
                "source": "exa",
                "exchange": ticker_cfg.exchange,
                "ticker": ticker_cfg.ticker,
                "source_item_id": str(res_id),
                "query_label": query_label,
                "query_text": query_text,
                "domain": extract_domain(url),
                "title": title.strip(),
                "text": combined_text,
                "highlights": highlights,
                "url": url,
                "published_date": pub_date,
                "language_hint": lang_hint,
                "source_tier": source_tier,
                "source_credibility_weight": float(credibility_weight),
                "has_full_body": bool(text_body.strip()),
            }
        )
    return rows


def ingest_exa_news(
    tickers: list[TickerConfig],
    exchange_windows: dict[str, ExchangeWindowBounds],
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch bilingual institutional news via Exa for all selected tickers with automatic Tier-2 backfill."""
    exa_cfg = sources_cfg["news"]["exa"]
    smoke_cfg = sources_cfg["smoke_test"]["exa"]
    smoke_test = mode == "smoke"

    api_key = os.environ.get("EXA_API_KEY", "").strip()
    exa_client: Exa | None = None

    all_records: list[dict[str, Any]] = []

    for tcfg in tickers:
        window = exchange_windows[tcfg.exchange]
        raw_path = RAW_DIR / "news" / "exa" / tcfg.exchange / f"{tcfg.ticker}_{end_date_str}.jsonl"
        ckey = make_cache_key("news", "exa", tcfg.exchange, tcfg.ticker, end_date_str, mode)

        if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
            logger.info("[exa] Cache HIT for %s (%s)", tcfg.ticker, raw_path.name)
            all_records.extend(read_jsonl(raw_path))
            continue

        if exa_client is None:
            if not api_key:
                raise RuntimeError("EXA_API_KEY is missing in environment/.env")
            exa_client = Exa(api_key=api_key)

        start_iso = window.exchange_warmup_lead_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = window.exchange_window_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        num_a = int(smoke_cfg["query_a_num_results"] if smoke_test else exa_cfg["query_a_num_results"])
        num_b = int(smoke_cfg["query_b_num_results"] if smoke_test else exa_cfg["query_b_num_results"])
        min_for_no_backfill = int(
            smoke_cfg["tier2_trigger_min_articles"] if smoke_test else exa_cfg["tier2_trigger_min_articles"]
        )
        num_t2 = int(smoke_cfg["tier2_num_results"] if smoke_test else exa_cfg["tier2_num_results"])

        query_a = exa_cfg["query_a_template"].format(company_en=tcfg.company_en, ticker=tcfg.ticker)
        query_b = exa_cfg["query_b_templates"][tcfg.exchange].format(
            company_local=tcfg.company_local,
            code_short=tcfg.code_short,
            exchange_code=tcfg.exchange_code,
        )
        local_lang = "zh" if tcfg.exchange == "HKEX" else "ko"

        ticker_raw: list[dict[str, Any]] = []
        total_cost = 0.0

        try:
            logger.info("[exa] Query A (EN Tier-1) for %s (n=%d)", tcfg.ticker, num_a)
            res_a, cost_a = _run_exa_search(
                exa=exa_client,
                query=query_a,
                num_results=num_a,
                start_iso=start_iso,
                end_iso=end_iso,
                contents=exa_cfg["contents"],
                search_type=exa_cfg["type"],
                category=exa_cfg["category"],
                include_domains=exa_cfg["tier1_domains_en"][tcfg.exchange],
            )
            total_cost += cost_a
            ticker_raw.extend(
                _exa_results_to_raw_records(
                    res_a, tcfg, "query_a_en_tier1", query_a, "tier1", exa_cfg["tier1_credibility_weight"], "en"
                )
            )

            logger.info("[exa] Query B (%s Tier-1) for %s (n=%d)", local_lang.upper(), tcfg.ticker, num_b)
            res_b, cost_b = _run_exa_search(
                exa=exa_client,
                query=query_b,
                num_results=num_b,
                start_iso=start_iso,
                end_iso=end_iso,
                contents=exa_cfg["contents"],
                search_type=exa_cfg["type"],
                category=exa_cfg["category"],
                include_domains=exa_cfg["tier1_domains_local"][tcfg.exchange],
            )
            total_cost += cost_b
            ticker_raw.extend(
                _exa_results_to_raw_records(
                    res_b, tcfg, "query_b_local_tier1", query_b, "tier1", exa_cfg["tier1_credibility_weight"], local_lang
                )
            )

            # Check if Tier-2 backfill is needed
            deduped_t1, _, _ = deduplicate_news_records(
                ticker_raw,
                similarity_threshold=float(exa_cfg["syndication_title_similarity_threshold"]),
                window_hours=int(exa_cfg["syndication_window_hours"]),
            )
            if len(deduped_t1) < min_for_no_backfill:
                logger.info(
                    "[exa] %s yielded %d Tier-1 articles (< %d); triggering Tier-2 backfill (n=%d)",
                    tcfg.ticker,
                    len(deduped_t1),
                    min_for_no_backfill,
                    num_t2,
                )
                res_t2, cost_t2 = _run_exa_search(
                    exa=exa_client,
                    query=query_a,
                    num_results=num_t2,
                    start_iso=start_iso,
                    end_iso=end_iso,
                    contents=exa_cfg["contents"],
                    search_type=exa_cfg["type"],
                    category=exa_cfg["category"],
                    exclude_domains=exa_cfg["tier2_exclude_domains"],
                )
                total_cost += cost_t2
                ticker_raw.extend(
                    _exa_results_to_raw_records(
                        res_t2, tcfg, "query_c_tier2_backfill", query_a, "tier2", exa_cfg["tier2_credibility_weight"], "en"
                    )
                )

            final_deduped, dropped_dedup, dropped_invalid = deduplicate_news_records(
                ticker_raw,
                similarity_threshold=float(exa_cfg["syndication_title_similarity_threshold"]),
                window_hours=int(exa_cfg["syndication_window_hours"]),
            )
            if not final_deduped:
                raise RuntimeError(
                    f"Exa returned 0 valid dated news articles for {tcfg.ticker} in [{start_iso}, {end_iso}]"
                )

            raw_count = write_jsonl_atomic(raw_path, ticker_raw)
            manifest_store.record(
                ManifestEntry(
                    cache_key=ckey,
                    stage="news",
                    source="exa",
                    exchange=tcfg.exchange,
                    key=tcfg.ticker,
                    end_date=end_date_str,
                    mode=mode,
                    status="ok",
                    raw_path=str(raw_path),
                    raw_count=raw_count,
                    kept_count=len(final_deduped),
                    dropped_empty_count=dropped_invalid,
                    dropped_dedup_count=dropped_dedup,
                    est_cost_usd=round(total_cost, 6),
                    updated_at_utc=utc_now_iso(),
                )
            )
            logger.info(
                "[exa] %s complete: raw=%d, kept=%d (dedup=%d, invalid_date=%d, cost=$%.4f)",
                tcfg.ticker,
                raw_count,
                len(final_deduped),
                dropped_dedup,
                dropped_invalid,
                total_cost,
            )
            all_records.extend(ticker_raw)

        except Exception as exc:
            manifest_store.record(
                ManifestEntry(
                    cache_key=ckey,
                    stage="news",
                    source="exa",
                    exchange=tcfg.exchange,
                    key=tcfg.ticker,
                    end_date=end_date_str,
                    mode=mode,
                    status="failed",
                    raw_path=str(raw_path),
                    est_cost_usd=round(total_cost, 6),
                    updated_at_utc=utc_now_iso(),
                    error_detail=str(exc),
                )
            )
            raise

    return all_records
