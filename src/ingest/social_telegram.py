"""Telegram public web channel ingestion via apify/cheerio-scraper (Phase 1 Section 1.5).

Dual-mode scraping across 12 live-verified channels:
- search_q: t.me/s/{handle}?q={term} + a.tme_messages_more[data-before] pagination
- full_feed: t.me/s/{handle} + a.tme_messages_more[data-before] pagination
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import quote

from apify_client import ApifyClient
from apify_client.errors import RateLimitError, ServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ingest.manifest import (
    ManifestStore,
    check_apify_budget,
    make_cache_key,
    read_jsonl,
    utc_now_iso,
    write_jsonl_atomic,
)
from src.ingest.schemas import (
    RAW_DIR,
    ExchangeCode,
    ExchangeWindowBounds,
    ManifestEntry,
    RunMode,
    TelegramChannelConfig,
    TickerConfig,
)

logger = logging.getLogger(__name__)

# Cheerio Scraper v3 pageFunction verified against live t.me/s/{handle} HTML structure.
TELEGRAM_PAGE_FUNCTION = r"""
async function pageFunction(context) {
    const { $, request, enqueueRequest, log } = context;
    const ud = request.userData || {};
    const handle = ud.handle;
    const channelName = ud.channel_name || handle;
    const mode = ud.mode || 'full_feed';
    const queryTerm = ud.query_term || null;
    const exchange = ud.exchange;
    const pageNum = Number(ud.page_num || 1);
    const maxPages = Number(ud.max_pages || 25);
    const warmupLeadMs = Number(ud.warmup_lead_ms || 0);

    const parseCompact = (raw) => {
        if (!raw) return 0;
        const cleaned = String(raw).replace(/,/g, '').replace(/\s+/g, ' ').trim();
        const m = cleaned.match(/([\d.]+)\s*([KMBkmb])?/);
        if (!m) return 0;
        const num = parseFloat(m[1]);
        if (Number.isNaN(num)) return 0;
        const suffix = (m[2] || '').toUpperCase();
        if (suffix === 'K') return Math.round(num * 1000);
        if (suffix === 'M') return Math.round(num * 1000000);
        if (suffix === 'B') return Math.round(num * 1000000000);
        return Math.round(num);
    };

    const headerCounterRaw = $('.tgme_header_counter').first().text().trim();
    const channelSubscribers = parseCompact(headerCounterRaw);

    const items = [];
    let oldestMs = null;

    $('.tgme_widget_message').each((_, el) => {
        const $el = $(el);
        const dataPost = $el.attr('data-post') || '';
        const msgId = dataPost.includes('/') ? dataPost.split('/').pop() : dataPost;
        const timeAttr = $el.find('time[datetime]').first().attr('datetime') || null;
        if (!msgId || !timeAttr) return;

        const tsMs = Date.parse(timeAttr);
        if (!Number.isNaN(tsMs)) {
            if (oldestMs === null || tsMs < oldestMs) {
                oldestMs = tsMs;
            }
        }

        const $textEl = $el.find('.tgme_widget_message_text').first();
        // Preserve line breaks cleanly
        const text = $textEl.length ? $textEl.text().trim() : '';
        const viewsRaw = $el.find('.tgme_widget_message_views').first().text().trim();
        const views = parseCompact(viewsRaw);

        let reactionsTotal = 0;
        $el.find('.tgme_reaction').each((__, rEl) => {
            const rTxt = $(rEl).text().trim();
            const c = parseCompact(rTxt);
            reactionsTotal += c;
        });

        items.push({
            source: 'telegram',
            exchange,
            channel_handle: handle,
            channel_name: channelName,
            channel_subscribers: channelSubscribers,
            channel_subscribers_raw: headerCounterRaw,
            scrape_mode: mode,
            query_term: queryTerm,
            data_post: dataPost,
            message_id: String(msgId),
            text,
            published_utc: timeAttr,
            views,
            views_raw: viewsRaw,
            reactions_total: reactionsTotal,
            url: dataPost ? `https://t.me/${dataPost}` : request.url,
            page_num: pageNum,
        });
    });

    // Cursor pagination: follow a.tme_messages_more[data-before] only while within warmup lead & page cap
    const $more = $('a.tme_messages_more[data-before]').first();
    const beforeCursor = $more.attr('data-before');
    const href = $more.attr('href');
    if (
        beforeCursor &&
        pageNum < maxPages &&
        items.length > 0 &&
        (oldestMs === null || oldestMs >= warmupLeadMs)
    ) {
        let nextUrl = '';
        if (href && href.startsWith('/')) {
            nextUrl = `https://t.me${href}`;
        } else if (href && href.startsWith('http')) {
            nextUrl = href;
        } else if (mode === 'search_q' && queryTerm) {
            nextUrl = `https://t.me/s/${handle}?q=${encodeURIComponent(queryTerm)}&before=${beforeCursor}`;
        } else {
            nextUrl = `https://t.me/s/${handle}?before=${beforeCursor}`;
        }
        await enqueueRequest({
            url: nextUrl,
            userData: {
                ...ud,
                page_num: pageNum + 1,
            },
        });
    }

    log.info(`[telegram] @${handle} (${mode}${queryTerm ? ':' + queryTerm : ''}) p${pageNum}: ${items.length} msgs`);
    return items;
}
"""


def _extract_run_cost_usd(run: Any) -> float:
    for attr in ("usage_total_usd", "external_charged_total_usd"):
        val = getattr(run, attr, None)
        if val is not None:
            return float(val)
    if isinstance(run, dict):
        for k in ("usageTotalUsd", "usage_total_usd"):
            if run.get(k) is not None:
                return float(run[k])
    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type((RateLimitError, ServerError, ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _call_cheerio_telegram(
    client: ApifyClient,
    actor_id: str,
    run_input: dict[str, Any],
    memory_mbytes: int,
    run_timeout_secs: int,
) -> tuple[str, list[dict[str, Any]], float]:
    run = client.actor(actor_id).call(
        run_input=run_input,
        memory_mbytes=memory_mbytes,
        run_timeout=timedelta(seconds=run_timeout_secs),
        force_permission_level="LIMITED_PERMISSIONS",
    )
    if run is None:
        raise RuntimeError(f"Apify actor {actor_id} returned None run object")
    status = getattr(run, "status", None)
    run_id = getattr(run, "id", "")
    if status != "SUCCEEDED":
        raise RuntimeError(
            f"Apify actor {actor_id} failed (run_id={run_id}, status={status}, "
            f"msg={getattr(run, 'status_message', None)})"
        )
    dataset_id = getattr(run, "default_dataset_id", None)
    if not dataset_id:
        raise RuntimeError(f"Apify run {run_id} missing default_dataset_id")
    items = list(client.dataset(dataset_id).iterate_items())
    cost = _extract_run_cost_usd(run)
    return run_id, items, cost


def _select_channels_for_exchange(
    sources_cfg: dict[str, Any],
    exchange: ExchangeCode,
    smoke_test: bool,
) -> list[TelegramChannelConfig]:
    raw_channels = sources_cfg["social"]["telegram"]["channels"]
    enabled = [
        TelegramChannelConfig.model_validate(ch)
        for ch in raw_channels
        if ch.get("enabled", True) and ch["exchange"].upper() == exchange
    ]
    if not smoke_test:
        return enabled

    # In smoke test: pick 1 search_q + 1 full_feed channel (up to max_channels_per_exchange)
    cap = int(sources_cfg["smoke_test"]["telegram"]["max_channels_per_exchange"])
    picked: list[TelegramChannelConfig] = []
    for mode_pref in ("search_q", "full_feed"):
        for ch in enabled:
            if ch.mode == mode_pref and ch not in picked:
                picked.append(ch)
                break
    for ch in enabled:
        if len(picked) >= cap:
            break
        if ch not in picked:
            picked.append(ch)
    return picked[:cap]


def ingest_telegram(
    exchange: ExchangeCode,
    tickers: list[TickerConfig],
    window: ExchangeWindowBounds,
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Scrape public Telegram channels for an exchange via apify/cheerio-scraper."""
    smoke_test = mode == "smoke"
    tg_cfg = sources_cfg["social"]["telegram"]
    actor_id = tg_cfg["actor_id"]
    key_label = f"channels_{exchange.lower()}"
    raw_path = RAW_DIR / "social" / "telegram" / exchange / f"{key_label}_{end_date_str}.jsonl"
    ckey = make_cache_key("social", "telegram", exchange, key_label, end_date_str, mode)

    if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
        logger.info("[telegram] Cache HIT for %s (%s)", exchange, raw_path.name)
        return read_jsonl(raw_path)

    api_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("APIFY_API_TOKEN is missing in environment/.env")

    hard_stop = float(sources_cfg["budget_guard"]["apify_hard_stop_monthly_usd"])
    check_apify_budget(api_token, hard_stop_usd=hard_stop, estimated_next_run_usd=0.05)

    channels = _select_channels_for_exchange(sources_cfg, exchange, smoke_test=smoke_test)
    if not channels:
        raise RuntimeError(f"No enabled Telegram channels configured for exchange {exchange}")

    warmup_lead_ms = int(window.exchange_warmup_lead_utc.timestamp() * 1000)

    # Deduplicate search_q terms across selected tickers
    q_terms: list[str] = []
    seen_terms: set[str] = set()
    max_terms_per_ticker = int(sources_cfg["smoke_test"]["telegram"]["max_terms_per_ticker"]) if smoke_test else 99
    for tcfg in tickers:
        if tcfg.exchange != exchange:
            continue
        for term in tcfg.telegram_q_terms[:max_terms_per_ticker]:
            if term not in seen_terms:
                seen_terms.add(term)
                q_terms.append(term)

    smoke_page_cap = int(sources_cfg["smoke_test"]["telegram"]["max_pages_per_stream"])
    start_urls: list[dict[str, Any]] = []
    total_page_budget = 0

    for ch in channels:
        page_cap = min(ch.max_pages, smoke_page_cap) if smoke_test else ch.max_pages
        if ch.mode == "search_q":
            for term in q_terms:
                url = f"https://t.me/s/{ch.handle}?q={quote(term)}"
                start_urls.append(
                    {
                        "url": url,
                        "userData": {
                            "handle": ch.handle,
                            "channel_name": ch.name,
                            "mode": "search_q",
                            "query_term": term,
                            "exchange": exchange,
                            "page_num": 1,
                            "max_pages": page_cap,
                            "warmup_lead_ms": warmup_lead_ms,
                        },
                    }
                )
                total_page_budget += page_cap
        else:
            url = f"https://t.me/s/{ch.handle}"
            start_urls.append(
                {
                    "url": url,
                    "userData": {
                        "handle": ch.handle,
                        "channel_name": ch.name,
                        "mode": "full_feed",
                        "query_term": None,
                        "exchange": exchange,
                        "page_num": 1,
                        "max_pages": page_cap,
                        "warmup_lead_ms": warmup_lead_ms,
                    },
                }
            )
            total_page_budget += page_cap

    run_input = {
        "startUrls": start_urls,
        "pageFunction": TELEGRAM_PAGE_FUNCTION,
        "proxyConfiguration": tg_cfg["proxy_configuration"],
        "maxConcurrency": int(tg_cfg["max_concurrency"]),
        "maxPagesPerCrawl": total_page_budget + 10,
        "maxRequestRetries": 3,
        "ignoreSslErrors": False,
        "debugLog": False,
    }

    client = ApifyClient(api_token)
    logger.info(
        "[telegram] Starting %s for %s: %d start URLs across %d channels (mode=%s, maxPages=%d)",
        actor_id,
        exchange,
        len(start_urls),
        len(channels),
        mode,
        total_page_budget,
    )

    try:
        run_id, items, cost_usd = _call_cheerio_telegram(
            client=client,
            actor_id=actor_id,
            run_input=run_input,
            memory_mbytes=int(tg_cfg["memory_mbytes"]),
            run_timeout_secs=int(tg_cfg["run_timeout_secs"]),
        )
        if not items:
            raise RuntimeError(
                f"Telegram scraper returned 0 items for {exchange} (actor={actor_id}, run_id={run_id})"
            )

        raw_count = write_jsonl_atomic(raw_path, items)
        manifest_store.record(
            ManifestEntry(
                cache_key=ckey,
                stage="social",
                source="telegram",
                exchange=exchange,
                key=key_label,
                end_date=end_date_str,
                mode=mode,
                status="ok",
                actor_id=actor_id,
                apify_run_id=run_id,
                raw_path=str(raw_path),
                raw_count=raw_count,
                kept_count=raw_count,
                est_cost_usd=round(cost_usd, 6),
                updated_at_utc=utc_now_iso(),
            )
        )
        logger.info(
            "[telegram] %s complete: %d raw messages, run_id=%s, cost=$%.4f",
            exchange,
            raw_count,
            run_id,
            cost_usd,
        )
        return items
    except Exception as exc:
        manifest_store.record(
            ManifestEntry(
                cache_key=ckey,
                stage="social",
                source="telegram",
                exchange=exchange,
                key=key_label,
                end_date=end_date_str,
                mode=mode,
                status="failed",
                actor_id=actor_id,
                raw_path=str(raw_path),
                est_cost_usd=0.0,
                updated_at_utc=utc_now_iso(),
                error_detail=str(exc),
            )
        )
        raise
