"""Naver Finance (종목토론실) ingestion via apify/cheerio-scraper JSON mode (Phase 1 Section 1.5).

Uses verified live 2026 JSON endpoint:
  https://m.stock.naver.com/front-api/discussion/list?discussionType=domesticStock&itemCode={krx_code}&pageSize=100
Returns full post bodies (contentSwReplaced) natively in Pass 1 with cursor pagination via offset=lastOffset.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

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
    ExchangeWindowBounds,
    ManifestEntry,
    RunMode,
    TickerConfig,
)

logger = logging.getLogger(__name__)

NAVER_PAGE_FUNCTION = r"""
async function pageFunction(context) {
    const { request, enqueueRequest, log } = context;
    const ud = request.userData || {};
    const ticker = ud.ticker;
    const krxCode = ud.krx_code;
    const pageSize = Number(ud.page_size || 100);
    const pageNum = Number(ud.page_num || 1);
    const maxPages = Number(ud.max_pages || 400);
    const warmupLeadKst = String(ud.warmup_lead_kst || '');
    const referer = ud.referer;

    let payload = context.json;
    if (!payload && context.body) {
        const rawStr = typeof context.body === 'string'
            ? context.body
            : context.body.toString('utf8');
        payload = JSON.parse(rawStr);
    }

    if (!payload || !payload.isSuccess || !payload.result) {
        throw new Error(`Naver JSON API invalid response for ${krxCode} p${pageNum}: ${JSON.stringify(payload).slice(0, 300)}`);
    }

    const result = payload.result;
    const posts = Array.isArray(result.posts) ? result.posts : [];
    const lastOffset = result.lastOffset;

    const items = [];
    let oldestWrittenAt = null;

    for (const p of posts) {
        const writtenAt = p.writtenAt || '';
        if (writtenAt && (oldestWrittenAt === null || writtenAt < oldestWrittenAt)) {
            oldestWrittenAt = writtenAt;
        }
        const postId = String(p.id || '');
        items.push({
            source: 'naver_finance',
            exchange: 'KRX',
            ticker,
            krx_code: krxCode,
            source_item_id: postId,
            title: p.title || '',
            text: p.contentSwReplaced || p.contentSwReplacedButImg || '',
            written_at_kst: writtenAt,
            views: Number(p.viewCount || 0),
            likes: Number(p.recommendCount || 0),
            dislikes: Number(p.notRecommendCount || 0),
            is_holder_verified: Boolean(p.isHolderVerified),
            is_cleanbot_passed: p.isCleanbotPassed !== false,
            author_nickname: (p.writer && p.writer.nickname) ? p.writer.nickname : '',
            has_full_body: true,
            url: `https://m.stock.naver.com/domestic/stock/${krxCode}/discussion/${postId}`,
            page_num: pageNum,
        });
    }

    if (
        posts.length > 0 &&
        lastOffset !== null &&
        lastOffset !== undefined &&
        pageNum < maxPages &&
        (!oldestWrittenAt || !warmupLeadKst || oldestWrittenAt >= warmupLeadKst)
    ) {
        const nextUrl = `https://m.stock.naver.com/front-api/discussion/list?discussionType=domesticStock&itemCode=${krxCode}&pageSize=${pageSize}&offset=${lastOffset}`;
        await enqueueRequest({
            url: nextUrl,
            headers: {
                Referer: referer,
                Accept: 'application/json, text/plain, */*',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            },
            userData: {
                ...ud,
                page_num: pageNum + 1,
            },
        });
    }

    log.info(`[naver] ${ticker} p${pageNum}: ${items.length} posts (oldest=${oldestWrittenAt})`);
    return items;
}
"""


def _extract_run_cost_usd(run: Any) -> float:
    for attr in ("usage_total_usd", "external_charged_total_usd"):
        val = getattr(run, attr, None)
        if val is not None:
            return float(val)
    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type((RateLimitError, ServerError, ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _call_cheerio_naver(
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
            f"Apify Naver run failed (actor={actor_id}, run_id={run_id}, status={status}, "
            f"msg={getattr(run, 'status_message', None)})"
        )
    dataset_id = getattr(run, "default_dataset_id", None)
    if not dataset_id:
        raise RuntimeError(f"Apify run {run_id} missing default_dataset_id")
    items = list(client.dataset(dataset_id).iterate_items())
    cost = _extract_run_cost_usd(run)
    return run_id, items, cost


def ingest_naver_finance(
    tickers: list[TickerConfig],
    window: ExchangeWindowBounds,
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Batch-scrape Naver Finance discussion API for uncached KRX tickers with granular per-ticker caching."""
    krx_tickers = [t for t in tickers if t.exchange == "KRX"]
    if not krx_tickers:
        return []

    naver_cfg = sources_cfg["social"]["naver_finance"]
    actor_id = naver_cfg["actor_id"]
    smoke_test = mode == "smoke"
    page_size = int(naver_cfg["page_size"])
    max_pages = (
        int(sources_cfg["smoke_test"]["naver"]["max_pages_per_ticker"])
        if smoke_test
        else int(naver_cfg["max_pages_per_ticker"])
    )

    cached_items: list[dict[str, Any]] = []
    to_fetch: list[TickerConfig] = []

    for tcfg in krx_tickers:
        raw_path = RAW_DIR / "social" / "naver_finance" / "KRX" / f"{tcfg.ticker}_{end_date_str}.jsonl"
        ckey = make_cache_key("social", "naver_finance", "KRX", tcfg.ticker, end_date_str, mode)
        if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
            logger.info("[naver] Cache HIT for %s (%s)", tcfg.ticker, raw_path.name)
            cached_items.extend(read_jsonl(raw_path))
        else:
            to_fetch.append(tcfg)

    if not to_fetch:
        return cached_items

    api_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("APIFY_API_TOKEN is missing in environment/.env")

    hard_stop = float(sources_cfg["budget_guard"]["apify_hard_stop_monthly_usd"])
    check_apify_budget(api_token, hard_stop_usd=hard_stop, estimated_next_run_usd=0.05)

    warmup_lead_kst = window.exchange_warmup_lead_utc.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")
    start_urls: list[dict[str, Any]] = []
    for tcfg in to_fetch:
        url = naver_cfg["api_url_template"].format(krx_code=tcfg.exchange_code, page_size=page_size)
        referer = naver_cfg["referer_template"].format(krx_code=tcfg.exchange_code)
        start_urls.append(
            {
                "url": url,
                "headers": {
                    "Referer": referer,
                    "Accept": "application/json, text/plain, */*",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                    ),
                },
                "userData": {
                    "ticker": tcfg.ticker,
                    "krx_code": tcfg.exchange_code,
                    "page_size": page_size,
                    "page_num": 1,
                    "max_pages": max_pages,
                    "warmup_lead_kst": warmup_lead_kst,
                    "referer": referer,
                },
            }
        )

    run_input = {
        "startUrls": start_urls,
        "pageFunction": NAVER_PAGE_FUNCTION,
        "additionalMimeTypes": ["application/json"],
        "proxyConfiguration": naver_cfg["proxy_configuration"],
        "maxConcurrency": int(naver_cfg["max_concurrency"]),
        "maxPagesPerCrawl": len(to_fetch) * max_pages + 10,
        "maxRequestRetries": 3,
        "ignoreSslErrors": False,
        "debugLog": False,
    }

    client = ApifyClient(api_token)
    logger.info(
        "[naver] Starting %s for %d KRX ticker(s) (mode=%s, max_pages/ticker=%d, lead_kst=%s)",
        actor_id,
        len(to_fetch),
        mode,
        max_pages,
        warmup_lead_kst,
    )

    try:
        run_id, fetched_items, cost_usd = _call_cheerio_naver(
            client=client,
            actor_id=actor_id,
            run_input=run_input,
            memory_mbytes=int(naver_cfg["memory_mbytes"]),
            run_timeout_secs=int(naver_cfg["run_timeout_secs"]),
        )
    except Exception as exc:
        for tcfg in to_fetch:
            raw_path = RAW_DIR / "social" / "naver_finance" / "KRX" / f"{tcfg.ticker}_{end_date_str}.jsonl"
            ckey = make_cache_key("social", "naver_finance", "KRX", tcfg.ticker, end_date_str, mode)
            manifest_store.record(
                ManifestEntry(
                    cache_key=ckey,
                    stage="social",
                    source="naver_finance",
                    exchange="KRX",
                    key=tcfg.ticker,
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

    # Partition fetched posts by ticker, persist immutable raw JSONL, and fail fast if any ticker got 0 posts
    by_ticker: dict[str, list[dict[str, Any]]] = {t.ticker: [] for t in to_fetch}
    for item in fetched_items:
        t = item.get("ticker")
        if t in by_ticker:
            by_ticker[t].append(item)

    cost_per_ticker = cost_usd / max(1, len(to_fetch))
    for tcfg in to_fetch:
        rows = by_ticker[tcfg.ticker]
        raw_path = RAW_DIR / "social" / "naver_finance" / "KRX" / f"{tcfg.ticker}_{end_date_str}.jsonl"
        ckey = make_cache_key("social", "naver_finance", "KRX", tcfg.ticker, end_date_str, mode)
        if not rows:
            err = f"Naver Finance returned 0 posts for {tcfg.ticker} (actor={actor_id}, run_id={run_id})"
            manifest_store.record(
                ManifestEntry(
                    cache_key=ckey,
                    stage="social",
                    source="naver_finance",
                    exchange="KRX",
                    key=tcfg.ticker,
                    end_date=end_date_str,
                    mode=mode,
                    status="failed",
                    actor_id=actor_id,
                    apify_run_id=run_id,
                    raw_path=str(raw_path),
                    est_cost_usd=round(cost_per_ticker, 6),
                    updated_at_utc=utc_now_iso(),
                    error_detail=err,
                )
            )
            raise RuntimeError(err)

        raw_count = write_jsonl_atomic(raw_path, rows)
        manifest_store.record(
            ManifestEntry(
                cache_key=ckey,
                stage="social",
                source="naver_finance",
                exchange="KRX",
                key=tcfg.ticker,
                end_date=end_date_str,
                mode=mode,
                status="ok",
                actor_id=actor_id,
                apify_run_id=run_id,
                raw_path=str(raw_path),
                raw_count=raw_count,
                kept_count=raw_count,
                est_cost_usd=round(cost_per_ticker, 6),
                updated_at_utc=utc_now_iso(),
            )
        )
        logger.info(
            "[naver] %s complete: %d full-body posts (run_id=%s, est_cost=$%.4f)",
            tcfg.ticker,
            raw_count,
            run_id,
            cost_per_ticker,
        )

    return cached_items + fetched_items
