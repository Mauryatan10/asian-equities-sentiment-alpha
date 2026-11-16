"""Standby social backup adapters: AAStocks (HKEX) and Blind (KRX).

STRICT POLICY (Phase 1 Section 1.9):
These adapters are DISABLED by default and NEVER auto-invoked on primary failures.
They run ONLY when explicitly requested via `--use-backup aastocks` or `--use-backup blind`.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import quote

from apify_client import ApifyClient

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

AASTOCKS_PAGE_FUNCTION = r"""
async function pageFunction(context) {
    const { $, request } = context;
    const ud = request.userData || {};
    const items = [];
    $('div.newshead4, div.newscontent4, a[href*="news-content"]').each((_, el) => {
        const $el = $(el);
        const $link = $el.is('a') ? $el : $el.find('a').first();
        const title = $link.text().trim();
        const href = $link.attr('href') || '';
        if (!title || !href) return;
        const url = href.startsWith('http') ? href : `https://www.aastocks.com${href}`;
        const timeText = $el.closest('div').find('.inline_block, .newstime4').first().text().trim();
        items.push({
            source: 'aastocks',
            exchange: 'HKEX',
            ticker: ud.ticker,
            source_item_id: url,
            title,
            text: title,
            published_raw: timeText,
            url,
            has_full_body: false,
        });
    });
    return items;
}
"""


def ingest_aastocks_backup(
    tickers: list[TickerConfig],
    window: ExchangeWindowBounds,
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Standby HKEX backup adapter for AAStocks (invoked ONLY via --use-backup aastocks)."""
    hkex_tickers = [t for t in tickers if t.exchange == "HKEX"]
    if not hkex_tickers:
        return []

    cfg = sources_cfg["social"]["backups"]["aastocks"]
    actor_id = cfg["actor_id"]
    key_label = "aastocks_hkex"
    raw_path = RAW_DIR / "social" / "aastocks" / "HKEX" / f"{key_label}_{end_date_str}.jsonl"
    ckey = make_cache_key("social", "aastocks", "HKEX", key_label, end_date_str, mode)

    if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
        return read_jsonl(raw_path)

    api_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("APIFY_API_TOKEN is missing")

    check_apify_budget(
        api_token,
        hard_stop_usd=float(sources_cfg["budget_guard"]["apify_hard_stop_monthly_usd"]),
        estimated_next_run_usd=0.05,
    )

    start_urls = [
        {
            "url": cfg["url_template"].format(code_5d=t.exchange_code),
            "userData": {"ticker": t.ticker, "code_5d": t.exchange_code},
        }
        for t in hkex_tickers
    ]
    client = ApifyClient(api_token)
    run = client.actor(actor_id).call(
        run_input={
            "startUrls": start_urls,
            "pageFunction": AASTOCKS_PAGE_FUNCTION,
            "proxyConfiguration": cfg["proxy_configuration"],
            "maxConcurrency": 3,
            "maxPagesPerCrawl": int(cfg["max_pages_per_ticker"]) * len(hkex_tickers),
        },
        memory_mbytes=int(cfg["memory_mbytes"]),
        run_timeout=timedelta(seconds=int(cfg["run_timeout_secs"])),
        force_permission_level="LIMITED_PERMISSIONS",
    )
    if run is None or getattr(run, "status", None) != "SUCCEEDED":
        raise RuntimeError(f"AAStocks backup run failed: {getattr(run, 'status_message', None)}")

    items = list(client.dataset(run.default_dataset_id).iterate_items())
    if not items:
        raise RuntimeError(f"AAStocks backup returned 0 items (run_id={run.id})")

    raw_count = write_jsonl_atomic(raw_path, items)
    manifest_store.record(
        ManifestEntry(
            cache_key=ckey,
            stage="social",
            source="aastocks",
            exchange="HKEX",
            key=key_label,
            end_date=end_date_str,
            mode=mode,
            status="ok",
            actor_id=actor_id,
            apify_run_id=run.id,
            raw_path=str(raw_path),
            raw_count=raw_count,
            kept_count=raw_count,
            est_cost_usd=float(getattr(run, "usage_total_usd", 0.0) or 0.0),
            updated_at_utc=utc_now_iso(),
        )
    )
    return items


def ingest_blind_backup(
    tickers: list[TickerConfig],
    window: ExchangeWindowBounds,
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Standby KRX backup adapter for Blind via hypebridge/blind-post-scraper (ONLY via --use-backup blind)."""
    krx_tickers = [t for t in tickers if t.exchange == "KRX"]
    if not krx_tickers:
        return []

    cfg = sources_cfg["social"]["backups"]["blind"]
    actor_id = cfg["actor_id"]
    key_label = "blind_krx"
    raw_path = RAW_DIR / "social" / "blind" / "KRX" / f"{key_label}_{end_date_str}.jsonl"
    ckey = make_cache_key("social", "blind", "KRX", key_label, end_date_str, mode)

    if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
        return read_jsonl(raw_path)

    api_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("APIFY_API_TOKEN is missing")

    max_charge = Decimal(str(0.10 if mode == "smoke" else cfg["max_total_charge_usd"]))
    check_apify_budget(
        api_token,
        hard_stop_usd=float(sources_cfg["budget_guard"]["apify_hard_stop_monthly_usd"]),
        estimated_next_run_usd=float(max_charge),
    )

    start_urls = [
        {"url": f"https://www.teamblind.com/search/{quote(t.company_en)}"}
        for t in krx_tickers
    ]
    max_items = 5 if mode == "smoke" else int(cfg["max_posts_per_ticker"]) * len(krx_tickers)

    # Uses Pre-Flight verified inputSchema keys for hypebridge/blind-post-scraper
    run_input = {
        "startUrls": start_urls,
        "maxItems": max_items,
        "scrapePostsFromLists": True,
        "scrapeComments": False,
        "maxComments": 0,
        "maxPages": 2 if mode == "smoke" else 10,
        "maxConcurrency": 3,
    }
    client = ApifyClient(api_token)
    run = client.actor(actor_id).call(
        run_input=run_input,
        memory_mbytes=int(cfg["memory_mbytes"]),
        run_timeout=timedelta(seconds=int(cfg["run_timeout_secs"])),
        max_total_charge_usd=max_charge,
    )
    if run is None or getattr(run, "status", None) != "SUCCEEDED":
        raise RuntimeError(f"Blind backup run failed: {getattr(run, 'status_message', None)}")

    items = list(client.dataset(run.default_dataset_id).iterate_items())
    if not items:
        raise RuntimeError(f"Blind backup returned 0 items (run_id={run.id})")

    raw_count = write_jsonl_atomic(raw_path, items)
    manifest_store.record(
        ManifestEntry(
            cache_key=ckey,
            stage="social",
            source="blind",
            exchange="KRX",
            key=key_label,
            end_date=end_date_str,
            mode=mode,
            status="ok",
            actor_id=actor_id,
            apify_run_id=run.id,
            raw_path=str(raw_path),
            raw_count=raw_count,
            kept_count=raw_count,
            est_cost_usd=float(getattr(run, "usage_total_usd", 0.0) or 0.0),
            updated_at_utc=utc_now_iso(),
        )
    )
    return items
