"""Standby news backup adapter using tavily-python (Phase 1 Section 1.6 & 1.9).

STRICT POLICY:
NEVER auto-invoked on Exa failures. Runs ONLY when explicitly requested via `--use-backup tavily`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tavily import TavilyClient

from src.ingest.manifest import (
    ManifestStore,
    make_cache_key,
    read_jsonl,
    utc_now_iso,
    write_jsonl_atomic,
)
from src.ingest.news_exa import canonicalize_url, extract_domain
from src.ingest.schemas import (
    RAW_DIR,
    ExchangeWindowBounds,
    ManifestEntry,
    RunMode,
    TickerConfig,
)

logger = logging.getLogger(__name__)


def ingest_tavily_backup(
    tickers: list[TickerConfig],
    exchange_windows: dict[str, ExchangeWindowBounds],
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Standby Tavily news ingestion (invoked ONLY via --use-backup tavily)."""
    cfg = sources_cfg["news"]["tavily_backup"]
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    client: TavilyClient | None = None
    all_rows: list[dict[str, Any]] = []

    for tcfg in tickers:
        raw_path = RAW_DIR / "news" / "tavily" / tcfg.exchange / f"{tcfg.ticker}_{end_date_str}.jsonl"
        ckey = make_cache_key("news", "tavily", tcfg.exchange, tcfg.ticker, end_date_str, mode)

        if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
            all_rows.extend(read_jsonl(raw_path))
            continue

        if client is None:
            if not api_key:
                raise RuntimeError("TAVILY_API_KEY is missing in environment/.env")
            client = TavilyClient(api_key=api_key)

        query = f"{tcfg.company_en} ({tcfg.ticker}) stock earnings analyst rating catalyst"
        max_res = 5 if mode == "smoke" else int(cfg["max_results"])
        resp = client.search(query=query, topic=cfg["topic"], max_results=max_res, days=21)
        results = resp.get("results", []) if isinstance(resp, dict) else []
        rows: list[dict[str, Any]] = []
        for r in results:
            url = r.get("url", "")
            rows.append(
                {
                    "source": "tavily",
                    "exchange": tcfg.exchange,
                    "ticker": tcfg.ticker,
                    "source_item_id": canonicalize_url(url) or url,
                    "domain": extract_domain(url),
                    "title": (r.get("title") or "").strip(),
                    "text": (r.get("content") or "").strip(),
                    "url": url,
                    "published_date": r.get("published_date"),
                    "language_hint": "en",
                    "source_tier": "tier2",
                    "source_credibility_weight": float(cfg["credibility_weight"]),
                    "has_full_body": bool((r.get("content") or "").strip()),
                }
            )
        if not rows:
            raise RuntimeError(f"Tavily backup returned 0 articles for {tcfg.ticker}")
        raw_count = write_jsonl_atomic(raw_path, rows)
        manifest_store.record(
            ManifestEntry(
                cache_key=ckey,
                stage="news",
                source="tavily",
                exchange=tcfg.exchange,
                key=tcfg.ticker,
                end_date=end_date_str,
                mode=mode,
                status="ok",
                raw_path=str(raw_path),
                raw_count=raw_count,
                kept_count=raw_count,
                est_cost_usd=0.0,
                updated_at_utc=utc_now_iso(),
            )
        )
        all_rows.extend(rows)

    return all_rows
