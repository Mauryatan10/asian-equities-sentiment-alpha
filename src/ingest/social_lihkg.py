"""LIHKG (財經台) ingestion via claude_code_reviewer/lihkg-scraper-en (Phase 1 Section 1.5).

Uses verbatim Pre-Flight-verified inputSchema keys and strict API-level max_total_charge_usd caps:
- Step 1 (mode='search'): keyword search with dateFrom early-stop ($0.002/thread; cap $0.65 full / $0.02 smoke)
- Step 2 (mode='thread'): top engagement threads for OP own_text + replies (cap $0.35 full / $0.03 smoke)
- aiLabels ALWAYS False (never pay +$0.002/row).
- Total LIHKG spend strictly <= $1.00 full / <= $0.05 smoke.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from decimal import Decimal
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


def _extract_run_cost_usd(run: Any) -> float:
    # Pay-per-event actors report event charges via charged_event_counts + pricing_info
    total_ppe = 0.0
    charged = getattr(run, "charged_event_counts", None) or {}
    pricing = getattr(run, "pricing_info", None)
    if charged and pricing is not None:
        ppe_obj = getattr(pricing, "pricing_per_event", None)
        if ppe_obj is None and isinstance(pricing, dict):
            ppe_obj = pricing.get("pricingPerEvent") or pricing.get("pricing_per_event")
        events_map = None
        if ppe_obj is not None:
            events_map = getattr(ppe_obj, "actor_charge_events", None)
            if events_map is None and isinstance(ppe_obj, dict):
                events_map = ppe_obj.get("actorChargeEvents") or ppe_obj.get("actor_charge_events")
        if isinstance(events_map, dict):
            for ev_name, count in charged.items():
                ev_meta = events_map.get(ev_name)
                price = None
                if ev_meta is not None:
                    price = getattr(ev_meta, "event_price_usd", None)
                    if price is None and isinstance(ev_meta, dict):
                        price = ev_meta.get("eventPriceUsd") or ev_meta.get("event_price_usd")
                if price is not None:
                    total_ppe += float(count) * float(price)
    if total_ppe > 0.0:
        return total_ppe

    for attr in ("external_charged_total_usd", "usage_total_usd"):
        val = getattr(run, attr, None)
        if val is not None and float(val) > 0:
            return float(val)
    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type((RateLimitError, ServerError, ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _call_lihkg_actor(
    client: ApifyClient,
    actor_id: str,
    run_input: dict[str, Any],
    memory_mbytes: int,
    run_timeout_secs: int,
    max_total_charge_usd: Decimal,
) -> tuple[str, list[dict[str, Any]], float]:
    run = client.actor(actor_id).call(
        run_input=run_input,
        memory_mbytes=memory_mbytes,
        run_timeout=timedelta(seconds=run_timeout_secs),
        max_total_charge_usd=max_total_charge_usd,
        force_permission_level="LIMITED_PERMISSIONS",
    )
    if run is None:
        raise RuntimeError(f"Apify actor {actor_id} returned None run object")
    status = getattr(run, "status", None)
    run_id = getattr(run, "id", "")
    # Note: Pay-per-event actors gracefully finish with SUCCEEDED when max_total_charge_usd is reached
    if status not in ("SUCCEEDED",):
        raise RuntimeError(
            f"Apify LIHKG run failed (actor={actor_id}, run_id={run_id}, status={status}, "
            f"msg={getattr(run, 'status_message', None)})"
        )
    dataset_id = getattr(run, "default_dataset_id", None)
    if not dataset_id:
        raise RuntimeError(f"Apify run {run_id} missing default_dataset_id")
    items = list(client.dataset(dataset_id).iterate_items())
    cost = _extract_run_cost_usd(run)
    return run_id, items, cost


def _engagement_score(row: dict[str, Any]) -> int:
    likes = int(row.get("like_count") or 0)
    dislikes = int(row.get("dislike_count") or 0)
    replies = int(row.get("reply_count") or 0)
    return likes + dislikes + replies


def ingest_lihkg(
    tickers: list[TickerConfig],
    window: ExchangeWindowBounds,
    sources_cfg: dict[str, Any],
    manifest_store: ManifestStore,
    end_date_str: str,
    mode: RunMode = "full",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Scrape LIHKG threads & top-thread replies for HKEX tickers within strict $1.00 ($0.05 smoke) hard cap."""
    hkex_tickers = [t for t in tickers if t.exchange == "HKEX"]
    if not hkex_tickers:
        return []

    lihkg_cfg = sources_cfg["social"]["lihkg"]
    smoke_cfg = sources_cfg["smoke_test"]["lihkg"]
    smoke_test = mode == "smoke"
    actor_id = lihkg_cfg["actor_id"]
    key_label = "threads_hkex"
    raw_path = RAW_DIR / "social" / "lihkg" / "HKEX" / f"{key_label}_{end_date_str}.jsonl"
    ckey = make_cache_key("social", "lihkg", "HKEX", key_label, end_date_str, mode)

    if manifest_store.is_cached_ok(ckey, force_refresh=force_refresh):
        logger.info("[lihkg] Cache HIT (%s)", raw_path.name)
        return read_jsonl(raw_path)

    api_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("APIFY_API_TOKEN is missing in environment/.env")

    search_cap = Decimal(
        str(smoke_cfg["search_max_charge_usd"] if smoke_test else lihkg_cfg["search_max_total_charge_usd"])
    )
    detail_cap = Decimal(
        str(smoke_cfg["detail_max_charge_usd"] if smoke_test else lihkg_cfg["detail_max_total_charge_usd"])
    )
    total_cap = search_cap + detail_cap

    hard_stop = float(sources_cfg["budget_guard"]["apify_hard_stop_monthly_usd"])
    check_apify_budget(api_token, hard_stop_usd=hard_stop, estimated_next_run_usd=float(total_cap))

    # Build deduplicated keyword list
    keywords: list[str] = []
    seen_kw: set[str] = set()
    for tcfg in hkex_tickers:
        for kw in tcfg.lihkg_keywords:
            if kw not in seen_kw:
                seen_kw.add(kw)
                keywords.append(kw)
    if smoke_test:
        keywords = keywords[: int(smoke_cfg["max_keywords"])]

    # dateFrom format verified from inputSchema: MM/DD/YY
    warmup_hkt = window.exchange_warmup_lead_utc.astimezone(ZoneInfo("Asia/Hong_Kong"))
    date_from_str = warmup_hkt.strftime("%m/%d/%y")

    search_max_items = (
        int(smoke_cfg["search_max_items"]) if smoke_test else int(lihkg_cfg["search_max_items_per_keyword"])
    )
    search_input = {
        "mode": "search",
        "keywords": keywords,
        "order": lihkg_cfg["search_order"],
        "dateFrom": date_from_str,
        "maxItems": search_max_items,
        "aiLabels": False,
        "proxyConfiguration": lihkg_cfg["proxy_configuration"],
    }

    client = ApifyClient(api_token)
    logger.info(
        "[lihkg] Step 1 (search): keywords=%s, dateFrom=%s, maxItems=%d, hard_cap=$%s",
        keywords,
        date_from_str,
        search_max_items,
        search_cap,
    )

    try:
        run1_id, search_rows, cost1 = _call_lihkg_actor(
            client=client,
            actor_id=actor_id,
            run_input=search_input,
            memory_mbytes=int(lihkg_cfg["memory_mbytes"]),
            run_timeout_secs=int(lihkg_cfg["run_timeout_secs"]),
            max_total_charge_usd=search_cap,
        )
        if not search_rows:
            raise RuntimeError(
                f"LIHKG search returned 0 threads (actor={actor_id}, run_id={run1_id}, keywords={keywords})"
            )

        # Deduplicate search threads by thread_id, preferring highest engagement
        thread_by_id: dict[str, dict[str, Any]] = {}
        for row in search_rows:
            tid = str(row.get("thread_id") or "")
            if not tid:
                continue
            prev = thread_by_id.get(tid)
            if prev is None or _engagement_score(row) > _engagement_score(prev):
                thread_by_id[tid] = row

        # Step 2 (thread details + replies) on top engagement threads within strict remaining cap
        max_detail_threads = (
            int(smoke_cfg["detail_max_threads"])
            if smoke_test
            else int(lihkg_cfg["detail_top_threads_per_ticker"]) * len(hkex_tickers)
        )
        max_replies = (
            int(smoke_cfg["detail_max_replies"]) if smoke_test else int(lihkg_cfg["detail_max_replies_per_thread"])
        )

        # Filter candidates for Step 2 to threads created within the 14-day window (avoiding pre-warmup early floors)
        warmup_lead_iso = window.exchange_warmup_lead_utc.strftime("%Y-%m-%dT%H:%M:%S")
        in_window_threads = [
            row
            for row in thread_by_id.values()
            if (row.get("created_at") or "") >= warmup_lead_iso
        ]
        candidate_pool = in_window_threads if in_window_threads else list(thread_by_id.values())

        # Pick diverse top threads across keywords/tickers
        top_threads_sorted = sorted(candidate_pool, key=_engagement_score, reverse=True)
        selected_tids: list[str] = []
        seen_tids: set[str] = set()
        # First try up to per-ticker quota by matching source_query or title
        per_ticker_quota = 1 if smoke_test else int(lihkg_cfg["detail_top_threads_per_ticker"])
        for tcfg in hkex_tickers:
            picked_for_t = 0
            kws_lower = [k.lower() for k in (tcfg.lihkg_keywords + tcfg.cjk_aliases + tcfg.ascii_aliases)]
            for row in top_threads_sorted:
                tid = str(row["thread_id"])
                if tid in seen_tids:
                    continue
                hay = f"{row.get('title', '')} {row.get('source_query', '')}".lower()
                if any(kw in hay for kw in kws_lower):
                    selected_tids.append(tid)
                    seen_tids.add(tid)
                    picked_for_t += 1
                    if picked_for_t >= per_ticker_quota or len(selected_tids) >= max_detail_threads:
                        break
        # Top up with overall highest engagement if slots remain
        for row in top_threads_sorted:
            if len(selected_tids) >= max_detail_threads:
                break
            tid = str(row["thread_id"])
            if tid not in seen_tids:
                selected_tids.append(tid)
                seen_tids.add(tid)

        run2_id = None
        cost2 = 0.0
        detail_rows: list[dict[str, Any]] = []
        if selected_tids and detail_cap > Decimal("0"):
            detail_input = {
                "mode": "thread",
                "threadIds": selected_tids,
                "maxItems": len(selected_tids),
                "maxReplies": max_replies,
                "aiLabels": False,
                "proxyConfiguration": lihkg_cfg["proxy_configuration"],
            }
            logger.info(
                "[lihkg] Step 2 (thread details): %d threads x maxReplies=%d, hard_cap=$%s",
                len(selected_tids),
                max_replies,
                detail_cap,
            )
            run2_id, detail_rows, cost2 = _call_lihkg_actor(
                client=client,
                actor_id=actor_id,
                run_input=detail_input,
                memory_mbytes=int(lihkg_cfg["memory_mbytes"]),
                run_timeout_secs=int(lihkg_cfg["run_timeout_secs"]),
                max_total_charge_usd=detail_cap,
            )

        # Merge Step 1 listings + Step 2 full OP bodies & replies
        final_records: list[dict[str, Any]] = []
        enriched_op_by_tid: dict[str, dict[str, Any]] = {}
        reply_rows: list[dict[str, Any]] = []

        for drow in detail_rows:
            tid = str(drow.get("thread_id") or "")
            if drow.get("reply_id") is not None or drow.get("floor") is not None:
                # Reply row — attach parent thread title & URL for reliable ticker attribution
                parent = thread_by_id.get(tid, {})
                reply_rows.append(
                    {
                        "source": "lihkg",
                        "exchange": "HKEX",
                        "record_kind": "reply",
                        "thread_id": tid,
                        "reply_id": str(drow.get("reply_id") or f"{tid}_f{drow.get('floor')}"),
                        "floor": drow.get("floor"),
                        "title": parent.get("title") or "",
                        "text": drow.get("own_text") or drow.get("content") or "",
                        "published_utc": drow.get("created_at") or parent.get("last_reply_at") or parent.get("created_at"),
                        "likes": int(drow.get("like_count") or 0),
                        "dislikes": int(drow.get("dislike_count") or 0),
                        "reply_count": int(parent.get("reply_count") or 0),
                        "category": parent.get("category") or "財經台",
                        "url": parent.get("thread_url") or f"https://lihkg.com/thread/{tid}/page/1",
                        "has_full_body": True,
                    }
                )
            elif tid:
                enriched_op_by_tid[tid] = drow

        for tid, base in thread_by_id.items():
            op = enriched_op_by_tid.get(tid)
            body_text = ""
            has_full = False
            if op is not None:
                body_text = op.get("own_text") or op.get("content") or ""
                has_full = bool(body_text.strip())
            published_raw = (
                (op.get("created_at") if op else None)
                or base.get("created_at")
                or base.get("last_reply_at")
            )
            final_records.append(
                {
                    "source": "lihkg",
                    "exchange": "HKEX",
                    "record_kind": "thread_op",
                    "thread_id": tid,
                    "reply_id": None,
                    "floor": 1,
                    "title": (op.get("title") if op else None) or base.get("title") or "",
                    "text": body_text,
                    "published_utc": published_raw,
                    "last_reply_at": base.get("last_reply_at"),
                    "likes": int((op.get("like_count") if op and op.get("like_count") is not None else base.get("like_count")) or 0),
                    "dislikes": int((op.get("dislike_count") if op and op.get("dislike_count") is not None else base.get("dislike_count")) or 0),
                    "reply_count": int((op.get("reply_count") if op and op.get("reply_count") is not None else base.get("reply_count")) or 0),
                    "category": (op.get("category") if op else None) or base.get("category") or "財經台",
                    "url": base.get("thread_url") or f"https://lihkg.com/thread/{tid}/page/1",
                    "has_full_body": has_full,
                }
            )

        final_records.extend(reply_rows)
        raw_count = write_jsonl_atomic(raw_path, final_records)
        total_cost = cost1 + cost2
        run_ids_joined = f"{run1_id},{run2_id}" if run2_id else run1_id

        manifest_store.record(
            ManifestEntry(
                cache_key=ckey,
                stage="social",
                source="lihkg",
                exchange="HKEX",
                key=key_label,
                end_date=end_date_str,
                mode=mode,
                status="ok",
                actor_id=actor_id,
                apify_run_id=run_ids_joined,
                raw_path=str(raw_path),
                raw_count=raw_count,
                kept_count=raw_count,
                est_cost_usd=round(total_cost, 6),
                updated_at_utc=utc_now_iso(),
            )
        )
        logger.info(
            "[lihkg] Complete: %d threads + %d replies = %d raw rows (runs=%s, total_cost=$%.4f)",
            len(thread_by_id),
            len(reply_rows),
            raw_count,
            run_ids_joined,
            total_cost,
        )
        return final_records

    except Exception as exc:
        manifest_store.record(
            ManifestEntry(
                cache_key=ckey,
                stage="social",
                source="lihkg",
                exchange="HKEX",
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
