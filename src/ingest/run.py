"""Modular CLI entrypoint for Phase 1: Data Ingestion & Temporal Alignment.

Usage:
  python -m src.ingest.run \
    --stage {preflight|prices|social|news|align|all} \
    --exchange {hkex|krx|all} \
    [--tickers 0700.HK,005930.KS] \
    [--end-date YYYY-MM-DD] \
    [--smoke-test] \
    [--force-refresh] \
    [--use-backup {aastocks|blind|tavily}]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.ingest.align import align_events_to_bars, verify_phase1_alignment, write_events_parquet
from src.ingest.manifest import ManifestStore, make_cache_key, read_jsonl, utc_now_iso
from src.ingest.news_exa import ingest_exa_news
from src.ingest.news_tavily_backup import ingest_tavily_backup
from src.ingest.normalize import normalize_events
from src.ingest.preflight import run_preflight
from src.ingest.prices import ingest_prices, load_exchange_windows_from_bars
from src.ingest.schemas import (
    BARS_PARQUET_PATH,
    PROJECT_ROOT,
    RAW_DIR,
    ExchangeCode,
    ExchangeWindowBounds,
    ManifestEntry,
    RunMode,
    TickerConfig,
    load_sources,
    load_universe,
)
from src.ingest.social_backups import ingest_aastocks_backup, ingest_blind_backup
from src.ingest.social_lihkg import ingest_lihkg
from src.ingest.social_naver import ingest_naver_finance
from src.ingest.social_telegram import ingest_telegram

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_raw_events_from_disk(
    tickers: list[TickerConfig],
    end_date_str: str,
    use_backup: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load raw social & news JSONL snapshots for the selected tickers & end_date."""
    exchanges = sorted({t.exchange for t in tickers})
    raw_social: list[dict[str, Any]] = []
    raw_news: list[dict[str, Any]] = []

    for ex in exchanges:
        tg_path = RAW_DIR / "social" / "telegram" / ex / f"channels_{ex.lower()}_{end_date_str}.jsonl"
        if tg_path.exists():
            raw_social.extend(read_jsonl(tg_path))

    if "HKEX" in exchanges:
        lihkg_path = RAW_DIR / "social" / "lihkg" / "HKEX" / f"threads_hkex_{end_date_str}.jsonl"
        if lihkg_path.exists():
            raw_social.extend(read_jsonl(lihkg_path))
        if use_backup == "aastocks":
            aa_path = RAW_DIR / "social" / "aastocks" / "HKEX" / f"aastocks_hkex_{end_date_str}.jsonl"
            if aa_path.exists():
                raw_social.extend(read_jsonl(aa_path))

    if "KRX" in exchanges:
        for t in tickers:
            if t.exchange == "KRX":
                nf_path = RAW_DIR / "social" / "naver_finance" / "KRX" / f"{t.ticker}_{end_date_str}.jsonl"
                if nf_path.exists():
                    raw_social.extend(read_jsonl(nf_path))
        if use_backup == "blind":
            bl_path = RAW_DIR / "social" / "blind" / "KRX" / f"blind_krx_{end_date_str}.jsonl"
            if bl_path.exists():
                raw_social.extend(read_jsonl(bl_path))

    for t in tickers:
        exa_path = RAW_DIR / "news" / "exa" / t.exchange / f"{t.ticker}_{end_date_str}.jsonl"
        if exa_path.exists():
            raw_news.extend(read_jsonl(exa_path))
        if use_backup == "tavily":
            tav_path = RAW_DIR / "news" / "tavily" / t.exchange / f"{t.ticker}_{end_date_str}.jsonl"
            if tav_path.exists():
                raw_news.extend(read_jsonl(tav_path))

    return raw_social, raw_news


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Asian Equities Sentiment Alpha — Data Ingestion & Temporal Alignment"
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "prices", "social", "news", "align", "all"],
        required=True,
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--exchange",
        choices=["hkex", "krx", "all"],
        default="all",
        help="Target exchange scope.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Optional comma-separated ticker override (e.g. 0700.HK,005930.KS).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional anchor end date YYYY-MM-DD (defaults to latest completed 1h bar as of run time).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal low-cost smoke test (1 ticker/exchange, tiny Apify/Exa caps, ~$0.02-$0.05).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass manifest cache and re-fetch from remote APIs.",
    )
    parser.add_argument(
        "--use-backup",
        choices=["aastocks", "blind", "tavily"],
        default=None,
        help="Explicitly invoke a standby backup adapter (never auto-invoked).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    _configure_logging(args.verbose)

    universe = load_universe()
    sources_cfg = load_sources()
    manifest_store = ManifestStore()

    mode: RunMode = "smoke" if args.smoke_test else "full"
    run_utc = datetime.now(timezone.utc)
    end_date_str = args.end_date or run_utc.strftime("%Y-%m-%d")

    explicit_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    tickers = universe.get_tickers(
        exchange=args.exchange,
        explicit_tickers=explicit_tickers,
        smoke_test=args.smoke_test,
        smoke_tickers_by_exchange=sources_cfg["smoke_test"]["tickers_by_exchange"],
    )
    if not tickers and args.stage != "preflight":
        logger.error("No tickers selected for --exchange=%s --tickers=%s", args.exchange, args.tickers)
        return 1

    exchanges_in_scope: list[ExchangeCode] = sorted({t.exchange for t in tickers})
    logger.info(
        "Phase 1 Runner | stage=%s | mode=%s | end_date=%s | exchanges=%s | tickers=%s | backup=%s",
        args.stage,
        mode,
        end_date_str,
        exchanges_in_scope,
        [t.ticker for t in tickers],
        args.use_backup,
    )

    try:
        # 1. Pre-Flight Verification Gate ($0.00)
        if args.stage in ("preflight", "all"):
            preflight_report = run_preflight(manifest_store=manifest_store)
            if args.stage == "preflight":
                print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
                return 0

        # 2. Prices (1h OHLCV & Per-Exchange 14-Trading-Day Calendars)
        bars_df: pd.DataFrame | None = None
        exchange_windows: dict[ExchangeCode, ExchangeWindowBounds]
        if args.stage in ("prices", "all"):
            bars_df, exchange_windows = ingest_prices(
                universe=universe,
                tickers=tickers,
                manifest_store=manifest_store,
                end_date_str=args.end_date,
                mode=mode,
                force_refresh=args.force_refresh,
                run_utc=run_utc,
            )
        else:
            exchange_windows = load_exchange_windows_from_bars()
            bars_df = pd.read_parquet(BARS_PARQUET_PATH)

        # 3. Social Sentiment Scraping (Apify)
        if args.stage in ("social", "all"):
            for ex in exchanges_in_scope:
                ex_tickers = [t for t in tickers if t.exchange == ex]
                ingest_telegram(
                    exchange=ex,
                    tickers=ex_tickers,
                    window=exchange_windows[ex],
                    sources_cfg=sources_cfg,
                    manifest_store=manifest_store,
                    end_date_str=end_date_str,
                    mode=mode,
                    force_refresh=args.force_refresh,
                )

            if "HKEX" in exchanges_in_scope:
                ingest_lihkg(
                    tickers=tickers,
                    window=exchange_windows["HKEX"],
                    sources_cfg=sources_cfg,
                    manifest_store=manifest_store,
                    end_date_str=end_date_str,
                    mode=mode,
                    force_refresh=args.force_refresh,
                )
                if args.use_backup == "aastocks":
                    ingest_aastocks_backup(
                        tickers=tickers,
                        window=exchange_windows["HKEX"],
                        sources_cfg=sources_cfg,
                        manifest_store=manifest_store,
                        end_date_str=end_date_str,
                        mode=mode,
                        force_refresh=args.force_refresh,
                    )

            if "KRX" in exchanges_in_scope:
                ingest_naver_finance(
                    tickers=tickers,
                    window=exchange_windows["KRX"],
                    sources_cfg=sources_cfg,
                    manifest_store=manifest_store,
                    end_date_str=end_date_str,
                    mode=mode,
                    force_refresh=args.force_refresh,
                )
                if args.use_backup == "blind":
                    ingest_blind_backup(
                        tickers=tickers,
                        window=exchange_windows["KRX"],
                        sources_cfg=sources_cfg,
                        manifest_store=manifest_store,
                        end_date_str=end_date_str,
                        mode=mode,
                        force_refresh=args.force_refresh,
                    )

        # 4. Institutional News (Exa primary; Tavily ONLY if --use-backup tavily)
        if args.stage in ("news", "all"):
            ingest_exa_news(
                tickers=tickers,
                exchange_windows=exchange_windows,
                sources_cfg=sources_cfg,
                manifest_store=manifest_store,
                end_date_str=end_date_str,
                mode=mode,
                force_refresh=args.force_refresh,
            )
            if args.use_backup == "tavily":
                ingest_tavily_backup(
                    tickers=tickers,
                    exchange_windows=exchange_windows,
                    sources_cfg=sources_cfg,
                    manifest_store=manifest_store,
                    end_date_str=end_date_str,
                    mode=mode,
                    force_refresh=args.force_refresh,
                )

        # 5. Normalize + Zero-Look-Ahead Temporal Alignment + Verification Assertions
        if args.stage in ("align", "all"):
            assert bars_df is not None
            raw_social, raw_news = _load_raw_events_from_disk(
                tickers=tickers,
                end_date_str=end_date_str,
                use_backup=args.use_backup,
            )
            normalized_events, norm_stats = normalize_events(
                raw_social_records=raw_social,
                raw_news_records=raw_news,
                universe=universe,
                active_tickers=tickers,
                exchange_windows=exchange_windows,
                sources_cfg=sources_cfg,
            )

            events_df = align_events_to_bars(
                events=normalized_events,
                bars_df=bars_df,
                universe=universe,
            )
            out_events_path = write_events_parquet(events_df)
            logger.info("[align] Wrote %d aligned events to %s", len(events_df), out_events_path)

            # Record per-source normalization counters into phase1_manifest.json
            for src_name, st in norm_stats.items():
                ckey = make_cache_key("align", src_name, args.exchange.upper(), "summary", end_date_str, mode)
                manifest_store.record(
                    ManifestEntry(
                        cache_key=ckey,
                        stage="align",
                        source=src_name,
                        exchange=args.exchange.upper(),
                        key="summary",
                        end_date=end_date_str,
                        mode=mode,
                        status="ok",
                        raw_path=str(out_events_path),
                        raw_count=st.raw_count,
                        kept_count=st.kept_count,
                        dropped_empty_count=st.dropped_empty_count,
                        dropped_cleanbot_count=st.dropped_cleanbot_count,
                        dropped_pre_warmup_count=st.dropped_pre_warmup_count,
                        dropped_dedup_count=st.dropped_dedup_count,
                        est_cost_usd=0.0,
                        updated_at_utc=utc_now_iso(),
                    )
                )

            verification = verify_phase1_alignment(
                bars_df=bars_df,
                events_df=events_df,
                active_tickers=tickers,
                universe=universe,
            )
            print(json.dumps(verification, indent=2, ensure_ascii=False))

    except Exception as exc:
        logger.exception("Phase 1 pipeline failed loudly (strict fail-fast): %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
