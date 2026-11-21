"""Modular CLI entrypoint for Phase 2: Feature Engineering & Multi-Agent Processing.

Usage:
  python -m src.features.run \
    --stage {preflight|llm|technical|aggregate|all} \
    --llm-provider {nvidia|gemini|groq} \
    [--nim-model nvidia/nemotron-3.5-lightning-30b-a3b] \
    [--tickers 0700.HK,005930.KS] \
    [--smoke-test] \
    [--force-refresh]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from src.features.alpha_factors import (
    compute_alpha_factors_and_snapshot,
    compute_full_corpus_stats,
    compute_impact_weight_coverage_pct,
    verify_ticker_phase2_outputs,
)
from src.features.cultural_nlp_agent import LLMScoreCheckpointStore, run_cultural_nlp_for_ticker
from src.features.graph import build_ticker_phase2_graph, export_mermaid_diagram
from src.features.llm_client import CloudLLMClient, TokenBucketRateLimiter
from src.features.preflight import run_phase2_preflight
from src.features.quant_technical_agent import compute_quant_technical_features
from src.features.router import route_ticker_events
from src.features.schemas import (
    EVENTS_SCORED_PATH,
    EVENTS_SCORED_SCHEMA,
    FEATURES_1H_PATH,
    FEATURES_1H_SCHEMA,
    LIVE_SNAPSHOT_PATH,
    LIVE_SNAPSHOT_SCHEMA,
    LLM_CACHE_PATH,
    MERMAID_GRAPH_PATH,
    PHASE2_MANIFEST_PATH,
    PROCESSED_DIR,
    LLMProviderName,
    Phase2Manifest,
    TickerPhase2Stats,
    load_phase2_config,
)
from src.ingest.schemas import (
    BARS_PARQUET_PATH,
    EVENTS_PARQUET_PATH,
    PROJECT_ROOT,
    TickerConfig,
    load_universe,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_parquet_strict(df: pd.DataFrame, schema: pa.Schema, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cols = [field.name for field in schema]
    missing = set(ordered_cols) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required schema columns for {path.name}: {sorted(missing)}")
    trimmed = df[ordered_cols].copy()
    table = pa.Table.from_pandas(trimmed, schema=schema, preserve_index=False, safe=False)
    pq.write_table(table, path, compression="zstd")
    return path


def _write_manifest(manifest: Phase2Manifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Asian Equities Sentiment Alpha — Feature Engineering & Multi-Agent Processing"
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "llm", "technical", "aggregate", "all"],
        required=True,
        help="Phase 2 stage to execute.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["nvidia", "gemini", "groq"],
        default="nvidia",
        help="Cloud LLM provider (strict fail-fast; never auto-switched).",
    )
    parser.add_argument(
        "--nim-model",
        type=str,
        default=None,
        help="Optional model override (e.g. nvidia/nemotron-3.5-lightning-30b-a3b or minimaxai/minimax-m3.1-flash).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Optional comma-separated ticker subset (e.g. 0700.HK,005930.KS).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test (0700.HK & 005930.KS, 10 LLM items/ticker, ~10s).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass phase2_llm_scores.jsonl checkpoint cache and re-score targeted tickers.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def _select_tickers(
    explicit_tickers_arg: str | None,
    smoke_test: bool,
    smoke_tickers: list[str],
) -> list[TickerConfig]:
    universe = load_universe()
    if explicit_tickers_arg:
        wanted = [t.strip() for t in explicit_tickers_arg.split(",") if t.strip()]
        return universe.get_tickers(exchange="all", explicit_tickers=wanted)
    if smoke_test:
        return universe.get_tickers(exchange="all", explicit_tickers=smoke_tickers)
    return universe.get_tickers(exchange="all")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    _configure_logging(args.verbose)

    cfg = load_phase2_config()
    provider: LLMProviderName = args.llm_provider
    model_override: str | None = args.nim_model
    resolved_model = model_override or cfg.llm.providers[provider].cultural_nlp_model
    run_started_utc = datetime.now(timezone.utc).isoformat()

    try:
        # 1. Preflight Gate
        if args.stage == "preflight":
            report = run_phase2_preflight(cfg=cfg, provider=provider, model_override=model_override)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        tickers = _select_tickers(args.tickers, args.smoke_test, cfg.smoke_test.tickers)
        if not tickers:
            logger.error("No tickers selected for Phase 2.")
            return 1

        is_full_universe = (not args.smoke_test) and (args.tickers is None) and (len(tickers) == 10)
        out_dir = PROCESSED_DIR if is_full_universe else (PROCESSED_DIR / "smoke")
        events_out_path = EVENTS_SCORED_PATH if is_full_universe else (out_dir / "events_scored.parquet")
        features_out_path = FEATURES_1H_PATH if is_full_universe else (out_dir / "features_1h.parquet")
        live_out_path = LIVE_SNAPSHOT_PATH if is_full_universe else (out_dir / "live_snapshot.parquet")
        manifest_out_path = (
            PHASE2_MANIFEST_PATH
            if is_full_universe
            else (PHASE2_MANIFEST_PATH.parent / "phase2_manifest_smoke.json")
        )

        logger.info(
            "Phase 2 Runner | stage=%s | provider=%s | model=%s | smoke=%s | tickers=%s",
            args.stage,
            provider,
            resolved_model,
            args.smoke_test,
            [t.ticker for t in tickers],
        )

        bars_all = pd.read_parquet(BARS_PARQUET_PATH)
        events_all = pd.read_parquet(EVENTS_PARQUET_PATH)
        ticker_set = {t.ticker for t in tickers}

        bars_by_ticker = {
            t.ticker: bars_all[bars_all["ticker"] == t.ticker].sort_values("bar_close_utc").reset_index(drop=True)
            for t in tickers
        }
        events_by_ticker = {
            t.ticker: events_all[events_all["ticker"] == t.ticker].reset_index(drop=True)
            for t in tickers
        }

        checkpoint_store = LLMScoreCheckpointStore(LLM_CACHE_PATH)
        shared_limiter = TokenBucketRateLimiter(cfg.llm.rpm_limit)

        # Stage-specific direct node execution
        if args.stage == "technical":
            tech_frames: list[pd.DataFrame] = []
            for t in tickers:
                b_df = bars_by_ticker[t.ticker]
                tech_df = compute_quant_technical_features(b_df, cfg)
                tech_df.insert(0, "ticker", t.ticker)
                tech_df["window_role"] = b_df["window_role"].to_numpy()
                tech_frames.append(tech_df)
            combined = pd.concat(tech_frames, ignore_index=True)
            active_nulls = int(combined.loc[combined["window_role"] == "active_10d", "poc_price"].isna().sum())
            logger.info("[technical] Computed causal technicals & KDE S/R for %d rows across %d tickers (active_kde_nulls=%d).", len(combined), len(tickers), active_nulls)
            print(
                json.dumps(
                    {
                        "stage": "technical",
                        "tickers": [t.ticker for t in tickers],
                        "rows": len(combined),
                        "active_kde_nulls": active_nulls,
                    },
                    indent=2,
                )
            )
            return 0

        if args.stage == "llm":
            client = CloudLLMClient(
                cfg=cfg,
                provider=provider,
                model_override=model_override,
                rate_limiter=shared_limiter,
            )
            smoke_max = cfg.smoke_test.max_llm_items_per_ticker if args.smoke_test else None
            batch_override = cfg.smoke_test.batch_size if args.smoke_test else None
            scored_frames: list[pd.DataFrame] = []
            stats_out: dict[str, Any] = {}
            for t in tickers:
                routed_df, dedup_map, rep_df = route_ticker_events(
                    events_by_ticker[t.ticker], cfg, smoke_max_items=smoke_max
                )
                scored_df, llm_stats = run_cultural_nlp_for_ticker(
                    ticker=t.ticker,
                    company_en=t.company_en,
                    company_local=t.company_local,
                    bars_df=bars_by_ticker[t.ticker],
                    routed_df=routed_df,
                    dedup_map=dedup_map,
                    rep_df=rep_df,
                    cfg=cfg,
                    client=client,
                    checkpoint_store=checkpoint_store,
                    batch_size_override=batch_override,
                    force_refresh=args.force_refresh,
                )
                scored_frames.append(scored_df)
                stats_out[t.ticker] = llm_stats
            combined_events = pd.concat(scored_frames, ignore_index=True)
            _write_parquet_strict(combined_events, EVENTS_SCORED_SCHEMA, events_out_path)
            print(json.dumps({"stage": "llm", "output": str(events_out_path), "stats": stats_out}, indent=2))
            return 0

        if args.stage == "aggregate":
            if not events_out_path.exists():
                raise FileNotFoundError(
                    f"Cannot run --stage aggregate: {events_out_path} not found. Run --stage llm or --stage all first."
                )
            scored_all = pd.read_parquet(events_out_path)
            feat_frames: list[pd.DataFrame] = []
            live_rows: list[dict[str, Any]] = []
            per_ticker_manifest_agg: dict[str, TickerPhase2Stats] = {}
            for t in tickers:
                b_df = bars_by_ticker[t.ticker]
                e_raw = events_by_ticker[t.ticker]
                e_scored = scored_all[scored_all["ticker"] == t.ticker].reset_index(drop=True)
                tech_df = compute_quant_technical_features(b_df, cfg)
                fc_df = compute_full_corpus_stats(b_df, e_raw, cfg)
                f1h_df, snap_row = compute_alpha_factors_and_snapshot(
                    ticker=t.ticker,
                    exchange=t.exchange,
                    company_en=t.company_en,
                    company_local=t.company_local,
                    bars_df=b_df,
                    tech_df=tech_df,
                    full_corpus_df=fc_df,
                    scored_events_df=e_scored,
                    cfg=cfg,
                )
                verify_ticker_phase2_outputs(
                    ticker=t.ticker,
                    bars_df=b_df,
                    raw_events_df=e_raw,
                    scored_events_df=e_scored,
                    features_1h_df=f1h_df,
                    live_snapshot_row=snap_row,
                    smoke_mode=args.smoke_test,
                )
                feat_frames.append(f1h_df)
                live_rows.append(snap_row)

                scored_only = e_scored[e_scored["is_llm_scored"]]
                per_ticker_manifest_agg[t.ticker] = TickerPhase2Stats(
                    ticker=t.ticker,
                    exchange=t.exchange,
                    total_events=len(e_scored),
                    llm_routed_events=len(scored_only),
                    unique_deduped_texts_sent=0,
                    dedup_calls_saved=0,
                    llm_calls=0,
                    bisect_retries=0,
                    cache_hits=len(scored_only),
                    mean_bar_impact_weight_coverage_pct=compute_impact_weight_coverage_pct(e_scored, cfg),
                    verified_holder_llm_count=int((scored_only["routing_bucket"] == "naver_verified").sum()),
                    fresh_bucket_llm_count=int((scored_only["routing_bucket"] == "naver_fresh").sum()),
                    sarcasm_rate=(
                        round(float(scored_only["is_sarcastic"].fillna(False).mean()), 4)
                        if len(scored_only) > 0
                        else 0.0
                    ),
                    live_event_count=int(snap_row["live_event_count"]),
                    active_10d_bars=int((f1h_df["window_role"] == "active_10d").sum()),
                )

            features_all = pd.concat(feat_frames, ignore_index=True)
            live_all = pd.DataFrame(live_rows)
            if is_full_universe:
                assert len(features_all) == len(bars_all) == 910, (
                    f"Expected 910 features_1h rows, got {len(features_all)}"
                )
                assert len(live_all) == 10, f"Expected 10 live_snapshot rows, got {len(live_all)}"
            _write_parquet_strict(features_all, FEATURES_1H_SCHEMA, features_out_path)
            _write_parquet_strict(live_all, LIVE_SNAPSHOT_SCHEMA, live_out_path)
            agg_manifest = Phase2Manifest(
                run_started_utc=run_started_utc,
                run_completed_utc=datetime.now(timezone.utc).isoformat(),
                llm_provider=provider,
                llm_model=resolved_model,
                batch_size=cfg.smoke_test.batch_size if args.smoke_test else cfg.llm.batch_size,
                rpm_limit=cfg.llm.rpm_limit,
                total_events=len(scored_all),
                llm_routed_events=sum(s.llm_routed_events for s in per_ticker_manifest_agg.values()),
                unique_deduped_texts_sent=0,
                dedup_calls_saved=0,
                total_llm_calls=0,
                bisect_retries=0,
                cache_hits=sum(s.cache_hits for s in per_ticker_manifest_agg.values()),
                per_ticker_stats=per_ticker_manifest_agg,
                assertions_passed=True,
                mermaid_graph_path=str(MERMAID_GRAPH_PATH) if MERMAID_GRAPH_PATH.exists() else "",
            )
            _write_manifest(agg_manifest, manifest_out_path)
            print(
                json.dumps(
                    {
                        "stage": "aggregate",
                        "features_1h": str(features_out_path),
                        "live_snapshot": str(live_out_path),
                        "manifest": str(manifest_out_path),
                        "feature_rows": len(features_all),
                        "live_rows": len(live_all),
                        "assertions_passed": True,
                    },
                    indent=2,
                )
            )
            return 0

        # =====================================================================
        # Full End-to-End Stage ("all"): Per-Ticker LangGraph StateGraph Runs
        # =====================================================================
        # Check if any uncached events require remote LLM calls before requiring an API key
        smoke_max = cfg.smoke_test.max_llm_items_per_ticker if args.smoke_test else None
        needs_llm_client = bool(args.force_refresh)
        if not needs_llm_client:
            for t in tickers:
                _, dmap, rep_df = route_ticker_events(
                    events_by_ticker[t.ticker], cfg, smoke_max_items=smoke_max
                )
                for rep_eid in rep_df["event_id"].astype(str):
                    dups = dmap.get(rep_eid, [rep_eid])
                    if not any(checkpoint_store.get(e, expected_model=resolved_model) for e in dups):
                        needs_llm_client = True
                        break
                if needs_llm_client:
                    break

        client: CloudLLMClient | None = (
            CloudLLMClient(
                cfg=cfg,
                provider=provider,
                model_override=model_override,
                rate_limiter=shared_limiter,
            )
            if needs_llm_client
            else None
        )
        compiled_graph = build_ticker_phase2_graph(
            cfg=cfg,
            client=client,
            checkpoint_store=checkpoint_store,
            smoke_mode=args.smoke_test,
            force_refresh=args.force_refresh,
            expected_model=resolved_model,
        )
        mermaid_path = export_mermaid_diagram(compiled_graph, MERMAID_GRAPH_PATH)
        logger.info("[graph] Exported LangGraph Mermaid diagram to %s", mermaid_path)

        # Execute up to max_concurrency per-ticker graphs sharing the single 25 RPM rate limiter
        results_by_ticker: dict[str, dict[str, Any]] = {}
        max_workers = max(1, min(cfg.llm.max_concurrency, len(tickers)))

        def _run_one(t_cfg: TickerConfig) -> tuple[str, dict[str, Any]]:
            logger.info("[graph:%s] Starting per-ticker Phase 2 StateGraph...", t_cfg.ticker)
            init_state = {
                "ticker": t_cfg.ticker,
                "exchange": t_cfg.exchange,
                "company_en": t_cfg.company_en,
                "company_local": t_cfg.company_local,
                "bars_df": bars_by_ticker[t_cfg.ticker],
                "events_df": events_by_ticker[t_cfg.ticker],
            }
            final_state = compiled_graph.invoke(init_state)
            logger.info(
                "[graph:%s] Completed | scored=%d | llm_calls=%d | cache_hits=%d | coverage=%.1f%%",
                t_cfg.ticker,
                final_state["ticker_stats"]["llm_routed_events"],
                final_state["ticker_stats"]["llm_calls"],
                final_state["ticker_stats"]["cache_hits"],
                final_state["ticker_stats"]["mean_bar_impact_weight_coverage_pct"],
            )
            return t_cfg.ticker, final_state

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, t): t.ticker for t in tickers}
            for fut in as_completed(futures):
                tkr, state = fut.result()
                results_by_ticker[tkr] = state

        # Concatenate in deterministic universe ticker order
        ordered_scored: list[pd.DataFrame] = []
        ordered_features: list[pd.DataFrame] = []
        ordered_live: list[dict[str, Any]] = []
        per_ticker_manifest: dict[str, TickerPhase2Stats] = {}

        for t in tickers:
            st = results_by_ticker[t.ticker]
            ordered_scored.append(st["scored_events_df"])
            ordered_features.append(st["features_1h_df"])
            ordered_live.append(st["live_snapshot_row"])
            per_ticker_manifest[t.ticker] = TickerPhase2Stats.model_validate(st["ticker_stats"])

        events_scored_df = pd.concat(ordered_scored, ignore_index=True)
        features_1h_df = pd.concat(ordered_features, ignore_index=True)
        live_snapshot_df = pd.DataFrame(ordered_live)

        # Global cross-universe mandatory assertions (Section 2.6)
        if is_full_universe:
            assert len(events_scored_df) == len(events_all), (
                f"Mismatch between events_scored ({len(events_scored_df)}) and events_all ({len(events_all)})"
            )
            assert len(features_1h_df) == len(bars_all), (
                f"Mismatch between features_1h ({len(features_1h_df)}) and bars_all ({len(bars_all)})"
            )
            assert len(live_snapshot_df) == len(tickers), (
                f"Expected {len(tickers)} live_snapshot rows, got {len(live_snapshot_df)}"
            )
            scored_count = int(events_scored_df["is_llm_scored"].sum())
            assert scored_count > 0, f"Expected positive is_llm_scored rows, got {scored_count}"

        _write_parquet_strict(events_scored_df, EVENTS_SCORED_SCHEMA, events_out_path)
        _write_parquet_strict(features_1h_df, FEATURES_1H_SCHEMA, features_out_path)
        _write_parquet_strict(live_snapshot_df, LIVE_SNAPSHOT_SCHEMA, live_out_path)

        manifest = Phase2Manifest(
            run_started_utc=run_started_utc,
            run_completed_utc=datetime.now(timezone.utc).isoformat(),
            llm_provider=provider,
            llm_model=resolved_model,
            batch_size=cfg.smoke_test.batch_size if args.smoke_test else cfg.llm.batch_size,
            rpm_limit=cfg.llm.rpm_limit,
            total_events=len(events_scored_df),
            llm_routed_events=sum(s.llm_routed_events for s in per_ticker_manifest.values()),
            unique_deduped_texts_sent=sum(s.unique_deduped_texts_sent for s in per_ticker_manifest.values()),
            dedup_calls_saved=sum(s.dedup_calls_saved for s in per_ticker_manifest.values()),
            total_llm_calls=sum(s.llm_calls for s in per_ticker_manifest.values()),
            bisect_retries=sum(s.bisect_retries for s in per_ticker_manifest.values()),
            cache_hits=sum(s.cache_hits for s in per_ticker_manifest.values()),
            per_ticker_stats=per_ticker_manifest,
            assertions_passed=all(results_by_ticker[t.ticker].get("assertions_ok", False) for t in tickers),
            mermaid_graph_path=str(mermaid_path),
        )
        _write_manifest(manifest, manifest_out_path)

        logger.info(
            "Phase 2 COMPLETE | events_scored=%d | features_1h=%d | live_snapshot=%d | manifest=%s",
            len(events_scored_df),
            len(features_1h_df),
            len(live_snapshot_df),
            manifest_out_path,
        )
        print(json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False))

    except Exception as exc:
        logger.exception("Phase 2 pipeline failed loudly (strict fail-fast): %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
