"""Per-ticker LangGraph StateGraph wiring and Mermaid diagram export (Section 2.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.features.alpha_factors import (
    compute_alpha_factors_and_snapshot,
    compute_full_corpus_stats,
    compute_impact_weight_coverage_pct,
    verify_ticker_phase2_outputs,
)
from src.features.cultural_nlp_agent import LLMScoreCheckpointStore, run_cultural_nlp_for_ticker
from src.features.llm_client import CloudLLMClient
from src.features.quant_technical_agent import compute_quant_technical_features
from src.features.router import route_ticker_events
from src.features.schemas import (
    MERMAID_GRAPH_PATH,
    Phase2Config,
    TickerPhase2State,
)


def build_ticker_phase2_graph(
    cfg: Phase2Config,
    client: CloudLLMClient | None,
    checkpoint_store: LLMScoreCheckpointStore,
    smoke_mode: bool = False,
    force_refresh: bool = False,
    expected_model: str | None = None,
) -> Any:
    """Compile a per-ticker LangGraph StateGraph implementing the exact Section 2.2 topology:
      START -> router -> cultural_nlp -> alpha_factors -> verify -> END
      START -> quant_technical        -> alpha_factors
      START -> full_corpus_stats      -> alpha_factors
    """
    smoke_max = cfg.smoke_test.max_llm_items_per_ticker if smoke_mode else None
    batch_override = cfg.smoke_test.batch_size if smoke_mode else None

    def router_node(state: TickerPhase2State) -> dict[str, Any]:
        routed_df, dedup_map, rep_df = route_ticker_events(
            events_df=state["events_df"],
            cfg=cfg,
            smoke_max_items=smoke_max,
        )
        return {
            "routed_events_df": routed_df,
            "dedup_map": dedup_map,
            "representative_events_df": rep_df,
        }

    def cultural_nlp_node(state: TickerPhase2State) -> dict[str, Any]:
        scored_df, llm_stats = run_cultural_nlp_for_ticker(
            ticker=state["ticker"],
            company_en=state["company_en"],
            company_local=state["company_local"],
            bars_df=state["bars_df"],
            routed_df=state["routed_events_df"],
            dedup_map=state["dedup_map"],
            rep_df=state["representative_events_df"],
            cfg=cfg,
            client=client,
            checkpoint_store=checkpoint_store,
            batch_size_override=batch_override,
            force_refresh=force_refresh,
            expected_model=expected_model,
        )
        return {
            "scored_events_df": scored_df,
            "llm_stats": llm_stats,
        }

    def quant_technical_node(state: TickerPhase2State) -> dict[str, Any]:
        tech_df = compute_quant_technical_features(
            bars_df=state["bars_df"],
            cfg=cfg,
        )
        return {"tech_features_df": tech_df}

    def full_corpus_stats_node(state: TickerPhase2State) -> dict[str, Any]:
        fc_df = compute_full_corpus_stats(
            bars_df=state["bars_df"],
            events_df=state["events_df"],
            cfg=cfg,
        )
        return {"full_corpus_stats_df": fc_df}

    def alpha_factors_node(state: TickerPhase2State) -> dict[str, Any]:
        features_1h_df, live_snapshot_row = compute_alpha_factors_and_snapshot(
            ticker=state["ticker"],
            exchange=state["exchange"],
            company_en=state["company_en"],
            company_local=state["company_local"],
            bars_df=state["bars_df"],
            tech_df=state["tech_features_df"],
            full_corpus_df=state["full_corpus_stats_df"],
            scored_events_df=state["scored_events_df"],
            cfg=cfg,
        )

        scored_df = state["scored_events_df"]
        scored_only = scored_df[scored_df["is_llm_scored"]]
        llm_stats = state.get("llm_stats", {})
        verified_llm_cnt = int((scored_only["routing_bucket"] == "naver_verified").sum())
        fresh_llm_cnt = int((scored_only["routing_bucket"] == "naver_fresh").sum())
        sarcasm_rate = (
            round(float(scored_only["is_sarcastic"].fillna(False).mean()), 4)
            if len(scored_only) > 0
            else 0.0
        )
        coverage_pct = compute_impact_weight_coverage_pct(scored_df, cfg)

        ticker_stats = {
            "ticker": state["ticker"],
            "exchange": state["exchange"],
            "total_events": len(scored_df),
            "llm_routed_events": int(llm_stats.get("llm_routed_events", len(scored_only))),
            "unique_deduped_texts_sent": int(llm_stats.get("unique_deduped_texts_sent", 0)),
            "dedup_calls_saved": int(llm_stats.get("dedup_calls_saved", 0)),
            "llm_calls": int(llm_stats.get("llm_calls", 0)),
            "bisect_retries": int(llm_stats.get("bisect_retries", 0)),
            "cache_hits": int(llm_stats.get("cache_hits", 0)),
            "mean_bar_impact_weight_coverage_pct": coverage_pct,
            "verified_holder_llm_count": verified_llm_cnt,
            "fresh_bucket_llm_count": fresh_llm_cnt,
            "sarcasm_rate": sarcasm_rate,
            "live_event_count": int(live_snapshot_row["live_event_count"]),
            "active_10d_bars": int((features_1h_df["window_role"] == "active_10d").sum()),
        }

        return {
            "features_1h_df": features_1h_df,
            "live_snapshot_row": live_snapshot_row,
            "ticker_stats": ticker_stats,
        }

    def verify_node(state: TickerPhase2State) -> dict[str, Any]:
        ok = verify_ticker_phase2_outputs(
            ticker=state["ticker"],
            bars_df=state["bars_df"],
            raw_events_df=state["events_df"],
            scored_events_df=state["scored_events_df"],
            features_1h_df=state["features_1h_df"],
            live_snapshot_row=state["live_snapshot_row"],
            smoke_mode=smoke_mode,
        )
        return {"assertions_ok": ok}

    builder = StateGraph(TickerPhase2State)
    builder.add_node("router", router_node)
    builder.add_node("cultural_nlp", cultural_nlp_node)
    builder.add_node("quant_technical", quant_technical_node)
    builder.add_node("full_corpus_stats", full_corpus_stats_node)
    builder.add_node("alpha_factors", alpha_factors_node)
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "router")
    builder.add_edge(START, "quant_technical")
    builder.add_edge(START, "full_corpus_stats")
    builder.add_edge("router", "cultural_nlp")
    builder.add_edge(["cultural_nlp", "quant_technical", "full_corpus_stats"], "alpha_factors")
    builder.add_edge("alpha_factors", "verify")
    builder.add_edge("verify", END)

    return builder.compile()


def export_mermaid_diagram(compiled_graph: Any, out_path: Path = MERMAID_GRAPH_PATH) -> Path:
    """Export compiled LangGraph StateGraph topology as Mermaid markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_src = compiled_graph.get_graph().draw_mermaid()
    out_path.write_text(mermaid_src, encoding="utf-8")
    return out_path
