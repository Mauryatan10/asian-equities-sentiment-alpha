"""CulturalNLPAgent: Single-ticker batched Cloud-LLM scoring with append-only JSONL checkpointing,
auto-bisect missing-ID / malformed-JSON recovery, price-target sanity bounding, and zero-collapse impact weights."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.llm_client import CloudLLMClient
from src.features.prompts import PromptDomain, build_system_prompt, build_user_prompt
from src.features.schemas import (
    LLM_CACHE_PATH,
    Phase2Config,
    ScoredEventItem,
)

logger = logging.getLogger(__name__)


class LLMScoreCheckpointStore:
    """Thread-safe append-only JSONL checkpoint store keyed by event_id."""

    def __init__(self, path: Path = LLM_CACHE_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Latest record per (event_id, sentiment_source) and fallback latest per event_id
        self._by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._scores: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    eid = str(rec["event_id"])
                    src_model = str(rec.get("sentiment_source") or "")
                    self._scores[eid] = rec
                    if src_model:
                        self._by_key[(eid, src_model)] = rec
                except Exception as exc:
                    logger.warning("Skipping corrupt checkpoint line %d in %s: %s", line_no, self.path, exc)

    def get(self, event_id: str, expected_model: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if expected_model:
                return self._by_key.get((event_id, expected_model))
            return self._scores.get(event_id)

    def append_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    eid = str(rec["event_id"])
                    src_model = str(rec.get("sentiment_source") or "")
                    self._scores[eid] = rec
                    if src_model:
                        self._by_key[(eid, src_model)] = rec
                fh.flush()

    def size(self) -> int:
        with self._lock:
            return len(self._scores)


def _extract_valid_items(
    parsed: dict[str, Any],
    wire_to_real: dict[str, str],
) -> dict[str, ScoredEventItem]:
    """Validate and extract ScoredEventItems, mapping compact wire IDs (E0..E24) back to canonical event_ids."""
    raw_items = parsed.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"LLM response JSON missing 'items' list (keys={list(parsed.keys())})")

    valid_real_ids = set(wire_to_real.values())
    found: dict[str, ScoredEventItem] = {}
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        raw_eid = str(entry.get("event_id") or "").strip()
        real_eid = wire_to_real.get(raw_eid)
        if real_eid is None and raw_eid in valid_real_ids:
            real_eid = raw_eid
        if real_eid is None or real_eid in found:
            continue
        try:
            normalized_entry = dict(entry)
            normalized_entry["event_id"] = real_eid
            item = ScoredEventItem.model_validate(normalized_entry)
            found[real_eid] = item
        except Exception as exc:
            logger.debug("Item validation failed for event_id=%s: %s", real_eid, exc)
    return found


def score_batch_with_bisect(
    client: CloudLLMClient,
    batch_rows: list[dict[str, Any]],
    domain: PromptDomain,
    ticker: str,
    company_en: str,
    company_local: str,
    cfg: Phase2Config,
    depth: int = 0,
    singleton_retries: int = 0,
) -> tuple[dict[str, ScoredEventItem], int, int]:
    """Score a single-ticker batch of items with automatic bisection on malformed JSON or omitted IDs.

    Uses compact wire IDs (E0..E{k-1}) on the LLM wire to avoid 64-char SHA-256 copy typos
    and save ~1,500 tokens/batch, then maps deterministically back to canonical event_ids.
    Tracks singleton_retries separately from bisection depth so any isolated 1-item failure
    always receives up to bisect_max_retries dedicated retry attempts.

    Returns (scored_map, llm_calls_made, bisect_retries_used).
    """
    if not batch_rows:
        return {}, 0, 0

    expected_ids = {str(r["event_id"]) for r in batch_rows}
    wire_to_real: dict[str, str] = {}
    wire_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(batch_rows):
        wire_id = f"E{idx}"
        wire_to_real[wire_id] = str(row["event_id"])
        wrow = dict(row)
        wrow["event_id"] = wire_id
        wire_rows.append(wrow)

    sys_prompt = build_system_prompt(domain, ticker, company_en, company_local, cfg)
    usr_prompt = build_user_prompt(wire_rows, domain, ticker, company_en, company_local, cfg)

    calls = 0
    bisect_retries = 0
    found: dict[str, ScoredEventItem] = {}

    try:
        calls += 1
        parsed, _ = client.chat_json(sys_prompt, usr_prompt)
        found = _extract_valid_items(parsed, wire_to_real)
    except ValueError as exc:
        logger.warning(
            "[cultural_nlp:%s/%s] Malformed JSON on batch size=%d (depth=%d, singleton_retries=%d): %s",
            ticker,
            domain,
            len(batch_rows),
            depth,
            singleton_retries,
            exc,
        )
        found = {}

    missing_rows = [r for r in batch_rows if str(r["event_id"]) not in found]
    if not missing_rows:
        return found, calls, bisect_retries

    max_bisect = cfg.llm.bisect_max_retries

    # Dedicated singleton retry budget (independent of prior halving depth)
    if len(missing_rows) == 1:
        next_singleton_try = singleton_retries + 1 if len(batch_rows) == 1 else 1
        if len(batch_rows) == 1 and singleton_retries >= max_bisect:
            raise RuntimeError(
                f"[cultural_nlp:{ticker}/{domain}] Unresolved singleton event_id={missing_rows[0]['event_id']} "
                f"after {singleton_retries} singleton retries (bisect depth={depth})."
            )
        bisect_retries += 1
        sub_found, sub_calls, sub_bisects = score_batch_with_bisect(
            client=client,
            batch_rows=missing_rows,
            domain=domain,
            ticker=ticker,
            company_en=company_en,
            company_local=company_local,
            cfg=cfg,
            depth=depth,
            singleton_retries=next_singleton_try,
        )
        found.update(sub_found)
        return found, calls + sub_calls, bisect_retries + sub_bisects

    # Bisect multi-item missing subset into two halves
    if depth >= max_bisect + 4:
        missing_ids = [str(r["event_id"]) for r in missing_rows]
        raise RuntimeError(
            f"[cultural_nlp:{ticker}/{domain}] Unresolved {len(missing_ids)} missing event_ids after bisection: "
            f"{missing_ids[:5]}"
        )

    mid = len(missing_rows) // 2
    left_rows = missing_rows[:mid]
    right_rows = missing_rows[mid:]
    logger.warning(
        "[cultural_nlp:%s/%s] Missing %d/%d IDs — bisecting into %d + %d (depth=%d)",
        ticker,
        domain,
        len(missing_rows),
        len(batch_rows),
        len(left_rows),
        len(right_rows),
        depth + 1,
    )
    bisect_retries += 1

    left_found, l_calls, l_bisects = score_batch_with_bisect(
        client=client,
        batch_rows=left_rows,
        domain=domain,
        ticker=ticker,
        company_en=company_en,
        company_local=company_local,
        cfg=cfg,
        depth=depth + 1,
        singleton_retries=0,
    )
    right_found, r_calls, r_bisects = score_batch_with_bisect(
        client=client,
        batch_rows=right_rows,
        domain=domain,
        ticker=ticker,
        company_en=company_en,
        company_local=company_local,
        cfg=cfg,
        depth=depth + 1,
        singleton_retries=0,
    )
    found.update(left_found)
    found.update(right_found)

    still_missing = expected_ids - set(found.keys())
    if still_missing:
        raise RuntimeError(
            f"[cultural_nlp:{ticker}/{domain}] Still missing {len(still_missing)} event_ids after bisection: "
            f"{sorted(still_missing)[:5]}"
        )
    return found, calls + l_calls + r_calls, bisect_retries + l_bisects + r_bisects


def compute_impact_weights(df: pd.DataFrame, cfg: Phase2Config) -> np.ndarray:
    """Zero-collapse per-event impact weight w_k (Section 2.5):
    - Social: log(1 + effective_reach) * (1 + log(1 + likes + dislikes + reposts + comments))
              * (1 + 0.5 * I(is_holder_verified)) * (0.5 if repost else 1.0) * confidence
    - News:   source_credibility_weight * confidence
    - Unscored rows: NaN
    """
    af = cfg.alpha_factors
    n = len(df)
    weights = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return weights

    scored_mask = df["is_llm_scored"].to_numpy(dtype=bool)
    if not scored_mask.any():
        return weights

    is_news = (df["event_type"].astype(str) == "news").to_numpy()
    conf = df["confidence"].fillna(0.0).to_numpy(dtype=np.float64)

    # News weights
    news_scored = scored_mask & is_news
    if news_scored.any():
        cred = df["source_credibility_weight"].fillna(1.0).to_numpy(dtype=np.float64)
        weights[news_scored] = cred[news_scored] * conf[news_scored]

    # Social weights
    soc_scored = scored_mask & (~is_news)
    if soc_scored.any():
        reach = np.maximum(df["effective_reach"].fillna(1).to_numpy(dtype=np.float64), 0.0)
        likes = np.maximum(df["likes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        dislikes = np.maximum(df["dislikes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        reposts = np.maximum(df["reposts"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        comments = np.maximum(df["comments"].fillna(0).to_numpy(dtype=np.float64), 0.0)
        eng = likes + dislikes + reposts + comments

        verified = (df["is_holder_verified"].fillna(False).astype(bool)).to_numpy(dtype=np.float64)
        is_repost = (df["is_cross_channel_repost"].fillna(False).astype(bool)).to_numpy()
        repost_mult = np.where(is_repost, af.repost_weight_factor, 1.0)

        w_soc = (
            np.log1p(reach)
            * (1.0 + np.log1p(eng))
            * (1.0 + af.holder_verified_weight_boost * verified)
            * repost_mult
            * conf
        )
        weights[soc_scored] = w_soc[soc_scored]

    return weights


def run_cultural_nlp_for_ticker(
    ticker: str,
    company_en: str,
    company_local: str,
    bars_df: pd.DataFrame,
    routed_df: pd.DataFrame,
    dedup_map: dict[str, list[str]],
    rep_df: pd.DataFrame,
    cfg: Phase2Config,
    client: CloudLLMClient | None,
    checkpoint_store: LLMScoreCheckpointStore,
    batch_size_override: int | None = None,
    force_refresh: bool = False,
    expected_model: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score all representative routed events for one ticker, fan back across duplicates,
    bound price targets against causal ref_close, and compute zero-collapse impact weights."""
    batch_size = batch_size_override or cfg.llm.batch_size
    model_label = (
        expected_model
        or (client.model if client is not None else cfg.llm.providers[cfg.llm.default_provider].cultural_nlp_model)
    )

    scored_by_eid: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    llm_calls = 0
    bisect_retries = 0
    unique_texts_sent = 0

    # Check cache per representative (or any duplicate sharing its exact text_dedup_key) matched to model_label
    pending_reps: list[dict[str, Any]] = []
    for _, row in rep_df.iterrows():
        rep_eid = str(row["event_id"])
        dup_eids = dedup_map.get(rep_eid, [rep_eid])
        cached: dict[str, Any] | None = None
        if not force_refresh:
            for cand_eid in dup_eids:
                hit = checkpoint_store.get(cand_eid, expected_model=model_label)
                if hit is not None:
                    cached = hit
                    break
        if cached is not None:
            cache_hits += 1
            for dup_eid in dup_eids:
                rec = dict(cached)
                rec["event_id"] = dup_eid
                scored_by_eid[dup_eid] = rec
        else:
            pending_reps.append(row.to_dict())

    if pending_reps:
        if client is None:
            raise RuntimeError(
                f"[cultural_nlp:{ticker}] {len(pending_reps)} uncached events require an LLM client, "
                "but client is None."
            )
        # Group strictly by prompt_domain so every batch has 1 exact (ticker, prompt_domain)
        by_domain: dict[PromptDomain, list[dict[str, Any]]] = {}
        for r in pending_reps:
            dom = str(r["prompt_domain"])  # type: ignore[assignment]
            by_domain.setdefault(dom, []).append(r)  # type: ignore[arg-type]

        # Build all single-ticker, single-domain batches and execute concurrently
        # (global concurrency and 25 RPM cap are enforced inside CloudLLMClient)
        batch_jobs: list[tuple[PromptDomain, int, int, list[dict[str, Any]]]] = []
        for domain, domain_rows in by_domain.items():
            total_batches = (len(domain_rows) + batch_size - 1) // batch_size
            for b_idx in range(total_batches):
                start = b_idx * batch_size
                batch_rows = domain_rows[start : start + batch_size]
                batch_jobs.append((domain, b_idx + 1, total_batches, batch_rows))

        def _exec_batch(
            job: tuple[PromptDomain, int, int, list[dict[str, Any]]],
        ) -> tuple[PromptDomain, int, int, int, dict[str, ScoredEventItem], int, int]:
            dom, b_num, tot_b, b_rows = job
            b_scored, c_made, b_used = score_batch_with_bisect(
                client=client,
                batch_rows=b_rows,
                domain=dom,
                ticker=ticker,
                company_en=company_en,
                company_local=company_local,
                cfg=cfg,
            )
            return dom, b_num, tot_b, len(b_rows), b_scored, c_made, b_used

        batch_workers = max(1, min(cfg.llm.max_concurrency, len(batch_jobs)))
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            futs = [pool.submit(_exec_batch, job) for job in batch_jobs]
            for fut in as_completed(futs):
                domain, b_num, total_batches, rep_cnt, batch_scored, calls_made, bisects_used = fut.result()
                unique_texts_sent += rep_cnt
                llm_calls += calls_made
                bisect_retries += bisects_used

                now_iso = datetime.now(timezone.utc).isoformat()
                to_persist: list[dict[str, Any]] = []
                for rep_eid, item in batch_scored.items():
                    base_rec = item.model_dump()
                    base_rec["sentiment_source"] = model_label
                    base_rec["scored_at_utc"] = now_iso
                    for dup_eid in dedup_map.get(rep_eid, [rep_eid]):
                        rec = dict(base_rec)
                        rec["event_id"] = dup_eid
                        scored_by_eid[dup_eid] = rec
                        to_persist.append(rec)

                checkpoint_store.append_batch(to_persist)
                logger.info(
                    "[cultural_nlp:%s/%s] Batch %d/%d done (%d reps -> %d events, calls=%d, bisects=%d)",
                    ticker,
                    domain,
                    b_num,
                    total_batches,
                    rep_cnt,
                    len(to_persist),
                    calls_made,
                    bisects_used,
                )

    # Map scored fields onto routed_df (exact 1:1 row order)
    out = routed_df.copy()

    # Build bar_close_utc -> close lookup for causal price target sanity bounding
    bars_sorted = bars_df.sort_values("bar_close_utc")
    close_by_bar: dict[pd.Timestamp, float] = {
        pd.Timestamp(ts): float(c)
        for ts, c in zip(bars_sorted["bar_close_utc"], bars_sorted["close"])
    }
    latest_close = float(bars_sorted["close"].iloc[-1])
    pt_min_ratio = cfg.alpha_factors.price_target_min_ratio
    pt_max_ratio = cfg.alpha_factors.price_target_max_ratio

    n = len(out)
    sent_source_arr: list[str | None] = [None] * n
    sent_score_arr = np.full(n, np.nan, dtype=np.float64)
    direction_arr: list[str | None] = [None] * n
    confidence_arr = np.full(n, np.nan, dtype=np.float64)
    sarcastic_arr: list[bool | None] = [None] * n
    hype_arr = np.full(n, np.nan, dtype=np.float64)
    catalyst_arr: list[str | None] = [None] * n
    pt_arr = np.full(n, np.nan, dtype=np.float64)
    summary_arr: list[str | None] = [None] * n

    eids = out["event_id"].astype(str).to_numpy()
    is_scored_flags = out["is_llm_scored"].to_numpy(dtype=bool)
    assigned_bars = out["assigned_bar_close_utc"].tolist()

    for i in range(n):
        if not is_scored_flags[i]:
            continue
        eid = eids[i]
        rec = scored_by_eid.get(eid)
        if rec is None:
            raise RuntimeError(
                f"[cultural_nlp:{ticker}] Missing LLM score for routed event_id={eid} (strict fail-fast)."
            )
        sent_source_arr[i] = str(rec.get("sentiment_source") or model_label)
        sent_score_arr[i] = float(rec["sentiment_score"])
        direction_arr[i] = str(rec["direction"])
        confidence_arr[i] = float(rec["confidence"])
        sarcastic_arr[i] = bool(rec["is_sarcastic"])
        hype_arr[i] = float(rec["hype_score"])
        catalyst_arr[i] = str(rec["catalyst_type"])
        summary_arr[i] = str(rec["english_summary"])

        raw_pt = rec.get("price_target_mentioned")
        if raw_pt is not None:
            try:
                pt_val = float(raw_pt)
                bar_ts = assigned_bars[i]
                ref_close = (
                    close_by_bar.get(pd.Timestamp(bar_ts), latest_close)
                    if pd.notna(bar_ts)
                    else latest_close
                )
                if (pt_min_ratio * ref_close) <= pt_val <= (pt_max_ratio * ref_close):
                    pt_arr[i] = pt_val
            except (TypeError, ValueError):
                pass

    out["sentiment_source"] = pd.Series(sent_source_arr, index=out.index, dtype="string")
    out["sentiment_score"] = sent_score_arr
    out["direction"] = pd.Series(direction_arr, index=out.index, dtype="string")
    out["confidence"] = confidence_arr
    out["is_sarcastic"] = pd.Series(sarcastic_arr, index=out.index, dtype="boolean")
    out["hype_score"] = hype_arr
    out["catalyst_type"] = pd.Series(catalyst_arr, index=out.index, dtype="string")
    out["price_target_mentioned"] = pt_arr
    out["english_summary"] = pd.Series(summary_arr, index=out.index, dtype="string")
    out["impact_weight"] = compute_impact_weights(out, cfg)

    routed_total = int(is_scored_flags.sum())
    unique_total = len(rep_df)
    llm_stats = {
        "ticker": ticker,
        "llm_routed_events": routed_total,
        "unique_reps": unique_total,
        "unique_deduped_texts_sent": unique_texts_sent,
        "dedup_calls_saved": max(0, routed_total - unique_total),
        "llm_calls": llm_calls,
        "bisect_retries": bisect_retries,
        "cache_hits": cache_hits,
    }
    return out, llm_stats
