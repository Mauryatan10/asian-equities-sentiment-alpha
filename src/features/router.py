"""Deterministic 3-bucket per-bar Naver router and pre-LLM identical-text deduplicator."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src.features.prompts import resolve_prompt_domain
from src.features.schemas import Phase2Config, RouterConfig

_WS_RE = re.compile(r"\s+")


def normalize_text_for_dedup(title: str, text: str) -> str:
    combined = f"{title or ''}\n{text or ''}"
    return _WS_RE.sub(" ", combined).strip().lower()


def compute_text_dedup_key(ticker: str, prompt_domain: str, title: str, text: str) -> str:
    norm = normalize_text_for_dedup(title, text)
    raw = f"{ticker}|{prompt_domain}|{norm}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_naver_quota(group_size: int, router_cfg: RouterConfig) -> int:
    """Rich-sqrt dynamic quota: N(M) = min(M, clip(round(2.0 * sqrt(M)), 6, 50))."""
    if group_size <= 0:
        return 0
    raw = int(round(router_cfg.sqrt_multiplier * math.sqrt(group_size)))
    clipped = max(router_cfg.min_quota, min(router_cfg.max_quota, raw))
    return min(group_size, clipped)


def compute_naver_priority_scores(df: pd.DataFrame, router_cfg: RouterConfig) -> np.ndarray:
    """Vectorized Naver priority score:
    priority_k = log(1 + views) + 2.0 * log(1 + likes + dislikes)
                 + 1.5 * I(is_holder_verified) + 0.5 * I(char_len >= 30)
    """
    pw = router_cfg.priority_weights
    views = np.maximum(df["views"].fillna(0).to_numpy(dtype=np.float64), 0.0)
    likes = np.maximum(df["likes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
    dislikes = np.maximum(df["dislikes"].fillna(0).to_numpy(dtype=np.float64), 0.0)
    verified = (df["is_holder_verified"].fillna(False).astype(bool)).to_numpy(dtype=np.float64)

    titles = df["title"].fillna("").astype(str)
    texts = df["text"].fillna("").astype(str)
    char_lens = (titles.str.strip() + " " + texts.str.strip()).str.strip().str.len().to_numpy(dtype=np.int64)
    long_flag = (char_lens >= pw.min_char_len_threshold).astype(np.float64)

    return (
        pw.views_log * np.log1p(views)
        + pw.engagement_log * np.log1p(likes + dislikes)
        + pw.holder_verified_bonus * verified
        + pw.min_char_len_bonus * long_flag
    )


def _deterministic_sort_indices(sub: pd.DataFrame) -> np.ndarray:
    """Return positional indices of `sub` sorted by:
    priority_score DESC, effective_event_utc DESC, event_id ASC.
    """
    # Resolution-safe UTC epoch seconds (works across pandas 2.x/3.x us/ns dtypes)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_us = (
        (pd.to_datetime(sub["effective_event_utc"], utc=True) - epoch)
        .dt.total_seconds()
        .to_numpy(dtype=np.float64)
    )
    prio = sub["priority_score"].to_numpy(dtype=np.float64)
    eids = sub["event_id"].astype(str).to_numpy()

    order = sorted(
        range(len(sub)),
        key=lambda i: (-prio[i], -ts_us[i], eids[i]),
    )
    return np.array(order, dtype=np.int64)


def route_ticker_events(
    events_df: pd.DataFrame,
    cfg: Phase2Config,
    smoke_max_items: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """Route a single ticker's events using the deterministic 3-bucket Naver strategy
    and compute pre-LLM identical-text deduplication.

    Zero Look-Ahead Guarantee:
      Naver routing operates STRICTLY inside each (ticker, assigned_bar_close_utc) bar,
      treating post_window_live (assigned_bar_close_utc is NaT) as one separate live group.

    Returns:
      - routed_df: copy of events_df with `priority_score`, `routing_bucket`, `is_llm_scored`,
        `prompt_domain`, and `text_dedup_key` attached (exact 1:1 row order preserved).
      - dedup_map: mapping rep_event_id -> list of all routed event_ids sharing the same
        (ticker, prompt_domain, normalized_text) key.
      - rep_df: DataFrame of unique representative rows to send to CulturalNLPAgent.
    """
    router_cfg = cfg.router
    df = events_df.copy()

    if df.empty:
        df["priority_score"] = pd.Series(dtype="float64")
        df["routing_bucket"] = pd.Series(dtype="string")
        df["is_llm_scored"] = pd.Series(dtype="bool")
        df["prompt_domain"] = pd.Series(dtype="string")
        df["text_dedup_key"] = pd.Series(dtype="string")
        return df, {}, df.iloc[0:0].copy()

    n_rows = len(df)
    priority_arr = np.full(n_rows, np.nan, dtype=np.float64)
    bucket_arr = np.full(n_rows, "non_naver", dtype=object)
    scored_arr = np.ones(n_rows, dtype=bool)

    naver_mask = (df["source"].astype(str) == "naver_finance").to_numpy()
    if naver_mask.any():
        naver_positions = np.flatnonzero(naver_mask)
        naver_sub = df.iloc[naver_positions].copy()
        naver_prios = compute_naver_priority_scores(naver_sub, router_cfg)
        priority_arr[naver_positions] = naver_prios
        naver_sub["priority_score"] = naver_prios
        naver_sub["_orig_pos"] = naver_positions

        # Default all Naver posts to tail unscored before selecting per-bar quotas
        bucket_arr[naver_positions] = "naver_tail_unscored"
        scored_arr[naver_positions] = False

        # Group strictly by assigned_bar_close_utc (NaT = post_window_live group)
        bar_keys = naver_sub["assigned_bar_close_utc"]
        is_live_mask = bar_keys.isna()

        group_slices: list[tuple[bool, pd.DataFrame]] = []
        # Historical bars
        hist_naver = naver_sub[~is_live_mask]
        if not hist_naver.empty:
            for _, grp in hist_naver.groupby("assigned_bar_close_utc", sort=True):
                group_slices.append((False, grp))
        # Live post-window group
        live_naver = naver_sub[is_live_mask]
        if not live_naver.empty:
            group_slices.append((True, live_naver))

        for is_live_group, grp in group_slices:
            m = len(grp)
            quota = compute_naver_quota(m, router_cfg)
            if quota <= 0:
                continue

            sorted_idx = _deterministic_sort_indices(grp)
            grp_sorted = grp.iloc[sorted_idx]

            orig_pos_sorted = grp_sorted["_orig_pos"].to_numpy(dtype=np.int64)
            is_verified_sorted = (
                grp_sorted["is_holder_verified"].fillna(False).astype(bool).to_numpy()
            )

            if is_live_group:
                # Latest 60 min of post_window_live
                ev_utc = pd.to_datetime(grp_sorted["effective_event_utc"], utc=True)
                max_live_utc = ev_utc.max()
                lag_min = (max_live_utc - ev_utc).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
                is_fresh_sorted = lag_min <= router_cfg.fresh_window_minutes
            else:
                lag_min = grp_sorted["lag_minutes_to_bar_close"].fillna(np.inf).to_numpy(dtype=np.float64)
                is_fresh_sorted = lag_min <= router_cfg.fresh_window_minutes

            selected_local = np.zeros(m, dtype=bool)

            # Bucket 1: Verified Shareholders (up to ceil(0.30 * N))
            b1_cap = min(math.ceil(router_cfg.verified_reserve_ratio * quota), quota)
            b1_candidates = np.flatnonzero(is_verified_sorted)
            b1_pick = b1_candidates[:b1_cap]
            selected_local[b1_pick] = True
            bucket_arr[orig_pos_sorted[b1_pick]] = "naver_verified"
            scored_arr[orig_pos_sorted[b1_pick]] = True

            # Bucket 2: Fresh Recency (up to ceil(0.15 * N) from remaining unselected)
            remaining_after_b1 = quota - len(b1_pick)
            b2_cap = min(math.ceil(router_cfg.fresh_reserve_ratio * quota), remaining_after_b1)
            b2_candidates = np.flatnonzero((~selected_local) & is_fresh_sorted)
            b2_pick = b2_candidates[:b2_cap]
            selected_local[b2_pick] = True
            bucket_arr[orig_pos_sorted[b2_pick]] = "naver_fresh"
            scored_arr[orig_pos_sorted[b2_pick]] = True

            # Bucket 3: Overall Priority Fill (remainder)
            b3_cap = quota - len(b1_pick) - len(b2_pick)
            if b3_cap > 0:
                b3_candidates = np.flatnonzero(~selected_local)
                b3_pick = b3_candidates[:b3_cap]
                selected_local[b3_pick] = True
                bucket_arr[orig_pos_sorted[b3_pick]] = "naver_priority"
                scored_arr[orig_pos_sorted[b3_pick]] = True

    df["priority_score"] = priority_arr
    df["routing_bucket"] = pd.Series(bucket_arr, index=df.index, dtype="string")
    df["is_llm_scored"] = scored_arr

    # Optional smoke-test cap (keep balanced sample across news + social while capping total LLM items)
    if smoke_max_items is not None and smoke_max_items > 0:
        routed_positions = np.flatnonzero(df["is_llm_scored"].to_numpy())
        if len(routed_positions) > smoke_max_items:
            routed_sub = df.iloc[routed_positions]
            # Pick a deterministic mix covering news and social
            news_pos = routed_positions[routed_sub["event_type"].to_numpy() == "news"]
            social_pos = routed_positions[routed_sub["event_type"].to_numpy() != "news"]
            n_news_keep = min(len(news_pos), max(2, smoke_max_items // 3))
            n_soc_keep = min(len(social_pos), smoke_max_items - n_news_keep)
            # Pick latest events so active_10d & live are represented
            keep_pos = np.concatenate([news_pos[-n_news_keep:], social_pos[-n_soc_keep:]])
            keep_set = set(keep_pos.tolist())
            new_scored = np.zeros(n_rows, dtype=bool)
            for p in keep_set:
                new_scored[p] = True
            # Mark demoted Naver rows as naver_tail_unscored and demoted non-Naver rows as smoke_unscored
            demoted_naver = naver_mask & (~new_scored)
            demoted_non_naver = (~naver_mask) & (~new_scored)
            bucket_arr[demoted_naver] = "naver_tail_unscored"
            bucket_arr[demoted_non_naver] = "smoke_unscored"
            df["routing_bucket"] = pd.Series(bucket_arr, index=df.index, dtype="string")
            df["is_llm_scored"] = new_scored

    # Assign prompt_domain & text_dedup_key
    domains = [
        resolve_prompt_domain(et, ex)
        for et, ex in zip(df["event_type"].astype(str), df["exchange"].astype(str))
    ]
    df["prompt_domain"] = pd.Series(domains, index=df.index, dtype="string")

    dedup_keys = [
        compute_text_dedup_key(tkr, dom, ttl, txt)
        for tkr, dom, ttl, txt in zip(
            df["ticker"].astype(str),
            domains,
            df["title"].fillna("").astype(str),
            df["text"].fillna("").astype(str),
        )
    ]
    df["text_dedup_key"] = pd.Series(dedup_keys, index=df.index, dtype="string")

    # Build dedup_map ONLY for routed (is_llm_scored == True) rows
    routed_only = df[df["is_llm_scored"]].copy()
    dedup_map: dict[str, list[str]] = {}
    rep_indices: list[Any] = []

    if not routed_only.empty:
        # Sort deterministically by (prompt_domain, event_id) before picking 1 representative per key
        routed_only_sorted = routed_only.sort_values(
            by=["prompt_domain", "effective_event_utc", "event_id"],
            ascending=[True, True, True],
        )
        key_to_rep: dict[str, str] = {}
        for idx, row in routed_only_sorted.iterrows():
            dkey = str(row["text_dedup_key"])
            eid = str(row["event_id"])
            if dkey not in key_to_rep:
                key_to_rep[dkey] = eid
                dedup_map[eid] = [eid]
                rep_indices.append(idx)
            else:
                rep_eid = key_to_rep[dkey]
                dedup_map[rep_eid].append(eid)

    rep_df = df.loc[rep_indices].copy() if rep_indices else df.iloc[0:0].copy()
    return df, dedup_map, rep_df
