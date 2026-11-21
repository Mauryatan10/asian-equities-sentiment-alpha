"""Phase 2 Pre-Flight Verification Gate:
1. Authenticated GET /v1/models check verifying configured model IDs exist (never EOL 410).
2. 1-item Cantonese (HKEX) + 1-item Korean (KRX) JSON-mode sarcasm sanity test."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.features.llm_client import CloudLLMClient
from src.features.prompts import build_system_prompt, build_user_prompt
from src.features.schemas import LLMProviderName, Phase2Config, ScoredEventItem

logger = logging.getLogger(__name__)


def run_phase2_preflight(
    cfg: Phase2Config,
    provider: LLMProviderName | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    client = CloudLLMClient(cfg=cfg, provider=provider, model_override=model_override)
    prov_cfg = client.provider_cfg
    target_model = client.model

    logger.info("[preflight] Checking GET /v1/models on provider=%s (%s)", client.provider, prov_cfg.base_url)
    t0 = time.perf_counter()
    available_models = client.list_model_ids()
    models_latency_s = round(time.perf_counter() - t0, 3)

    # Some providers return "models/gemini-2.5-flash" while accepting "gemini-2.5-flash"
    normalized_models = {m.removeprefix("models/") for m in available_models}
    cultural_found = (target_model in available_models) or (target_model in normalized_models)
    if not cultural_found:
        raise RuntimeError(
            f"[preflight] Model '{target_model}' not found in GET /v1/models for provider '{client.provider}'. "
            f"Sample available models: {available_models[:15]}"
        )

    # Sanity Test 1: HKEX Cantonese sarcasm
    hk_item = {
        "event_id": "preflight_hkex_cantonese_001",
        "source": "lihkg",
        "title": "燈神又唱好騰訊見800",
        "text": "多謝大戶派貨比散戶接火棒，高追坐艇蟹貨一地，好野又插水",
    }
    hk_sys = build_system_prompt("hkex_social", "0700.HK", "Tencent Holdings", "騰訊", cfg)
    hk_usr = build_user_prompt([hk_item], "hkex_social", "0700.HK", "Tencent Holdings", "騰訊", cfg)
    t1 = time.perf_counter()
    hk_parsed, _ = client.chat_json(hk_sys, hk_usr, max_tokens=512)
    hk_latency_s = round(time.perf_counter() - t1, 3)
    hk_items = hk_parsed.get("items") or []
    if len(hk_items) != 1:
        raise RuntimeError(f"[preflight] Expected 1 HKEX item, got {len(hk_items)}: {hk_parsed}")
    hk_scored = ScoredEventItem.model_validate(hk_items[0])

    # Sanity Test 2: KRX Korean sarcasm
    kr_item = {
        "event_id": "preflight_krx_korean_001",
        "source": "naver_finance",
        "title": "역시 명품주 삼성전자 ㅋㅋㅋ",
        "text": "오늘도 설거지 제대로 당했네 ㅋㅋㅋ 한강물 온도 체크하러 갑니다",
        "is_holder_verified": True,
    }
    kr_sys = build_system_prompt("krx_social", "005930.KS", "Samsung Electronics", "삼성전자", cfg)
    kr_usr = build_user_prompt([kr_item], "krx_social", "005930.KS", "Samsung Electronics", "삼성전자", cfg)
    t2 = time.perf_counter()
    kr_parsed, _ = client.chat_json(kr_sys, kr_usr, max_tokens=512)
    kr_latency_s = round(time.perf_counter() - t2, 3)
    kr_items = kr_parsed.get("items") or []
    if len(kr_items) != 1:
        raise RuntimeError(f"[preflight] Expected 1 KRX item, got {len(kr_items)}: {kr_parsed}")
    kr_scored = ScoredEventItem.model_validate(kr_items[0])

    # Verify sarcasm reversal produced bearish scores
    if hk_scored.sentiment_score >= 0.0:
        raise RuntimeError(
            f"[preflight] HKEX Cantonese sarcasm test failed expected bearish sentiment: {hk_scored.model_dump()}"
        )
    if kr_scored.sentiment_score >= 0.0:
        raise RuntimeError(
            f"[preflight] KRX Korean sarcasm test failed expected bearish sentiment: {kr_scored.model_dump()}"
        )

    report = {
        "status": "ok",
        "provider": client.provider,
        "base_url": prov_cfg.base_url,
        "cultural_nlp_model": target_model,
        "cultural_nlp_model_verified": cultural_found,
        "available_model_count": len(available_models),
        "models_latency_s": models_latency_s,
        "hkex_cantonese_sanity": {
            "latency_s": hk_latency_s,
            "result": hk_scored.model_dump(),
        },
        "krx_korean_sanity": {
            "latency_s": kr_latency_s,
            "result": kr_scored.model_dump(),
        },
    }
    logger.info(
        "[preflight] PASSED | model=%s | HKEX=%.2f (sarc=%s, %.2fs) | KRX=%.2f (sarc=%s, %.2fs)",
        target_model,
        hk_scored.sentiment_score,
        hk_scored.is_sarcastic,
        hk_latency_s,
        kr_scored.sentiment_score,
        kr_scored.is_sarcastic,
        kr_latency_s,
    )
    return report
