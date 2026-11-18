"""Single-ticker scoped prompts and HKEX/KRX financial slang glossaries for CulturalNLPAgent."""

from __future__ import annotations

import json
from typing import Any, Literal

from src.features.schemas import Phase2Config, SlangEntry

PromptDomain = Literal["hkex_social", "krx_social", "news"]


def resolve_prompt_domain(event_type: str, exchange: str) -> PromptDomain:
    if str(event_type).lower() == "news":
        return "news"
    if str(exchange).upper() == "HKEX":
        return "hkex_social"
    return "krx_social"


def _format_glossary(entries: list[SlangEntry]) -> str:
    if not entries:
        return ""
    lines = ["LOCAL FINANCIAL SLANG & SARCASM GLOSSARY:"]
    for entry in entries:
        lines.append(f"- {entry.term}: {entry.meaning}")
    return "\n".join(lines)


_COMMON_JSON_CONTRACT = """OUTPUT FORMAT (STRICT JSON OBJECT ONLY — NO MARKDOWN, NO EXTRA PROSE):
Return a single JSON object with key "items", containing one object per input item in the EXACT same order, echoing every input "event_id" verbatim:
{
  "items": [
    {
      "event_id": "<exact input event_id>",
      "sentiment_score": <float in [-1.0, 1.0], true INTENDED market sentiment toward target_ticker after sarcasm/irony reversal>,
      "direction": "<bullish | bearish | neutral>",
      "confidence": <float in [0.0, 1.0]>,
      "is_sarcastic": <true | false>,
      "hype_score": <float in [0.0, 1.0], FOMO/meme euphoria intensity>,
      "catalyst_type": "<earnings | guidance | analyst_rating | macro_policy | product_tech | flow_squeeze | retail_rumor | other>",
      "price_target_mentioned": <number in local exchange currency (HKD for HKEX, KRW for KRX) if explicitly stated as a target price for target_ticker, else null>,
      "english_summary": "<concise English translation/catalyst summary, strictly <= 120 characters>"
    }
  ]
}
RULES:
1. Score sentiment SPECIFICALLY toward {ticker} ({company_display}); ignore sentiment directed at other companies or competitors mentioned in the same post.
2. If the post uses sarcasm, reverse-jinxing, or dark humor (e.g., praising a crashing stock), reverse the surface wording to reflect the speaker's TRUE bearish/bullish intent and set "is_sarcastic": true.
3. direction MUST align with sentiment_score: > +0.15 => "bullish", < -0.15 => "bearish", else "neutral".
4. price_target_mentioned MUST be null unless a concrete numeric target price for {ticker} is stated (do NOT put percentages, years, dates, or ticker codes there).
5. You MUST include every single input event_id exactly once in "items"."""


def build_system_prompt(
    domain: PromptDomain,
    ticker: str,
    company_en: str,
    company_local: str,
    cfg: Phase2Config,
) -> str:
    company_display = f"{company_en} / {company_local}"
    contract = (
        _COMMON_JSON_CONTRACT
        .replace("{ticker}", ticker)
        .replace("{company_display}", company_display)
    )

    if domain == "hkex_social":
        glossary = _format_glossary(cfg.slang_glossaries.get("hkex_cantonese", []))
        role = (
            f"You are an expert Hong Kong equities cultural NLP analyst specializing in HKEX retail forums "
            f"(LIHKG 財經台) and Chinese/Cantonese financial Telegram channels.\n"
            f"Your single target asset for this entire batch is {ticker} ({company_display}, currency: HKD).\n"
            f"Detect Cantonese slang, CBBC bull/bear warrant chatter, reverse-indicator irony (燈神), and whale-distribution complaints accurately."
        )
        return f"{role}\n\n{glossary}\n\n{contract}"

    if domain == "krx_social":
        glossary = _format_glossary(cfg.slang_glossaries.get("krx_naver", []))
        role = (
            f"You are an expert South Korean equities cultural NLP analyst specializing in KRX retail boards "
            f"(Naver Finance 종목토론실) and Korean stock Telegram channels.\n"
            f"Your single target asset for this entire batch is {ticker} ({company_display}, currency: KRW).\n"
            f"Detect Korean retail slang, dark humor (한강물 온도, ㅋㅋㅋ), exit-liquidity warnings (설거지), "
            f"weak-hand shakeout narratives (개미털기), and verified-shareholder conviction accurately."
        )
        return f"{role}\n\n{glossary}\n\n{contract}"

    # Institutional news
    currency = "HKD" if ticker.upper().endswith(".HK") else "KRW"
    role = (
        f"You are a senior Asian equities institutional news analyst covering HKEX and KRX.\n"
        f"Your single target asset for this entire batch is {ticker} ({company_display}, currency: {currency}).\n"
        f"Evaluate formal financial journalism objectively: earnings beats/misses, forward guidance, sell-side "
        f"analyst rating or target-price revisions, regulatory/macro policy shifts, supply-chain/product news, "
        f"and institutional flow catalysts. Set is_sarcastic=false for straight institutional reporting."
    )
    return f"{role}\n\n{contract}"


def build_user_prompt(
    items: list[dict[str, Any]],
    domain: PromptDomain,
    ticker: str,
    company_en: str,
    company_local: str,
    cfg: Phase2Config,
) -> str:
    company_display = f"{company_en} / {company_local}"
    if domain == "news":
        title_cap = cfg.llm.news_title_max_chars
        text_cap = cfg.llm.news_text_max_chars
    else:
        title_cap = cfg.llm.social_title_max_chars
        text_cap = cfg.llm.social_text_max_chars

    payload_items: list[dict[str, Any]] = []
    for row in items:
        title = str(row.get("title") or "").strip()[:title_cap]
        text = str(row.get("text") or "").strip()[:text_cap]
        entry: dict[str, Any] = {
            "event_id": str(row["event_id"]),
            "target_ticker": ticker,
            "company_name": company_display,
            "source": str(row.get("source") or ""),
            "title": title,
            "text": text,
        }
        if row.get("is_holder_verified") is True:
            entry["holder_verified"] = True
        payload_items.append(entry)

    header = (
        f"Score the following {len(payload_items)} items SPECIFICALLY for {ticker} ({company_display}). "
        f"Return JSON with key \"items\" containing exactly {len(payload_items)} scored objects."
    )
    return f"{header}\n\n{json.dumps(payload_items, ensure_ascii=False)}"
