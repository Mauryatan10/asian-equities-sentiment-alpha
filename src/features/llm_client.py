"""Unified OpenAI-compatible Cloud LLM client (NVIDIA NIM primary; Gemini & Groq backups)
with shared 25 RPM token-bucket limiter, concurrency semaphore, and 5x Retry-After backoff."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.features.schemas import LLMProviderName, Phase2Config

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", flags=re.DOTALL | re.IGNORECASE)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class TokenBucketRateLimiter:
    """Thread-safe token bucket enforcing a strict requests-per-minute (RPM) cap."""

    def __init__(self, rpm: int) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be > 0, got {rpm}")
        self.rpm = float(rpm)
        self.min_interval = 60.0 / self.rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self.min_interval
                    return
                wait_s = self._next_allowed - now
            time.sleep(min(wait_s, 1.0))


def strip_thinking_and_fences(raw_text: str) -> str:
    """Remove accidental <think>...</think> reasoning blocks and markdown code fences."""
    cleaned = _THINK_BLOCK_RE.sub("", raw_text or "").strip()
    m = _CODE_FENCE_RE.match(cleaned)
    if m:
        cleaned = m.group(1).strip()
    return cleaned


def parse_llm_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM text after stripping <think> tags and code fences."""
    cleaned = strip_thinking_and_fences(raw_text)
    if not cleaned:
        raise ValueError("Empty LLM response content after stripping thinking/fences.")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to locate outermost JSON object if trailing text leaked
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from LLM, got {type(parsed).__name__}")
    return parsed


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        secs = float(val)
        return max(0.0, min(secs, 120.0))
    except (TypeError, ValueError):
        return None


class CloudLLMClient:
    """Strict fail-fast Cloud LLM client backed by the official OpenAI SDK."""

    def __init__(
        self,
        cfg: Phase2Config,
        provider: LLMProviderName | None = None,
        model_override: str | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        self.cfg = cfg
        self.provider: LLMProviderName = provider or cfg.llm.default_provider
        if self.provider not in cfg.llm.providers:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self.provider_cfg = cfg.llm.providers[self.provider]
        self.model = model_override or self.provider_cfg.cultural_nlp_model
        api_key = os.getenv(self.provider_cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing required environment variable {self.provider_cfg.api_key_env} "
                f"for provider '{self.provider}'. Set it in .env (strict fail-fast; never auto-switches)."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.provider_cfg.base_url,
            timeout=cfg.llm.request_timeout_seconds,
            max_retries=0,  # Managed explicitly below with Retry-After + 5x exponential backoff
        )
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(cfg.llm.rpm_limit)
        self._semaphore = threading.BoundedSemaphore(cfg.llm.max_concurrency)
        self._stats_lock = threading.Lock()
        self.total_calls = 0

    def list_model_ids(self) -> list[str]:
        """Authenticated GET /v1/models check."""
        self.rate_limiter.acquire()
        resp = self._client.models.list()
        return [str(m.id) for m in getattr(resp, "data", [])]

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Call chat.completions in JSON mode with rate limiting and 5x Retry-After backoff.
        Returns (parsed_json_dict, raw_content_str).
        """
        backoff_schedule = self.cfg.llm.retry_backoff_seconds
        max_attempts = max(1, self.cfg.llm.retry_max_attempts)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.llm.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.llm.max_tokens if max_tokens is None else max_tokens,
        }
        if self.provider_cfg.use_response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        if self.provider_cfg.extra_body:
            kwargs["extra_body"] = self.provider_cfg.extra_body

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            sleep_s: float | None = None
            with self._semaphore:
                self.rate_limiter.acquire()
                try:
                    with self._stats_lock:
                        self.total_calls += 1
                    resp = self._client.chat.completions.create(**kwargs)
                    content = (resp.choices[0].message.content or "") if resp.choices else ""
                    parsed = parse_llm_json_object(content)
                    return parsed, content
                except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                    last_exc = exc
                    retry_after = _extract_retry_after_seconds(exc)
                    base_wait = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
                    sleep_s = max(float(base_wait), retry_after or 0.0)
                    logger.warning(
                        "[llm:%s/%s] Retryable transport/rate error (attempt %d/%d): %s — sleeping %.1fs",
                        self.provider,
                        self.model,
                        attempt,
                        max_attempts,
                        exc,
                        sleep_s,
                    )
                except APIStatusError as exc:
                    last_exc = exc
                    status_code = getattr(exc, "status_code", None)
                    if status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
                        retry_after = _extract_retry_after_seconds(exc)
                        base_wait = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
                        sleep_s = max(float(base_wait), retry_after or 0.0)
                        logger.warning(
                            "[llm:%s/%s] HTTP %s (attempt %d/%d): %s — sleeping %.1fs",
                            self.provider,
                            self.model,
                            status_code,
                            attempt,
                            max_attempts,
                            exc,
                            sleep_s,
                        )
                    else:
                        # Non-retryable HTTP error (e.g. 401/403/404/410) -> fail loudly
                        raise
                except (json.JSONDecodeError, ValueError) as exc:
                    # Malformed JSON is handed back to caller for auto-bisect recovery
                    raise ValueError(f"Malformed JSON from {self.provider}/{self.model}: {exc}") from exc

            # Sleep OUTSIDE self._semaphore so healthy batches are never starved during backoff
            if sleep_s is not None and attempt < max_attempts:
                time.sleep(sleep_s)

        raise RuntimeError(
            f"LLM call failed after {max_attempts} attempts ({self.provider}/{self.model}): {last_exc}"
        ) from last_exc
