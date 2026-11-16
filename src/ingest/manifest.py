"""Manifest management, idempotent cache lookup, atomic JSON/JSONL persistence, and Apify free-tier budget guard."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from src.ingest.schemas import (
    MANIFEST_PATH,
    ManifestEntry,
    Phase1Manifest,
    RunMode,
)

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cache_key(
    stage: str,
    source: str,
    exchange: str,
    key: str,
    end_date: str,
    mode: RunMode,
) -> str:
    return f"{stage}|{source}|{exchange.upper()}|{key}|{end_date}|{mode}"


class ManifestStore:
    """Atomic JSON manifest manager supporting idempotent caching across Phase 1 runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or MANIFEST_PATH
        self.manifest = self._load()

    def _load(self) -> Phase1Manifest:
        if not self.path.exists():
            return Phase1Manifest()
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return Phase1Manifest.model_validate(data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.manifest.model_dump(mode="json")
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp_name, self.path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def is_cached_ok(self, cache_key: str, force_refresh: bool = False) -> bool:
        if force_refresh:
            return False
        entry = self.manifest.entries.get(cache_key)
        if entry is None or entry.status != "ok":
            return False
        if entry.raw_path:
            return Path(entry.raw_path).exists()
        return True

    def get_entry(self, cache_key: str) -> ManifestEntry | None:
        return self.manifest.entries.get(cache_key)

    def record(self, entry: ManifestEntry) -> None:
        self.manifest.entries[entry.cache_key] = entry
        self.save()

    def set_exchange_window(self, exchange: str, window_dict: dict[str, Any]) -> None:
        self.manifest.exchange_windows[exchange.upper()] = window_dict
        self.save()

    def set_preflight(self, report: dict[str, Any]) -> None:
        self.manifest.preflight = report
        self.save()


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to an immutable JSONL snapshot atomically. Returns record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                count += 1
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class BudgetExceededError(RuntimeError):
    """Raised when Apify free-tier spend approaches the hard safety stop."""


def check_apify_budget(
    api_token: str,
    hard_stop_usd: float = 4.00,
    estimated_next_run_usd: float = 0.0,
) -> dict[str, float]:
    """$0.00 Apify free-tier usage guard via GET /v2/users/me/limits.

    Refuses to start new paid actor runs if monthlyUsageUsd + estimated_next_run_usd
    exceeds hard_stop_usd (default $4.00 on the $5.00 free tier).
    """
    resp = requests.get(
        "https://api.apify.com/v2/users/me/limits",
        headers={"Authorization": f"Bearer {api_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    max_monthly = float(data.get("limits", {}).get("maxMonthlyUsageUsd", 5.0))
    used_monthly = float(data.get("current", {}).get("monthlyUsageUsd", 0.0))
    remaining = max(0.0, max_monthly - used_monthly)
    effective_stop = min(hard_stop_usd, max_monthly - 0.25)

    logger.info(
        "Apify free-tier balance check: used=$%.4f / $%.2f (remaining=$%.4f, guard=$%.2f)",
        used_monthly,
        max_monthly,
        remaining,
        effective_stop,
    )

    if used_monthly + estimated_next_run_usd > effective_stop:
        raise BudgetExceededError(
            f"Apify budget guard triggered! Monthly usage=${used_monthly:.4f} + "
            f"est_next=${estimated_next_run_usd:.4f} exceeds safety cap ${effective_stop:.2f} "
            f"(monthly limit=${max_monthly:.2f}). Aborting before starting actor run."
        )

    return {
        "max_monthly_usd": max_monthly,
        "used_monthly_usd": used_monthly,
        "remaining_usd": remaining,
    }
