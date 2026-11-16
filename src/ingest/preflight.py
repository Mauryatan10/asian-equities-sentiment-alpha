"""$0.00 Pre-Flight Verification Gate (Phase 1 Section 1.1).

Verifies before any paid actor execution:
1. Apify free-tier balance & limits via GET /v2/users/me/limits ($0.00).
2. Verbatim inputSchema keys for LIHKG and Blind actors via Apify Actor Metadata API ($0.00)
   and validates that config/sources.yaml verified_input_keys match live schemas.
3. 2-ticker yfinance 60m probe (0700.HK, 005930.KS) verifying exact returned bar-open
   timestamps and timezone offsets against config/universe.yaml bar_schedule_local.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
import yfinance as yf

from src.ingest.manifest import ManifestStore, check_apify_budget, utc_now_iso
from src.ingest.schemas import load_sources, load_universe

logger = logging.getLogger(__name__)


def _fetch_actor_schema(actor_slug: str, api_token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_token}"}
    act_resp = requests.get(
        f"https://api.apify.com/v2/acts/{actor_slug}",
        headers=headers,
        timeout=20,
    )
    act_resp.raise_for_status()
    act_data = act_resp.json().get("data", {})
    tagged = act_data.get("taggedBuilds", {})
    build_info = tagged.get("latest") or tagged.get("version-3") or next(iter(tagged.values()), {})
    build_id = build_info.get("buildId")
    if not build_id:
        raise RuntimeError(f"Actor {actor_slug} has no tagged builds in metadata response")

    build_resp = requests.get(
        f"https://api.apify.com/v2/actor-builds/{build_id}",
        headers=headers,
        timeout=20,
    )
    build_resp.raise_for_status()
    bdata = build_resp.json().get("data", {})
    schema = bdata.get("inputSchema")
    if isinstance(schema, str):
        schema = json.loads(schema)
    props = (schema or {}).get("properties", {})
    return {
        "actor_slug": actor_slug,
        "build_id": build_id,
        "build_number": build_info.get("buildNumber"),
        "required": (schema or {}).get("required", []),
        "property_keys": sorted(props.keys()),
        "pricing_infos": act_data.get("pricingInfos", [])[-1:],
    }


def _probe_yfinance_60m(ticker: str, expected_tz: str, allowed_open_times: set[str]) -> dict[str, Any]:
    df = yf.Ticker(ticker).history(period="5d", interval="60m", auto_adjust=False, prepost=False)
    if df.empty:
        raise RuntimeError(f"yfinance 60m probe returned 0 bars for {ticker}")

    actual_tz = str(df.index.tz)
    distinct_times = sorted({ts.strftime("%H:%M") for ts in df.index})
    unknown_times = [t for t in distinct_times if t not in allowed_open_times]

    if actual_tz != expected_tz:
        raise RuntimeError(f"Timezone mismatch for {ticker}: expected {expected_tz}, got {actual_tz}")
    if unknown_times:
        raise RuntimeError(
            f"Unmapped yfinance 60m open times for {ticker}: {unknown_times} "
            f"(expected subset of {sorted(allowed_open_times)})"
        )

    return {
        "ticker": ticker,
        "rows": int(len(df)),
        "timezone": actual_tz,
        "distinct_local_open_times": distinct_times,
        "latest_bar_open": df.index[-1].isoformat(),
    }


def run_preflight(manifest_store: ManifestStore | None = None) -> dict[str, Any]:
    """Run the $0.00 Pre-Flight Verification Gate and persist results to phase1_manifest.json."""
    api_token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not api_token:
        raise RuntimeError("APIFY_API_TOKEN is missing in environment/.env")

    sources = load_sources()
    universe = load_universe()

    # 1. Check Apify free-tier balance ($0.00)
    hard_stop = float(sources["budget_guard"]["apify_hard_stop_monthly_usd"])
    budget_info = check_apify_budget(api_token, hard_stop_usd=hard_stop)

    # 2. Verify verbatim inputSchema keys for LIHKG and Blind actors ($0.00)
    actor_checks: dict[str, Any] = {}
    targets = [
        ("lihkg", "claude_code_reviewer~lihkg-scraper-en", sources["social"]["lihkg"]["verified_input_keys"]),
        ("blind", "hypebridge~blind-post-scraper", sources["social"]["backups"]["blind"]["verified_input_keys"]),
    ]
    for name, slug, expected_keys in targets:
        info = _fetch_actor_schema(slug, api_token)
        live_keys = set(info["property_keys"])
        missing_keys = [k for k in expected_keys if k not in live_keys]
        if missing_keys:
            raise RuntimeError(
                f"Pre-flight schema mismatch for {slug}: expected keys {missing_keys} "
                f"not found in live inputSchema keys {sorted(live_keys)}"
            )
        info["verified_expected_keys_ok"] = True
        actor_checks[name] = info
        logger.info(
            "Pre-flight OK [%s]: build=%s, verified %d input keys",
            slug,
            info["build_number"],
            len(expected_keys),
        )

    # 3. Verify yfinance 60m probe for 0700.HK and 005930.KS ($0.00)
    yf_checks: dict[str, Any] = {}
    for ex_code, probe_ticker in [("HKEX", "0700.HK"), ("KRX", "005930.KS")]:
        ex_cfg = universe.exchanges[ex_code]  # type: ignore[index]
        allowed = set(ex_cfg.bar_schedule_local.keys())
        probe = _probe_yfinance_60m(probe_ticker, ex_cfg.timezone, allowed)
        yf_checks[probe_ticker] = probe
        logger.info(
            "Pre-flight OK [yfinance %s]: tz=%s, open_times=%s",
            probe_ticker,
            probe["timezone"],
            probe["distinct_local_open_times"],
        )

    report = {
        "status": "ok",
        "cost_usd": 0.0,
        "checked_at_utc": utc_now_iso(),
        "apify_budget": budget_info,
        "actor_schemas": actor_checks,
        "yfinance_probe": yf_checks,
    }

    if manifest_store is not None:
        manifest_store.set_preflight(report)

    return report
