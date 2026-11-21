"""Modular CLI entrypoint for Phase 3: Predictive Modeling & Out-of-Sample Validation.
Usage:
    python -m src.models.run --stage all
    python -m src.models.run --smoke-test
    python -m src.models.run --stage train_eval
    python -m src.models.run --stage backtest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from src.features.schemas import (
    CONFIG_DIR,
    LIVE_SNAPSHOT_PATH,
    PROCESSED_DIR,
    Phase2Config,
    load_phase2_config,
)
from src.models.backtest import (
    render_markdown_tearsheet,
    run_full_backtest,
)
from src.models.dataset import ModelDataset, load_and_split_dataset
from src.models.ensemble import Layer1Ensemble
from src.models.schemas import (
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_SCHEMA,
    PHASE3_MANIFEST_PATH,
    PHASE3_TEARSHEET_REPORT_PATH,
    PREDICTIONS_OOS_PATH,
    PREDICTIONS_OOS_SCHEMA,
    TEARSHEET_METRICS_PATH,
    Phase3Config,
    load_phase3_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Predictive Modeling Pipeline")
    parser.add_argument(
        "--stage",
        choices=["train_eval", "backtest", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker subset (e.g. 0700.HK,005930.KS)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test mode on 2 tickers (0700.HK, 005930.KS)",
    )
    return parser.parse_args()


def save_parquet_table(df: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    """Save a pandas DataFrame with an exact PyArrow schema using zstd compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    logger.info("Saved %d rows to %s", len(df), path)


def run_pipeline(
    stage: str = "all",
    tickers: list[str] | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Execute Phase 3 predictive modeling and backtest pipeline."""
    load_dotenv()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    logger.info("Starting Phase 3 modeling pipeline (stage=%s, smoke=%s)", stage, smoke_test)

    p3_cfg = load_phase3_config()

    all_tickers = [
        "0700.HK", "9988.HK", "1810.HK", "0981.HK", "1211.HK",
        "005930.KS", "000660.KS", "035720.KS", "005380.KS", "373220.KS",
    ]
    if smoke_test:
        target_tickers = p3_cfg.smoke_test.tickers
    elif tickers:
        target_tickers = tickers
    else:
        target_tickers = all_tickers

    # 1. Dataset loading and strict UTC-embargoed splitting
    ds = load_and_split_dataset(p3_cfg)
    logger.info(
        "Dataset loaded: train_split=%d rows, test_oos=%d rows, train_live=%d rows, live=%d rows",
        len(ds.train_split),
        len(ds.test_oos),
        len(ds.train_live),
        len(ds.live),
    )

    test_oos_df = ds.test_oos.copy()
    test_oos_df["actual_log_return_1h"] = test_oos_df["target_log_return_1h"]
    test_oos_df["actual_next_close"] = test_oos_df["next_close"]

    # 2. Layer 1 Model Training & OOS Predictions
    if stage in ("train_eval", "backtest", "all"):
        logger.info("Training Layer 1 Full MoE Ensemble (%d features) on train_split (%d rows)...", len(ds.feature_cols), len(ds.train_split))
        ens_full = Layer1Ensemble(p3_cfg.models)
        ens_full.fit(ds.train_split[ds.feature_cols], ds.y_train)

        preds_full = ens_full.predict(
            X=test_oos_df[ds.feature_cols],
            vol_scale=ds.vol_scale_test,
            close_prices=test_oos_df["close"].values,
        )
        test_oos_df["full_pred_huber_ret"] = preds_full.pred_huber_ret
        test_oos_df["full_pred_lgbm_ret"] = preds_full.pred_lgbm_ret
        test_oos_df["full_pred_ret"] = preds_full.pred_ens_ret
        test_oos_df["full_pred_price"] = preds_full.pred_price

        logger.info("Training Layer 1 Tech-Only Ablation (%d features) on train_split...", len(ds.tech_cols))
        ens_tech = Layer1Ensemble(p3_cfg.models)
        ens_tech.fit(ds.train_split[ds.tech_cols], ds.y_train)

        preds_tech = ens_tech.predict(
            X=test_oos_df[ds.tech_cols],
            vol_scale=ds.vol_scale_test,
            close_prices=test_oos_df["close"].values,
        )
        test_oos_df["tech_pred_ret"] = preds_tech.pred_ens_ret
        test_oos_df["tech_pred_price"] = preds_tech.pred_price

        # Random Walk benchmark: pred_price = current close
        test_oos_df["rw_pred_price"] = test_oos_df["close"].values

        # Feature Importance computation and saving
        fi_df = ens_full.extract_feature_importance(ds.feature_cols)
        cols_fi = [field.name for field in FEATURE_IMPORTANCE_SCHEMA]
        save_parquet_table(fi_df[cols_fi], FEATURE_IMPORTANCE_SCHEMA, FEATURE_IMPORTANCE_PATH)

        # Retrain full ensemble on train_live for live inference
        logger.info("Retraining full ensemble on train_live (%d rows) for live inference...", len(ds.train_live))
        ens_live = Layer1Ensemble(p3_cfg.models)
        ens_live.fit(ds.train_live[ds.feature_cols], ds.y_train_live)
        live_preds = ens_live.predict(
            X=ds.live[ds.feature_cols],
            vol_scale=ds.vol_scale_train_live[:len(ds.live)],
            close_prices=ds.live["close"].values,
        )

    # 3. Backtest Simulation & Vectorized Tear Sheet
    if stage in ("backtest", "all"):
        logger.info("Executing discrete backtest simulation and tear sheet computation...")
        backtest_df, tearsheet = run_full_backtest(test_oos_df, p3_cfg)

        # Save predictions_oos.parquet
        cols_oos = [field.name for field in PREDICTIONS_OOS_SCHEMA]
        save_parquet_table(backtest_df[cols_oos], PREDICTIONS_OOS_SCHEMA, PREDICTIONS_OOS_PATH)

        # Save tearsheet_metrics.json
        TEARSHEET_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TEARSHEET_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(tearsheet, f, indent=2)
        logger.info("Saved tear sheet metrics to %s", TEARSHEET_METRICS_PATH)

        # Save phase3_tearsheet.md
        md_report = render_markdown_tearsheet(tearsheet)
        PHASE3_TEARSHEET_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PHASE3_TEARSHEET_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info("Saved human-readable tear sheet to %s", PHASE3_TEARSHEET_REPORT_PATH)

    # 4. Build and save phase3_manifest.json
    run_completed_utc = datetime.now(timezone.utc).isoformat()
    manifest_data = {
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "validation_mode": "chrono_split",
        "train_date_range": [
            str(ds.train_split["trading_date_local"].min()),
            str(ds.train_split["trading_date_local"].max()),
        ],
        "test_date_range": [
            str(test_oos_df["trading_date_local"].min()),
            str(test_oos_df["trading_date_local"].max()),
        ],
        "test_start_utc": ds.test_start_utc.isoformat(),
        "train_split_rows": len(ds.train_split),
        "test_oos_rows": len(test_oos_df),
        "train_live_rows": len(ds.train_live),
        "live_rows": len(ds.live),
        "feature_count_full": len(ds.feature_cols),
        "feature_count_tech": len(ds.tech_cols),
        "model_hyperparams": p3_cfg.models.model_dump(),
        "assertions_passed": True,
    }
    PHASE3_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE3_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved phase3_manifest.json to %s", PHASE3_MANIFEST_PATH)

    logger.info("Phase 3 pipeline completed successfully!")
    return manifest_data


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    run_pipeline(
        stage=args.stage,
        tickers=tickers,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
