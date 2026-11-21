"""Learned Mixture-of-Experts (MoE) Ensemble: HuberRegressor + shallow LGBMRegressor
routed dynamically via Ridge Logistic / Softmax Gating Network on Market State Vector."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.models.schemas import ModelsConfig, Phase3Config

logger = logging.getLogger(__name__)


@dataclass
class Layer1Predictions:
    pred_huber_ret: np.ndarray
    pred_lgbm_ret: np.ndarray
    pred_ens_ret: np.ndarray
    pred_price: np.ndarray
    gating_weights_linear: np.ndarray | None = None
    gating_weights_tree: np.ndarray | None = None


class LearnedMoEEnsemble:
    """Learned Mixture-of-Experts (MoE) Ensemble with state-dependent Softmax gating."""

    def __init__(self, cfg: ModelsConfig) -> None:
        self.cfg = cfg
        self.pipe_huber = Pipeline([
            ("scaler", RobustScaler()),
            (
                "huber",
                HuberRegressor(
                    epsilon=cfg.huber.epsilon,
                    alpha=cfg.huber.alpha,
                    max_iter=cfg.huber.max_iter,
                ),
            ),
        ])
        self.lgbm = LGBMRegressor(
            objective=cfg.lgbm.objective,
            alpha=cfg.lgbm.alpha,
            max_depth=cfg.lgbm.max_depth,
            num_leaves=cfg.lgbm.num_leaves,
            min_child_samples=cfg.lgbm.min_child_samples,
            learning_rate=cfg.lgbm.learning_rate,
            n_estimators=cfg.lgbm.n_estimators,
            colsample_bytree=cfg.lgbm.colsample_bytree,
            subsample=cfg.lgbm.subsample,
            subsample_freq=cfg.lgbm.subsample_freq,
            reg_alpha=cfg.lgbm.reg_alpha,
            reg_lambda=cfg.lgbm.reg_lambda,
            random_state=cfg.lgbm.random_state,
            verbosity=cfg.lgbm.verbosity,
        )
        self.feature_names: list[str] = []
        self.state_scaler = RobustScaler()
        self.state_feature_names: list[str] = (
            list(cfg.moe_gating.state_features)
            if hasattr(cfg, "moe_gating") and hasattr(cfg.moe_gating, "state_features")
            else ["realized_vol_12h", "atr_pct", "sentiment_dispersion_24h"]
        )

        # Prior log-odds: ln(p_tree / p_linear) = ln(0.40 / 0.60)
        huber_w = cfg.blend.huber_weight
        lgbm_w = cfg.blend.lgbm_weight
        self.prior_b = float(np.log(max(lgbm_w, 1e-6) / max(huber_w, 1e-6)))
        self.W_gate: np.ndarray = np.zeros(len(self.state_feature_names), dtype=np.float64)
        self.b_gate: float = self.prior_b
        self.mean_gating_linear: float = huber_w
        self.mean_gating_tree: float = lgbm_w

    def _extract_state_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.state_feature_names):
                return X[self.state_feature_names].fillna(0.0).to_numpy(dtype=np.float64)
        return None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        state_df: pd.DataFrame | None = None,
    ) -> LearnedMoEEnsemble:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
            X_arr = X

        # Fit base experts on full in-sample dataset
        self.pipe_huber.fit(X_arr, y)
        self.lgbm.fit(X_arr, y)

        # Check if MoE Gating is enabled and state features are available
        s_arr = None
        if state_df is not None and all(col in state_df.columns for col in self.state_feature_names):
            s_arr = state_df[self.state_feature_names].fillna(0.0).to_numpy(dtype=np.float64)
        else:
            s_arr = self._extract_state_matrix(X)

        moe_enabled = getattr(self.cfg.moe_gating, "enabled", True) if hasattr(self.cfg, "moe_gating") else True
        if moe_enabled and s_arr is not None and len(s_arr) >= 20:
            self._fit_gating_network(X_arr, y, s_arr)
        else:
            self.W_gate = np.zeros(len(self.state_feature_names), dtype=np.float64)
            self.b_gate = self.prior_b
            self.mean_gating_linear = self.cfg.blend.huber_weight
            self.mean_gating_tree = self.cfg.blend.lgbm_weight

        return self

    def _fit_gating_network(self, X_arr: np.ndarray, y: np.ndarray, s_arr: np.ndarray) -> None:
        """Fit Ridge Logistic Gating router over out-of-fold predictions."""
        n = len(y)
        oof_huber = np.zeros(n, dtype=np.float64)
        oof_lgbm = np.zeros(n, dtype=np.float64)

        cv_folds = getattr(self.cfg.moe_gating, "cv_folds", 5) if hasattr(self.cfg, "moe_gating") else 5
        kf = KFold(n_splits=cv_folds, shuffle=False)

        for train_idx, val_idx in kf.split(X_arr):
            huber_cv = Pipeline([
                ("scaler", RobustScaler()),
                (
                    "huber",
                    HuberRegressor(
                        epsilon=self.cfg.huber.epsilon,
                        alpha=self.cfg.huber.alpha,
                        max_iter=self.cfg.huber.max_iter,
                    ),
                ),
            ])
            lgbm_cv = LGBMRegressor(
                objective=self.cfg.lgbm.objective,
                alpha=self.cfg.lgbm.alpha,
                max_depth=self.cfg.lgbm.max_depth,
                num_leaves=self.cfg.lgbm.num_leaves,
                min_child_samples=self.cfg.lgbm.min_child_samples,
                learning_rate=self.cfg.lgbm.learning_rate,
                n_estimators=self.cfg.lgbm.n_estimators,
                colsample_bytree=self.cfg.lgbm.colsample_bytree,
                subsample=self.cfg.lgbm.subsample,
                subsample_freq=self.cfg.lgbm.subsample_freq,
                reg_alpha=self.cfg.lgbm.reg_alpha,
                reg_lambda=self.cfg.lgbm.reg_lambda,
                random_state=self.cfg.lgbm.random_state,
                verbosity=self.cfg.lgbm.verbosity,
            )

            huber_cv.fit(X_arr[train_idx], y[train_idx])
            lgbm_cv.fit(X_arr[train_idx], y[train_idx])

            oof_huber[val_idx] = huber_cv.predict(X_arr[val_idx])
            oof_lgbm[val_idx] = lgbm_cv.predict(X_arr[val_idx])

        # Scale state variables using strictly training data
        s_scaled = self.state_scaler.fit_transform(s_arr)

        l2_reg = getattr(self.cfg.moe_gating, "l2_reg", 1.0) if hasattr(self.cfg, "moe_gating") else 1.0
        n_dim = s_scaled.shape[1]

        def loss_fn(w: np.ndarray) -> float:
            logits = s_scaled @ w + self.prior_b
            p_tree = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            p_linear = 1.0 - p_tree
            y_pred = p_linear * oof_huber + p_tree * oof_lgbm
            mse = float(np.mean((y - y_pred) ** 2))
            reg = float(l2_reg * np.sum(w ** 2))
            return mse + reg

        init_w = np.zeros(n_dim, dtype=np.float64)
        res = minimize(loss_fn, init_w, method="L-BFGS-B")
        if res.success:
            self.W_gate = res.x
            self.b_gate = self.prior_b
        else:
            self.W_gate = np.zeros(n_dim, dtype=np.float64)
            self.b_gate = self.prior_b

        train_logits = s_scaled @ self.W_gate + self.b_gate
        train_p_tree = 1.0 / (1.0 + np.exp(-np.clip(train_logits, -20.0, 20.0)))
        self.mean_gating_tree = float(np.mean(train_p_tree))
        self.mean_gating_linear = float(1.0 - self.mean_gating_tree)
        logger.info(
            "Learned MoE Gating fitted: W_gate=%s, b_gate=%.4f (mean weights: linear=%.3f, tree=%.3f)",
            self.W_gate, self.b_gate, self.mean_gating_linear, self.mean_gating_tree
        )

    def compute_gating_weights(
        self,
        X: pd.DataFrame | np.ndarray,
        state_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute sample-wise softmax gating allocation weights w_linear, w_tree."""
        n = len(X)
        s_arr = None
        if state_df is not None and all(col in state_df.columns for col in self.state_feature_names):
            s_arr = state_df[self.state_feature_names].fillna(0.0).to_numpy(dtype=np.float64)
        else:
            s_arr = self._extract_state_matrix(X)

        if s_arr is not None and len(self.W_gate) == s_arr.shape[1] and np.any(self.W_gate != 0):
            try:
                s_scaled = self.state_scaler.transform(s_arr)
                logits = s_scaled @ self.W_gate + self.b_gate
                w_tree = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
                w_linear = 1.0 - w_tree
                return w_linear, w_tree
            except Exception:
                pass

        w_lin = np.full(n, self.cfg.blend.huber_weight, dtype=np.float64)
        w_tr = np.full(n, self.cfg.blend.lgbm_weight, dtype=np.float64)
        return w_lin, w_tr

    def predict_raw(
        self,
        X: pd.DataFrame | np.ndarray,
        state_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        pred_huber_y = self.pipe_huber.predict(X_arr)
        pred_lgbm_y = self.lgbm.predict(X_arr)

        w_linear, w_tree = self.compute_gating_weights(X, state_df=state_df)
        pred_ens_y = w_linear * pred_huber_y + w_tree * pred_lgbm_y
        return pred_huber_y, pred_lgbm_y, pred_ens_y

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        vol_scale: np.ndarray,
        close_prices: np.ndarray,
        state_df: pd.DataFrame | None = None,
    ) -> Layer1Predictions:
        pred_huber_y = self.pipe_huber.predict(X.values if isinstance(X, pd.DataFrame) else X)
        pred_lgbm_y = self.lgbm.predict(X.values if isinstance(X, pd.DataFrame) else X)
        w_linear, w_tree = self.compute_gating_weights(X, state_df=state_df)
        pred_ens_y = w_linear * pred_huber_y + w_tree * pred_lgbm_y

        pred_huber_ret = pred_huber_y * vol_scale
        pred_lgbm_ret = pred_lgbm_y * vol_scale
        pred_ens_ret = pred_ens_y * vol_scale
        pred_price = close_prices * np.exp(pred_ens_ret)

        return Layer1Predictions(
            pred_huber_ret=pred_huber_ret,
            pred_lgbm_ret=pred_lgbm_ret,
            pred_ens_ret=pred_ens_ret,
            pred_price=pred_price,
            gating_weights_linear=w_linear,
            gating_weights_tree=w_tree,
        )

    def extract_feature_importance(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        names = feature_names or self.feature_names
        huber_coef = self.pipe_huber.named_steps["huber"].coef_
        lgbm_gain = self.lgbm.booster_.feature_importance(importance_type="gain")

        norm_huber = np.abs(huber_coef) / max(float(np.sum(np.abs(huber_coef))), 1e-12)
        norm_lgbm = lgbm_gain / max(float(np.sum(lgbm_gain)), 1e-12)
        combined_score = (
            self.mean_gating_linear * norm_huber
            + self.mean_gating_tree * norm_lgbm
        )

        fi = pd.DataFrame({
            "feature_name": names,
            "feature_group": [
                "tech_kde" if i < 18 else "sentiment_alpha" for i in range(len(names))
            ],
            "huber_coef_signed": huber_coef,
            "huber_coef_abs": np.abs(huber_coef),
            "lgbm_gain_importance": lgbm_gain,
            "combined_score": combined_score,
        })
        fi = fi.sort_values("combined_score", ascending=False).reset_index(drop=True)
        fi["combined_rank"] = np.arange(1, len(names) + 1, dtype=np.int32)
        return fi

    # Compatibility alias
    compute_feature_importance = extract_feature_importance


# Backward compatible alias
Layer1Ensemble = LearnedMoEEnsemble
