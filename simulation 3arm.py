#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-arm Multimodal Causal Meta-Learning Simulation for Individualized
Treatment Selection in Non-Invasive Neuromodulation.

This script implements the three-arm extension of the simulation framework,
with Sham control, tVNS, and Neurofeedback (NF) as treatment arms.

Treatment coding: T ∈ {0 = Sham, 1 = tVNS, 2 = NF}
Sample: n = 810, with 270 individuals per arm (exact 1:1:1 randomisation).
Features: p = 42 (20 EEG-like + 10 HRV-like + 10 psychometric + Age + Sex).

Estimands (per individual i):
    τ_i^NF        = Y_i(2) - Y_i(0)         absolute NF effect (vs sham)
    τ_i^tVNS      = Y_i(1) - Y_i(0)         absolute tVNS effect (vs sham)
    τ_i^comp      = Y_i(2) - Y_i(1)         comparative NF vs tVNS effect

Outputs CSV tables of recovery metrics and SHAP rankings.

Author: Seyedeh Zeinab Molaeizadeh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Generative parameters for the three-arm simulation."""
    n_per_arm: int = 270
    n_eeg: int = 20
    n_hrv: int = 10
    n_psych: int = 10
    age_mean: float = 50.0
    age_sd: float = 15.0
    sex_prob: float = 0.5
    outcome_noise_sd: float = 0.40       # ε ~ N(0, 0.40²)
    treatment_effect_noise_sd: float = 0.30  # η ~ N(0, 0.30²)
    eeg_ar1_rho: float = 0.30
    hrv_lognormal_sigma: float = 0.50
    psy_corr: float = 0.40
    random_seed: int = 42

    @property
    def n_samples(self) -> int:
        return 3 * self.n_per_arm

    @property
    def n_features(self) -> int:
        return self.n_eeg + self.n_hrv + self.n_psych + 2  # +Age, +Sex


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the gradient-boosted base learners and CV."""
    n_splits: int = 5
    random_seed: int = 42
    n_estimators: int = 60
    learning_rate: float = 0.05
    max_depth: int = 2
    subsample: float = 0.85


# Treatment labels
T_SHAM = 0
T_TVNS = 1
T_NF = 2
ARM_NAMES = {T_SHAM: "Sham", T_TVNS: "tVNS", T_NF: "NF"}


# =============================================================================
# Synthetic data generation (three-arm)
# =============================================================================

def autoregressive_covariance(n_features: int, rho: float) -> np.ndarray:
    """AR(1) covariance matrix Σ[i,j] = ρ^|i−j|."""
    idx = np.arange(n_features)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_synthetic_dataset(config: SimulationConfig) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate features, three potential outcomes, observed outcome, and ground-truth ITEs.

    DGP (sister's specification):
        b(X)        = 0.50·X̄_EEG[8:11] − 0.30·X̄_HRV[1:3] + 0.15·X̄_PSY[1:3]
                      + 0.05·Age_norm + 0.03·Sex
        τ_NF(X)     = 0.80·X̄_EEG[8:11] + 0.30·X̄_PSY[1:3] + η_NF
        τ_tVNS(X)   = −0.70·X̄_HRV[1:3] + 0.20·X̄_PSY[1:3] + η_tVNS
        Y(0)        = b(X) + ε_0                       (Sham)
        Y(1)        = b(X) + τ_tVNS(X) + ε_1           (tVNS)
        Y(2)        = b(X) + τ_NF(X) + ε_2             (NF)

    where η ~ N(0, σ_η²), ε ~ N(0, σ_ε²), Age_norm = (Age − 50)/15.
    Indices 8:11 (1-based 8–11 inclusive) → Python slice [7:11] over 0-indexed.
    Indices 1:3 (1-based 1–3 inclusive) → Python slice [0:3] over 0-indexed.
    """
    rng = np.random.default_rng(config.random_seed)
    n = config.n_samples

    # EEG: AR(1) Gaussian
    eeg_cov = autoregressive_covariance(config.n_eeg, rho=config.eeg_ar1_rho)
    eeg = rng.multivariate_normal(np.zeros(config.n_eeg), eeg_cov, n)

    # HRV: log-normal, then standardised
    hrv_raw = rng.lognormal(mean=0.0, sigma=config.hrv_lognormal_sigma, size=(n, config.n_hrv))
    hrv = (hrv_raw - hrv_raw.mean(axis=0)) / hrv_raw.std(axis=0)

    # PSY: Gaussian with compound-symmetry covariance
    psy_cov = config.psy_corr * np.ones((config.n_psych, config.n_psych)) + \
              (1.0 - config.psy_corr) * np.eye(config.n_psych)
    psy = rng.multivariate_normal(np.zeros(config.n_psych), psy_cov, n)

    # Demographic
    age = rng.normal(config.age_mean, config.age_sd, n)
    age_norm = (age - config.age_mean) / config.age_sd
    sex = rng.binomial(1, config.sex_prob, n).astype(float)

    # Assemble feature matrix
    x = np.hstack([eeg, hrv, psy, age_norm.reshape(-1, 1), sex.reshape(-1, 1)])
    feature_names = (
        [f"EEG_{i+1}" for i in range(config.n_eeg)]
        + [f"HRV_{i+1}" for i in range(config.n_hrv)]
        + [f"PSY_{i+1}" for i in range(config.n_psych)]
        + ["Age_norm", "Sex"]
    )

    # Generative drivers (1-based indices 8–11 for EEG, 1–3 for HRV, 1–3 for PSY)
    alpha_eeg = eeg[:, 7:11].mean(axis=1)        # X̄_EEG[8:11]
    hrv_index = hrv[:, 0:3].mean(axis=1)         # X̄_HRV[1:3]
    psy_mod = psy[:, 0:3].mean(axis=1)           # X̄_PSY[1:3]

    # Prognostic component
    b_x = (
        0.50 * alpha_eeg
        - 0.30 * hrv_index
        + 0.15 * psy_mod
        + 0.05 * age_norm
        + 0.03 * sex
    )

    # Conditional treatment-effect functions
    eta_nf = rng.normal(0.0, config.treatment_effect_noise_sd, n)
    eta_tvns = rng.normal(0.0, config.treatment_effect_noise_sd, n)
    tau_nf_x = 0.80 * alpha_eeg + 0.30 * psy_mod + eta_nf
    tau_tvns_x = -0.70 * hrv_index + 0.20 * psy_mod + eta_tvns

    # Outcome noise (independent per arm)
    eps_0 = rng.normal(0.0, config.outcome_noise_sd, n)
    eps_1 = rng.normal(0.0, config.outcome_noise_sd, n)
    eps_2 = rng.normal(0.0, config.outcome_noise_sd, n)

    # Three potential outcomes
    y_sham = b_x + eps_0
    y_tvns = b_x + tau_tvns_x + eps_1
    y_nf = b_x + tau_nf_x + eps_2

    # Stratified 1:1:1 assignment
    treatment = np.concatenate([
        np.full(config.n_per_arm, T_SHAM, dtype=int),
        np.full(config.n_per_arm, T_TVNS, dtype=int),
        np.full(config.n_per_arm, T_NF, dtype=int),
    ])
    rng.shuffle(treatment)

    # Observed outcome under consistency
    y_obs = np.where(
        treatment == T_NF, y_nf,
        np.where(treatment == T_TVNS, y_tvns, y_sham),
    )

    # Ground-truth ITEs (noisy unit-level realisations)
    tau_nf_true = y_nf - y_sham
    tau_tvns_true = y_tvns - y_sham
    tau_comp_true = y_nf - y_tvns

    # True optimal treatment per unit (among realised potential outcomes)
    pot = np.column_stack([y_sham, y_tvns, y_nf])
    optimal_arm = np.argmax(pot, axis=1)

    df = pd.DataFrame(x, columns=feature_names)
    df["Age"] = age
    df["Treatment"] = treatment
    df["Treatment_label"] = pd.Series(treatment).map(ARM_NAMES).values
    df["Outcome"] = y_obs
    df["Y_Sham"] = y_sham
    df["Y_tVNS"] = y_tvns
    df["Y_NF"] = y_nf
    df["True_tau_NF"] = tau_nf_true
    df["True_tau_tVNS"] = tau_tvns_true
    df["True_tau_comparative"] = tau_comp_true
    df["Optimal_arm"] = optimal_arm
    df["Optimal_arm_label"] = pd.Series(optimal_arm).map(ARM_NAMES).values

    arrays: Dict[str, np.ndarray] = {
        "X": x,
        "T": treatment,
        "Y_obs": y_obs,
        "Y_sham": y_sham,
        "Y_tvns": y_tvns,
        "Y_nf": y_nf,
        "tau_nf_true": tau_nf_true,
        "tau_tvns_true": tau_tvns_true,
        "tau_comp_true": tau_comp_true,
        "feature_names": np.array(feature_names, dtype=object),
    }
    return df, arrays


# =============================================================================
# Meta-learners (three-arm)
# =============================================================================

def make_preprocessor() -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_gb(config: ModelConfig, seed_offset: int = 0) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        loss="squared_error",
        random_state=config.random_seed + seed_offset,
    )


def s_learner_3arm(x_tr: np.ndarray, t_tr: np.ndarray, y_tr: np.ndarray,
                   x_te: np.ndarray, cfg: ModelConfig) -> Dict[str, np.ndarray]:
    """S-learner: single regressor on (X, T) → Y. Predict at all three T values."""
    x_aug = np.column_stack([x_tr, t_tr.astype(float)])
    model = make_gb(cfg, seed_offset=10)
    model.fit(x_aug, y_tr)
    n_te = x_te.shape[0]
    pred_sham = model.predict(np.column_stack([x_te, np.full(n_te, T_SHAM, dtype=float)]))
    pred_tvns = model.predict(np.column_stack([x_te, np.full(n_te, T_TVNS, dtype=float)]))
    pred_nf = model.predict(np.column_stack([x_te, np.full(n_te, T_NF, dtype=float)]))
    return {
        "tau_nf": pred_nf - pred_sham,
        "tau_tvns": pred_tvns - pred_sham,
        "tau_comp": pred_nf - pred_tvns,
        "mu_sham": pred_sham,
        "mu_tvns": pred_tvns,
        "mu_nf": pred_nf,
    }


def t_learner_3arm(x_tr: np.ndarray, t_tr: np.ndarray, y_tr: np.ndarray,
                   x_te: np.ndarray, cfg: ModelConfig) -> Dict[str, np.ndarray]:
    """T-learner: separate regressor per arm."""
    m_sham = make_gb(cfg, seed_offset=20)
    m_tvns = make_gb(cfg, seed_offset=21)
    m_nf = make_gb(cfg, seed_offset=22)
    m_sham.fit(x_tr[t_tr == T_SHAM], y_tr[t_tr == T_SHAM])
    m_tvns.fit(x_tr[t_tr == T_TVNS], y_tr[t_tr == T_TVNS])
    m_nf.fit(x_tr[t_tr == T_NF], y_tr[t_tr == T_NF])
    pred_sham = m_sham.predict(x_te)
    pred_tvns = m_tvns.predict(x_te)
    pred_nf = m_nf.predict(x_te)
    return {
        "tau_nf": pred_nf - pred_sham,
        "tau_tvns": pred_tvns - pred_sham,
        "tau_comp": pred_nf - pred_tvns,
        "mu_sham": pred_sham,
        "mu_tvns": pred_tvns,
        "mu_nf": pred_nf,
    }


def x_learner_3arm(x_tr: np.ndarray, t_tr: np.ndarray, y_tr: np.ndarray,
                   x_te: np.ndarray, cfg: ModelConfig
                   ) -> Tuple[Dict[str, np.ndarray], GradientBoostingRegressor, GradientBoostingRegressor]:
    """
    X-learner for three arms with Sham as common reference.

    Stage 1: outcome model on sham units only:  μ_Sham(x) ≈ E[Y | X=x, T=Sham].

    Stage 2: imputed treatment effects on the active-treatment side:
        For T=NF units:    D_i^NF   = Y_i − μ_Sham(X_i)
        For T=tVNS units:  D_i^tVNS = Y_i − μ_Sham(X_i)

    Stage 3: second-stage regressors:
        τ̂_NF(x)   = ĝ_NF(x)    fitted on (X_i, D_i^NF)   for T=NF units
        τ̂_tVNS(x) = ĝ_tVNS(x)  fitted on (X_i, D_i^tVNS) for T=tVNS units

    Returns the second-stage estimators (effect_nf, effect_tvns) so that
    SHAP can be computed on the trained models.
    """
    m_sham = make_gb(cfg, seed_offset=40)
    m_sham.fit(x_tr[t_tr == T_SHAM], y_tr[t_tr == T_SHAM])

    nf_mask = t_tr == T_NF
    tvns_mask = t_tr == T_TVNS
    d_nf = y_tr[nf_mask] - m_sham.predict(x_tr[nf_mask])
    d_tvns = y_tr[tvns_mask] - m_sham.predict(x_tr[tvns_mask])

    effect_nf = make_gb(cfg, seed_offset=50)
    effect_tvns = make_gb(cfg, seed_offset=51)
    effect_nf.fit(x_tr[nf_mask], d_nf)
    effect_tvns.fit(x_tr[tvns_mask], d_tvns)

    tau_nf = effect_nf.predict(x_te)
    tau_tvns = effect_tvns.predict(x_te)
    return {
        "tau_nf": tau_nf,
        "tau_tvns": tau_tvns,
        "tau_comp": tau_nf - tau_tvns,
    }, effect_nf, effect_tvns


# =============================================================================
# Metrics
# =============================================================================

def pehe(true_ite: np.ndarray, pred_ite: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred_ite - true_ite) ** 2)))


def ate_error(true_ite: np.ndarray, pred_ite: np.ndarray) -> float:
    return float(abs(np.mean(pred_ite) - np.mean(true_ite)))


def calibration_slope(true_ite: np.ndarray, pred_ite: np.ndarray) -> float:
    if np.std(pred_ite) < 1e-9:
        return float("nan")
    model = LinearRegression()
    model.fit(pred_ite.reshape(-1, 1), true_ite)
    return float(model.coef_[0])


def three_way_assignment_accuracy(
    tau_nf_pred: np.ndarray, tau_tvns_pred: np.ndarray,
    y_sham: np.ndarray, y_tvns: np.ndarray, y_nf: np.ndarray,
) -> float:
    """3-way assignment accuracy: predicted optimal arm matches realised optimal arm.

    Predicted arm: argmax over (0, τ̂_tVNS, τ̂_NF), where 0 stands for Sham (no effect).
        - Sham (0) if max(τ̂_NF, τ̂_tVNS) ≤ 0
        - tVNS (1) if τ̂_tVNS > τ̂_NF and τ̂_tVNS > 0
        - NF (2)   if τ̂_NF > τ̂_tVNS and τ̂_NF > 0
    Realised optimal arm: argmax over (Y(0), Y(1), Y(2)).
    """
    pred_effects = np.column_stack([np.zeros_like(tau_nf_pred), tau_tvns_pred, tau_nf_pred])
    pred_arm = np.argmax(pred_effects, axis=1)
    actual_outcomes = np.column_stack([y_sham, y_tvns, y_nf])
    optimal_arm = np.argmax(actual_outcomes, axis=1)
    return float(np.mean(pred_arm == optimal_arm))


def policy_metrics(
    tau_nf_pred: np.ndarray, tau_tvns_pred: np.ndarray,
    y_sham: np.ndarray, y_tvns: np.ndarray, y_nf: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Policy value, regret vs oracle, improvement over random 3-arm policy.

    Returns: (policy_value, oracle_value, regret, improvement_over_random)
    """
    pred_effects = np.column_stack([np.zeros_like(tau_nf_pred), tau_tvns_pred, tau_nf_pred])
    pred_arm = np.argmax(pred_effects, axis=1)
    actual_outcomes = np.column_stack([y_sham, y_tvns, y_nf])
    policy_outcome = actual_outcomes[np.arange(len(pred_arm)), pred_arm]
    optimal_outcome = np.max(actual_outcomes, axis=1)
    random_outcome = actual_outcomes.mean(axis=1)
    policy_value = float(np.mean(policy_outcome))
    oracle_value = float(np.mean(optimal_outcome))
    regret = float(oracle_value - policy_value)
    improvement_over_random = float(policy_value - np.mean(random_outcome))
    return policy_value, oracle_value, regret, improvement_over_random


def summarise(feature_set: str, model_name: str,
              preds: Dict[str, np.ndarray], arrays: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    tau_nf_pred = preds["tau_nf"]
    tau_tvns_pred = preds["tau_tvns"]
    tau_comp_pred = preds["tau_comp"]
    tau_nf_true = arrays["tau_nf_true"]
    tau_tvns_true = arrays["tau_tvns_true"]
    tau_comp_true = arrays["tau_comp_true"]
    y_sham = arrays["Y_sham"]
    y_tvns = arrays["Y_tvns"]
    y_nf = arrays["Y_nf"]
    pv, ov, regret, improv = policy_metrics(tau_nf_pred, tau_tvns_pred, y_sham, y_tvns, y_nf)
    return {
        "Feature_Set": feature_set,
        "Model": model_name,
        "PEHE_NF": pehe(tau_nf_true, tau_nf_pred),
        "PEHE_tVNS": pehe(tau_tvns_true, tau_tvns_pred),
        "PEHE_comparative": pehe(tau_comp_true, tau_comp_pred),
        "R2_NF": float(r2_score(tau_nf_true, tau_nf_pred)),
        "R2_tVNS": float(r2_score(tau_tvns_true, tau_tvns_pred)),
        "R2_comparative": float(r2_score(tau_comp_true, tau_comp_pred)),
        "ATE_error_NF": ate_error(tau_nf_true, tau_nf_pred),
        "ATE_error_tVNS": ate_error(tau_tvns_true, tau_tvns_pred),
        "ATE_error_comparative": ate_error(tau_comp_true, tau_comp_pred),
        "Assignment_Accuracy_3way": three_way_assignment_accuracy(
            tau_nf_pred, tau_tvns_pred, y_sham, y_tvns, y_nf),
        "Policy_Value": pv,
        "Oracle_Value": ov,
        "Regret": regret,
        "Policy_Improvement_Over_Random": improv,
        "Calibration_Slope_NF": calibration_slope(tau_nf_true, tau_nf_pred),
        "Calibration_Slope_tVNS": calibration_slope(tau_tvns_true, tau_tvns_pred),
        "Calibration_Slope_comparative": calibration_slope(tau_comp_true, tau_comp_pred),
    }


# =============================================================================
# Evaluation pipeline
# =============================================================================

def evaluate_feature_set(
    x: np.ndarray, treatment: np.ndarray, y_obs: np.ndarray,
    arrays: Dict[str, np.ndarray],
    feature_indices: List[int], feature_set_name: str,
    model_cfg: ModelConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, GradientBoostingRegressor]]:
    """5-fold stratified CV; aggregates out-of-fold predictions; returns metrics."""
    x_sub = x[:, feature_indices]
    skf = StratifiedKFold(n_splits=model_cfg.n_splits, shuffle=True,
                          random_state=model_cfg.random_seed)

    preds: Dict[str, Dict[str, np.ndarray]] = {
        "S-learner": {"tau_nf": np.zeros(x_sub.shape[0]), "tau_tvns": np.zeros(x_sub.shape[0]),
                      "tau_comp": np.zeros(x_sub.shape[0])},
        "T-learner": {"tau_nf": np.zeros(x_sub.shape[0]), "tau_tvns": np.zeros(x_sub.shape[0]),
                      "tau_comp": np.zeros(x_sub.shape[0])},
        "X-learner": {"tau_nf": np.zeros(x_sub.shape[0]), "tau_tvns": np.zeros(x_sub.shape[0]),
                      "tau_comp": np.zeros(x_sub.shape[0])},
    }

    final_x_models: Dict[str, GradientBoostingRegressor] = {}

    for train_idx, test_idx in skf.split(x_sub, treatment):
        x_tr_raw, x_te_raw = x_sub[train_idx], x_sub[test_idx]
        t_tr = treatment[train_idx]
        y_tr = y_obs[train_idx]

        pre = make_preprocessor()
        x_tr = pre.fit_transform(x_tr_raw)
        x_te = pre.transform(x_te_raw)

        s_out = s_learner_3arm(x_tr, t_tr, y_tr, x_te, model_cfg)
        t_out = t_learner_3arm(x_tr, t_tr, y_tr, x_te, model_cfg)
        x_out, eff_nf, eff_tvns = x_learner_3arm(x_tr, t_tr, y_tr, x_te, model_cfg)

        for k in ("tau_nf", "tau_tvns", "tau_comp"):
            preds["S-learner"][k][test_idx] = s_out[k]
            preds["T-learner"][k][test_idx] = t_out[k]
            preds["X-learner"][k][test_idx] = x_out[k]

        final_x_models = {"effect_nf": eff_nf, "effect_tvns": eff_tvns}

    rows = []
    for model_name in ("S-learner", "T-learner", "X-learner"):
        rows.append(summarise(feature_set_name, model_name, preds[model_name], arrays))
    return pd.DataFrame(rows), preds["X-learner"], final_x_models


# =============================================================================
# Bootstrap intervals
# =============================================================================

def bootstrap_ci(values_true: np.ndarray, values_pred: np.ndarray,
                 metric_fn, n_boot: int = 1000, seed: int = 42,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for paired (true, pred) metric."""
    rng = np.random.default_rng(seed)
    n = len(values_true)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        estimates[b] = metric_fn(values_true[idx], values_pred[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(metric_fn(values_true, values_pred)), float(lo), float(hi)


# =============================================================================
# Robustness analyses
# =============================================================================

def evaluate_outcome_noise_robustness(
    arrays: Dict[str, np.ndarray], model_cfg: ModelConfig,
    sigma_levels: List[float]) -> pd.DataFrame:
    """Re-run X-learner under added outcome noise. Multimodal configuration."""
    rng = np.random.default_rng(model_cfg.random_seed + 100)
    rows = []
    x = arrays["X"]
    treatment = arrays["T"]
    base_y = arrays["Y_obs"]
    for sigma in sigma_levels:
        if sigma == 0.0:
            y_noisy = base_y.copy()
        else:
            y_noisy = base_y + rng.normal(0.0, sigma, len(base_y))

        skf = StratifiedKFold(n_splits=model_cfg.n_splits, shuffle=True,
                              random_state=model_cfg.random_seed)
        tau_nf_oof = np.zeros(len(base_y))
        tau_tvns_oof = np.zeros(len(base_y))
        for tr_idx, te_idx in skf.split(x, treatment):
            pre = make_preprocessor()
            x_tr = pre.fit_transform(x[tr_idx])
            x_te = pre.transform(x[te_idx])
            x_out, _, _ = x_learner_3arm(x_tr, treatment[tr_idx], y_noisy[tr_idx], x_te, model_cfg)
            tau_nf_oof[te_idx] = x_out["tau_nf"]
            tau_tvns_oof[te_idx] = x_out["tau_tvns"]

        tau_comp_oof = tau_nf_oof - tau_tvns_oof
        rows.append({
            "Outcome_Noise_SD": sigma,
            "PEHE_NF": pehe(arrays["tau_nf_true"], tau_nf_oof),
            "PEHE_tVNS": pehe(arrays["tau_tvns_true"], tau_tvns_oof),
            "PEHE_comparative": pehe(arrays["tau_comp_true"], tau_comp_oof),
            "R2_NF": r2_score(arrays["tau_nf_true"], tau_nf_oof),
            "R2_tVNS": r2_score(arrays["tau_tvns_true"], tau_tvns_oof),
            "R2_comparative": r2_score(arrays["tau_comp_true"], tau_comp_oof),
            "Assignment_Accuracy_3way": three_way_assignment_accuracy(
                tau_nf_oof, tau_tvns_oof, arrays["Y_sham"], arrays["Y_tvns"], arrays["Y_nf"]),
        })
    return pd.DataFrame(rows)


def evaluate_missingness_robustness(
    arrays: Dict[str, np.ndarray], model_cfg: ModelConfig,
    missing_rates: List[float]) -> pd.DataFrame:
    """Re-run X-learner under MCAR feature missingness. Multimodal configuration."""
    rng = np.random.default_rng(model_cfg.random_seed + 200)
    rows = []
    x_base = arrays["X"]
    treatment = arrays["T"]
    y = arrays["Y_obs"]
    for rate in missing_rates:
        if rate == 0.0:
            x_use = x_base.copy()
        else:
            mask = rng.random(x_base.shape) < rate
            x_use = x_base.copy()
            x_use[mask] = np.nan

        skf = StratifiedKFold(n_splits=model_cfg.n_splits, shuffle=True,
                              random_state=model_cfg.random_seed)
        tau_nf_oof = np.zeros(len(y))
        tau_tvns_oof = np.zeros(len(y))
        for tr_idx, te_idx in skf.split(x_use, treatment):
            pre = make_preprocessor()
            x_tr = pre.fit_transform(x_use[tr_idx])
            x_te = pre.transform(x_use[te_idx])
            x_out, _, _ = x_learner_3arm(x_tr, treatment[tr_idx], y[tr_idx], x_te, model_cfg)
            tau_nf_oof[te_idx] = x_out["tau_nf"]
            tau_tvns_oof[te_idx] = x_out["tau_tvns"]

        tau_comp_oof = tau_nf_oof - tau_tvns_oof
        rows.append({
            "Missingness_Rate": rate,
            "PEHE_NF": pehe(arrays["tau_nf_true"], tau_nf_oof),
            "PEHE_tVNS": pehe(arrays["tau_tvns_true"], tau_tvns_oof),
            "PEHE_comparative": pehe(arrays["tau_comp_true"], tau_comp_oof),
            "R2_NF": r2_score(arrays["tau_nf_true"], tau_nf_oof),
            "R2_tVNS": r2_score(arrays["tau_tvns_true"], tau_tvns_oof),
            "R2_comparative": r2_score(arrays["tau_comp_true"], tau_comp_oof),
            "Assignment_Accuracy_3way": three_way_assignment_accuracy(
                tau_nf_oof, tau_tvns_oof, arrays["Y_sham"], arrays["Y_tvns"], arrays["Y_nf"]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# SHAP analysis (X-learner second-stage)
# =============================================================================

def compute_shap_rankings(arrays: Dict[str, np.ndarray], model_cfg: ModelConfig,
                          ) -> pd.DataFrame:
    """Train X-learner on full data; compute SHAP on both second-stage models."""
    try:
        import shap
    except ImportError:
        return pd.DataFrame()

    x = arrays["X"]
    treatment = arrays["T"]
    y = arrays["Y_obs"]
    feature_names = arrays["feature_names"]

    pre = make_preprocessor()
    x_std = pre.fit_transform(x)

    _, eff_nf, eff_tvns = x_learner_3arm(x_std, treatment, y, x_std, model_cfg)

    # Use shap.TreeExplainer for gradient-boosted trees
    explainer_nf = shap.TreeExplainer(eff_nf)
    explainer_tvns = shap.TreeExplainer(eff_tvns)
    phi_nf = explainer_nf.shap_values(x_std)
    phi_tvns = explainer_tvns.shap_values(x_std)

    mean_abs_nf = np.abs(phi_nf).mean(axis=0)
    mean_abs_tvns = np.abs(phi_tvns).mean(axis=0)
    phi_comp = phi_nf - phi_tvns
    mean_abs_comp = np.abs(phi_comp).mean(axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP_NF": mean_abs_nf,
        "MeanAbsSHAP_tVNS": mean_abs_tvns,
        "MeanAbsSHAP_comparative": mean_abs_comp,
    })
    df = df.sort_values("MeanAbsSHAP_comparative", ascending=False).reset_index(drop=True)
    df["Rank_comparative"] = df.index + 1
    return df


# =============================================================================
# Sanity checks
# =============================================================================

def sanity_checks(df: pd.DataFrame, arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Internal consistency checks on the generated dataset."""
    rows = []
    rows.append(("n_total", df.shape[0]))
    rows.append(("n_sham", int((df['Treatment'] == T_SHAM).sum())))
    rows.append(("n_tvns", int((df['Treatment'] == T_TVNS).sum())))
    rows.append(("n_nf", int((df['Treatment'] == T_NF).sum())))
    rows.append(("p_features", arrays["X"].shape[1]))
    rows.append(("mean_tau_NF", float(arrays["tau_nf_true"].mean())))
    rows.append(("mean_tau_tVNS", float(arrays["tau_tvns_true"].mean())))
    rows.append(("mean_tau_comp", float(arrays["tau_comp_true"].mean())))
    rows.append(("std_tau_NF", float(arrays["tau_nf_true"].std())))
    rows.append(("std_tau_tVNS", float(arrays["tau_tvns_true"].std())))
    rows.append(("std_tau_comp", float(arrays["tau_comp_true"].std())))
    rows.append(("corr_tau_NF_EEG_alpha", float(np.corrcoef(arrays["tau_nf_true"], arrays["X"][:, 7:11].mean(axis=1))[0, 1])))
    rows.append(("corr_tau_tVNS_HRV_index", float(np.corrcoef(arrays["tau_tvns_true"], arrays["X"][:, 20:23].mean(axis=1))[0, 1])))
    return pd.DataFrame(rows, columns=["Check", "Value"])


# =============================================================================
# Main pipeline
# =============================================================================

def run_simulation(out_dir: Path) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig()
    model_cfg = ModelConfig()

    df, arrays = generate_synthetic_dataset(sim_cfg)
    df.to_csv(out_dir / "synthetic_dataset_3arm.csv", index=False)

    # Define feature configurations
    n_eeg, n_hrv, n_psych = sim_cfg.n_eeg, sim_cfg.n_hrv, sim_cfg.n_psych
    n_total = arrays["X"].shape[1]
    eeg_idx = list(range(0, n_eeg))
    hrv_idx = list(range(n_eeg, n_eeg + n_hrv))
    psy_idx = list(range(n_eeg + n_hrv, n_eeg + n_hrv + n_psych))
    demo_idx = list(range(n_eeg + n_hrv + n_psych, n_total))
    multimodal_idx = list(range(n_total))

    configs = {
        "EEG-only": eeg_idx,
        "HRV-only": hrv_idx,
        "Psychometric-only": psy_idx,
        "Multimodal": multimodal_idx,
    }

    # Main performance table
    all_rows = []
    multimodal_x_preds = None
    multimodal_x_models = None
    for name, idx in configs.items():
        rows_df, x_preds, x_models = evaluate_feature_set(
            arrays["X"], arrays["T"], arrays["Y_obs"], arrays, idx, name, model_cfg)
        all_rows.append(rows_df)
        if name == "Multimodal":
            multimodal_x_preds = x_preds
            multimodal_x_models = x_models

    performance = pd.concat(all_rows, ignore_index=True)
    performance.to_csv(out_dir / "model_performance_summary_3arm.csv", index=False)

    # Bootstrap CIs for multimodal X-learner (R²-ITE recovery, each estimand)
    boot_rows = []
    for label, true_arr, pred_arr in [
        ("R2_NF", arrays["tau_nf_true"], multimodal_x_preds["tau_nf"]),
        ("R2_tVNS", arrays["tau_tvns_true"], multimodal_x_preds["tau_tvns"]),
        ("R2_comparative", arrays["tau_comp_true"], multimodal_x_preds["tau_comp"]),
    ]:
        est, lo, hi = bootstrap_ci(true_arr, pred_arr,
                                    metric_fn=lambda t, p: r2_score(t, p),
                                    n_boot=1000, seed=42)
        boot_rows.append({"Metric": label, "Estimate": est, "CI_low": lo, "CI_high": hi})
    for label, true_arr, pred_arr in [
        ("PEHE_NF", arrays["tau_nf_true"], multimodal_x_preds["tau_nf"]),
        ("PEHE_tVNS", arrays["tau_tvns_true"], multimodal_x_preds["tau_tvns"]),
        ("PEHE_comparative", arrays["tau_comp_true"], multimodal_x_preds["tau_comp"]),
    ]:
        est, lo, hi = bootstrap_ci(true_arr, pred_arr, metric_fn=pehe, n_boot=1000, seed=42)
        boot_rows.append({"Metric": label, "Estimate": est, "CI_low": lo, "CI_high": hi})

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "bootstrap_cis_xlearner_3arm.csv", index=False)

    # Robustness
    noise_df = evaluate_outcome_noise_robustness(arrays, model_cfg, [0.0, 0.25, 0.50, 1.00])
    noise_df.to_csv(out_dir / "noise_robustness_xlearner_3arm.csv", index=False)

    miss_df = evaluate_missingness_robustness(arrays, model_cfg, [0.0, 0.10, 0.20, 0.30])
    miss_df.to_csv(out_dir / "missingness_robustness_xlearner_3arm.csv", index=False)

    # SHAP
    shap_df = compute_shap_rankings(arrays, model_cfg)
    if not shap_df.empty:
        shap_df.to_csv(out_dir / "shap_rankings_3arm.csv", index=False)

    # Sanity checks
    sanity_df = sanity_checks(df, arrays)
    sanity_df.to_csv(out_dir / "sanity_checks_3arm.csv", index=False)

    return {
        "performance": performance,
        "bootstrap": boot_df,
        "noise": noise_df,
        "missingness": miss_df,
        "shap": shap_df,
        "sanity": sanity_df,
    }


if __name__ == "__main__":
    out = Path("/home/claude/sim_3arm_outputs")
    results = run_simulation(out)
    print("=== Performance ===")
    print(results["performance"].to_string(index=False))
    print()
    print("=== Bootstrap CIs (X-learner, multimodal) ===")
    print(results["bootstrap"].to_string(index=False))
    print()
    print("=== Noise robustness (X-learner, multimodal) ===")
    print(results["noise"].to_string(index=False))
    print()
    print("=== Missingness robustness (X-learner, multimodal) ===")
    print(results["missingness"].to_string(index=False))
    if not results["shap"].empty:
        print()
        print("=== Top-15 SHAP features (X-learner, comparative effect) ===")
        print(results["shap"].head(15).to_string(index=False))
    print()
    print("=== Sanity checks ===")
    print(results["sanity"].to_string(index=False))
