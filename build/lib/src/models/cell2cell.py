from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import get_settings
from src.utils.io import save_artifact, save_json_report, timestamp_slug
from src.utils.logging import configure_logging


logger = configure_logging("cell2cell_model")

NUMERIC_FEATURES = [
    "monthly_revenue",
    "monthly_minutes",
    "total_recurring_charge",
    "director_assisted_calls",
    "overage_minutes",
    "roaming_calls",
    "perc_change_minutes",
    "perc_change_revenues",
    "dropped_calls",
    "blocked_calls",
    "unanswered_calls",
    "customer_care_calls",
    "threeway_calls",
    "received_calls",
    "outbound_calls",
    "inbound_calls",
    "peak_calls_in_out",
    "off_peak_calls_in_out",
    "dropped_blocked_calls",
    "call_forwarding_calls",
    "call_waiting_calls",
    "months_in_service",
    "unique_subs",
    "active_subs",
    "handsets",
    "handset_models",
    "current_equipment_days",
    "age_hh1",
    "age_hh2",
    "retention_calls",
    "retention_offers_accepted",
    "referrals_made_by_subscriber",
    "income_group",
    "adjustments_to_credit_rating",
    "handset_price",
    "monthly_margin",
    "call_problem_volume",
    "usage_mix_total",
    "overage_ratio",
    "call_problem_rate",
    "revenue_per_minute",
    "equipment_tenure_ratio",
    "retention_pressure_score",
    "household_age_gap",
    "high_value_customer",
    "high_call_problem_flag",
]

CATEGORICAL_FEATURES = [
    "children_in_hh",
    "handset_refurbished",
    "handset_web_capable",
    "truck_owner",
    "rv_owner",
    "homeownership",
    "buys_via_mail_order",
    "responds_to_mail_offers",
    "opt_out_mailings",
    "non_us_travel",
    "owns_computer",
    "has_credit_card",
    "new_cellphone_user",
    "not_new_cellphone_user",
    "owns_motorcycle",
    "made_call_to_retention_team",
    "credit_rating_grouped",
    "prizm_code_grouped",
    "occupation_grouped",
    "marital_status",
    "service_area_grouped",
    "service_tenure_band",
]


@dataclass
class Cell2CellArtifacts:
    model_artifact_path: Path
    report_path: Path
    model_name: str
    model_version: str
    metrics: list[dict[str, Any]]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models(class_ratio: float, random_state: int) -> tuple[dict[str, Any], list[str]]:
    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    skipped: list[str] = []

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=280,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            min_child_weight=2,
            random_state=random_state,
            eval_metric="logloss",
            scale_pos_weight=max(1.0, class_ratio),
        )
    except Exception as error:
        logger.warning("Skipping xgboost because it is not runnable in this environment: %s", error)
        skipped.append("xgboost")

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    except Exception as error:
        logger.warning("Skipping lightgbm because it is not runnable in this environment: %s", error)
        skipped.append("lightgbm")

    return candidates, skipped


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", clone(estimator)),
        ]
    )


def _evaluate(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_eval, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_eval, probabilities)), 4),
        "precision": round(float(precision_score(y_eval, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_eval, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_eval, predictions, zero_division=0)), 4),
    }


def _cross_validate(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, folds: int) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
    }
    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv)
    return {
        "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
        "cv_average_precision": round(float(scores["test_average_precision"].mean()), 4),
        "cv_f1": round(float(scores["test_f1"].mean()), 4),
    }


def _resolve_original_feature(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)
    raw = transformed_name.replace("cat__", "", 1)
    for column in CATEGORICAL_FEATURES:
        if raw == column or raw.startswith(f"{column}_"):
            return column
    return raw


def aggregate_feature_importance(model: Pipeline) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "coef_"):
        raw_importance = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        raw_importance = np.abs(estimator.feature_importances_)
    else:
        raw_importance = np.zeros(len(feature_names))

    aggregated: dict[str, float] = {}
    for name, importance in zip(feature_names, raw_importance, strict=False):
        original = _resolve_original_feature(name)
        aggregated[original] = aggregated.get(original, 0.0) + float(importance)

    return [
        {"feature": feature, "importance": round(score, 6)}
        for feature, score in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]


def recommend_action(row: pd.Series, churn_probability: float) -> dict[str, Any]:
    if churn_probability >= 0.7:
        if row.get("retention_pressure_score", 0) > 0:
            action = "priority_retention_callback"
            rationale = "Customer already shows retention friction and remains high risk."
        elif row.get("monthly_margin", 0) > 20:
            action = "discount_offer"
            rationale = "High-value customer with elevated churn probability."
        else:
            action = "service_quality_outreach"
            rationale = "High churn risk with likely service-quality friction."
    elif churn_probability >= 0.4:
        if row.get("high_call_problem_flag", 0) == 1:
            action = "proactive_support_checkin"
            rationale = "Call-quality issues and medium churn risk."
        else:
            action = "loyalty_nudge"
            rationale = "Medium churn risk; reinforce plan value before churn accelerates."
    else:
        action = "upsell_or_cross_sell"
        rationale = "Low churn risk; focus on account expansion."

    return {
        "recommended_action": action,
        "action_rationale": rationale,
        "risk_segment": "high_risk" if churn_probability >= 0.7 else "medium_risk" if churn_probability >= 0.4 else "low_risk",
    }


def train_and_score_cell2cell(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> tuple[Cell2CellArtifacts, pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    model_version = timestamp_slug()

    labeled = train_df.copy()
    y = labeled["churn"].astype(int)
    X = labeled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=settings.train_test_split,
        random_state=settings.model_random_state,
        stratify=y,
    )

    class_ratio = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    candidates, skipped = build_candidate_models(class_ratio=class_ratio, random_state=settings.model_random_state)

    leaderboard: list[dict[str, Any]] = []
    candidate_models: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        model = _build_pipeline(estimator)
        model.fit(X_train, y_train)
        candidate_models[name] = model
        result = {
            "model_name": name,
            "model_version": model_version,
            **_evaluate(model, X_valid, y_valid),
            **_cross_validate(model, X_train, y_train, settings.cv_folds),
        }
        leaderboard.append(result)
        logger.info("Cell2Cell candidate %s AP=%s", name, result["average_precision"])

    leaderboard.sort(key=lambda row: (row["average_precision"], row["roc_auc"]), reverse=True)
    champion_name = leaderboard[0]["model_name"]

    champion_model = _build_pipeline(candidates[champion_name])
    champion_model.fit(X, y)
    global_importance = aggregate_feature_importance(champion_model)

    artifact = {
        "model_name": champion_name,
        "model_version": model_version,
        "pipeline": champion_model,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "leaderboard": leaderboard,
        "global_feature_importance": global_importance,
        "skipped_optional_models": skipped,
    }
    artifact_path = save_artifact(artifact, "kaggle_cell2cell_model.joblib")

    report_payload = {
        "dataset_slug": "jpacse/datasets-for-churn-telecom",
        "model_version": model_version,
        "champion_model": champion_name,
        "leaderboard": leaderboard,
        "skipped_optional_models": skipped,
        "global_feature_importance": global_importance[:20],
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_churn_rate": round(float(train_df["churn"].mean()), 4),
    }
    report_path = save_json_report(report_payload, "kaggle_cell2cell_model_report")

    holdout_features = holdout_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    holdout_scores = champion_model.predict_proba(holdout_features)[:, 1]
    holdout_predictions = holdout_df[["customer_id"]].copy()
    holdout_predictions["churn_probability"] = holdout_scores.round(6)
    holdout_predictions["predicted_churn"] = (holdout_scores >= 0.5).astype(int)

    recommendation_rows = []
    for idx, probability in enumerate(holdout_scores):
        recommendation = recommend_action(holdout_df.iloc[idx], float(probability))
        recommendation_rows.append(recommendation)
    holdout_predictions = pd.concat([holdout_predictions, pd.DataFrame(recommendation_rows)], axis=1)

    validation_scores = candidate_models[champion_name].predict_proba(X_valid)[:, 1]
    validation_predictions = labeled.loc[X_valid.index, ["customer_id", "churn"]].copy()
    validation_predictions["churn_probability"] = validation_scores.round(6)
    validation_predictions["predicted_churn"] = (validation_scores >= 0.5).astype(int)

    return (
        Cell2CellArtifacts(
            model_artifact_path=artifact_path,
            report_path=Path(report_path),
            model_name=champion_name,
            model_version=model_version,
            metrics=leaderboard,
        ),
        validation_predictions.reset_index(drop=True),
        holdout_predictions.reset_index(drop=True),
    )
