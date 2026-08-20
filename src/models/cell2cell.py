from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.registry import register_model_source
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
    "handset_price_missing",
    "age_hh1_missing",
    "age_hh2_missing",
    "perc_change_minutes_missing",
    "perc_change_revenues_missing",
    "monthly_revenue_missing",
    "monthly_minutes_missing",
    "total_call_events",
    "customer_care_rate",
    "dropped_rate",
    "blocked_rate",
    "unanswered_rate",
    "retention_offer_rate",
    "active_sub_ratio",
    "handset_model_density",
    "minutes_per_month_service",
    "revenue_per_month_service",
    "high_usage_low_revenue_flag",
    "retention_but_no_accept_flag",
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
    "equipment_age_bucket",
    "service_area_prefix",
    "credit_prizm_combo",
    "marital_home_combo",
]

WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LGBM_WARNING_PATTERN = "X does not have valid feature names, but LGBMClassifier was fitted with feature names"


@dataclass
class Cell2CellArtifacts:
    model_artifact_path: Path
    report_path: Path
    model_name: str
    model_version: str
    metrics: list[dict[str, Any]]


class WeightedBlendModel:
    """Weighted ensemble over independently trained base estimators."""

    def __init__(
        self,
        estimator_defs: dict[str, Any],
        weights: dict[str, float],
        threshold: float,
    ) -> None:
        self.estimator_defs = estimator_defs
        self.weights = {name: weight for name, weight in weights.items() if weight > 0}
        self.threshold = threshold
        self.preprocessor: ColumnTransformer | None = None
        self.models: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WeightedBlendModel":
        self.preprocessor = build_preprocessor()
        transformed = self.preprocessor.fit_transform(X)
        self.models = {}
        for name, weight in self.weights.items():
            if weight <= 0:
                continue
            estimator = clone(self.estimator_defs[name])
            estimator.fit(transformed, y)
            self.models[name] = estimator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or not self.models:
            raise RuntimeError("WeightedBlendModel must be fitted before prediction.")
        transformed = self.preprocessor.transform(X)
        positive_probability = np.zeros(transformed.shape[0], dtype=float)
        for name, estimator in self.models.items():
            positive_probability += self.weights[name] * _predict_positive_proba_raw(estimator, transformed)
        return np.column_stack([1 - positive_probability, positive_probability])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


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
            n_estimators=600,
            max_depth=16,
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
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.3,
            min_child_weight=4,
            gamma=0.1,
            tree_method="hist",
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
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=60,
            reg_lambda=1.0,
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


def _predict_positive_proba_raw(estimator: Any, transformed: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{LGBM_WARNING_PATTERN}.*", category=UserWarning)
        return estimator.predict_proba(transformed)[:, 1]


def _predict_positive_proba(model: Any, X_eval: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{LGBM_WARNING_PATTERN}.*", category=UserWarning)
        return model.predict_proba(X_eval)[:, 1]


def _evaluate_from_probabilities(y_eval: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{LGBM_WARNING_PATTERN}.*", category=UserWarning)
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


def _aggregate_feature_importance_from_estimator(
    preprocessor: ColumnTransformer,
    estimator: Any,
    model_weight: float = 1.0,
) -> dict[str, float]:
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
        aggregated[original] = aggregated.get(original, 0.0) + float(importance) * model_weight
    return aggregated


def aggregate_feature_importance(model: Pipeline) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    aggregated = _aggregate_feature_importance_from_estimator(preprocessor, estimator)
    return [
        {"feature": feature, "importance": round(score, 6)}
        for feature, score in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]


def aggregate_blend_feature_importance(model: WeightedBlendModel) -> list[dict[str, Any]]:
    if model.preprocessor is None:
        return []
    aggregated: dict[str, float] = {}
    for name, estimator in model.models.items():
        partial = _aggregate_feature_importance_from_estimator(
            model.preprocessor,
            estimator,
            model_weight=model.weights[name],
        )
        for feature, importance in partial.items():
            aggregated[feature] = aggregated.get(feature, 0.0) + importance
    return [
        {"feature": feature, "importance": round(score, 6)}
        for feature, score in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]


def _search_best_blend(validation_predictions: dict[str, np.ndarray], y_valid: pd.Series) -> tuple[dict[str, float], np.ndarray, float]:
    model_names = list(validation_predictions)
    if len(model_names) == 1:
        name = model_names[0]
        probabilities = validation_predictions[name]
        return {name: 1.0}, probabilities, float(average_precision_score(y_valid, probabilities))

    best_weights: dict[str, float] | None = None
    best_probabilities: np.ndarray | None = None
    best_score = -np.inf

    for raw_weights in product(WEIGHT_GRID, repeat=len(model_names)):
        if not np.isclose(sum(raw_weights), 1.0):
            continue
        probabilities = np.zeros(len(y_valid), dtype=float)
        for name, weight in zip(model_names, raw_weights, strict=False):
            probabilities += validation_predictions[name] * weight
        score = float(average_precision_score(y_valid, probabilities))
        if score > best_score:
            best_score = score
            best_weights = dict(zip(model_names, raw_weights, strict=False))
            best_probabilities = probabilities

    if best_weights is None or best_probabilities is None:
        name = model_names[0]
        probabilities = validation_predictions[name]
        return {name: 1.0}, probabilities, float(average_precision_score(y_valid, probabilities))

    best_weights = {name: weight for name, weight in best_weights.items() if weight > 0}
    return best_weights, best_probabilities, best_score


def _optimize_threshold(y_valid: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_valid, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def _training_defaults(features: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        defaults[column] = float(features[column].median())
    for column in CATEGORICAL_FEATURES:
        defaults[column] = str(features[column].mode().iloc[0])
    return defaults


def _numeric_baseline(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline = {}
    for column in NUMERIC_FEATURES:
        series = features[column]
        baseline[column] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=0)), 4),
            "p10": round(float(series.quantile(0.10)), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "p50": round(float(series.quantile(0.50)), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "p90": round(float(series.quantile(0.90)), 4),
            "missing_rate": round(float(series.isna().mean()), 4),
        }
    return baseline


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
        "risk_segment": (
            "high_risk"
            if churn_probability >= 0.7
            else "medium_risk" if churn_probability >= 0.4 else "low_risk"
        ),
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
    validation_predictions_by_model: dict[str, np.ndarray] = {}
    for name, estimator in candidates.items():
        model = _build_pipeline(estimator)
        model.fit(X_train, y_train)
        candidate_models[name] = model
        probabilities = _predict_positive_proba(model, X_valid)
        validation_predictions_by_model[name] = probabilities
        result = {
            "model_name": name,
            "model_version": model_version,
            **_evaluate_from_probabilities(y_valid, probabilities),
            **_cross_validate(model, X_train, y_train, settings.cv_folds),
        }
        leaderboard.append(result)
        logger.info("Cell2Cell candidate %s AP=%s", name, result["average_precision"])

    leaderboard.sort(key=lambda row: (row["average_precision"], row["roc_auc"]), reverse=True)
    best_single = leaderboard[0]
    blend_weights, blend_probabilities, blend_ap = _search_best_blend(validation_predictions_by_model, y_valid)
    blend_threshold, blend_best_f1 = _optimize_threshold(y_valid, blend_probabilities)

    use_blend = len(blend_weights) > 1 and blend_ap >= best_single["average_precision"]
    if use_blend:
        champion_name = "weighted_blend"
        champion_probabilities = blend_probabilities
        decision_threshold = blend_threshold
        champion_metrics = _evaluate_from_probabilities(y_valid, champion_probabilities, threshold=decision_threshold)
        champion_metrics["optimized_f1"] = round(blend_best_f1, 4)
        champion_details = {
            "blend_weights": {name: round(weight, 4) for name, weight in blend_weights.items()},
            "decision_threshold": round(decision_threshold, 4),
        }
    else:
        champion_name = best_single["model_name"]
        champion_probabilities = validation_predictions_by_model[champion_name]
        decision_threshold, best_f1 = _optimize_threshold(y_valid, champion_probabilities)
        champion_metrics = _evaluate_from_probabilities(y_valid, champion_probabilities, threshold=decision_threshold)
        champion_metrics["optimized_f1"] = round(best_f1, 4)
        champion_details = {
            "blend_weights": {champion_name: 1.0},
            "decision_threshold": round(decision_threshold, 4),
        }

    if use_blend:
        final_model: WeightedBlendModel | Pipeline = WeightedBlendModel(
            estimator_defs=candidates,
            weights=blend_weights,
            threshold=decision_threshold,
        )
        final_model.fit(X, y)
        global_importance = aggregate_blend_feature_importance(final_model)
    else:
        final_model = _build_pipeline(candidates[champion_name])
        final_model.fit(X, y)
        global_importance = aggregate_feature_importance(final_model)

    artifact = {
        "model_name": champion_name,
        "model_version": model_version,
        "model_source": "kaggle_cell2cell",
        "pipeline": final_model,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "leaderboard": leaderboard,
        "global_feature_importance": global_importance,
        "skipped_optional_models": skipped,
        "decision_threshold": decision_threshold,
        "blend_weights": champion_details["blend_weights"],
        "input_defaults": _training_defaults(train_df),
        "training_baseline": _numeric_baseline(train_df),
    }
    artifact_path = save_artifact(artifact, "kaggle_cell2cell_model.joblib")

    report_payload = {
        "dataset_slug": "jpacse/datasets-for-churn-telecom",
        "model_version": model_version,
        "champion_model": champion_name,
        "leaderboard": leaderboard,
        "champion_validation_metrics": champion_metrics,
        "decision_threshold": round(decision_threshold, 4),
        "blend_weights": champion_details["blend_weights"],
        "skipped_optional_models": skipped,
        "global_feature_importance": global_importance[:20],
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_churn_rate": round(float(train_df["churn"].mean()), 4),
    }
    report_path = save_json_report(report_payload, "kaggle_cell2cell_model_report")

    register_model_source(
        "kaggle_cell2cell",
        model_name=champion_name,
        model_version=model_version,
        artifact_name="kaggle_cell2cell_model.joblib",
        report_name="kaggle_cell2cell_model_report.json",
        activate=True,
        extra_metadata={
            "prediction_report_name": "kaggle_cell2cell_holdout_predictions.csv",
            "validation_report_name": "kaggle_cell2cell_validation_predictions.csv",
            "feature_dataset_name": "kaggle_cell2cell_holdout_latest.parquet",
        },
    )

    holdout_features = holdout_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    holdout_scores = final_model.predict_proba(holdout_features)[:, 1]
    holdout_predictions = holdout_df[["customer_id"]].copy()
    holdout_predictions["churn_probability"] = holdout_scores.round(6)
    holdout_predictions["predicted_churn"] = (holdout_scores >= decision_threshold).astype(int)

    recommendation_rows = []
    for idx, probability in enumerate(holdout_scores):
        recommendation = recommend_action(holdout_df.iloc[idx], float(probability))
        recommendation_rows.append(recommendation)
    holdout_predictions = pd.concat([holdout_predictions, pd.DataFrame(recommendation_rows)], axis=1)

    validation_predictions = labeled.loc[X_valid.index, ["customer_id", "churn"]].copy()
    validation_predictions["churn_probability"] = champion_probabilities.round(6)
    validation_predictions["predicted_churn"] = (champion_probabilities >= decision_threshold).astype(int)

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
