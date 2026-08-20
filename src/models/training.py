from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import text

from src.features.engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.registry import register_model_source
from src.utils.config import get_settings
from src.utils.db import ModelRun, SessionLocal
from src.utils.io import save_artifact, save_json_report, timestamp_slug
from src.utils.logging import configure_logging


logger = configure_logging("model_training")
LGBM_WARNING_PATTERN = "X does not have valid feature names, but LGBMClassifier was fitted with feature names"


@dataclass
class TrainingArtifacts:
    model_bundle_path: Path
    explainer_bundle_path: Path
    leaderboard: list[dict[str, Any]]
    champion_model: str
    model_version: str


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
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    skipped: list[str] = []

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=240,
            max_depth=4,
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
            n_estimators=300,
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


def _split_dataset(features: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = features["target_churn"].astype(int)
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X,
        y,
        test_size=settings.train_test_split,
        random_state=settings.model_random_state,
        stratify=stratify,
    )


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _cross_validate_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, folds: int) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{LGBM_WARNING_PATTERN}.*", category=UserWarning)
        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=None)
    return {
        "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
        "cv_average_precision": round(float(scores["test_average_precision"].mean()), 4),
        "cv_f1": round(float(scores["test_f1"].mean()), 4),
    }


def _evaluate_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{LGBM_WARNING_PATTERN}.*", category=UserWarning)
        probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }


def aggregate_feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "coef_"):
        raw_importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        raw_importance = np.abs(model.feature_importances_)
    else:
        raw_importance = np.zeros(len(feature_names))

    aggregated: dict[str, float] = {}
    for name, importance in zip(feature_names, raw_importance, strict=False):
        original = _resolve_original_feature(name)
        aggregated[original] = aggregated.get(original, 0.0) + float(importance)

    ranked = [
        {"feature": feature, "importance": round(score, 6)}
        for feature, score in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]
    return ranked


def _resolve_original_feature(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1)
    raw = transformed_name.replace("cat__", "", 1)
    for column in CATEGORICAL_FEATURES:
        if raw == column or raw.startswith(f"{column}_"):
            return column
    return raw


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


def train_models(features: pd.DataFrame) -> TrainingArtifacts:
    settings = get_settings()
    model_version = timestamp_slug()
    X_train, X_test, y_train, y_test = _split_dataset(features, settings)
    class_ratio = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    candidates, skipped = build_candidate_models(class_ratio=class_ratio, random_state=settings.model_random_state)

    leaderboard: list[dict[str, Any]] = []
    trained_models: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        holdout = _evaluate_holdout(pipeline, X_test, y_test)
        cross_val = _cross_validate_model(pipeline, X_train, y_train, settings.cv_folds)
        result = {
            "model_name": name,
            "model_version": model_version,
            **holdout,
            **cross_val,
        }
        leaderboard.append(result)
        logger.info("Trained %s with holdout AP %s", name, holdout["average_precision"])

    leaderboard.sort(key=lambda row: (row["average_precision"], row["roc_auc"]), reverse=True)
    champion_name = leaderboard[0]["model_name"]
    champion_model = trained_models[champion_name]
    explainer_model = trained_models["logistic_regression"]

    champion_importance = aggregate_feature_importance(champion_model)
    explainer_importance = aggregate_feature_importance(explainer_model)
    defaults = _training_defaults(features)
    baseline_stats = _numeric_baseline(features)

    model_bundle = {
        "model_name": champion_name,
        "model_version": model_version,
        "pipeline": champion_model,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "leaderboard": leaderboard,
        "global_feature_importance": champion_importance,
        "input_defaults": defaults,
        "training_baseline": baseline_stats,
        "risk_thresholds": {
            "medium": settings.risk_medium_threshold,
            "high": settings.risk_high_threshold,
        },
        "skipped_optional_models": skipped,
    }
    explainer_bundle = {
        "model_name": "logistic_regression",
        "model_version": model_version,
        "pipeline": explainer_model,
        "global_feature_importance": explainer_importance,
        "input_defaults": defaults,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    model_bundle_path = save_artifact(model_bundle, "champion_model.joblib")
    explainer_bundle_path = save_artifact(explainer_bundle, "explainer_model.joblib")
    save_json_report(
        {
            "model_version": model_version,
            "champion_model": champion_name,
            "leaderboard": leaderboard,
            "skipped_optional_models": skipped,
            "global_feature_importance": champion_importance[:15],
        },
        "model_leaderboard",
    )
    pd.DataFrame(champion_importance).to_csv(
        settings.report_dir / "feature_importance_latest.csv", index=False
    )

    session = SessionLocal()
    session.execute(text("UPDATE model_runs SET selected = 0"))
    for row in leaderboard:
        session.add(
            ModelRun(
                model_name=row["model_name"],
                artifact_path=str(model_bundle_path if row["model_name"] == champion_name else ""),
                metrics_json=row,
                selected=row["model_name"] == champion_name,
            )
        )
    session.commit()
    session.close()

    register_model_source(
        "demo",
        model_name=champion_name,
        model_version=model_version,
        artifact_name="champion_model.joblib",
        report_name="model_leaderboard.json",
        activate=True,
        extra_metadata={
            "prediction_report_name": "predictions_latest.csv",
            "feature_dataset_name": "customer_features_latest.parquet",
        },
    )

    return TrainingArtifacts(
        model_bundle_path=model_bundle_path,
        explainer_bundle_path=explainer_bundle_path,
        leaderboard=leaderboard,
        champion_model=champion_name,
        model_version=model_version,
    )
