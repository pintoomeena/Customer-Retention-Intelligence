from __future__ import annotations

import ast
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.registry import resolve_model_source
from src.utils.config import get_settings
from src.utils.io import load_latest_dataframe


SOURCE_OPTIONS = {
    "auto": "Auto (active model source)",
    "demo": "Synthetic demo pipeline",
    "kaggle": "Kaggle Cell2Cell pipeline",
}


st.set_page_config(page_title="Churn Command Center", page_icon=":bar_chart:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(225, 196, 153, 0.22), transparent 28%),
            linear-gradient(180deg, #f6efe5 0%, #fffaf5 45%, #f3efe8 100%);
        color: #1f2937;
        font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }
    .block-container {
        padding-top: 1.25rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(146, 64, 14, 0.15);
        border-radius: 16px;
        padding: 0.8rem;
        box-shadow: 0 12px 30px rgba(120, 53, 15, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _latest_timestamp(paths: list[Path]) -> float:
    available = [path.stat().st_mtime for path in paths if path.exists()]
    return max(available) if available else 0.0


def _format_timestamp(timestamp: float) -> str:
    if not timestamp:
        return "Unknown"
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _empty_frame(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [])


def _stringify_customer_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_id" in df.columns:
        frame = df.copy()
        frame["customer_id"] = frame["customer_id"].astype(str)
        return frame
    return df


def _coalesce_column(frame: pd.DataFrame, primary: str, fallback: str | None = None, default: Any = "") -> pd.Series:
    if primary in frame.columns:
        series = frame[primary]
    elif fallback and fallback in frame.columns:
        series = frame[fallback]
    else:
        series = pd.Series([default] * len(frame), index=frame.index)
    return series.fillna(default)


def _normalize_demo_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    frame = _stringify_customer_ids(predictions)
    frame["recommended_action"] = _coalesce_column(frame, "recommended_action", fallback="action_type")
    frame["rationale"] = _coalesce_column(frame, "rationale")
    frame["channel"] = _coalesce_column(frame, "channel")
    frame["experiment_group"] = _coalesce_column(frame, "experiment_group", default="treatment")
    frame["status"] = _coalesce_column(frame, "status", default="queued")
    frame["actual_churn"] = frame.get("actual_churn", pd.Series(index=frame.index, dtype="float64"))
    return frame


def _normalize_kaggle_predictions(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    frame = _stringify_customer_ids(predictions)
    frame["recommended_action"] = _coalesce_column(frame, "recommended_action")
    frame["rationale"] = _coalesce_column(frame, "action_rationale")
    frame["action_type"] = frame["recommended_action"]
    frame["channel"] = "review_queue"
    frame["experiment_group"] = "not_applicable"
    frame["status"] = "proposed"
    if "churn" in features.columns:
        actuals = _stringify_customer_ids(features[["customer_id", "churn"]]).rename(columns={"churn": "actual_churn"})
        frame = frame.merge(actuals, on="customer_id", how="left")
    else:
        frame["actual_churn"] = pd.NA
    return frame


def _top_importance_frame(report: dict[str, Any], importance: pd.DataFrame) -> pd.DataFrame:
    if not importance.empty:
        return importance.copy()
    values = report.get("global_feature_importance", [])
    return pd.DataFrame(values) if values else _empty_frame(["feature", "importance"])


def _load_demo_source(settings) -> dict[str, Any] | None:
    features_path = settings.processed_data_dir / "customer_features_latest.parquet"
    predictions_path = settings.report_dir / "predictions_latest.csv"
    importance_path = settings.report_dir / "feature_importance_latest.csv"
    leaderboard_path = settings.report_dir / "model_leaderboard.json"
    source_paths = [features_path, predictions_path, importance_path, leaderboard_path]
    if not features_path.exists() or not predictions_path.exists():
        return None

    features = _stringify_customer_ids(load_latest_dataframe("customer_features"))
    predictions = _normalize_demo_predictions(
        pd.read_csv(predictions_path) if predictions_path.exists() else _empty_frame()
    )
    report = _read_json(leaderboard_path)
    importance = _top_importance_frame(
        report,
        pd.read_csv(importance_path) if importance_path.exists() else _empty_frame(["feature", "importance"]),
    )
    return {
        "key": "demo",
        "label": SOURCE_OPTIONS["demo"],
        "features": features,
        "predictions": predictions,
        "importance": importance,
        "report": report,
        "last_updated_ts": _latest_timestamp(source_paths),
        "last_updated_label": _format_timestamp(_latest_timestamp(source_paths)),
    }


def _load_kaggle_source(settings) -> dict[str, Any] | None:
    features_path = settings.processed_data_dir / "kaggle_cell2cell_holdout_latest.parquet"
    predictions_path = settings.report_dir / "kaggle_cell2cell_holdout_predictions.csv"
    report_path = settings.report_dir / "kaggle_cell2cell_model_report.json"
    source_paths = [features_path, predictions_path, report_path]
    if not features_path.exists() or not predictions_path.exists() or not report_path.exists():
        return None

    features = _stringify_customer_ids(pd.read_parquet(features_path))
    predictions = _normalize_kaggle_predictions(pd.read_csv(predictions_path), features)
    report = _read_json(report_path)
    importance = _top_importance_frame(report, _empty_frame(["feature", "importance"]))
    return {
        "key": "kaggle",
        "label": SOURCE_OPTIONS["kaggle"],
        "features": features,
        "predictions": predictions,
        "importance": importance,
        "report": report,
        "last_updated_ts": _latest_timestamp(source_paths),
        "last_updated_label": _format_timestamp(_latest_timestamp(source_paths)),
    }


def _available_sources() -> dict[str, dict[str, Any]]:
    settings = get_settings()
    sources: dict[str, dict[str, Any]] = {}
    for loader in (_load_demo_source, _load_kaggle_source):
        source = loader(settings)
        if source:
            sources[source["key"]] = source
    return sources


def _resolve_source(preference: str, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not sources:
        raise FileNotFoundError("No dashboard data sources are available.")
    if preference != "auto":
        if preference not in sources:
            raise FileNotFoundError(f"The '{preference}' dashboard source has not been generated yet.")
        return sources[preference]
    try:
        active_source = resolve_model_source("auto")
        if active_source in sources:
            return sources[active_source]
    except FileNotFoundError:
        pass
    return max(sources.values(), key=lambda source: source["last_updated_ts"])


def _safe_quantile(df: pd.DataFrame, column: str, quantile: float) -> float | None:
    if column not in df.columns:
        return None
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.quantile(quantile))


def _safe_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(row: pd.Series, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return str(value)


def _parse_explanation_payload(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (SyntaxError, ValueError, TypeError):
            return []


def _demo_rule_explanation(row: pd.Series, features: pd.DataFrame) -> dict[str, Any]:
    recency_high = _safe_quantile(features, "recency_days", 0.75) or 14
    sessions_low = _safe_quantile(features, "sessions_last_30d", 0.25) or 6
    adoption_low = _safe_quantile(features, "feature_adoption_ratio", 0.25) or 0.35
    tickets_high = _safe_quantile(features, "tickets_90d", 0.75) or 2

    risk_factors: list[dict[str, Any]] = []
    protective_factors: list[dict[str, Any]] = []

    if _safe_value(row, "recency_days") >= recency_high:
        risk_factors.append(
            {
                "feature": "recency_days",
                "message": f"No meaningful activity for {int(round(_safe_value(row, 'recency_days')))} days is a strong churn signal.",
            }
        )
    if _safe_value(row, "sessions_last_30d") <= sessions_low:
        risk_factors.append(
            {
                "feature": "sessions_last_30d",
                "message": f"Only {int(round(_safe_value(row, 'sessions_last_30d')))} sessions in the last 30 days points to weak engagement.",
            }
        )
    if _safe_value(row, "feature_adoption_ratio") <= adoption_low:
        risk_factors.append(
            {
                "feature": "feature_adoption_ratio",
                "message": f"Feature adoption is only {_safe_value(row, 'feature_adoption_ratio'):.0%}, leaving the account under-embedded.",
            }
        )
    if _safe_value(row, "tickets_90d") >= tickets_high:
        risk_factors.append(
            {
                "feature": "tickets_90d",
                "message": f"{int(round(_safe_value(row, 'tickets_90d')))} support tickets in 90 days suggest unresolved friction.",
            }
        )
    if _safe_value(row, "activity_trend_slope") < 0:
        risk_factors.append(
            {
                "feature": "activity_trend_slope",
                "message": f"Recent activity is trending down with slope {_safe_value(row, 'activity_trend_slope'):.2f}.",
            }
        )

    if _safe_value(row, "engagement_score") >= (_safe_quantile(features, "engagement_score", 0.75) or 35):
        protective_factors.append(
            {
                "feature": "engagement_score",
                "message": f"Engagement remains healthy at {_safe_value(row, 'engagement_score'):.1f}.",
            }
        )
    if _safe_value(row, "feature_adoption_ratio") >= (_safe_quantile(features, "feature_adoption_ratio", 0.75) or 0.65):
        protective_factors.append(
            {
                "feature": "feature_adoption_ratio",
                "message": f"Feature adoption is healthy at {_safe_value(row, 'feature_adoption_ratio'):.0%}.",
            }
        )
    if _safe_value(row, "avg_ticket_csat") >= (_safe_quantile(features, "avg_ticket_csat", 0.75) or 4.0):
        protective_factors.append(
            {
                "feature": "avg_ticket_csat",
                "message": f"Support satisfaction is strong at {_safe_value(row, 'avg_ticket_csat'):.1f}/5.",
            }
        )
    if _safe_value(row, "avg_payment_delay_days") <= (_safe_quantile(features, "avg_payment_delay_days", 0.25) or 1.0):
        protective_factors.append(
            {
                "feature": "avg_payment_delay_days",
                "message": f"Payments are mostly on time with {_safe_value(row, 'avg_payment_delay_days'):.1f} delayed days on average.",
            }
        )

    return {
        "risk_factors": risk_factors[:3],
        "protective_factors": protective_factors[:3],
        "mode": "rule_based",
    }


def _kaggle_rule_explanation(row: pd.Series, features: pd.DataFrame) -> dict[str, Any]:
    risk_factors: list[dict[str, Any]] = []
    protective_factors: list[dict[str, Any]] = []

    if _safe_value(row, "perc_change_minutes") < -50:
        risk_factors.append(
            {
                "feature": "perc_change_minutes",
                "message": f"Usage dropped by {abs(_safe_value(row, 'perc_change_minutes')):.0f} minutes versus the prior period.",
            }
        )
    if _safe_value(row, "perc_change_revenues") < -10:
        risk_factors.append(
            {
                "feature": "perc_change_revenues",
                "message": f"Revenue declined by {abs(_safe_value(row, 'perc_change_revenues')):.0f}, which often precedes telecom churn.",
            }
        )
    if _safe_value(row, "call_problem_rate") >= (_safe_quantile(features, "call_problem_rate", 0.75) or 0.12):
        risk_factors.append(
            {
                "feature": "call_problem_rate",
                "message": f"Call-quality friction is elevated with a problem rate of {_safe_value(row, 'call_problem_rate'):.2%}.",
            }
        )
    if _safe_value(row, "retention_pressure_score") > 0:
        risk_factors.append(
            {
                "feature": "retention_pressure_score",
                "message": "The account has already shown retention pressure, including prior save attempts or offer friction.",
            }
        )
    if _safe_value(row, "current_equipment_days") >= (_safe_quantile(features, "current_equipment_days", 0.75) or 600):
        risk_factors.append(
            {
                "feature": "current_equipment_days",
                "message": f"The current device has been in use for {int(round(_safe_value(row, 'current_equipment_days')))} days, a common renewal-risk pattern.",
            }
        )
    if _safe_value(row, "monthly_margin") <= (_safe_quantile(features, "monthly_margin", 0.25) or 0):
        risk_factors.append(
            {
                "feature": "monthly_margin",
                "message": f"Monthly margin is thin at {_safe_value(row, 'monthly_margin'):.2f}, limiting value perception.",
            }
        )

    if _safe_value(row, "months_in_service") >= (_safe_quantile(features, "months_in_service", 0.75) or 50):
        protective_factors.append(
            {
                "feature": "months_in_service",
                "message": f"The subscriber has {int(round(_safe_value(row, 'months_in_service')))} months of tenure, which usually supports retention.",
            }
        )
    if _safe_value(row, "monthly_revenue") >= (_safe_quantile(features, "monthly_revenue", 0.75) or 60):
        protective_factors.append(
            {
                "feature": "monthly_revenue",
                "message": f"Monthly revenue is strong at {_safe_value(row, 'monthly_revenue'):.2f}, which often makes rescue economics worthwhile.",
            }
        )
    if _safe_value(row, "call_problem_rate") <= (_safe_quantile(features, "call_problem_rate", 0.25) or 0.03):
        protective_factors.append(
            {
                "feature": "call_problem_rate",
                "message": f"Call quality is comparatively stable with only {_safe_value(row, 'call_problem_rate'):.2%} problematic events.",
            }
        )
    if _safe_value(row, "active_sub_ratio") >= 1.0:
        protective_factors.append(
            {
                "feature": "active_sub_ratio",
                "message": "All known subscriptions in the household remain active, which reduces immediate churn pressure.",
            }
        )

    return {
        "risk_factors": risk_factors[:3],
        "protective_factors": protective_factors[:3],
        "mode": "rule_based",
    }


def _build_customer_explanation(
    bundle: dict[str, Any],
    row: pd.Series,
    prediction_row: pd.Series | None,
) -> dict[str, Any]:
    if prediction_row is not None and "explanation_json" in prediction_row.index:
        parsed = _parse_explanation_payload(prediction_row.get("explanation_json"))
        if parsed:
            fallback = _demo_rule_explanation(row, bundle["features"])
            return {
                "risk_factors": parsed[:3],
                "protective_factors": fallback["protective_factors"],
                "mode": "saved_model_explanation",
            }
    if bundle["key"] == "demo":
        return _demo_rule_explanation(row, bundle["features"])
    return _kaggle_rule_explanation(row, bundle["features"])


def _customer_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    features = bundle["features"].copy()
    predictions = bundle["predictions"].copy()
    if predictions.empty:
        return features
    return features.merge(predictions, on="customer_id", how="left")


@st.cache_data(show_spinner=False, ttl=60)
def load_data(source_preference: str) -> dict[str, Any]:
    sources = _available_sources()
    selected = _resolve_source(source_preference, sources)
    return {
        "selected": selected,
        "available": {key: source["label"] for key, source in sources.items()},
    }


def render_source_summary(bundle: dict[str, Any], requested_source: str):
    st.sidebar.caption(f"Requested source: {SOURCE_OPTIONS[requested_source]}")
    st.sidebar.caption(f"Active source: {bundle['label']}")
    st.sidebar.caption(f"Last updated: {bundle['last_updated_label']}")
    report = bundle["report"]
    if report.get("champion_model"):
        st.sidebar.caption(f"Champion model: {report['champion_model']}")
    if report.get("model_version"):
        st.sidebar.caption(f"Model version: {report['model_version']}")
    st.sidebar.info(
        "The dashboard reads persisted reports and processed datasets. "
        "Run a pipeline, then click `Reload latest files` to refresh immediately."
    )


def render_overview(bundle: dict[str, Any]):
    features = bundle["features"]
    predictions = bundle["predictions"]
    report = bundle["report"]

    st.title("Churn Command Center")
    st.caption("Operational churn monitoring, explanation, and retention decisions.")

    total_users = len(features)
    high_risk = (
        int((predictions["risk_segment"] == "high_risk").sum())
        if not predictions.empty and "risk_segment" in predictions.columns
        else 0
    )
    avg_score = (
        float(predictions["churn_probability"].mean())
        if not predictions.empty and "churn_probability" in predictions.columns
        else 0.0
    )

    observed = None
    if bundle["key"] == "demo" and "target_churn" in features.columns:
        observed = float(features["target_churn"].mean())
    elif bundle["key"] == "kaggle":
        observed = report.get("train_churn_rate")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{total_users:,}")
    col2.metric(
        "Observed Churn Rate" if bundle["key"] == "demo" else "Training Churn Rate",
        f"{observed:.1%}" if observed is not None else "N/A",
    )
    col3.metric("High-Risk Users", f"{high_risk:,}")
    col4.metric("Avg Predicted Risk", f"{avg_score:.1%}")

    if bundle["key"] == "demo":
        by_plan = features.groupby("plan", dropna=False)["target_churn"].mean().reset_index()
        plan_fig = px.bar(
            by_plan,
            x="plan",
            y="target_churn",
            color="target_churn",
            color_continuous_scale="OrRd",
            title="Churn Rate by Plan",
            labels={"target_churn": "churn_rate"},
        )
        plan_fig.update_layout(coloraxis_showscale=False)

        cohort = (
            features.groupby(["cohort_month", "plan"], dropna=False)["target_churn"]
            .mean()
            .reset_index()
            .rename(columns={"target_churn": "churn_rate"})
        )
        cohort_fig = px.density_heatmap(
            cohort,
            x="cohort_month",
            y="plan",
            z="churn_rate",
            color_continuous_scale="YlOrRd",
            title="Cohort Churn Heatmap",
        )

        left, right = st.columns([1, 1.1])
        left.plotly_chart(plan_fig, width="stretch")
        right.plotly_chart(cohort_fig, width="stretch")

        segment = (
            features.groupby("customer_segment", dropna=False)[
                ["engagement_score", "target_churn", "sessions_last_30d"]
            ]
            .mean()
            .reset_index()
        )
        segment_fig = px.scatter(
            segment,
            x="sessions_last_30d",
            y="engagement_score",
            size="target_churn",
            color="customer_segment",
            title="Segment Behavior vs Churn",
        )
        st.plotly_chart(segment_fig, width="stretch")
        return

    merged = _customer_frame(bundle)
    risk_counts = (
        predictions.groupby("risk_segment", dropna=False).size().reset_index(name="customers").sort_values("customers")
    )
    risk_fig = px.bar(
        risk_counts,
        x="customers",
        y="risk_segment",
        orientation="h",
        color="risk_segment",
        title="Risk Segment Distribution",
        color_discrete_map={
            "low_risk": "#65a30d",
            "medium_risk": "#d97706",
            "high_risk": "#dc2626",
        },
    )

    histogram = px.histogram(
        predictions,
        x="churn_probability",
        nbins=30,
        title="Predicted Churn Probability Distribution",
        color_discrete_sequence=["#b45309"],
    )

    left, right = st.columns([1, 1.1])
    left.plotly_chart(risk_fig, width="stretch")
    right.plotly_chart(histogram, width="stretch")

    if {"service_tenure_band", "churn_probability"}.issubset(merged.columns):
        tenure = (
            merged.groupby("service_tenure_band", dropna=False)["churn_probability"]
            .mean()
            .reset_index()
            .sort_values("churn_probability", ascending=False)
        )
        tenure_fig = px.bar(
            tenure,
            x="service_tenure_band",
            y="churn_probability",
            color="churn_probability",
            color_continuous_scale="OrRd",
            title="Average Predicted Risk by Service Tenure Band",
        )
        tenure_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(tenure_fig, width="stretch")

    if {"monthly_margin", "call_problem_rate", "churn_probability"}.issubset(merged.columns):
        scatter = px.scatter(
            merged.sample(min(len(merged), 4000), random_state=42),
            x="monthly_margin",
            y="call_problem_rate",
            color="risk_segment",
            size="churn_probability",
            hover_data=["customer_id", "recommended_action"],
            title="Margin vs Call Friction",
        )
        st.plotly_chart(scatter, width="stretch")


def render_customer_explorer(bundle: dict[str, Any]):
    merged = _customer_frame(bundle)
    st.title("Customer Explorer")
    if merged.empty:
        st.info("Run a pipeline first so the dashboard has customers to explore.")
        return

    if "churn_probability" in merged.columns:
        options = merged.sort_values("churn_probability", ascending=False, na_position="last")["customer_id"].tolist()
    else:
        options = sorted(merged["customer_id"].tolist())
    customer_id = st.selectbox("Select customer", options=options)
    row = merged.loc[merged["customer_id"] == customer_id].iloc[0]
    probability = _safe_value(row, "churn_probability")
    explanation = _build_customer_explanation(bundle, row, row)

    score_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": f"Churn Probability for {customer_id}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#b45309"},
                "steps": [
                    {"range": [0, 40], "color": "#d1fae5"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#fecaca"},
                ],
            },
        )
    )
    st.plotly_chart(score_fig, width="stretch")

    metrics = st.columns(4)
    metrics[0].metric("Risk Segment", _safe_text(row, "risk_segment", "unknown").replace("_", " ").title())
    if bundle["key"] == "demo":
        metrics[1].metric("Engagement Score", f"{_safe_value(row, 'engagement_score'):.1f}")
        metrics[2].metric("Last Activity", f"{int(round(_safe_value(row, 'recency_days')))} days")
    else:
        metrics[1].metric("Monthly Revenue", f"{_safe_value(row, 'monthly_revenue'):.2f}")
        metrics[2].metric("Service Tenure", f"{int(round(_safe_value(row, 'months_in_service')))} months")
    metrics[3].metric("Recommended Action", _safe_text(row, "recommended_action", "review").replace("_", " ").title())

    rationale = _safe_text(row, "rationale")
    if rationale:
        st.caption(f"Action rationale: {rationale}")

    st.subheader("Why This Customer May Churn")
    if explanation["risk_factors"]:
        for factor in explanation["risk_factors"]:
            st.write(f"- {factor['message']}")
    else:
        st.write("- No high-confidence risk drivers were available in the persisted outputs.")

    st.subheader("Protective Factors")
    if explanation["protective_factors"]:
        for factor in explanation["protective_factors"]:
            st.write(f"- {factor['message']}")
    else:
        st.write("- No strong protective factors were detected for this customer.")

    if bundle["key"] == "demo":
        snapshot = {
            "plan": _safe_text(row, "plan"),
            "customer_segment": _safe_text(row, "customer_segment"),
            "sessions_last_30d": int(round(_safe_value(row, "sessions_last_30d"))),
            "feature_adoption_ratio": f"{_safe_value(row, 'feature_adoption_ratio'):.0%}",
            "tickets_90d": int(round(_safe_value(row, "tickets_90d"))),
        }
    else:
        snapshot = {
            "credit_rating": _safe_text(row, "credit_rating_grouped"),
            "service_tenure_band": _safe_text(row, "service_tenure_band"),
            "call_problem_rate": f"{_safe_value(row, 'call_problem_rate'):.2%}",
            "monthly_margin": f"{_safe_value(row, 'monthly_margin'):.2f}",
            "equipment_age_bucket": _safe_text(row, "equipment_age_bucket"),
        }
    st.subheader("Customer Snapshot")
    st.dataframe(pd.DataFrame([snapshot]), width="stretch", hide_index=True)


def render_insights(bundle: dict[str, Any]):
    features = bundle["features"]
    importance = bundle["importance"]
    report = bundle["report"]

    st.title("Insights")

    metrics = report.get("champion_validation_metrics", {})
    if metrics:
        cols = st.columns(4)
        cols[0].metric("Validation ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        cols[1].metric("Validation AP", f"{metrics.get('average_precision', 0):.4f}")
        cols[2].metric("Validation F1", f"{metrics.get('f1', 0):.4f}")
        cols[3].metric("Decision Threshold", f"{report.get('decision_threshold', 0):.4f}")

    if not importance.empty:
        fig_importance = px.bar(
            importance.head(12),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="sunsetdark",
            title="Global Feature Importance",
        )
        fig_importance.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig_importance, width="stretch")

    if bundle["key"] == "demo":
        trend = (
            features.groupby("cohort_month", dropna=False)[
                ["target_churn", "engagement_score", "feature_adoption_ratio"]
            ]
            .mean()
            .reset_index()
        )
        fig_trend = px.line(
            trend,
            x="cohort_month",
            y=["target_churn", "engagement_score", "feature_adoption_ratio"],
            markers=True,
            title="Churn, Engagement, and Adoption Trend",
        )
        st.plotly_chart(fig_trend, width="stretch")
    else:
        merged = _customer_frame(bundle)
        if {"credit_rating_grouped", "service_tenure_band", "churn_probability"}.issubset(merged.columns):
            slice_frame = (
                merged.groupby(["credit_rating_grouped", "service_tenure_band"], dropna=False)["churn_probability"]
                .mean()
                .reset_index()
            )
            heatmap = px.density_heatmap(
                slice_frame,
                x="service_tenure_band",
                y="credit_rating_grouped",
                z="churn_probability",
                color_continuous_scale="YlOrRd",
                title="Risk Heatmap by Credit Rating and Tenure Band",
            )
            st.plotly_chart(heatmap, width="stretch")

        if "recommended_action" in merged.columns:
            actions = (
                merged.groupby("recommended_action", dropna=False)
                .size()
                .reset_index(name="customers")
                .sort_values("customers", ascending=False)
            )
            action_fig = px.bar(
                actions.head(10),
                x="customers",
                y="recommended_action",
                orientation="h",
                color="customers",
                color_continuous_scale="OrRd",
                title="Top Recommended Actions",
            )
            action_fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(action_fig, width="stretch")

    leaderboard_rows = report.get("leaderboard", [])
    if leaderboard_rows:
        st.subheader("Model Leaderboard")
        st.dataframe(pd.DataFrame(leaderboard_rows), width="stretch", hide_index=True)


def render_action_center(bundle: dict[str, Any]):
    predictions = bundle["predictions"]
    st.title("Action Center")
    if predictions.empty:
        st.info("Run scoring first to populate the action center.")
        return

    high_risk = predictions.loc[predictions["risk_segment"] == "high_risk"].copy()
    action_columns = ["customer_id", "churn_probability", "recommended_action", "rationale", "risk_segment"]
    optional_columns = ["experiment_group", "status", "channel", "actual_churn"]
    display_columns = [column for column in action_columns + optional_columns if column in high_risk.columns]

    st.dataframe(
        high_risk[display_columns].sort_values("churn_probability", ascending=False),
        width="stretch",
        hide_index=True,
    )

    if "experiment_group" in predictions.columns and predictions["experiment_group"].nunique(dropna=True) > 1:
        ab = (
            predictions.groupby("experiment_group", dropna=False)[["actual_churn", "churn_probability"]]
            .mean(numeric_only=True)
            .reset_index()
            .rename(columns={"actual_churn": "observed_churn", "churn_probability": "avg_predicted_risk"})
        )
        fig_ab = px.bar(
            ab,
            x="experiment_group",
            y=["observed_churn", "avg_predicted_risk"],
            barmode="group",
            title="A/B Monitoring: Control vs Treatment",
        )
        st.plotly_chart(fig_ab, width="stretch")
        return

    action_mix = (
        predictions.groupby("recommended_action", dropna=False)
        .size()
        .reset_index(name="customers")
        .sort_values("customers", ascending=False)
    )
    fig_actions = px.bar(
        action_mix.head(10),
        x="recommended_action",
        y="customers",
        color="customers",
        color_continuous_scale="OrRd",
        title="Recommended Action Mix",
    )
    fig_actions.update_layout(xaxis_title=None, coloraxis_showscale=False)
    st.plotly_chart(fig_actions, width="stretch")


def main():
    requested_source = st.sidebar.selectbox(
        "Data source",
        options=list(SOURCE_OPTIONS),
        format_func=lambda option: SOURCE_OPTIONS[option],
        index=0,
    )

    if st.sidebar.button("Reload latest files"):
        load_data.clear()
        st.rerun()

    try:
        data = load_data(requested_source)
    except FileNotFoundError as error:
        st.error(
            "Dashboard inputs are missing. Run `python3 scripts/bootstrap_demo.py` "
            "or `python3 scripts/run_kaggle_cell2cell_pipeline.py` first."
        )
        st.caption(str(error))
        return

    bundle = data["selected"]
    render_source_summary(bundle, requested_source)

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Customer Explorer", "Insights", "Action Center"],
        index=0,
    )

    if page == "Overview":
        render_overview(bundle)
    elif page == "Customer Explorer":
        render_customer_explorer(bundle)
    elif page == "Insights":
        render_insights(bundle)
    else:
        render_action_center(bundle)


if __name__ == "__main__":
    main()
