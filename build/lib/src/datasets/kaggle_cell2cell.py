from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.utils.config import get_settings
from src.utils.io import save_dataframe_version
from src.utils.logging import configure_logging


logger = configure_logging("kaggle_cell2cell")

DATASET_SLUG = "jpacse/datasets-for-churn-telecom"
TRAIN_FILENAME = "cell2celltrain.csv"
HOLDOUT_FILENAME = "cell2cellholdout.csv"

RAW_SUBDIR = "kaggle_cell2cell"
TRAIN_DATASET_NAME = "kaggle_cell2cell_train"
HOLDOUT_DATASET_NAME = "kaggle_cell2cell_holdout"

NUMERIC_COLUMNS = [
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
]


def download_or_locate_cell2cell_dataset(force_download: bool = False) -> tuple[Path, Path]:
    settings = get_settings()
    raw_dir = settings.raw_data_dir / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    local_train = raw_dir / TRAIN_FILENAME
    local_holdout = raw_dir / HOLDOUT_FILENAME
    if local_train.exists() and local_holdout.exists() and not force_download:
        return local_train, local_holdout

    import kagglehub

    download_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    source_train = download_dir / TRAIN_FILENAME
    source_holdout = download_dir / HOLDOUT_FILENAME
    shutil.copy2(source_train, local_train)
    shutil.copy2(source_holdout, local_holdout)
    logger.info("Downloaded Kaggle dataset %s into %s", DATASET_SLUG, raw_dir)
    return local_train, local_holdout


def load_raw_cell2cell(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, holdout_path = download_or_locate_cell2cell_dataset(force_download=force_download)
    return pd.read_csv(train_path), pd.read_csv(holdout_path)


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "Unknown", "NaN": "Unknown"})
        .fillna("Unknown")
    )


def _collapse_rare_categories(series: pd.Series, top_n: int = 20, other_label: str = "Other") -> pd.Series:
    top_values = series.value_counts(dropna=False).head(top_n).index
    return series.where(series.isin(top_values), other_label)


def prepare_cell2cell_frames(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rename_map = {
        "CustomerID": "customer_id",
        "Churn": "churn",
        "MonthlyRevenue": "monthly_revenue",
        "MonthlyMinutes": "monthly_minutes",
        "TotalRecurringCharge": "total_recurring_charge",
        "DirectorAssistedCalls": "director_assisted_calls",
        "OverageMinutes": "overage_minutes",
        "RoamingCalls": "roaming_calls",
        "PercChangeMinutes": "perc_change_minutes",
        "PercChangeRevenues": "perc_change_revenues",
        "DroppedCalls": "dropped_calls",
        "BlockedCalls": "blocked_calls",
        "UnansweredCalls": "unanswered_calls",
        "CustomerCareCalls": "customer_care_calls",
        "ThreewayCalls": "threeway_calls",
        "ReceivedCalls": "received_calls",
        "OutboundCalls": "outbound_calls",
        "InboundCalls": "inbound_calls",
        "PeakCallsInOut": "peak_calls_in_out",
        "OffPeakCallsInOut": "off_peak_calls_in_out",
        "DroppedBlockedCalls": "dropped_blocked_calls",
        "CallForwardingCalls": "call_forwarding_calls",
        "CallWaitingCalls": "call_waiting_calls",
        "MonthsInService": "months_in_service",
        "UniqueSubs": "unique_subs",
        "ActiveSubs": "active_subs",
        "ServiceArea": "service_area",
        "Handsets": "handsets",
        "HandsetModels": "handset_models",
        "CurrentEquipmentDays": "current_equipment_days",
        "AgeHH1": "age_hh1",
        "AgeHH2": "age_hh2",
        "ChildrenInHH": "children_in_hh",
        "HandsetRefurbished": "handset_refurbished",
        "HandsetWebCapable": "handset_web_capable",
        "TruckOwner": "truck_owner",
        "RVOwner": "rv_owner",
        "Homeownership": "homeownership",
        "BuysViaMailOrder": "buys_via_mail_order",
        "RespondsToMailOffers": "responds_to_mail_offers",
        "OptOutMailings": "opt_out_mailings",
        "NonUSTravel": "non_us_travel",
        "OwnsComputer": "owns_computer",
        "HasCreditCard": "has_credit_card",
        "RetentionCalls": "retention_calls",
        "RetentionOffersAccepted": "retention_offers_accepted",
        "NewCellphoneUser": "new_cellphone_user",
        "NotNewCellphoneUser": "not_new_cellphone_user",
        "ReferralsMadeBySubscriber": "referrals_made_by_subscriber",
        "IncomeGroup": "income_group",
        "OwnsMotorcycle": "owns_motorcycle",
        "AdjustmentsToCreditRating": "adjustments_to_credit_rating",
        "HandsetPrice": "handset_price",
        "MadeCallToRetentionTeam": "made_call_to_retention_team",
        "CreditRating": "credit_rating",
        "PrizmCode": "prizm_code",
        "Occupation": "occupation",
        "MaritalStatus": "marital_status",
    }

    def _prep(df: pd.DataFrame, split: str) -> pd.DataFrame:
        frame = df.rename(columns=rename_map).copy()
        frame["dataset_split"] = split
        frame["churn"] = (
            frame["churn"]
            .map({"Yes": 1, "No": 0})
            .astype("Int64")
        )

        for column in NUMERIC_COLUMNS:
            frame[column] = frame[column].replace({"?": pd.NA, "Unknown": pd.NA})
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        categorical_columns = [
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
            "credit_rating",
            "prizm_code",
            "occupation",
            "marital_status",
            "service_area",
        ]
        for column in categorical_columns:
            frame[column] = _normalize_yes_no(frame[column])

        frame["monthly_margin"] = frame["monthly_revenue"] - frame["total_recurring_charge"]
        frame["call_problem_volume"] = (
            frame["dropped_calls"].fillna(0)
            + frame["blocked_calls"].fillna(0)
            + frame["unanswered_calls"].fillna(0)
            + frame["customer_care_calls"].fillna(0)
        )
        frame["usage_mix_total"] = (
            frame["received_calls"].fillna(0)
            + frame["outbound_calls"].fillna(0)
            + frame["inbound_calls"].fillna(0)
        )
        frame["overage_ratio"] = frame["overage_minutes"] / (frame["monthly_minutes"].abs() + 1)
        frame["call_problem_rate"] = frame["call_problem_volume"] / (frame["monthly_minutes"].abs() + 1)
        frame["revenue_per_minute"] = frame["monthly_revenue"] / (frame["monthly_minutes"].abs() + 1)
        frame["equipment_tenure_ratio"] = frame["current_equipment_days"] / ((frame["months_in_service"] * 30.4) + 1)
        frame["retention_pressure_score"] = (
            frame["retention_calls"].fillna(0)
            + frame["retention_offers_accepted"].fillna(0)
            + frame["made_call_to_retention_team"].eq("Yes").astype(int)
        )
        frame["household_age_gap"] = (frame["age_hh1"].fillna(0) - frame["age_hh2"].fillna(0)).abs()
        frame["service_tenure_band"] = pd.cut(
            frame["months_in_service"].fillna(0),
            bins=[-1, 6, 12, 24, 48, 120],
            labels=["0_6m", "7_12m", "13_24m", "25_48m", "49m_plus"],
        ).astype(str)

        frame = frame.replace([float("inf"), float("-inf")], pd.NA)
        return frame

    prepared_train = _prep(train_df, "train")
    prepared_holdout = _prep(holdout_df, "holdout")

    def _apply_grouping(frame: pd.DataFrame, column: str, top_values: pd.Index, new_column: str) -> pd.DataFrame:
        frame[new_column] = frame[column].where(frame[column].isin(top_values), "Other")
        return frame

    service_area_top = prepared_train["service_area"].value_counts(dropna=False).head(25).index
    occupation_top = prepared_train["occupation"].value_counts(dropna=False).head(12).index
    credit_rating_top = prepared_train["credit_rating"].value_counts(dropna=False).head(6).index
    prizm_code_top = prepared_train["prizm_code"].value_counts(dropna=False).head(6).index

    for frame in (prepared_train, prepared_holdout):
        _apply_grouping(frame, "service_area", service_area_top, "service_area_grouped")
        _apply_grouping(frame, "occupation", occupation_top, "occupation_grouped")
        _apply_grouping(frame, "credit_rating", credit_rating_top, "credit_rating_grouped")
        _apply_grouping(frame, "prizm_code", prizm_code_top, "prizm_code_grouped")

    high_value_threshold = prepared_train["monthly_revenue"].quantile(0.75)
    high_call_problem_threshold = prepared_train["call_problem_volume"].quantile(0.75)
    for frame in (prepared_train, prepared_holdout):
        frame["high_value_customer"] = (frame["monthly_revenue"] >= high_value_threshold).astype(int)
        frame["high_call_problem_flag"] = (
            frame["call_problem_volume"] >= high_call_problem_threshold
        ).astype(int)

    if persist:
        save_dataframe_version(prepared_train, TRAIN_DATASET_NAME)
        save_dataframe_version(prepared_holdout, HOLDOUT_DATASET_NAME)

    return prepared_train, prepared_holdout
