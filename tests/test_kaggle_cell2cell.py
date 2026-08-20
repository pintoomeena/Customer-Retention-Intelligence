from __future__ import annotations

import pandas as pd

from src.datasets.kaggle_cell2cell import prepare_cell2cell_frames


def test_prepare_cell2cell_frames_creates_model_features():
    train = pd.DataFrame(
        [
            {
                "CustomerID": 1,
                "Churn": "Yes",
                "MonthlyRevenue": "24.00",
                "MonthlyMinutes": "219.0",
                "TotalRecurringCharge": "22.0",
                "DirectorAssistedCalls": "0.25",
                "OverageMinutes": "0.0",
                "RoamingCalls": "0.0",
                "PercChangeMinutes": "-157.0",
                "PercChangeRevenues": "-19.0",
                "DroppedCalls": "0.7",
                "BlockedCalls": "0.7",
                "UnansweredCalls": "6.3",
                "CustomerCareCalls": "0.0",
                "ThreewayCalls": "0.0",
                "ReceivedCalls": "97.2",
                "OutboundCalls": "0.0",
                "InboundCalls": "0.0",
                "PeakCallsInOut": "58.0",
                "OffPeakCallsInOut": "24.0",
                "DroppedBlockedCalls": "1.3",
                "CallForwardingCalls": "0.0",
                "CallWaitingCalls": "0.3",
                "MonthsInService": "61",
                "UniqueSubs": "2",
                "ActiveSubs": "1",
                "ServiceArea": "SEAPOR503",
                "Handsets": "2.0",
                "HandsetModels": "2.0",
                "CurrentEquipmentDays": "361.0",
                "AgeHH1": "62.0",
                "AgeHH2": "0.0",
                "ChildrenInHH": "No",
                "HandsetRefurbished": "No",
                "HandsetWebCapable": "Yes",
                "TruckOwner": "No",
                "RVOwner": "No",
                "Homeownership": "Known",
                "BuysViaMailOrder": "Yes",
                "RespondsToMailOffers": "Yes",
                "OptOutMailings": "No",
                "NonUSTravel": "No",
                "OwnsComputer": "Yes",
                "HasCreditCard": "Yes",
                "RetentionCalls": "1",
                "RetentionOffersAccepted": "0",
                "NewCellphoneUser": "No",
                "NotNewCellphoneUser": "No",
                "ReferralsMadeBySubscriber": "0",
                "IncomeGroup": "4",
                "OwnsMotorcycle": "No",
                "AdjustmentsToCreditRating": "0",
                "HandsetPrice": "30",
                "MadeCallToRetentionTeam": "Yes",
                "CreditRating": "1-Highest",
                "PrizmCode": "Suburban",
                "Occupation": "Professional",
                "MaritalStatus": "No",
            }
        ]
    )
    holdout = train.assign(Churn=pd.NA)

    prepared_train, prepared_holdout = prepare_cell2cell_frames(train, holdout, persist=False)

    assert prepared_train.loc[0, "churn"] == 1
    assert pd.isna(prepared_holdout.loc[0, "churn"])
    assert {
        "monthly_margin",
        "call_problem_volume",
        "service_area_grouped",
        "service_tenure_band",
        "handset_price_missing",
        "customer_care_rate",
        "equipment_age_bucket",
        "credit_prizm_combo",
    }.issubset(prepared_train.columns)
