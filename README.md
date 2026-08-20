# Customer Retention Intelligence Platform

An end-to-end machine learning platform for **customer churn prediction, risk analysis, retention recommendations, and model monitoring**.

The system takes customer, activity, transaction, and support data, engineers behavioral and commercial features, trains multiple ML models, provides customer-level explanations, and delivers retention recommendations through an API and interactive dashboard.

## 🚀 Key Features

* Customer data ingestion and feature engineering
* Behavioral, engagement, payment, support, and RFM-based features
* Churn prediction using Logistic Regression, Random Forest, XGBoost, and LightGBM
* Customer-level risk scoring and explainability
* Automated retention action recommendations
* FastAPI service for prediction, explanation, and recommendations
* Streamlit dashboard for portfolio and customer analysis
* Data drift and model performance monitoring
* Batch pipelines for training, scoring, and monitoring
* Docker support for deployment

## 🏗️ Architecture

```text
Customer Data
     ↓
Data Ingestion & Cleaning
     ↓
Feature Engineering
     ↓
EDA & Analysis
     ↓
Model Training
     ↓
Churn Risk Prediction
     ↓
Explainability & Retention Actions
     ↓
FastAPI + Streamlit Dashboard
     ↓
Monitoring
```

## 📁 Project Structure

```text
.
├── dashboard/              # Streamlit dashboard
├── data/                   # Raw data, processed data and artifacts
├── scripts/                # Training, scoring and monitoring pipelines
├── src/
│   ├── api/                # FastAPI application
│   ├── datasets/           # Dataset loaders
│   ├── features/           # Cleaning and feature engineering
│   ├── models/             # Training, prediction and explainability
│   ├── monitoring/         # Drift and performance monitoring
│   ├── retention/          # Retention recommendation logic
│   └── utils/              # Configuration and utilities
├── tests/                  # Unit and API tests
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
└── pyproject.toml
```

## 📊 Modeling

The feature pipeline uses:

* Customer tenure and cohort information
* Engagement and activity behavior
* Revenue and payment behavior
* Support interactions and satisfaction
* RFM-based customer value signals
* Churn-risk indicators

Candidate models include:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

Models are evaluated using classification and ranking metrics, and the best-performing model is registered for downstream scoring.

## 🔍 Explainability & Retention

The platform provides both **global feature importance** and **individual customer explanations**.

Customers are classified into:

* `Low Risk`
* `Medium Risk`
* `High Risk`

Based on customer behavior, the system recommends appropriate retention actions such as:

* Discount offers
* Support outreach
* Feature adoption campaigns
* Nurture campaigns

## 🖥️ Dashboard

The Streamlit dashboard provides four main views:

* **Overview** — portfolio-level KPIs and risk metrics
* **Customer Explorer** — individual customer analysis
* **Insights** — risk and feature-level analysis
* **Action Center** — prioritized retention actions

## ⚡ API

The FastAPI service provides endpoints for:

```text
GET  /health
GET  /ready
GET  /sources
GET  /model/info

POST /predict
POST /explain
POST /recommend
```

## 🛠️ Installation

```bash
git clone https://github.com/pintoomeena/Customer-Retention-Intelligence.git
cd Customer-Retention-Intelligence

python -m venv .venv
.venv\Scripts\activate

pip install ".[dev]"
```

## ▶️ Run the Project

Run the complete demo pipeline:

```bash
python scripts/bootstrap_demo.py
```

Start the API:

```bash
uvicorn src.api.app:app --reload
```

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

Or run everything using Docker:

```bash
docker compose up --build
```

## 🧪 Testing

```bash
pytest -q
```

## 👨‍💻 Project

**Customer Retention Intelligence Platform**

Built with **Python, SQL, Pandas, Scikit-learn, XGBoost, LightGBM, SHAP, FastAPI, Streamlit, Docker, and MLflow**.
