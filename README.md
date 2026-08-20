# Customer Retention Intelligence Platform

An end-to-end **machine learning platform for customer churn prediction, risk scoring, customer insights, and retention recommendations**.

The platform processes customer and behavioral data, performs feature engineering, trains and compares multiple machine learning models, scores customers by churn risk, and provides actionable retention recommendations through a **FastAPI backend and Streamlit dashboard**.

---

## 🚀 Key Features

- End-to-end **data ingestion and feature engineering**
- Behavioral, engagement, revenue, payment, support, and RFM-based features
- Churn prediction using:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Automated **model comparison and champion model selection**
- Customer-level **churn probability and risk segmentation**
- Global **feature importance and churn-driver analysis**
- Automated **retention action recommendations**
- **FastAPI** endpoints for prediction, explanation, and recommendations
- Interactive **Streamlit dashboard**
- Batch scoring and monitoring
- Model artifact and feature-store management
- Docker support
- Automated testing with Pytest

---

## 🏗️ System Architecture

```text
                    Customer Data
                         │
                         ▼
              Data Ingestion & Cleaning
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                    EDA & Analysis
                         │
                         ▼
             Model Training & Evaluation
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        Logistic      Random       XGBoost
       Regression     Forest          │
             │           │             │
             └───────────┴─────────────┘
                         │
                      LightGBM
                         │
                         ▼
                Champion Model
                         │
                         ▼
                 Risk Prediction
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Feature Importance       Risk Segmentation
             │                       │
             └───────────┬───────────┘
                         ▼
               Retention Recommendations
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          FastAPI              Streamlit
           Service              Dashboard
              │                     │
              └──────────┬──────────┘
                         ▼
                     Monitoring
```

---

## 📊 Model Performance

The demo pipeline trains and evaluates four classification models on a synthetic dataset containing **1,500 customers**.

| Model | Holdout ROC-AUC | Holdout Average Precision |
|---|---:|---:|
| Logistic Regression | 99.77% | 99.57% |
| Random Forest | 99.84% | 99.69% |
| XGBoost | 99.97% | 99.94% |
| **LightGBM** | **99.99%** | **99.98%** |

### 🏆 Champion Model

**LightGBM**

- **ROC-AUC:** 99.99%
- **Average Precision:** 99.98%

> **Note:** These results are from the synthetic demo dataset. Average Precision is reported instead of referring to it as classification accuracy.

---

## 🔬 Feature Engineering

The platform creates customer-level behavioral and commercial features.

### Customer & Tenure

- Customer tenure
- Signup date
- Cohort month
- Service tenure bands

### Engagement

- Sessions in the last 30 days
- Login frequency
- Activity decay
- Engagement score
- Feature adoption ratio

### Revenue & Payment

- Monthly revenue
- Average invoice amount
- Payment behavior
- Revenue-related risk signals

### Support

- Support interactions
- Customer service activity
- Support-related behavioral indicators

### RFM Signals

The pipeline incorporates RFM-style customer value signals:

- **Recency**
- **Frequency**
- **Monetary Value**

These features help identify behavioral patterns associated with customer churn.

---

## 🤖 Machine Learning Pipeline

The training pipeline evaluates multiple candidate models:

```text
Feature Dataset
      │
      ▼
Train / Validation Split
      │
      ▼
Feature Preparation
      │
      ├── Logistic Regression
      ├── Random Forest
      ├── XGBoost
      └── LightGBM
      │
      ▼
Model Evaluation
      │
      ▼
Model Leaderboard
      │
      ▼
Champion Model Selection
      │
      ▼
Model Artifact
```

The best-performing model is registered as the **champion model** and used by the downstream scoring pipeline.

---

## 🎯 Customer Risk Scoring

Each customer receives a predicted churn probability.

Customers are grouped into risk segments:

```text
                Churn Probability
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Low Risk    Medium Risk    High Risk
```

The scoring pipeline generates information such as:

- Customer ID
- Churn probability
- Risk segment
- Recommended action
- Recommendation rationale
- Experiment group
- Campaign status
- Channel
- Actual churn outcome

---

## 🔍 Explainability

The platform provides both global model insights and customer-level explanations.

### Global Feature Importance

The dashboard displays the most important features contributing to model predictions.

Examples include:

- Recency
- Frequency per week
- Tenure
- Activity duration
- Monthly revenue
- Activity trend
- Monthly customer value

### Customer-Level Explanation

For individual customers, the system provides a rationale for the recommended retention action based on their behavioral risk signals.

Example:

```text
Frequency per week is elevating risk.
Activity decay is elevating risk.
```

This makes the model output more actionable for a retention team.

---

## 💡 Retention Recommendations

The platform converts churn predictions into operational actions instead of stopping at a probability score.

Possible actions include:

- **Executive check-in**
- **Onboarding reactivation**
- **Discount offer**
- **Feature adoption campaign**
- **Support outreach**
- **Nurture campaign**
- **Holdout/control assignment**

Each recommendation can include a rationale explaining the behavioral signals behind the decision.

---

## 🖥️ Streamlit Dashboard

The Streamlit dashboard provides four main views.

### 1. Overview

Provides portfolio-level information:

- Total customers
- Observed churn rate
- High-risk customer count
- Average predicted risk
- Churn rate by plan
- Cohort churn analysis
- Customer segment behavior

### 2. Customer Explorer

Provides individual customer analysis:

- Customer profile
- Churn probability
- Risk segment
- Behavioral features
- Customer activity
- Recommended retention action

### 3. Model Insights

Provides model-level analysis:

- Champion model
- Model version
- Candidate model comparison
- ROC-AUC
- Average Precision
- Precision
- Recall
- F1 score
- Decision threshold
- Global feature importance
- Churn-driver analysis

### 4. Action Center

Provides a prioritized view of high-risk customers:

- Customer ID
- Churn probability
- Risk segment
- Recommended retention action
- Recommendation rationale
- Experiment group
- Campaign status
- Channel

The Action Center also supports **treatment vs. control monitoring** for retention campaigns.

---

## ⚡ FastAPI

The project includes a FastAPI service for model serving.

### Available Endpoints

```text
GET  /health
GET  /ready
GET  /sources
GET  /model/info

POST /predict
POST /explain
POST /recommend
```

### Prediction Workflow

```text
Client Request
      │
      ▼
    FastAPI
      │
      ▼
Feature Validation
      │
      ▼
Champion Model
      │
      ▼
Churn Probability
      │
      ├── Risk Segment
      ├── Explanation
      └── Recommendation
```

---

## 📈 Monitoring

The platform includes a batch monitoring pipeline that generates reports after scoring.

Monitoring artifacts can be used to track:

- Prediction batches
- Model performance
- Risk distributions
- Data behavior
- Treatment/control outcomes
- Model artifacts
- Feature artifacts

This provides a foundation for continuously evaluating the deployed churn model.

---

## 🔄 End-to-End Pipeline

The complete demo pipeline can be executed using one command:

```bash
python scripts/bootstrap_demo.py
```

The pipeline performs:

```text
1. Generate Demo Data
        ↓
2. Data Ingestion
        ↓
3. Feature Engineering
        ↓
4. EDA
        ↓
5. Model Training
        ↓
6. Model Comparison
        ↓
7. Champion Model Selection
        ↓
8. Customer Scoring
        ↓
9. Retention Recommendations
        ↓
10. Monitoring
```

---

## 📁 Project Structure

```text
Customer-Retention-Intelligence/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── artifacts/
│   └── reports/
│
├── scripts/
│   ├── bootstrap_demo.py
│   ├── run_training.py
│   ├── run_scoring.py
│   └── run_monitoring.py
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── datasets/
│   │   └── ...
│   │
│   ├── features/
│   │   └── engineering.py
│   │
│   ├── models/
│   │   ├── training.py
│   │   ├── scoring.py
│   │   ├── serving.py
│   │   └── registry.py
│   │
│   ├── monitoring/
│   │   └── ...
│   │
│   ├── retention/
│   │   └── ...
│   │
│   └── utils/
│       ├── config.py
│       ├── io.py
│       └── logging.py
│
├── tests/
│   └── ...
│
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM

### Backend

- FastAPI
- Uvicorn

### Dashboard & Visualization

- Streamlit
- Plotly

### Data & Storage

- CSV
- Parquet
- JSON
- Joblib

### Testing & Deployment

- Pytest
- Docker
- Docker Compose
- Git
- GitHub

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pintoomeena/Customer-Retention-Intelligence.git
```

### 2. Navigate to the Project

```bash
cd Customer-Retention-Intelligence
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

#### Windows PowerShell

```powershell
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install ".[dev]"
```

---

## ▶️ Running the Project

### Run the Complete Demo Pipeline

```bash
python scripts/bootstrap_demo.py
```

This generates the demo data, processes the features, trains the models, selects the champion model, scores customers, and creates monitoring artifacts.

### Start the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

### Start the FastAPI Service

```bash
uvicorn src.api.app:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker

The project also supports Docker deployment.

Run:

```bash
docker compose up --build
```

This starts the application services defined in the Docker Compose configuration.

---

## 🧪 Testing

Run the test suite:

```bash
pytest -q
```

The project includes tests for core pipeline and API functionality.

---

## 📌 Project Highlights

- Processed **1,500 customers** through the complete demo pipeline
- Built an end-to-end **production-style ML workflow**
- Compared **4 machine learning models**
- Selected **LightGBM as the champion model**
- Achieved **99.99% holdout ROC-AUC**
- Achieved **99.98% holdout Average Precision**
- Implemented automated **customer risk segmentation**
- Generated **customer-specific retention recommendations**
- Built an interactive **Streamlit decision-support dashboard**
- Added **FastAPI model serving**
- Added **batch scoring and monitoring**
- Implemented **model registry and artifact management**
- Added **Docker deployment support**
- Added automated testing with **Pytest**

---

## ⚠️ Demo Dataset

The included bootstrap pipeline uses **synthetically generated customer data** for demonstration purposes.

Therefore, the reported model metrics should **not be interpreted as production performance** on a real-world customer churn dataset.

The project demonstrates:

- Machine learning pipeline engineering
- Feature engineering
- Model comparison and selection
- Customer risk scoring
- Explainability
- Retention decisioning
- Model serving
- Monitoring
- Dashboard development

---
