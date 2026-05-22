---
title: Customer Churn Predictor
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📉 Customer Churn Prediction — MLOps Pipeline

> **End-to-end MLOps project**: Telecom customer churn prediction with ensemble ML, SHAP explainability, SMOTE rebalancing, threshold optimisation, FastAPI serving, automated CI/CD, and MLflow experiment tracking — containerised with Docker and deployed to HuggingFace Spaces.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/Mechscientist26/customer-churn-predictor)
[![CI/CD](https://github.com/Mondip-Mech/customer_churn_predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Mondip-Mech/customer_churn_predictor/actions/workflows/ci.yml)

---

## 🏆 Results at a Glance

| Metric | Score |
|---|---|
| ROC-AUC | **0.8496** |
| Accuracy | 0.8006 |
| Recall (Churn) | 0.6631 |
| Precision (Churn) | 0.6502 |
| F1 Score | 0.6566 |
| F2 Score (recall-weighted) | 0.6606 |

> **Production model**: Voting Classifier — soft voting ensemble of GradientBoosting + LogisticRegression + AdaBoost

---

## 🏗️ MLOps Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Telco Churn CSV  →  Pandas  →  ColumnTransformer Pipeline      │
│  StandardScaler · OrdinalEncoder · OneHotEncoder · SMOTE        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      MODEL LAYER                                │
│  VotingClassifier (soft)                                        │
│  ├── GradientBoostingClassifier  (n_estimators=300, depth=1)    │
│  ├── LogisticRegression          (liblinear, C=1.0)             │
│  └── AdaBoostClassifier          (n_estimators=100, lr=0.3)     │
│  ↓                                                              │
│  Threshold Optimiser  →  Business Cost = FP×$25 + FN×$200      │
│  SHAP Explainer       →  Per-customer feature attribution       │
│  MLflow Registry      →  Staging → Production promotion         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     SERVING LAYER                               │
│  FastAPI  (port 8000)   Streamlit Dashboard  (port 8501)        │
│  ├── POST /predict       ├── Predict tab                        │
│  ├── POST /predict/batch ├── SHAP Explanation tab               │
│  ├── GET  /health        ├── Business Impact tab                │
│  ├── GET  /metrics       └── Model Performance tab              │
│  └── GET  /predictions                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      OPS LAYER                                  │
│  GitHub Actions CI/CD  →  Test → Lint → Docker Build            │
│  Docker Compose        →  API + Streamlit + MLflow together      │
│  SQLite Prediction Log →  Every inference persisted             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis

### Churn Distribution

![Churn Distribution](reports/figures/Churn%20Distributions.png)

> **26.5% churn rate** — moderate class imbalance addressed using SMOTE in the training pipeline.

---

### Contract Type vs Churn

![Contract Distribution](reports/figures/Customer%20contract%20distribution.png)

> Month-to-month customers churn at **~43%** vs only **~3%** for two-year contracts — the strongest single predictor.

---

### Tenure vs Churn

![Tenure vs Churn](reports/figures/Tenure%20vs%20Churn.png)

> Churned customers have a median tenure of **~10 months** vs **~38 months** for retained customers. Early-life engagement is critical.

---

### Internet Service & Gender vs Churn

![Internet Service](reports/figures/Churn%20Distribution%20w.r.t.%20Internet%20Service%20and%20Gender.png)

> Fiber optic customers churn at **~42%** — twice the DSL rate. Gender has no significant impact.

---

### Payment Method vs Churn

![Payment Method](reports/figures/Customer%20Payment%20Method%20distribution%20w.r.t.%20Churn.png)

> Electronic check customers churn at **~45%**. Auto-pay customers are significantly more loyal.

---

### Online Security vs Churn

![Online Security](reports/figures/Churn%20w.r.t%20Online%20Security.png)

> Customers without online security have **2× the churn rate** — a strong upsell opportunity.

---

### Tech Support vs Churn

![Tech Support](reports/figures/Chrun%20distribution%20w.r.t.%20TechSupport.png)

> No tech support → significantly higher churn. Customers who feel supported stay longer.

---

### Partner & Dependents vs Churn

| Partner Status | Dependents |
|---|---|
| ![Partner](reports/figures/Chrun%20distribution%20w.r.t.%20Partners.png) | ![Dependents](reports/figures/Dependents%20distribution.png) |

> Customers without partners or dependents show **significantly higher churn** — they have less reason to stay on a family plan.

---

### Senior Citizen vs Churn

![Senior Citizen](reports/figures/Chrun%20distribution%20w.r.t.%20Senior%20Citizen.png)

> Senior citizens churn at **~42%** vs 24% overall — a segment requiring dedicated support.

---

### Paperless Billing vs Churn

![Paperless Billing](reports/figures/Chrun%20distribution%20w.r.t.%20Paperless%20Billing.png)

> Paperless billing customers show higher churn — likely correlated with the electronic check payment pattern.

---

## 🤖 Model Comparison

| Model | CV ROC-AUC | CV Accuracy | Notes |
|---|---|---|---|
| **Voting Classifier (tuned)** | **0.8496** | **0.8012** | ✅ Production model |
| Gradient Boosting (tuned) | 0.8495 | 0.7998 | Near-identical to Voting |
| AdaBoost (tuned) | 0.8464 | 0.7961 | Slightly weaker recall |
| Logistic Regression (tuned) | 0.8460 | 0.7934 | Most interpretable |
| Kernel SVM | 0.8390 | 0.7950 | Best single-kernel model |
| SVM Linear | 0.8350 | 0.7910 | Slow on large data |
| Random Forest | 0.8310 | 0.7880 | Strong baseline |
| Naive Bayes | 0.8120 | 0.7560 | Poor precision |
| KNN | 0.7820 | 0.7690 | Sensitive to scaling |
| Decision Tree | 0.7180 | 0.7350 | Overfits without pruning |

All models evaluated with **Stratified 10-fold CV** on the training set. Top 4 tuned using `RandomizedSearchCV` + `GridSearchCV`.

---

## ⚖️ Class Imbalance — SMOTE

| Dataset | Samples | Churn=1 | Churn=0 | Rate |
|---|---|---|---|---|
| Original (train) | 5,634 | 1,497 | 4,137 | 26.6% |
| After SMOTE | 8,274 | 4,137 | 4,137 | 50.0% |
| Test set | 1,409 | 372 | 1,037 | 26.4% |

SMOTE creates synthetic minority samples in feature space — improving recall without distorting the test distribution.

---

## 🎯 Threshold Optimisation

Default threshold (0.5) treats all errors equally. In churn:

| Error Type | Business Cost |
|---|---|
| False Negative (missed churner) | **$200** — lost revenue |
| False Positive (unnecessary offer) | **$25** — retention offer cost |

Sweeping thresholds 0.10 → 0.90 finds the optimal point minimising `FP×$25 + FN×$200`. The optimal threshold sits around **0.35–0.40**, significantly increasing recall.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) answers *why* the model predicts churn for each customer:

- **Global beeswarm** — feature importance across all customers
- **Global bar chart** — mean absolute SHAP values ranking
- **Individual waterfall** — step-by-step explanation per prediction

Red bars push toward churn · Blue bars push away from churn.

---

## 🚀 FastAPI Service

```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe — model loaded status |
| `/predict` | POST | Single customer churn prediction |
| `/predict/batch` | POST | Batch scoring up to 500 customers |
| `/predictions` | GET | Recent prediction log |
| `/metrics` | GET | Aggregate stats for drift monitoring |
| `/docs` | GET | Interactive Swagger UI |

**Every prediction is logged** to `predictions.db` (SQLite) with timestamp, features, probability, risk level, and latency.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

| Test Suite | Coverage |
|---|---|
| `test_schema.py` | Pydantic input validation — valid inputs, boundary violations, invalid enums |
| `test_api.py` | API endpoints — health, predict, batch, monitoring, 422 error handling |

**27 tests total** — all passing in CI.

---

## 🔄 CI/CD Pipeline (GitHub → HuggingFace Spaces)

Every push to `main` triggers the full pipeline automatically:

```
Push to main
    │
    ├── Test job     →  pytest tests/ (schema + API)
    ├── Lint job     →  ruff check api/ tests/
    └── Deploy job   →  git push → HuggingFace Spaces (auto Docker build)
```

HuggingFace picks up the new commit, rebuilds the Docker image, and re-serves the app — zero manual steps after the initial setup.

**One-time setup — add `HF_TOKEN` to GitHub secrets:**
1. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (role: **Write**)
2. In your GitHub repo → **Settings → Secrets → Actions** → **New secret**
3. Name: `HF_TOKEN`, Value: your token

See `.github/workflows/ci.yml`

---

## 📁 Project Structure

```
customer_churn_predictor/
│
├── api/                              # FastAPI service
│   ├── main.py                       # 5 endpoints with prediction logging
│   ├── schema.py                     # Pydantic input/output validation
│   └── logger.py                     # SQLite prediction logger
│
├── tests/                            # 27 automated tests
│   ├── test_api.py                   # API integration tests
│   └── test_schema.py                # Schema unit tests
│
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions CI/CD
│
├── metrics/
│   └── model_performance.json        # Pre-computed metrics for dashboard
│
├── reports/figures/                  # EDA visualisations
│
├── Customer Churn Prediction.ipynb   # Full analysis (10 sections)
├── app.py                            # Streamlit dashboard (4 tabs)
├── config.py                         # Paths, feature lists, cost constants
├── final_churn_model.pkl             # Production VotingClassifier pipeline
├── background_data.pkl               # SHAP background dataset (100 samples)
├── Dockerfile                        # Streamlit container
├── Dockerfile.api                    # FastAPI container
├── docker-compose.yml                # Runs API + Streamlit + MLflow together
└── requirements.txt
```

---

## 🖥️ Streamlit Dashboard — 4 Tabs

| Tab | Description |
|---|---|
| **Predict** | Customer input form → risk gauge → churn probability → retention recommendation |
| **SHAP Explanation** | Waterfall + bar chart showing *why* this customer is predicted to churn |
| **Business Impact** | Adjustable FP/FN costs + customer base slider → threshold sensitivity analysis |
| **Model Performance** | Confusion matrix, ROC curve, model comparison table, SMOTE before/after |

---

## 🚀 Deployment

### Live on HuggingFace Spaces
The app is deployed automatically via GitHub Actions on every push to `main`.

👉 **[huggingface.co/spaces/Mechscientist26/customer-churn-predictor](https://huggingface.co/spaces/Mechscientist26/customer-churn-predictor)**

### Run Locally

**Streamlit App**
```bash
git clone https://github.com/Mondip-Mech/customer_churn_predictor.git
cd customer_churn_predictor
pip install -r requirements.txt
streamlit run app.py
# http://localhost:8501
```

**FastAPI Service**
```bash
uvicorn api.main:app --reload --port 8000
# http://localhost:8000/docs
```

**Docker (local)**
```bash
docker build -t churn-predictor .
docker run -p 7860:7860 churn-predictor
# http://localhost:7860
```

**Everything together (Docker Compose)**
```bash
docker-compose up
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
```

---

## 📦 MLflow Experiment Tracking

All 10 model runs logged with parameters, CV metrics, and artifacts:

```bash
mlflow ui    # http://localhost:5000
```

- Section 10 of the notebook logs all experiments retroactively
- `MLflowGridSearchCV` wrapper auto-logs every hyperparameter candidate as a nested run
- Production model registered in the **MLflow Model Registry** with `Staging → Production` promotion

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **ML** | scikit-learn, imbalanced-learn |
| **Explainability** | SHAP |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Testing** | pytest, httpx |
| **CI/CD** | GitHub Actions |
| **Containerisation** | Docker, Docker Compose |
| **Data** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |

---

## 👤 Author

**Mondip Mech** — Machine Learning & Data Science

[![GitHub](https://img.shields.io/badge/GitHub-Mondip--Mech-181717?style=flat&logo=github)](https://github.com/Mondip-Mech)
