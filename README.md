# 🛡️ COD Shield AI

### Machine Learning for Cash-on-Delivery (COD) Order Risk Assessment

COD Shield AI is a machine-learning portfolio project exploring how order characteristics can be used to identify potentially abusive Cash-on-Delivery (COD) ordering patterns.

The project combines synthetic data generation, Logistic Regression, campaign-based model evaluation, decision-threshold optimization, and an interactive Streamlit dashboard.

> **Research disclaimer:** This is a synthetic-data demonstration. The model has not been validated on real-world fraud data and must not be used to make actual customer or order decisions.

---

## 📌 Project Overview

Cash-on-Delivery orders can present operational challenges when orders are repeatedly placed but not confirmed or collected.

COD Shield AI explores whether observable order characteristics can help distinguish legitimate orders from simulated abuse patterns.

The prototype provides:

- Synthetic COD order data generation
- Machine-learning-based risk scoring
- Decision-threshold optimization
- Campaign-based train/test separation
- Interactive risk assessment
- Model evaluation and visualization
- A reusable Python risk-scoring interface

---

## 🎯 Project Objectives

1. Generate a reproducible synthetic COD order dataset.
2. Train a Logistic Regression classification model.
3. Evaluate performance on held-out campaigns.
4. Optimize the classification threshold using validation data.
5. Visualize classification performance through a Streamlit dashboard.
6. Document limitations and provide reproducible execution steps.

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and machine learning |
| pandas | Dataset manipulation |
| NumPy | Numerical operations |
| scikit-learn | Preprocessing, model training, and evaluation |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| SQLite | Local recipient-protection reporting |
| pytest | Automated testing |
| joblib | Model persistence |

---

## 🧠 Machine Learning Approach

### Model

The prototype uses Logistic Regression within a scikit-learn preprocessing pipeline.

### Features

The model uses the following input features:

- `order_value_inr`
- `orders_last_15m`
- `failed_confirmations`
- `distinct_accounts`
- `confirmation_status`
- `payment_method`

The synthetic target variable is:

- `is_simulated_abuse`

Campaign identifiers are used to separate groups during evaluation and are not included as model features.

### Training and Evaluation

The current pipeline uses:

- A campaign-based outer training/test split
- A separate validation split within the training data
- Decision-threshold selection using validation F1-score
- A final model fitted on the complete outer training set
- Evaluation on the held-out test set

Campaign overlap between the outer training and test sets is checked by the training script.

---

## 📊 Model Performance

The following results were obtained from the current synthetic dataset using the selected decision threshold of `0.30`.

| Metric | Result |
|---|---:|
| Accuracy | 93.2% |
| Abuse precision | 70.7% |
| Abuse recall | 63.6% |
| Abuse F1-score | 67.0% |
| Average precision | 73.4% |
| ROC-AUC | 91.4% |
| Test orders | 1,010 |
| Test campaigns | 101 |
| Selected threshold | 0.30 |

### Confusion Matrix

| Actual Class | Predicted Legitimate | Predicted Abuse |
|---|---:|---:|
| Legitimate | 871 | 29 |
| Simulated abuse | 40 | 70 |

### Interpretation

At the selected threshold, the model identified 70 of the 110 simulated-abuse cases in the held-out test set.

It also flagged 29 legitimate cases as simulated abuse and missed 40 simulated-abuse cases.

These results illustrate the trade-off between detecting more positive cases and generating additional false positives.

**Important:** The model scores are not calibrated real-world fraud probabilities. All performance results are based on artificially generated data.

---

## 🖥️ Streamlit Dashboard

The application includes the following pages:

### 1. Analytics Dashboard

Provides an overview of the generated dataset and its characteristics.

### 2. AI Risk Checker

Allows users to enter fictional order characteristics and view:

- Model score
- Risk category
- Decision threshold
- Suggested next action

### 3. Recipient Protection

Provides a prototype interface for recipient-reported incidents and related reporting workflows.

### 4. Model Evaluation

Displays:

- Classification metrics
- Confusion matrix
- ROC curve
- Precision–Recall curve
- Classification report
- Research limitations

---

## 📂 Project Structure

```text
cod-shield-ai/
│
├── app.py
├── dashboard.py
├── app/
│   └── pages/
│       ├── 1_Analytics_Dashboard.py
│       ├── 2_AI_Risk_Checker.py
│       ├── 3_Recipient_Protection.py
│       └── 4_Model_Evaluation.py
│
├── src/
│   ├── data_generator.py
│   ├── analyze_data.py
│   ├── train_model.py
│   ├── risk_engine.py
│   └── database.py
│
├── data/
│   ├── processed/
│   │   └── synthetic_orders.csv
│   └── cod_shield.db
│
├── models/
│   ├── cod_risk_model.joblib
│   └── decision_threshold.json
│
├── test_risk_engine.py
├── requirements.txt
└── README.md