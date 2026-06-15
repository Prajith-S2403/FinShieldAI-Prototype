# 🛡️ FinShieldAI — Mule Account Detection Dashboard

AI-powered AML intelligence platform built with Streamlit + XGBoost.

## 📁 Project Structure

```
finshieldai/
├── app.py               # Main Streamlit dashboard
├── generate_data.py     # Demo CSV generator
├── demo_accounts.csv    # 500 banking accounts (auto-generated)
└── requirements.txt     # Python dependencies
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate demo data
```bash
python generate_data.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📊 Features

| Feature | Description |
|---|---|
| **KPI Cards** | Total Accounts, Mule Detected, Critical Risk, Avg Risk Score |
| **Risk Score Histogram** | Distribution overlaid by account type |
| **Mule vs Normal Donut** | Proportion of flagged vs clean accounts |
| **Risk Category Bar** | Count of Critical / High / Medium / Low accounts |
| **Account Deep Dive** | Risk gauge, feature bar chart, reason codes |
| **STR Report** | Auto-generated Suspicious Transaction Report per account |
| **Bulk CSV Export** | All flagged accounts with reason codes + recommended actions |

## 🎯 Risk Categories

| Category | Risk Score |
|---|---|
| 🔴 Critical | ≥ 0.75 |
| 🟠 High | ≥ 0.50 |
| 🟡 Medium | ≥ 0.25 |
| 🟢 Low | < 0.25 |

## 🧠 Model

- **Algorithm**: XGBoost Classifier
- **Features**: sweep_ratio, drawdown_score, credit_debit_ratio, community_risk_score, transaction_velocity, account_age_days, govt_alert_flag
- **Class Imbalance**: Handled via `scale_pos_weight`
