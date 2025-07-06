# 🔮 Quantum Causal XAI

A modular, explainable AI framework for **counterparty credit risk assessment**, leveraging **quantum-inspired feature embedding**, **causal discovery**, **interpretable modeling (TCAV)**, and **stress testing** on the Home Credit Default Risk dataset.

---

## 📌 Overview

This repository implements a **Quantum Causal Explainable AI (XAI)** pipeline that enhances traditional machine learning with interpretable and robust layers. Designed for financial applications, the system focuses on **credit risk prediction** while providing transparency through **causal graphs** and **concept-based attribution**.


---

## 🔍 Methodology

### 1. **Data Preprocessing**
- Loads a 5,000-record sample from the Home Credit dataset.
- Selects key features:
  - `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`
  - `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`
- Fills missing values using median imputation.
- Outputs: `X`, `y`, `feature_names`.

---

### 2. **Quantum-Inspired Feature Embedding**
- Applies cosine/sine transformations to simulate quantum kernels.
- Uses `RBFSampler` to generate 15 additional Fourier features.
- Concatenates all into `X_quantum (5000 x 21)`.

---

### 3. **Causal Discovery**
- Computes correlation and mutual information among the first 6 quantum-transformed features.
- Builds a **directed causal graph** based on:
  - Corr > 0.3
  - MI > 0.1
- Visualizes causal dependencies.

---

### 4. **Risk Prediction**
- Splits data into train/test (80/20).
- Uses **SMOTE** to handle class imbalance.
- Trains a `RandomForestClassifier` with 100 trees.
- Outputs: predictions, AUC-ROC, and metrics.

---

### 5. **Explainability (TCAV)**
- Defines concepts:
  - **Liquidity** → Feature 0
  - **Credit Exposure** → Features 1 & 2
  - **Credit Score** → Features 3, 4 & 5
- Perturbs concept-specific indices in test set.
- Computes **TCAV scores**: concept sensitivity → model output.

---

### 6. **Stress Testing**
- Simulates economic scenarios:
  - **Interest Rate Shock**
  - **Liquidity Crisis**
  - **Credit Crunch**
- Applies stress multipliers to `X_test`.
- Measures:
  - Risk delta
  - TCAV variation
- Reveals model robustness under uncertainty.

---

### 7. **Visualization and Summary**
- Plots:
  - Feature Importance (RF)
  - Risk Distribution
  - Stress Impact (Bar Chart)
  - TCAV Heatmap
- Outputs: PNG charts and textual performance summary.

---

## 📊 Results

| Module              | Outcome |
|---------------------|---------|
| Risk Prediction     | Accurate, AUC-ROC evaluated |
| Causal Discovery    | Clear directed graph showing inter-feature relations |
| TCAV                | Concept-level model attribution |
| Stress Testing      | Shows model behavior under real-world scenarios |

---

👨‍🔬 Authors
**Parth Bramhecha
Shubham Chandaratre
Rajdeep Thakur
**
