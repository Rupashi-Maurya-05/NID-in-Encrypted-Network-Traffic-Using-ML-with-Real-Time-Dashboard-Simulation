# Network Intrusion Detection System for Encrypted Traffic

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A **two-stage Machine Learning Intrusion Detection System (IDS)** for **encrypted network traffic**, combining:

- **Stage 1:** Autoencoder for anomaly detection
- **Stage 2:** XGBoost for 15-class attack classification
- **Confidence-gated decision logic** to reduce false alarms

The system detects cyber attacks using only **flow-level statistical features extracted from packet headers**, requiring:

- ❌ No packet payload inspection
- ❌ No TLS/SSL decryption
- ✅ Encryption-agnostic detection

---

## Project Overview

This project was developed during an internship at **SAG DRDO** under the guidance of **Dr. Sanjay Kumar**.

**Dataset**

- CIC-IDS2017
- Engelen et al. corrected version (IEEE SPW 2021)

**Final Performance**

| Metric | Value |
|---------|------:|
| Test Flows | 419,376 |
| Attack Classes | 15 |
| Final Macro F1 | **0.8937** |

---

# Quick Start

## 1. Clone repository

```bash
git clone https://github.com/yourusername/nids-project.git
cd nids-project
```

## 2. Create virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Launch dashboard

Models are already trained.

```bash
python dashboard/app_dash.py
```

Open

```
http://localhost:8050
```

---

# Project Structure

```
nids-project/
│
├── data/
│   ├── flows/
│   └── raw_pcaps/
│
├── processed/
│   ├── combined_flows.csv
│   ├── dashboard_simulation.csv
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── y_test.npy
│
├── models/
│   ├── autoencoder.keras
│   ├── xgboost.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── threshold.json
│
├── src/
│   ├── merge_clean.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train_autoencoder.py
│   ├── train_xgboost.py
│   ├── train_xgboost_weighted.py
│   ├── evaluate_autoencoder.py
│   ├── evaluate_final_pipeline.py
│   └── evaluate_val_vs_test.py
│
├── dashboard/
│   ├── app_dash.py
│   └── app.py
│
└── requirements.txt
```

---

# Reproducing the Complete Pipeline

Place the **Engelen corrected CSV files** inside

```
data/flows/
```

Then execute:

```bash
# Merge and clean
python src/merge_clean.py

# Exploratory Data Analysis
python src/eda.py

# Preprocessing
python src/preprocess.py

# Train Autoencoder
python src/train_autoencoder.py

# Train XGBoost
python src/train_xgboost.py

# Evaluate
python src/evaluate_autoencoder.py
python src/evaluate_final_pipeline.py
python src/evaluate_val_vs_test.py

# Launch Dashboard
python dashboard/app_dash.py
```

---

# Pipeline Architecture

```
                 Network Flow

                       │

                       ▼

             Feature Extraction
          (Header Statistics Only)

                       │

                       ▼

             MinMaxScaler Transform

                       │

                       ▼

              Autoencoder (Stage 1)

          Reconstruction Error (MSE)

                       │

          ┌────────────┴────────────┐

          │                         │

      Below Threshold          Above Threshold

          │                         │

          ▼                         ▼

      XGBoost                  Confidence Check

     Classification                │

          │                         │

          └────────────┬────────────┘

                       ▼

                Final Prediction
```

---

# Dataset

Dataset used:

**CIC-IDS2017 (Engelen Corrected Version)**

Original CICFlowMeter contains multiple documented feature extraction bugs.

The corrected dataset:

- fixes corrupted flow statistics
- merges attempted attack labels
- restores the intended 15 attack classes
- matches IEEE SPW 2021 benchmark

---

# Preprocessing Pipeline

The preprocessing stage performs:

- Removal of exact duplicate rows
- Removal of non-feature columns
- Stratified Train / Validation / Test split
- Label encoding
- BENIGN class undersampling
- SMOTE oversampling on minority attack classes
- MinMax scaling
- Saving processed NumPy arrays and preprocessing artifacts

---

# Model Architecture

## Stage 1 — Autoencoder

Architecture

```
78
↓

64

↓

32

↓

64

↓

78
```

Activation

- ReLU hidden layers
- Sigmoid output

Training

- BENIGN traffic only
- Early stopping
- Threshold = P95 reconstruction error

Purpose

Detect flows that significantly differ from normal encrypted traffic.

---

## Stage 2 — XGBoost

Configuration

- Objective: `multi:softprob`
- Tree Method: `hist`
- Early Stopping
- Maximum Trees: 500
- Best Iteration: 209

Outputs

- Attack class
- Class probabilities

---

# Confidence-Gated Decision Logic

The Autoencoder alone generates many false alarms.

Instead, predictions are overridden only when:

- Autoencoder detects anomaly
- XGBoost confidence < 0.90

This reduces false positives dramatically while preserving attack recall.

---

# Dashboard

The dashboard is implemented using **Plotly Dash**.

Features

- Live flow simulation
- Real-time attack detection
- Attack distribution visualization
- Autoencoder anomaly score timeline
- Grouped alert feed
- No chart flickering
- WebSocket component updates

---

# Key Design Decisions

| Decision | Selected | Alternative | Reason |
|-----------|----------|-------------|--------|
| Dataset | Engelen Corrected | Original CIC-IDS2017 | Fixes feature extraction bugs |
| Classes | Merge Attempted Labels | 25 Classes | Standard 15-class benchmark |
| Near Duplicates | Keep | Remove | Better operational realism |
| Source Port | Drop | Keep | Minimal learning signal |
| Split | Stratified Random | Day-wise | Preserves all attack classes |
| BENIGN Sampling | Undersample | Keep All | Reduce redundancy |
| Minority class balancing | SMOTE oversampling | Random oversampling / No balancing | Improves learning of rare attack classes by generating synthetic minority samples rather than duplicating existing ones |
| Threshold | P95 Validation | F1 Search | More principled |
| Decision Logic | Confidence Gate | Pure Threshold | Fewer false alarms |
| Dashboard | Plotly Dash | Streamlit | Smooth updates |

---

# Results

## Test Performance

| Pipeline | Macro F1 |
|------------|----------:|
| Autoencoder | 0.8584 |
| XGBoost | 0.8889 |
| Naive AE + XGBoost | 0.8776 |
| **Confidence-Gated Pipeline** | **0.8937** |

---

## Validation vs Test

| Pipeline | Validation | Test |
|------------|----------:|----------:|
| Autoencoder | 0.8588 | 0.8584 |
| XGBoost | 0.8853 | 0.8889 |
| Naive AE + XGBoost | 0.8880 | 0.9186 |
| **Confidence-Gated** | **0.8898** | **0.8937** |

The confidence-gated pipeline shows only a **+0.0039** Macro F1 difference between validation and test, indicating strong generalization.

---

# References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset*. ICISSP.

2. Engelen, G., Rimmer, V., & Joosen, W. (2021). *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*. IEEE SPW.

3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.

4. Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.

Dataset

https://www.unb.ca/cic/datasets/ids-2017.html

Corrected Dataset

https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html

Corrected CICFlowMeter

https://github.com/GintsEngelen/CICFlowMeter

---

# License

This repository is intended for academic and research purposes.

Please cite the original CIC-IDS2017 and Engelen et al. papers if using this work in research.
