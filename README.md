# Explainable PHM System using PCA-HMM for Turbofan Engines

## ⁕ Project Overview

This project implements an Explainable Prognostic Health Management (PHM) system for predicting the Remaining Useful Life (RUL) of turbofan engines.

The system uses dimensionality reduction and probabilistic state modeling to analyze degradation patterns and estimate failure progression.

---

## ⁕ Dataset

We use the NASA C-MAPSS Turbofan Engine Degradation Dataset.

- Each engine operates until failure.
- Multiple sensor measurements are recorded at each cycle.
- The objective is to estimate Remaining Useful Life (RUL).

Dataset includes:
- Engine ID
- Time in cycles
- Operational settings
- 21 sensor measurements

---

## ⁕ Methodology

The system pipeline consists of the following stages:

### 1️⃣ Data Preprocessing
- Generate Remaining Useful Life (RUL) labels
- Remove constant sensors
- Standardize sensor measurements

### 2️⃣ Principal Component Analysis (PCA)
- Reduce dimensionality
- Extract dominant degradation trends
- Remove noise and redundancy

### 3️⃣ Hidden Markov Model (HMM)
- Model degradation stages as hidden states
- Learn state transition probabilities
- Capture temporal degradation behavior

### 4️⃣ Expectation-Maximization (EM)
- Train HMM parameters iteratively
- Maximize likelihood of observed sequences

### 5️⃣ RUL Estimation
- Infer current hidden state
- Estimate time to failure state

---

## ⁕ Project Structure

```
PHM_PCA_HMM/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── pca_model.py
│   ├── hmm_model.py
│   ├── topology.py
│   ├── inference.py
│   └── evaluation.py
│
├── results/
├── models/
├── main.py
└── README.md
```

---

## ⁕ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- hmmlearn

---

## ⁕ Objective

To build a modular and explainable health monitoring system capable of:

- Modeling degradation behavior
- Identifying health states
- Estimating Remaining Useful Life
- Supporting predictive maintenance decisions

---

## ⁕ Team Members

- Karthik K Jash
- R Harishanker
- Bharat Krishna

---

## ⁕ Future Work

- Implement constrained left-right topology
- Compare with regression-based RUL models
- Improve state interpretability
- Deploy as real-time monitoring framework