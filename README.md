# Lung Cancer Risk Prediction
> Machine Learning · Clinical Study

A complete ML pipeline for early lung cancer risk prediction — classifying patients as **Low**, **Medium**, or **High** risk using clinical and lifestyle data.
From raw data to actionable risk predictions.

---

## The Problem

Lung cancer is a silent disease, often caught too late to treat effectively.

| | Challenge |
|---|---|
| 01 | **Late-stage symptoms** — Diagnosis depends on symptoms that only appear in advanced stages, when treatment options are severely limited |
| 02 | **Complex risk factors** — Environmental, genetic, and lifestyle factors interact in ways that are difficult to assess without systematic analysis |
| 03 | **No early-warning tool** — Clinicians lack a data-driven method to flag high-risk patients before symptoms emerge |
| 04 | **Delayed intervention** — Every stage of delay dramatically reduces survival rates, making proactive risk prediction critical |

---

## Project Objectives

**Four goals shaped this project:**

- 🎯 **Build a risk model** — Predict lung cancer risk as Low, Medium, or High across 5,000 patient records
- 🔍 **Identify key drivers** — Surface the most significant environmental, lifestyle, and biological risk factors
- ⚙️ **Full ML pipeline** — End-to-end steps: data cleaning, encoding, scaling, and multi-model training
- 🏥 **Real-world impact** — A clinically relevant tool supporting early intervention for high-risk patients

---

## Dataset

- **Samples:** 5,000 patient records
- **Features:** 24 clinical and lifestyle factors (Age, Smoking, Air Pollution, Genetic Risk, etc.)
- **Target:** `Level` — Low, Medium, or High cancer risk

---

## Models

| File | Algorithm |
|---|---|
| `Logistic_Regression.py` | Logistic Regression |
| `KNN_Model.py` | K-Nearest Neighbors (k=7) |
| `SVM_Model.py` | Support Vector Machine (RBF kernel) |
| `XGBoost_Model.py` | XGBoost Classifier |
| `Comparison.py` | All models compared side by side |

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| **XGBoost** | **87.80%** | **87.80%** |
| SVM | 85.80% | 85.76% |
| Logistic Regression | 84.00% | 83.85% |
| KNN | 77.50% | 76.99% |

XGBoost achieved the best performance across all metrics.

---

## How to Run

Ensure `lung_cancer.csv` is in the same folder, then:

```bash
python Data_Processing.py       # explore & verify data
python Logistic_Regression.py
python KNN_Model.py
python SVM_Model.py
python XGBoost_Model.py
python Comparison.py            # compare all models
```

---

## Dependencies

```
pandas  numpy  scikit-learn  xgboost  seaborn  matplotlib
```
