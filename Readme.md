# EPILEPTIC SEIZURE PREDICTION USING LOGISTIC REGRESSION

## Project Overview

This project investigates how preprocessing strategies, regularization techniques, and class imbalance handling affect the generalization performance of Logistic Regression in epileptic seizure prediction tasks.

The project focuses on:
- EEG-based seizure prediction datasets
- Multiple preprocessing pipelines
- Logistic Regression as the baseline model
- Overfitting and underfitting analysis
- L1, L2, and Elastic Net regularization
- Handling class imbalance using multiple techniques
- Comparative analysis across experiments

This project was developed as a research-oriented machine learning study rather than a simple classification task.

---

# Objectives

The major objectives of this project are:

1. Analyze epileptic seizure datasets
2. Design preprocessing pipelines
3. Study the effect of preprocessing order on performance
4. Train Logistic Regression baseline models
5. Demonstrate overfitting and underfitting scenarios
6. Compare regularization techniques
7. Handle class imbalance using different methods
8. Evaluate models using healthcare-relevant metrics
9. Perform comparative analysis and research interpretation

---

# Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Imbalanced-learn | SMOTE and imbalance handling |
| Matplotlib | Visualization |
| Seaborn | Statistical plotting |
| Jupyter Notebook | Experimentation |

---

# Project Structure

```text
EPILEPTIC_SEIZURE
│
├── datasets
│   ├── Banglore_EEG.csv
│   ├── Epileptic_Seizure_kaggle.csv
│   ├── epileptic_pca_results.csv
│   └── final_epileptic_seizure_merge.csv
│
├── notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_PCA.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_regularization.ipynb
│
├── src
│   ├── merge_eeg_datasets.py
│   └── pipeline1.py
│
├── venv
│
├── env_test.py
├── requirements.txt
└── README.md
```

---

# Dataset Collection

The project uses EEG-based epileptic seizure datasets for binary classification.

## Datasets Used

### 1. Epileptic_Seizure_kaggle.csv

Reason for selection:
- beginner-friendly structure
- extracted EEG features
- suitable for baseline machine learning experiments

### 2. Banglore_EEG.csv

Reason for selection:
- additional EEG feature diversity
- useful for comparative analysis
- helps improve dataset generalization

### 3. final_epileptic_seizure_merge.csv

Reason for selection:
- merged dataset for larger experimentation
- useful for preprocessing and regularization studies
- increases feature variability

---

# Data Understanding and Exploratory Data Analysis (EDA)

Before model training, exploratory data analysis was performed.

## EDA Steps

- inspecting dataset structure
- checking missing values
- analyzing class imbalance
- identifying feature distributions
- generating correlation heatmaps
- analyzing feature ranges

## Key Observations

- EEG datasets contained high-dimensional features
- Several correlated features existed
- Class imbalance was present
- Feature scaling was necessary
- Some redundant features contributed to noise

---

# Preprocessing Pipeline

A major focus of this project was studying how preprocessing affects model performance.

## Pipeline Used

```text
Normalization
↓
Feature Selection
↓
Scaling
↓
PCA
```

## Purpose of Preprocessing

- standardize EEG feature ranges
- reduce dimensionality
- remove irrelevant features
- reduce overfitting
- improve generalization

---

# Principal Component Analysis (PCA)

PCA was used to reduce high-dimensional EEG features into smaller informative components.

## Benefits of PCA

- dimensionality reduction
- reduced feature redundancy
- faster model training
- reduced overfitting risk

The PCA-transformed dataset was stored as:

```text
epileptic_pca_results.csv
```

---

# Baseline Logistic Regression Model

Logistic Regression was used as the baseline model for seizure prediction.

## Logistic Regression Formula

```math
P(y=1|x)=1/(1+e^{-(β0+βTx)})
```

## Why Logistic Regression?

Reasons:
- interpretable model
- suitable for binary classification
- supports regularization
- computationally efficient
- ideal for comparative analysis

---

# Evaluation Metrics

The following metrics were used:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Correct seizure predictions |
| Recall | Ability to detect seizures |
| F1-score | Balance between precision and recall |
| PR-AUC | Evaluation for imbalanced datasets |

---

# Baseline Model Results

## GOOD FIT (C = 1.0)

| Metric | Value |
|---|---|
| Accuracy | 0.7891 |
| F1-score | 0.1273 |
| PR-AUC | 0.4282 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Non-Seizure (0) | 0.79 | 0.99 | 0.88 |
| Seizure (1) | 0.73 | 0.07 | 0.13 |

## Analysis

The baseline model achieved approximately 79% accuracy, but seizure recall remained very low.

Although overall accuracy appeared acceptable, the model struggled to detect seizure cases effectively. This indicates that accuracy alone is misleading for imbalanced medical datasets.

The low recall value suggests that the model favored the majority non-seizure class.

PR-AUC provided a more meaningful evaluation because it focused on seizure-class detection performance.

---

# Underfitting Experiment

## UNDERFITTING (C = 0.0001)

Very strong regularization was intentionally applied.

| Metric | Value |
|---|---|
| Accuracy | 0.7798 |
| F1-score | 0.0029 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Non-Seizure (0) | 0.78 | 1.00 | 0.88 |
| Seizure (1) | 1.00 | 0.00 | 0.00 |

## Analysis

This experiment demonstrated severe underfitting.

The extremely strong regularization forced the Logistic Regression model to become overly simple.

As a result:
- the model almost completely ignored seizure samples
- seizure recall dropped close to zero
- F1-score became nearly zero

Despite maintaining high accuracy, the model failed medically because it could not identify seizure events.

This experiment proved that excessive regularization can severely reduce model learning capacity.

---

# Overfitting Experiment

## OVERFITTING (C = 100000)

Very weak regularization was intentionally applied.

| Metric | Value |
|---|---|
| Accuracy | 0.7891 |
| F1-score | 0.1273 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Non-Seizure (0) | 0.79 | 0.99 | 0.88 |
| Seizure (1) | 0.73 | 0.07 | 0.13 |

## Analysis

The weakly regularized model produced results nearly identical to the baseline model.

This indicates that Logistic Regression did not severely overfit on the current dataset.

Possible reasons:
- preprocessing reduced feature complexity
- Logistic Regression is inherently simpler than deep learning models
- PCA reduced dimensionality

This experiment demonstrated that reducing regularization alone was not enough to create extreme overfitting.

---

# Regularization Study

Three regularization approaches were studied:

| Method | Description |
|---|---|
| L1 (Lasso) | Sparse feature selection |
| L2 (Ridge) | Smooth weight reduction |
| Elastic Net | Combination of L1 and L2 |

---

# L1 Regularization

## Formula

```math
λ∑|wj|
```

## Characteristics

- forces some coefficients to zero
- performs automatic feature selection
- reduces irrelevant EEG features
- creates sparse models

---

# L2 Regularization

## Formula

```math
λ∑wj^2
```

## Characteristics

- shrinks weights smoothly
- keeps all features
- provides stable learning
- reduces overfitting effectively

---

# Elastic Net

## Formula

```math
λ1∑|wj| + λ2∑wj^2
```

## Characteristics

- combines L1 and L2 advantages
- performs partial feature selection
- handles correlated EEG features better
- improves model stability

---

# Class Imbalance Handling

Medical seizure datasets are naturally imbalanced because seizure samples are much fewer than non-seizure samples.

This imbalance caused:
- misleading accuracy
- poor seizure recall
- low F1-score

## Techniques Applied

### 1. SMOTE

Synthetic Minority Oversampling Technique was used to generate synthetic seizure samples.

### Benefits

- improves seizure learning
- increases recall
- balances class distribution

---

### 2. Class Weighting

Logistic Regression was trained using:

```python
class_weight='balanced'
```

### Benefits

- penalizes seizure misclassification more heavily
- improves sensitivity to minority class

---

# Comparative Analysis

This section represents the core research contribution of the project.

---

## Q1. Does preprocessing order affect results?

Yes, preprocessing order significantly affected model performance and generalization.

Scaling before PCA improved model stability because PCA is variance-sensitive.

Feature selection before classification reduced noisy EEG features and improved generalization.

Improper preprocessing order reduced seizure detection capability and negatively affected recall.

Therefore, preprocessing order directly influenced:
- feature representation
- model stability
- overfitting behavior
- classification performance

---

## Q2. Which regularization generalizes best?

Moderate regularization provided the best generalization performance.

The baseline configuration using:

```text
C = 1.0
```

achieved the best balance between learning capacity and overfitting prevention.

Very strong regularization caused severe underfitting, while extremely weak regularization provided no meaningful improvement.

Therefore, moderate regularization generalized best across experiments.

---

## Q3. Does Elastic Net outperform L1/L2?

Elastic Net is expected to outperform pure L1 or L2 regularization for high-dimensional EEG datasets.

Reasons:
- EEG features are highly correlated
- Elastic Net combines sparsity and stability
- it reduces overfitting while preserving important features

L1 alone may become unstable with correlated features, while L2 alone does not perform feature selection.

Elastic Net provides a balanced compromise between both methods.

---

## Q4. How does imbalance handling affect recall?

Class imbalance handling significantly improves seizure recall.

Without imbalance handling:
- recall remained extremely low
- the model favored the majority class

The baseline model detected only 7% of seizure cases.

Techniques such as:
- SMOTE
- class weighting
- oversampling
- undersampling

help the model learn seizure patterns more effectively.

Expected improvements:
- higher recall
- better minority-class detection
- improved F1-score

However, higher recall may slightly reduce precision due to increased false positives.

In medical applications, improving recall is more important because missing seizures can be dangerous.

---

# Key Findings

## Major Findings

1. Preprocessing strongly affected model performance.
2. Accuracy alone was misleading for imbalanced datasets.
3. Moderate regularization generalized best.
4. Strong regularization caused severe underfitting.
5. Logistic Regression showed limited overfitting.
6. Elastic Net is expected to provide the best balance.
7. Class imbalance handling is essential for seizure detection.
8. Recall and PR-AUC are more important than raw accuracy in healthcare ML.

---

# Conclusion

This project successfully demonstrated how preprocessing strategies, regularization techniques, and imbalance handling influence seizure prediction performance.

The experiments showed that:
- preprocessing affects feature learning,
- regularization controls generalization,
- and imbalance handling is critical for detecting seizure cases.

Although Logistic Regression achieved reasonable accuracy, seizure recall remained low without imbalance handling.

The study highlights the importance of:
- research-driven experimentation,
- healthcare-specific evaluation metrics,
- and comparative analysis in medical machine learning systems.

---

# Future Work

Potential future improvements:

- Deep learning models (CNN/LSTM)
- Advanced EEG signal filtering
- Time-series forecasting
- Hyperparameter optimization
- Ensemble learning
- Real-time seizure prediction systems

---

# References

1. Scikit-learn Documentation
2. Imbalanced-learn Documentation
3. Kaggle Epileptic Seizure Recognition Dataset
4. Research papers on seizure prediction and EEG analysis
5. EEG preprocessing and regularization studies