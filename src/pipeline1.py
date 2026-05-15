"""
regularization.py
─────────────────────────────────────────────────────────────────────────────
Importable module extracted from 04_regularization.ipynb.

Provides:
  - load_data()              → X, y
  - split_data()             → X_train, X_test, y_train, y_test
  - train_model()            → fitted LogisticRegression (default C=1.0)
  - train_underfit_model()   → fitted LogisticRegression (C=0.0001, strong regularization)
  - train_overfit_model()    → fitted LogisticRegression (C=100000, weak regularization)
  - evaluate_model()         → prints accuracy, F1, classification report
  - compute_pr_auc()         → precision, recall, thresholds, pr_auc
  - plot_pr_curve()          → displays Precision-Recall curve
  - plot_learning_curve()    → displays Learning Curve with train vs validation scores

Usage from any other file in your project:
──────────────────────────────────────────
    from src.regularization import (
        load_data, split_data,
        train_model, train_underfit_model, train_overfit_model,
        evaluate_model, compute_pr_auc, plot_pr_curve, plot_learning_curve,
    )

    X, y                             = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Good fit
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Underfitting experiment
    underfit_model = train_underfit_model(X_train, y_train)
    evaluate_model(underfit_model, X_test, y_test)
    plot_learning_curve(underfit_model, X_train, y_train, title="Underfitting")

    # Overfitting experiment
    overfit_model = train_overfit_model(X_train, y_train)
    evaluate_model(overfit_model, X_test, y_test)
    plot_learning_curve(overfit_model, X_train, y_train, title="Overfitting")
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import learning_curve, train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Epileptic_seizure\datasets\epileptic_pca_results.csv"
TARGET_COL   = "Label"
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH):
    """
    Load the epileptic seizure PCA dataset.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (all columns except Label)
    y : pd.Series     — target column (Label)
    """
    df = pd.read_csv(path)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    return X, y


def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ── Models ────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train, C: float = 1.0) -> LogisticRegression:
    """
    Fit a standard Logistic Regression (default C=1.0, balanced regularization).

    Parameters
    ----------
    X_train : array-like — training features
    y_train : array-like — training labels
    C       : float      — inverse regularization strength (default 1.0)

    Returns
    -------
    model : fitted LogisticRegression
    """
    model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_underfit_model(X_train, y_train) -> LogisticRegression:
    """
    Intentional UNDERFITTING experiment.

    Uses very strong regularization (C=0.0001).
    Small C → heavy penalty → model is too constrained to learn the data.

    Expected result on learning curves:
      - Both training AND validation scores are LOW
      - The two curves are close together (high bias, low variance)

    Returns
    -------
    model : fitted LogisticRegression with C=0.0001
    """
    print("=" * 50)
    print("UNDERFITTING EXPERIMENT  (C=0.0001)")
    print("Strong regularization — model is intentionally too simple")
    print("=" * 50)
    model = LogisticRegression(C=0.0001, max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_overfit_model(X_train, y_train) -> LogisticRegression:
    """
    Intentional OVERFITTING experiment.

    Uses very weak regularization (C=100000).
    Large C → almost no penalty → model memorises training data.

    Expected result on learning curves:
      - Training score is HIGH
      - Validation score is LOW (large gap between the two)
      - High variance, low bias

    Returns
    -------
    model : fitted LogisticRegression with C=100000
    """
    print("=" * 50)
    print("OVERFITTING EXPERIMENT  (C=100000)")
    print("Weak regularization — model is intentionally too complex")
    print("=" * 50)
    model = LogisticRegression(C=100000, max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> None:
    """
    Print accuracy, F1 score, and the full classification report.

    Parameters
    ----------
    model   : fitted classifier
    X_test  : array-like — test features
    y_test  : array-like — true labels
    """
    y_pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred))


def compute_pr_auc(model, X_test, y_test):
    """
    Compute the Precision-Recall curve and PR-AUC score.

    Parameters
    ----------
    model   : fitted classifier (must expose predict_proba)
    X_test  : array-like — test features
    y_test  : array-like — true labels

    Returns
    -------
    precision  : ndarray
    recall     : ndarray
    thresholds : ndarray
    pr_auc     : float
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)
    return precision, recall, thresholds, pr_auc


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_pr_curve(precision, recall) -> None:
    """
    Display the Precision-Recall curve.

    Parameters
    ----------
    precision : ndarray — from compute_pr_auc()
    recall    : ndarray — from compute_pr_auc()
    """
    import matplotlib.pyplot as plt

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


def plot_learning_curve(
    model,
    X_train,
    y_train,
    title:   str = "Learning Curve",
    cv:      int = 5,
    scoring: str = "accuracy",
    n_jobs:  int = -1,
) -> None:
    """
    Generate and plot the Learning Curve for a given model.

    Shows training score vs cross-validation score as training size grows.

    Interpretation
    --------------
    Scenario    | Observation
    ------------|------------------------------------------
    Underfit    | Both scores are LOW and close together
    Overfit     | Train score HIGH, validation score LOW (large gap)
    Good fit    | Both scores are HIGH and close together

    Parameters
    ----------
    model   : fitted or unfitted scikit-learn estimator
    X_train : array-like — training features
    y_train : array-like — training labels
    title   : str        — plot title (default "Learning Curve")
    cv      : int        — number of cross-validation folds (default 5)
    scoring : str        — scoring metric (default "accuracy")
    n_jobs  : int        — parallel jobs (-1 = use all cores)
    """
    import matplotlib.pyplot as plt

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_STATE,
    )

    # Mean and std across CV folds
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(9, 5))

    # Training score
    plt.plot(train_sizes, train_mean, "o-", color="royalblue", label="Training Score")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="royalblue",
    )

    # Validation score
    plt.plot(train_sizes, val_mean, "o-", color="tomato", label="Validation Score")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="tomato",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.capitalize())
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ── Convenience: run everything in one call ───────────────────────────────────
def run_full_pipeline(path: str = DATA_PATH):
    """
    Load data → split → train (good/under/over) → evaluate → PR-AUC
    → learning curves for all three models.

    Returns a dict with all key objects for further use.
    """
    X, y                             = load_data(path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Good fit ──────────────────────────────────────────────────────────────
    print("\n── GOOD FIT (C=1.0) ──")
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    precision, recall, thresholds, pr_auc = compute_pr_auc(model, X_test, y_test)
    plot_pr_curve(precision, recall)
    plot_learning_curve(model, X_train, y_train, title="Good Fit — C=1.0")

    # ── Underfitting ──────────────────────────────────────────────────────────
    print("\n── UNDERFITTING (C=0.0001) ──")
    underfit_model = train_underfit_model(X_train, y_train)
    evaluate_model(underfit_model, X_test, y_test)
    plot_learning_curve(underfit_model, X_train, y_train,
                        title="Underfitting — C=0.0001")

    # ── Overfitting ───────────────────────────────────────────────────────────
    print("\n── OVERFITTING (C=100000) ──")
    overfit_model = train_overfit_model(X_train, y_train)
    evaluate_model(overfit_model, X_test, y_test)
    plot_learning_curve(overfit_model, X_train, y_train,
                        title="Overfitting — C=100000")

    return {
        "model":          model,
        "underfit_model": underfit_model,
        "overfit_model":  overfit_model,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "precision":      precision,
        "recall":         recall,
        "thresholds":     thresholds,
        "pr_auc":         pr_auc,
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_pipeline()