import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans


def create_dataset(filepath="data/customer_churn.csv", n_samples=3500, random_state=42):
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    tenure = np.random.gamma(shape=2.0, scale=14.0, size=n_samples).clip(1, 72).astype(int)
    age = np.random.normal(42, 12, size=n_samples).clip(18, 85).astype(int)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.50, 0.28, 0.22])
    internet = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.35, 0.50, 0.15])
    payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples, p=[0.35, 0.20, 0.25, 0.20])
    security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62])
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.36, 0.64])
    paperless = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    num_tickets = np.random.poisson(lam=1.2, size=n_samples).clip(0, 10)

    monthly = np.where(
        internet == "Fiber optic", np.random.normal(88, 14, n_samples),
        np.where(internet == "DSL", np.random.normal(52, 10, n_samples), np.random.normal(24, 4, n_samples))
    ).clip(18.5, 125.0)

    total_charges = (monthly * tenure) + np.random.normal(0, 35, n_samples)
    total_charges = np.maximum(total_charges, monthly)
    charges_per_tenure = total_charges / (tenure + 1e-5)

    logits = (
        -2.2
        - 0.055 * tenure
        + 0.022 * monthly
        + 0.015 * (age > 60)
        + 1.35 * (contract == "Month-to-month")
        - 1.10 * (contract == "Two year")
        + 0.95 * (internet == "Fiber optic")
        + 0.55 * (payment == "Electronic check")
        - 0.75 * (security == "Yes")
        - 0.70 * (tech_support == "Yes")
        + 0.35 * (num_tickets >= 3)
        + 0.25 * (monthly > 90) * (tenure < 12)
    )
    p = 1 / (1 + np.exp(-logits))
    churn = np.random.binomial(1, p)

    df = pd.DataFrame({
        "TenureMonths": tenure,
        "CustomerAge": age,
        "MonthlyCharges": np.round(monthly, 2),
        "TotalCharges": np.round(total_charges, 2),
        "ChargesPerTenure": np.round(charges_per_tenure, 2),
        "SupportTickets": num_tickets,
        "ContractType": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "OnlineSecurity": security,
        "TechSupport": tech_support,
        "PaperlessBilling": paperless,
        "Dependents": dependents,
        "Churn": churn
    })

    nan_idx = np.random.choice(n_samples, size=25, replace=False)
    df.loc[nan_idx, "TotalCharges"] = np.nan
    df.to_csv(filepath, index=False)
    return df


def execute_pipeline():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    data_file = "data/customer_churn.csv"

    if not os.path.exists(data_file):
        df = create_dataset(data_file)
    else:
        df = pd.read_csv(data_file)

    # 1. Feature Preprocessing Pipeline
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_features = ["TenureMonths", "CustomerAge", "MonthlyCharges", "TotalCharges", "ChargesPerTenure", "SupportTickets"]
    categorical_features = ["ContractType", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling", "Dependents"]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Benchmarking all Classical ML Algorithms
    base_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis()
    }

    results = []
    trained_pipes = {}

    for name, clf in base_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        trained_pipes[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        })

    # Meta Stacking
    stack_estimators = [
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))
    ]
    meta_model = LogisticRegression(random_state=42)
    stack_clf = StackingClassifier(estimators=stack_estimators, final_estimator=meta_model, cv=5)
    stacking_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", stack_clf)
    ])
    stacking_pipe.fit(X_train, y_train)
    trained_pipes["Stacking Ensemble"] = stacking_pipe
    y_pred_st = stacking_pipe.predict(X_test)
    y_proba_st = stacking_pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": "Stacking Ensemble (Meta-Learner)",
        "Accuracy": accuracy_score(y_test, y_pred_st),
        "Precision": precision_score(y_test, y_pred_st, zero_division=0),
        "Recall": recall_score(y_test, y_pred_st, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_st, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_st)
    })

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    results_df.to_csv("plots/models_benchmark.csv", index=False)
    print("\n--- Model Benchmark Table ---")
    print(results_df.to_string(index=False))

    # Champion Tuning
    print("\n[INFO] Hyperparameter Tuning Champion Architecture...")
    tune_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 160],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 4]
    }

    grid = GridSearchCV(tune_pipeline, param_grid, scoring="f1", cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42), n_jobs=-1)
    grid.fit(X_train, y_train)
    champion_pipeline = grid.best_estimator_

    # Serializing
    joblib.dump(champion_pipeline, "models/champion_pipeline.joblib")
    joblib.dump(stacking_pipe, "models/stacking_pipeline.joblib")
    print("\n[SUCCESS] Production pipelines persisted to disk.")


if __name__ == "__main__":
    execute_pipeline()
