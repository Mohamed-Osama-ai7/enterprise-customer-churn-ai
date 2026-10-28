# Enterprise Customer Churn Prediction & Retention Platform

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/Mohamed-Osama-ai7/enterprise-customer-churn-ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade classical machine learning system for enterprise subscription retention. Combines rigorous feature engineering, 14 benchmarked classical and meta-ensemble models, unsupervised customer persona clustering, and an interactive diagnostic Streamlit web interface backed by containerized CI/CD.

---

## 📊 Benchmark Results (14 Classical Models)

Evaluated on out-of-sample holdout test data using point-in-time cross-validation:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **81.4%** | **69.9%** | **50.8%** | **0.589** | **0.834** |
| **Stacking Meta-Ensemble** | 80.4% | 68.5% | 46.4% | 0.554 | **0.828** |
| **Linear Discriminant Analysis (LDA)** | 81.0% | 68.7% | 50.3% | 0.580 | 0.824 |
| **AdaBoost Classifier** | 79.6% | 70.4% | 37.7% | 0.491 | 0.821 |
| **Extra Trees Classifier** | 80.0% | **72.6%** | 37.7% | 0.496 | 0.811 |
| **HistGradientBoosting** | 77.4% | 58.2% | 48.6% | 0.530 | 0.809 |
| **Gradient Boosting Classifier** | 78.7% | 62.5% | 46.4% | 0.533 | 0.807 |
| **Gaussian Naive Bayes** | 75.0% | 51.6% | **69.4%** | **0.592** | 0.805 |
| **Quadratic Discriminant Analysis (QDA)**| 74.7% | 51.2% | 67.8% | 0.584 | 0.799 |
| **Support Vector Machine (RBF)** | 79.3% | 67.3% | 40.4% | 0.505 | 0.798 |
| **Random Forest Classifier** | 78.4% | 64.5% | 38.8% | 0.485 | 0.790 |
| **Decision Tree Classifier** | 76.6% | 57.1% | 41.5% | 0.481 | 0.771 |
| **K-Nearest Neighbors (KNN)** | 76.3% | 56.0% | 43.2% | 0.488 | 0.751 |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data ["Data & Feature Engineering"]
        Raw[Enterprise Customer Records] --> Clean[Imputation: Median / Most-Frequent]
        Clean --> Scale[RobustScaler: Outlier Resilience]
        Clean --> OHE[OneHotEncoder ColumnTransformer]
        Scale --> FE[Feature Engineering: ChargesPerTenure, Service Ratios]
        OHE --> FE
    end

    subgraph Modeling ["Modeling & Ensemble Core"]
        FE --> Bench[14-Model Automated Benchmark]
        FE --> KMeans[K-Means Persona Segmentation]
        Bench --> Stack[Stacking Meta-Ensemble: Tree Models + Linear Meta-Learner]
        Bench --> Champ[Champion Pipeline Export: .joblib]
    end

    subgraph Serving ["Inference & Production UI"]
        Champ --> Streamlit[Streamlit Diagnostic Dashboard]
        KMeans --> Streamlit
        Streamlit --> Single[Real-Time Single Customer Diagnostic]
        Streamlit --> Batch[High-Throughput Batch Scoring & CSV Export]
        Streamlit --> Diag[ROC / Confusion Matrix / Feature Importance Plots]
    end

    subgraph DevOps ["Containerization & CI/CD"]
        Streamlit --> Docker[Docker & Docker-Compose]
        Docker --> GHA[GitHub Actions Automated Testing & Validation]
    end
```

---

## ⚙️ Core Technical Capabilities

### 1. Leak-Free Preprocessing Pipeline
- **Numerical Scaling**: `RobustScaler` protects against severe skew in tenure and total billing distributions.
- **Categorical Encoding**: Handled strictly within scikit-learn's `ColumnTransformer` with `handle_unknown='ignore'` to guarantee zero data leakage between training folds and test sets.
- **Domain Feature Derivations**: Automated creation of interaction features such as `ChargesPerTenure` and multi-service adoption ratios.

### 2. Multi-Model Architecture & Meta-Ensembling
- Automated benchmarking of 14 distinct algorithms across linear, kernel, tree-based, and probabilistic paradigms.
- **Stacking Meta-Ensemble**: Combines calibrated probabilities from top-performing tree ensembles and linear estimators into a regularized `LogisticRegression` meta-learner.

### 3. Customer Persona Segmentation
- Integrated `K-Means` unsupervised clustering to group customer accounts by behavioral usage and revenue velocity, enabling targeted retention playbooks alongside binary risk probabilities.

### 4. Interactive Diagnostic Application
- **Single-Customer Simulator**: Dynamic form allowing retention managers to test "what-if" contract amendments, bundle changes, or payment method switches to see real-time risk reduction.
- **Batch CSV Inference**: Upload raw enterprise batches, compute churn probabilities, and download prioritized retention rosters.
- **Explainability & Analytics**: In-app inspection of model ROC-AUC curves, feature importance rankings, and confusion matrices.

---

## 📁 Repository Structure

```
enterprise-customer-churn-ai/
├── data/                               # Dataset files
├── models/                             # Pre-trained pipeline artifacts
├── plots/                              # Benchmark visualizations & ROC curves
├── production_ml_system/               # Production deployment package
│   ├── app.py                          # Streamlit interactive application
│   ├── train.py                        # Full training, benchmarking, and export script
│   ├── Dockerfile                      # Production container image
│   ├── docker-compose.yml              # Multi-container orchestration
│   ├── requirements.txt                # Pinned production dependencies
│   ├── data/
│   │   ├── customer_churn.csv          # Base training dataset
│   │   └── customer_churn_with_clusters.csv
│   ├── models/
│   │   ├── champion_pipeline.joblib    # Serialized champion model
│   │   └── stacking_pipeline.joblib    # Serialized stacking ensemble
│   ├── plots/
│   │   ├── models_benchmark.csv        # Comprehensive performance metrics
│   │   ├── champion_confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── kmeans_clustering.png
│   │   ├── model_comparison.png
│   │   └── multi_model_roc.png
│   └── .github/workflows/
│       └── deploy.yml                  # Automated CI/CD workflow
├── requirements.txt                    # Top-level dependencies
└── README.md                           # Master project documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Mohamed-Osama-ai7/enterprise-customer-churn-ai.git
cd enterprise-customer-churn-ai/production_ml_system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Run with Docker

```bash
docker-compose up --build
```

### 4. Retrain Models & Reproduce Benchmarks

```bash
python train.py
```

This will run the full 14-model cross-validation pipeline, update `models_benchmark.csv`, regenerate all visualization plots, and serialize the champion pipeline to `models/champion_pipeline.joblib`.

---

## 📄 License & Attribution

Developed by **[Mohamed Osama](https://github.com/Mohamed-Osama-ai7)**.  
Released under the [MIT License](LICENSE).
