# Enterprise Customer Churn Prediction & Retention Engine
### Classical Machine Learning Retention Platform

---

## 🏛️ Project Architecture & Technical Scope

This repository houses an end-to-end machine learning system built strictly with classical algorithms (No Deep Learning), incorporating:

1. **Robust Feature Preprocessing & Derived Engineering:**
   - Missing value imputation (`median` for numerical, `most_frequent` for categorical).
   - Robust scaling (`RobustScaler`) resistant to billing outliers.
   - Leak-free OneHotEncoding encapsulated inside a strict `ColumnTransformer`.
   - Automated business metric derivations (`ChargesPerTenure`, service interaction ratios).

2. **13 Classical ML Models Benchmarked:**
   - `LogisticRegression` (Champion: 81.4% Accuracy, 0.834 ROC-AUC)
   - `K-Nearest Neighbors`
   - `Decision Tree Classifier`
   - `Random Forest Classifier`
   - `Extra Trees Classifier`
   - `Support Vector Classifier (RBF kernel)`
   - `Gaussian Naive Bayes`
   - `Gradient Boosting Classifier`
   - `AdaBoost Classifier`
   - `HistGradientBoosting Classifier`
   - `Linear Discriminant Analysis (LDA)`
   - `Quadratic Discriminant Analysis (QDA)`
   - **Stacking Meta-Ensemble** (Combining top tree models + linear estimators into a Logistic Regression Meta-Learner)

3. **Unsupervised Clustering:**
   - `K-Means` customer persona segmentation embedded into the dataset for cohort targeting.

4. **Interactive Web Application (Streamlit):**
   - Real-time single-customer diagnostic simulator with adjustable decision alert thresholds.
   - Batch CSV file inference with enriched probability tagging and export.
   - Interactive diagnostic tabs: Model comparison, ROC curves, confusion matrix, feature importances, and exploratory clustering.

5. **Deployment Ready:**
   - Docker & Docker-Compose multi-container setup.
   - GitHub Actions automated CI/CD pipeline for build and artifact testing.

---

## 🚀 Quickstart & Local Setup

```bash
# 1. Setup virtual environment
python -m venv venv
source venv/bin/activate # Windows: .\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Platform
streamlit run app.py
```

---

## 🐳 Docker Container Deployment

```bash
docker-compose up --build -d
```
The application will be serving at `http://localhost:8501`.
