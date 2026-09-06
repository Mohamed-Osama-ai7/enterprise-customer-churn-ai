# Enterprise Customer Churn Prediction & Retention Engine
### Production-Grade Classical Machine Learning Platform

---

## 🏛️ Project Architecture & Technical Scope

This repository houses an end-to-end industrial machine learning system built strictly with classical algorithms (No Deep Learning), incorporating:

1. **Robust Feature Preprocessing & Derived Engineering:**
   - Missing value imputation (`median` for numerical, `most_frequent` for categorical).
   - Robust scaling (`RobustScaler`) resistant to billing outliers.
   - Leak-free OneHotEncoding encapsulated inside a strict `ColumnTransformer`.
   - Automated business metric derivations (`ChargesPerTenure`, service interaction ratios).

2. **14 Classical ML Models Benchmarked:**
   - `LogisticRegression`
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
   - **Soft Voting Ensemble** (Weighted consensus probability model)

3. **Unsupervised Clustering:**
   - `K-Means` customer persona segmentation embedded into the dataset for cohort targeting.

4. **Production Web Application (Streamlit):**
   - Real-time single-customer diagnostic simulator with adjustable decision alert thresholds.
   - Batch CSV file inference with enriched probability tagging and export.
   - Interactive diagnostic tabs: Model comparison, ROC curves, confusion matrix, feature importances, and exploratory clustering.

5. **Deployment Ready:**
   - Streamlit Community Cloud.
   - Docker & Docker-Compose production setup.
   - GitHub Actions automated CI/CD pipeline for build and artifact testing.

---

## 🚀 Quickstart & Local Setup

```bash
# 1. Unzip the project folder
unzip production_ml_system.zip
cd production_ml_system

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit Platform
streamlit run app.py
```

---

## 🐳 Docker Container Deployment

```bash
docker-compose up --build -d
```
The application will be serving at `http://localhost:8501`.
