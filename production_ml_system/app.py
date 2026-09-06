import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Enterprise Churn AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive look
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 5px solid #2563eb;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    champ_path = "models/champion_pipeline.joblib"
    stack_path = "models/stacking_pipeline.joblib"
    
    if not os.path.exists(champ_path):
        st.error("Model artifacts missing. Please run `python train.py` first.")
        st.stop()
       # force cache update - v1.1
    champion = joblib.load(champ_path)
    stacking = joblib.load(stack_path) if os.path.exists(stack_path) else None
    return champion, stacking

champion_pipeline, stacking_pipeline = load_models()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("Model Controller")
    st.caption("Production ML Inference Pipeline v2.4")
    
    selected_architecture = st.selectbox(
        "Inference Engine",
        ["Tuned Champion (Gradient Boosting)", "Stacking Meta-Ensemble"]
    )
    
    confidence_threshold = st.slider(
        "Decision Threshold (Alert Level)",
        min_value=0.20, max_value=0.80, value=0.50, step=0.05,
        help="Adjusting classification cutoff for high-risk customer interventions."
    )
    
    st.divider()
    st.markdown("### System Health")
    st.success("🟢 Artifacts Loaded (In-Memory)")
    st.info("⚡ Latency: < 15ms")

st.title("Enterprise Customer Retention & Churn AI Engine")
st.markdown("Automated Classical Machine Learning Architecture featuring Stacking Ensembles, Outlier-Robust Preprocessing, and Dynamic Risk Scoring.")

tabs = st.tabs([
    "🎯 Real-Time Prediction",
    "📁 Batch CSV Inference",
    "📊 Benchmarking & Models",
    "🔍 Feature Importance",
    "🧭 Exploratory Data Analysis & Clusters"
])

active_pipeline = champion_pipeline if "Champion" in selected_architecture else stacking_pipeline

# TAB 1: Real-time inference
with tabs[0]:
    st.subheader("Customer Profile Telemetry")
    
    with st.form("single_prediction_form"):
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=14, step=1)
        with r1_c2:
            age = st.number_input("Customer Age", min_value=18, max_value=95, value=38, step=1)
        with r1_c3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=85.5, step=0.5)
        with r1_c4:
            total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=10000.0, value=1200.0, step=10.0)

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            tickets = st.number_input("Support Tickets (Last 90d)", min_value=0, max_value=15, value=2, step=1)
        with r2_c2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with r2_c3:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with r2_c4:
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

        r3_c1, r3_c2, r3_c3 = st.columns(3)
        with r3_c1:
            security = st.selectbox("Online Security Addon", ["No", "Yes"])
        with r3_c2:
            tech_support = st.selectbox("Tech Support Addon", ["No", "Yes"])
        with r3_c3:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        submitted = st.form_submit_button("Run Diagnostic Inference", type="primary")

    if submitted:
        # Automated Derived Feature Engineering (matches training pipeline)
        charges_per_tenure = total_charges / (tenure + 1e-5)
        
        payload = pd.DataFrame([{
            "TenureMonths": tenure,
            "CustomerAge": age,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "ChargesPerTenure": np.round(charges_per_tenure, 2),
            "SupportTickets": tickets,
            "ContractType": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "OnlineSecurity": security,
            "TechSupport": tech_support,
            "PaperlessBilling": paperless,
            "Dependents": dependents
        }])

        proba = active_pipeline.predict_proba(payload)[0][1]
        is_churn = proba >= confidence_threshold

        st.divider()
        res1, res2, res3 = st.columns([1.5, 1, 1])
        
        with res1:
            if is_churn:
                st.error("🚨 **PREDICTION: CRITICAL CHURN RISK DETECTED**")
                st.write("Customer shows high churn logit sensitivity. Recommended action: Route to Retention Specialist immediately.")
            else:
                st.success("✅ **PREDICTION: ACCOUNT RETAINED (STABLE)**")
                st.write("Customer profile indicators are healthy with strong retention propensity.")

        with res2:
            st.metric(label="Calculated Churn Probability", value=f"{proba * 100:.1f}%", delta=f"Cutoff: {confidence_threshold*100:.0f}%", delta_color="inverse")

        with res3:
            st.metric(label="Ratio Charges/Tenure", value=f"${charges_per_tenure:.1f}")

# TAB 2: Batch Inference
with tabs[1]:
    st.subheader("Batch Dataset Inference (CSV Export)")
    uploaded_file = st.file_uploader("Upload customer CSV file to score entire portfolio", type=["csv"])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Input Data Preview (First 5 records):", batch_df.head())
        
        if st.button("Score Batch File"):
            # Apply feature derivation if not present
            if "ChargesPerTenure" not in batch_df.columns:
                batch_df["ChargesPerTenure"] = batch_df["TotalCharges"] / (batch_df["TenureMonths"] + 1e-5)
            
            features_df = batch_df.drop(columns=["Churn"], errors="ignore")
            predictions = active_pipeline.predict(features_df)
            probabilities = active_pipeline.predict_proba(features_df)[:, 1]
            
            output_df = batch_df.copy()
            output_df["Predicted_Churn"] = predictions
            output_df["Churn_Probability"] = np.round(probabilities, 4)
            output_df["Risk_Level"] = np.where(probabilities > 0.65, "High", np.where(probabilities > 0.35, "Medium", "Low"))
            
            st.success(f"Successfully processed {len(output_df)} customer accounts.")
            st.dataframe(output_df.head(10))
            
            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Enriched Predictions CSV", data=csv, file_name="churn_predictions_scored.csv", mime="text/csv")

# TAB 3: Benchmarks & Model comparison
with tabs[2]:
    st.subheader("Multi-Model Benchmarking (14 Classical Machine Learning Architectures)")
    
    if os.path.exists("plots/models_benchmark.csv"):
        b_df = pd.read_csv("plots/models_benchmark.csv")
        st.dataframe(b_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#dbeafe"), use_container_width=True)
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if os.path.exists("plots/model_comparison.png"):
            st.image("plots/model_comparison.png", caption="F1-Score Comparison Across All 14 Algorithms")
    with col_b2:
        if os.path.exists("plots/multi_model_roc.png"):
            st.image("plots/multi_model_roc.png", caption="Comparative ROC-AUC of Top 5 Contenders")

# TAB 4: Feature Importance & Diagnostics
with tabs[3]:
    st.subheader("Model Explainability & Diagnostic Evaluation")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if os.path.exists("plots/feature_importance.png"):
            st.image("plots/feature_importance.png", caption="Gini Feature Importance (Champion Model)")
    with col_d2:
        if os.path.exists("plots/champion_confusion_matrix.png"):
            st.image("plots/champion_confusion_matrix.png", caption="Confusion Matrix on Unseen 20% Stratified Holdout")

# TAB 5: EDA & Clustering
with tabs[4]:
    st.subheader("Exploratory Data Analysis & Unsupervised Clustering")
    if os.path.exists("plots/eda_overview.png"):
        st.image("plots/eda_overview.png", caption="Global Feature Distributions and Heatmap")
    st.divider()
    if os.path.exists("plots/kmeans_clustering.png"):
        st.image("plots/kmeans_clustering.png", caption="Unsupervised K-Means Customer Persona Segmentation")
