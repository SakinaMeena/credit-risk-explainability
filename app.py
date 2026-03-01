
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(
    page_title="Credit Risk Explainability",
    page_icon="💳",
    layout="wide"
)

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/rf_fair.pkl', 'rb') as f:
        rf_fair = pickle.load(f)
    with open('models/clf_results.pkl', 'rb') as f:
        clf_results = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return preprocessor, rf_model, rf_fair, clf_results, feature_names

preprocessor, rf_model, rf_fair, clf_results, feature_names = load_artifacts()

# Sidebar
st.sidebar.title("Credit Risk Decision Support")
st.sidebar.markdown("""
This tool predicts the probability of a borrower experiencing serious 
financial distress within two years using an explainable machine learning 
model trained on 150,000 historical loan records.

**Model:** Random Forest (AUC-ROC: 0.8637)

**Explainability:** SHAP values for every prediction

**Fairness:** Audited for age-based bias using AIF360
""")

model_choice = st.sidebar.radio(
    "Select Model",
    options=['Standard Model', 'Fair Model (Reweighed)'],
    help="The Fair Model applies Reweighing to reduce age-based bias"
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with scikit-learn, SHAP, AIF360, and Streamlit")

# Main title
st.title("Explainable Credit Risk Prediction System")
st.markdown("Enter applicant details to generate a default risk prediction "
            "with SHAP explanation and fairness assessment.")
st.markdown("---")

# Input form
st.subheader("Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=45)
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0, max_value=25000,
        value=5400, step=100)
    revolving_util = st.slider(
        "Revolving Utilization (0-1)", 
        min_value=0.0, max_value=1.0, value=0.3, step=0.01)

with col2:
    debt_ratio = st.slider(
        "Debt Ratio", min_value=0.0, max_value=10.0,
        value=0.35, step=0.01)
    open_credit = st.slider(
        "Open Credit Lines", min_value=0, max_value=30, value=8)
    real_estate = st.slider(
        "Real Estate Loans", min_value=0, max_value=10, value=1)

with col3:
    late_30_59 = st.slider(
        "Times 30-59 Days Late", min_value=0, max_value=10, value=0)
    late_60_89 = st.slider(
        "Times 60-89 Days Late", min_value=0, max_value=10, value=0)
    late_90 = st.slider(
        "Times 90+ Days Late", min_value=0, max_value=10, value=0)
    dependents = st.slider(
        "Number of Dependents", min_value=0, max_value=10, value=0)

# Predict button
if st.button("Generate Prediction", type="primary"):

    # Build input dataframe
    input_data = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'DebtRatio':                            debt_ratio,
        'MonthlyIncome':                        monthly_income,
        'age':                                  age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfTimes90DaysLate':              late_90,
        'NumberOfOpenCreditLinesAndLoans':      open_credit,
        'NumberRealEstateLoansOrLines':         real_estate,
        'NumberOfDependents':                   dependents
    }])

    # Preprocess
    input_processed = preprocessor.transform(input_data)

    # Select model
    model = rf_fair if model_choice == 'Fair Model (Reweighed)' else rf_model

    # Predict
    default_prob = model.predict_proba(input_processed)[0][1]

    # Risk classification
    if default_prob < 0.15:
        risk_label = "LOW RISK"
        risk_color = "green"
    elif default_prob < 0.40:
        risk_label = "MEDIUM RISK"
        risk_color = "orange"
    else:
        risk_label = "HIGH RISK"
        risk_color = "red"

    st.markdown("---")
    st.subheader("Prediction Results")

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{default_prob:.1%}")
    col2.metric("Risk Classification", risk_label)
    col3.metric("Model Used", model_choice.split()[0])

    # Fairness flag
    if age < 40:
        st.warning(
            "⚠️ Fairness Flag: This applicant is under 40. "
            "The standard model has shown age-based bias toward younger "
            "borrowers (Disparate Impact: 0.766, below the 0.8 legal threshold). "
            "Consider using the Fair Model or manual review."
        )

    # SHAP explanation
    st.markdown("---")
    st.subheader("SHAP Explanation — Why this prediction?")

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(input_processed)

    if isinstance(shap_vals, list):
        shap_default = shap_vals[1][0]
    else:
        shap_default = shap_vals[0, :, 1]

    shap_df = pd.DataFrame({
        'feature':    feature_names,
        'shap_value': shap_default
    }).sort_values('shap_value')

    colors = ['#d73027' if v > 0 else '#4575b4'
              for v in shap_df['shap_value']]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value (impact on default probability)')
    ax.set_title(f'Feature Contributions — Default Probability: '
                 f'{default_prob:.1%}')
    plt.tight_layout()
    st.pyplot(fig)

    # Plain English explanation
    top_risk    = shap_df[shap_df['shap_value'] > 0].tail(3)
    top_protect = shap_df[shap_df['shap_value'] < 0].head(3)

    st.markdown("**Primary risk factors:**")
    for _, row in top_risk.iloc[::-1].iterrows():
        st.markdown(f"- **{row['feature']}** increased default "
                    f"probability by {abs(row['shap_value']):.3f}")

    st.markdown("**Primary protective factors:**")
    for _, row in top_protect.iterrows():
        st.markdown(f"- **{row['feature']}** reduced default "
                    f"probability by {abs(row['shap_value']):.3f}")

    # Model comparison
    st.markdown("---")
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(clf_results).set_index('model').round(4)
    results_df.columns = ['AUC-ROC', 'AUC-PR', 'F1']
    st.dataframe(results_df, use_container_width=True)

    # Fairness summary
    st.markdown("---")
    st.subheader("Fairness Audit Summary")

    fairness_data = {
        'Metric': ['Disparate Impact', 'Stat Parity Diff',
                   'Equal Opp Diff',   'Avg Odds Diff'],
        'Original Model': [0.7661, -0.2076, -0.1777, -0.1840],
        'Fair Model':     [0.9974, -0.0026, -0.0007,  0.0368],
        'Threshold':      ['≥ 0.80', '±0.10', '±0.10', '±0.10']
    }
    st.dataframe(
        pd.DataFrame(fairness_data).set_index('Metric'),
        use_container_width=True
    )
