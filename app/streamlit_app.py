from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

ARTIFACT_PATH = MODELS_DIR / "final_model_artifact.joblib"
TEST_DATA_PATH = DATA_DIR / "test_engineered.csv"
GLOBAL_SHAP_PATH = REPORTS_DIR / "global_shap_importance.csv"
RISK_COHORT_PATH = REPORTS_DIR / "risk_cohort_summary.csv"
RISK_PROFILE_PATH = REPORTS_DIR / "risk_profile_summary.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)


def pct_str(x: float) -> str:
    return f"{x * 100:.2f}%"


FEATURE_LABELS = {
    "num__long_contract": "Long-term contract",
    "num__tenure": "Tenure (months)",
    "num__total_charges": "Total charges",
    "num__fiber_optic_customer": "Fiber optic service",
    "cat__online_security_No": "No online security",
    "cat__tech_support_No": "No tech support",
    "num__monthly_charges": "Monthly charges",
    "num__avg_monthly_value_proxy": "Average monthly value",
    "cat__payment_method_Electronic check": "Electronic check payment",
    "cat__contract_Two year": "Two-year contract",
    "cat__paperless_billing_No": "No paperless billing",
    "cat__multiple_lines_No": "No multiple phone lines",
    "cat__contract_Month-to-month": "Month-to-month contract",
    "cat__contract_One year": "One-year contract",
    "cat__streaming_movies_Yes": "Movie streaming subscription",
}


def classify_risk(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Moderate"
    if score >= 0.20:
        return "Low"
    return "Very Low"


def build_scored_test_set(test_df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    model = artifact["pipeline"]

    X_test = test_df.drop(columns=["churn"]).copy()
    y_test = test_df["churn"].copy()

    scores = model.predict_proba(X_test)[:, 1]

    scored = X_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_score"] = scores
    scored["pred_default_050"] = (scored["churn_score"] >= 0.50).astype(int)
    scored["pred_validation_threshold"] = (
        scored["churn_score"] >= artifact["threshold"]
    ).astype(int)
    scored["risk_level"] = scored["churn_score"].apply(classify_risk)

    return scored.sort_values("churn_score", ascending=False).reset_index(drop=True)


def plot_score_distribution(scored_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scored_df["churn_score"], bins=20)
    ax.set_title("Distribution of customer risk scores")
    ax.set_xlabel("Churn risk score")
    ax.set_ylabel("Number of customers")
    plt.tight_layout()
    return fig


def plot_top_features(shap_df: pd.DataFrame, top_n: int = 10):
    plot_df = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = plot_df["feature"].map(FEATURE_LABELS).fillna(plot_df["feature"])
    ax.barh(labels, plot_df["mean_abs_shap"])
    ax.set_title(f"Top {top_n} features by SHAP contribution")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


def plot_risk_level_distribution(scored_df: pd.DataFrame):
    risk_counts = (
        scored_df["risk_level"]
        .value_counts()
        .reindex(["Very High", "High", "Moderate", "Low", "Very Low"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(risk_counts.index, risk_counts.values)
    ax.set_title("Customer Count by Risk Level")
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Number of customers")
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


def validate_required_files():
    required_paths = [
        ARTIFACT_PATH,
        TEST_DATA_PATH,
        GLOBAL_SHAP_PATH,
        RISK_COHORT_PATH,
        RISK_PROFILE_PATH,
    ]
    return [str(path) for path in required_paths if not path.exists()]


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
missing_files = validate_required_files()

if missing_files:
    st.error("Some required files are missing.")
    st.write("Please make sure the modeling and explainability notebooks have been executed.")
    st.write("Missing files:")
    for path in missing_files:
        st.write(f"- {path}")
    st.stop()

artifact = load_artifact(ARTIFACT_PATH)
test_df = load_csv(TEST_DATA_PATH)
global_shap_df = load_csv(GLOBAL_SHAP_PATH)
cohort_df = load_csv(RISK_COHORT_PATH)
profile_df = load_csv(RISK_PROFILE_PATH)

scored_test_df = build_scored_test_set(test_df, artifact)

test_results = artifact["test_results"].copy()
validation_results = artifact["validation_results"].copy()

model_name = artifact["model_name"]
validation_threshold = float(artifact["threshold"])
default_metrics = test_results.loc[test_results["setting"] == "default_0.50"].iloc[0]
lift_at_10 = default_metrics["recall_at_top10pct"] / 0.10


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Explore the results")

top_n_features = st.sidebar.slider(
    "Features to show in the SHAP chart",
    min_value=5,
    max_value=15,
    value=10,
)

top_k_customers = st.sidebar.slider(
    "Customer records to display",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
)

selected_risk_levels = st.sidebar.multiselect(
    "Risk levels in the customer table",
    options=["Very High", "High", "Moderate", "Low", "Very Low"],
    default=["Very High", "High", "Moderate", "Low", "Very Low"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Final model:** {model_name}")
st.sidebar.write("**Overview classification threshold:** 0.50")
st.sidebar.write(f"**Validation-selected threshold:** {validation_threshold:.2f}")


# -----------------------------------------------------------------------------
# Title and context
# -----------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    """
Explore the results of an end-to-end telecom churn prediction project:
model performance, SHAP explanations, customer risk profiles, and retention recommendations.
"""
)

st.caption(
    f"Model: {model_name} · Evaluation data: {len(test_df):,} held-out test customers "
    "from the IBM Telco Customer Churn dataset."
)

st.markdown("---")


# -----------------------------------------------------------------------------
# Section 1 - Model overview
# -----------------------------------------------------------------------------
st.header("1. Model performance")

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{default_metrics['roc_auc']:.4f}")
col2.metric("Churners captured in top 10%", pct_str(default_metrics["recall_at_top10pct"]))
col3.metric("Lift over random targeting", f"{lift_at_10:.2f}×")

st.info(
    f"Selecting the highest-risk 10% of test customers captures "
    f"**{pct_str(default_metrics['recall_at_top10pct'])} of all churners**—"
    f"approximately **{lift_at_10:.2f} times** the coverage of random targeting."
)

st.caption(
    "Precision, recall, F1, and the predicted churn rate below use a threshold of 0.50. "
    f"The validation-selected threshold of {validation_threshold:.2f} is also reported in the full test results."
)
col4, col5, col6, col7, col8 = st.columns(5)
col4.metric("Average precision (PR-AUC)", f"{default_metrics['pr_auc']:.4f}")
col5.metric("Precision", pct_str(default_metrics["precision"]))
col6.metric("Recall", pct_str(default_metrics["recall"]))
col7.metric("F1-score", f"{default_metrics['f1']:.4f}")
col8.metric("Predicted churn rate", pct_str(default_metrics["predicted_positive_rate"]))

with st.expander("See full test metrics table"):
    st.dataframe(test_results.round(4), use_container_width=True)

with st.expander("See validation model comparison"):
    st.dataframe(validation_results.round(4), use_container_width=True)


# -----------------------------------------------------------------------------
# Section 2 - Explainability
# -----------------------------------------------------------------------------
st.header("2. What influences the predictions?")

st.pyplot(plot_top_features(global_shap_df, top_n=top_n_features), clear_figure=True)

st.markdown(
    """
Contract commitment and tenure have the largest individual SHAP contributions.
Billing, internet service, support subscriptions, and payment method also influence the scores.
"""
)
st.caption(
    "Longer bars show larger average contributions to the model's predictions. "
    "This chart shows contribution size; it does not show whether a feature raises or lowers a customer's score."
)

with st.expander("View SHAP importance table"):
    st.dataframe(global_shap_df.head(20).round(4), use_container_width=True)


# -----------------------------------------------------------------------------
# Section 3 - Score distribution
# -----------------------------------------------------------------------------
st.header("3. Customer risk scores")
st.markdown(
    "Scores rank customers from lower to higher predicted churn risk. "
    "The summaries below describe the full test set."
)

st.pyplot(plot_score_distribution(scored_test_df), clear_figure=True)

score_summary = pd.DataFrame(
    {
        "metric": [
            "Average score",
            "Median score",
            "Max score",
            "Min score",
            "Customers with score >= 0.80",
            "Customers with score >= 0.50",
        ],
        "value": [
            round(scored_test_df["churn_score"].mean(), 4),
            round(scored_test_df["churn_score"].median(), 4),
            round(scored_test_df["churn_score"].max(), 4),
            round(scored_test_df["churn_score"].min(), 4),
            int((scored_test_df["churn_score"] >= 0.80).sum()),
            int((scored_test_df["churn_score"] >= 0.50).sum()),
        ],
    }
)

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Score Summary")
    st.dataframe(score_summary, use_container_width=True)

with col_b:
    st.subheader("Risk Cohort Summary")
    st.dataframe(cohort_df.round(4), use_container_width=True)

st.subheader("Customers by risk level")
st.caption(
    "Score bands: Very Low < 0.20; Low 0.20–<0.40; Moderate 0.40–<0.60; "
    "High 0.60–<0.80; Very High ≥ 0.80."
)
st.pyplot(plot_risk_level_distribution(scored_test_df), clear_figure=True)


# -----------------------------------------------------------------------------
# Section 4 - High-risk customers
# -----------------------------------------------------------------------------
st.header("4. Customers to prioritize")
st.markdown(
    "Test records are ordered by risk score, highest first. "
    "Use the sidebar to select risk levels and the number of records to display or download."
)

filtered_scored_df = scored_test_df[scored_test_df["risk_level"].isin(selected_risk_levels)].copy()

risk_columns = [
    "tenure",
    "internet_service",
    "contract",
    "payment_method",
    "monthly_charges",
    "tenure_group",
    "long_contract",
    "fiber_optic_customer",
    "actual_churn",
    "churn_score",
    "risk_level",
]

top_risk_customers = filtered_scored_df[risk_columns].head(top_k_customers).copy()
top_risk_customers["churn_score"] = top_risk_customers["churn_score"].round(4)
top_risk_customers["monthly_charges"] = top_risk_customers["monthly_charges"].round(2)

st.dataframe(top_risk_customers, use_container_width=True)

csv_export = top_risk_customers.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download highest-risk customers as CSV",
    data=csv_export,
    file_name="highest_risk_customers.csv",
    mime="text/csv",
)


# -----------------------------------------------------------------------------
# Section 5 - Risk profile comparison
# -----------------------------------------------------------------------------
st.header("5. High-risk and low-risk customer profiles")

st.markdown(
    """
The table compares the full test set with the highest-risk and lowest-risk 10% of customers. Values are percentages within each group.

All customers in the highest-risk group have month-to-month contracts, and 91.51% have less than 12 months of tenure.
Fiber optic service, electronic check payments, and the absence of tech support or online security are also common.
In the lowest-risk group, 89.62% have two-year contracts.
"""
)

st.dataframe(profile_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Section 6 - Recommendations
# -----------------------------------------------------------------------------
st.header("6. Retention recommendations")

st.markdown(
    """
1. **Start outreach early.** Prioritize new customers on month-to-month contracts for onboarding follow-ups and retention campaigns.
2. **Review the fiber optic customer experience.** Investigate service issues and support needs, particularly among customers without tech support or online security.
3. **Test offers for electronic-check customers.** Evaluate payment migration incentives and retention offers for this segment.
4. **Offer relevant contract and support options.** Test longer contracts and support bundles where they fit customer needs.
5. **Allocate contacts by risk score.** Match the number of selected customers to campaign capacity, then track response, retention, and cost.

These recommendations use the observed profiles to guide campaign design. Campaign results are needed to measure the effect of each offer.
"""
)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Customer Churn Prediction · Built with Streamlit · Model evaluation, SHAP explanations, and retention planning"
)
