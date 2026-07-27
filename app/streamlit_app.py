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
    ax.set_title("Predicted Churn Risk Distribution")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    plt.tight_layout()
    return fig


def plot_top_features(shap_df: pd.DataFrame, top_n: int = 10):
    plot_df = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_title(f"Top {top_n} Churn Drivers (SHAP)")
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
st.sidebar.title("Dashboard Controls")

top_n_features = st.sidebar.slider(
    "Number of top explanatory features",
    min_value=5,
    max_value=15,
    value=10,
)

top_k_customers = st.sidebar.slider(
    "Number of highest-risk customers to display",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
)

selected_risk_levels = st.sidebar.multiselect(
    "Filter risk levels",
    options=["Very High", "High", "Moderate", "Low", "Very Low"],
    default=["Very High", "High", "Moderate", "Low", "Very Low"],
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Final model:** {model_name}")
st.sidebar.write(f"**Validation-selected threshold:** {validation_threshold:.2f}")


# -----------------------------------------------------------------------------
# Title and context
# -----------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    """
This dashboard summarizes the final churn prediction pipeline:
- final model performance,
- explanatory drivers of churn,
- score distribution,
- highest-risk customers,
- business recommendations.
"""
)

st.info(
    """
**Key insights:** XGBoost is the final model.  
It achieves **84.82% ROC-AUC** and **65.19% PR-AUC** on the test set.  
The highest-risk profile is dominated by **new customers on month-to-month contracts with fiber optic service, electronic check payments, and no support/security services**.
"""
)

st.markdown("---")


# -----------------------------------------------------------------------------
# Section 1 - Model overview
# -----------------------------------------------------------------------------
st.header("1. Model Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", model_name)
col2.metric("ROC-AUC", pct_str(default_metrics["roc_auc"]))
col3.metric("PR-AUC", pct_str(default_metrics["pr_auc"]))
col4.metric("F1-score", pct_str(default_metrics["f1"]))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Precision", pct_str(default_metrics["precision"]))
col6.metric("Recall", pct_str(default_metrics["recall"]))
col7.metric("Recall@Top10%", pct_str(default_metrics["recall_at_top10pct"]))
col8.metric("Predicted Positive Rate", pct_str(default_metrics["predicted_positive_rate"]))

col9, col10 = st.columns(2)
col9.metric("Validation Threshold", f"{validation_threshold:.2f}")
col10.metric("Lift@Top10%", f"{lift_at_10:.2f}x")

with st.expander("See full test metrics table"):
    st.dataframe(test_results.round(4), use_container_width=True)

with st.expander("See validation model comparison"):
    st.dataframe(validation_results.round(4), use_container_width=True)


# -----------------------------------------------------------------------------
# Section 2 - Explainability
# -----------------------------------------------------------------------------
st.header("2. Top Churn Drivers")

st.pyplot(plot_top_features(global_shap_df, top_n=top_n_features), clear_figure=True)

st.markdown(
    """
**Main explanatory themes observed in the final model**
- contract commitment,
- customer tenure,
- billing level,
- support/security services,
- internet configuration,
- payment behavior.
"""
)

with st.expander("View SHAP importance table"):
    st.dataframe(global_shap_df.head(20).round(4), use_container_width=True)


# -----------------------------------------------------------------------------
# Section 3 - Score distribution
# -----------------------------------------------------------------------------
st.header("3. Predicted Churn Risk Distribution")

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

st.subheader("Risk Segmentation")
st.pyplot(plot_risk_level_distribution(scored_test_df), clear_figure=True)


# -----------------------------------------------------------------------------
# Section 4 - High-risk customers
# -----------------------------------------------------------------------------
st.header("4. Highest-Risk Customers to Prioritize")

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
st.header("5. High-Risk vs Low-Risk Profile Comparison")

st.markdown(
    """
This table compares the overall population with the top 10% highest-risk customers and the bottom 10% lowest-risk customers.

**Interpretation:** top-risk customers are overwhelmingly month-to-month, fiber-optic, electronic-check users with no support/security services, while bottom-risk customers are mostly long-tenure customers on long contracts.
"""
)

st.dataframe(profile_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Section 6 - Recommendations
# -----------------------------------------------------------------------------
st.header("6. Business Recommendations")

st.markdown(
    """
### Main takeaways
- The model identifies a clear high-risk profile: new customers, month-to-month contracts, fiber optic service, electronic check payments, and lack of support/security services.
- The lowest-risk customers tend to have long tenure, long contracts, lower monthly charges, and more stable service/payment configurations.

### Recommended actions
1. Prioritize new and month-to-month customers for early retention campaigns.
2. Investigate churn risk among fiber optic customers, especially when support/security services are absent.
3. Target electronic-check customers with payment migration or retention incentives.
4. Promote longer contracts and support/security bundles where relevant.
5. Use churn scores as a prioritization tool for retention operations, not only as a binary classification output.
"""
)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit | Final model: XGBoost | Artifacts loaded from models/ and reports/"
)