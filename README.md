# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn and identifying the main factors behind customer attrition.

This project was built with a simple objective in mind: not just train a model, but build a full pipeline that goes from raw customer data to business-oriented insights. The final result includes data loading with SQL, exploratory analysis, feature engineering, model comparison, explainability, and a Streamlit dashboard for decision support.

## Project overview

Customer churn is an important business problem in industries such as telecom, banking, and subscription services. Predicting churn is useful, but prediction alone is not enough. A good churn project should also help answer practical questions such as:

- Which customers are the most likely to leave?
- What patterns are associated with churn?
- How can a business prioritize retention actions?

That is the purpose of this project.

The dataset used here is the **IBM Telco Customer Churn** dataset. It contains 7,043 customer records and 21 raw features describing customer demographics, subscribed services, contract type, billing information, and churn status.

## Main objective

The goal was to build a clean and reproducible churn prediction pipeline able to:

- prepare the raw data properly,
- predict churn risk,
- explain the predictions,
- and highlight high-risk customer profiles in a way that is understandable from a business perspective.

## Dataset and data quality

The target variable was converted into binary form:

- `0` → customer retained
- `1` → customer churned

The data quality was generally good. The only notable issue was that 11 values were missing in `total_charges`. After inspection, all of them corresponded to customers with `tenure = 0`, which is consistent with newly joined customers who had not yet accumulated total charges.

This made the dataset suitable for modeling without major cleaning difficulties.

## Repository structure

```text
churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── sql/
│   ├── schema.sql
│   └── feature_queries.sql
├── src/
│   ├── config.py
│   └── load_data.py
├── app/
│   └── streamlit_app.py
├── models/
├── reports/
├── README.md
└── requirements.txt
````

## Exploratory analysis

The EDA already showed that churn was not random.

The global churn rate is **26.54%**, which means the problem is moderately imbalanced but still manageable. Several variables appeared strongly related to churn from the beginning.

Customers on **month-to-month contracts** had by far the highest churn rate. **Fiber optic customers** were also more exposed than DSL users or customers without internet service. The payment method mattered as well: **electronic check** was clearly associated with higher churn. Another strong pattern was tenure. New customers were much more likely to churn than long-term customers.

In practice, the EDA suggested a fairly intuitive story: customers who are new, less committed, and more exposed to costly or unstable service configurations are more likely to leave.

## Feature engineering

To make the dataset more informative for the models, I created several derived variables.

The most important engineered features were:

* `tenure_group`, to represent lifecycle stage
* `long_contract`, to distinguish committed customers from short-term ones
* `high_monthly_charges`
* `avg_monthly_value_proxy`
* `subscribed_services_count`
* `payment_group`
* `has_security_bundle`
* `fiber_optic_customer`

These features were designed not only to improve predictive performance, but also to make the final results easier to interpret.

For preprocessing, I used a reusable scikit-learn pipeline with:

* median imputation for numerical variables,
* most-frequent imputation for categorical variables,
* one-hot encoding,
* and scaling for numerical features.

## Models compared

Three models were tested:

* **Logistic Regression**
* **Random Forest**
* **XGBoost**

Logistic Regression was used as the baseline. Random Forest and XGBoost were added to capture more complex relationships in the data.

Because churn prediction is an imbalanced classification problem, I did not rely on accuracy. The evaluation focused on:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score
* Recall@Top10%

That last metric was especially important because churn models are often used to rank customers for retention campaigns rather than simply assign a yes/no label.

## Model performance

After validation, **XGBoost** was selected as the final model.

On the test set, its performance was:

| Metric        |  Value |
| ------------- | -----: |
| ROC-AUC       | 0.8482 |
| PR-AUC        | 0.6519 |
| Precision     | 0.5274 |
| Recall        | 0.7893 |
| F1-score      | 0.6323 |
| Recall@Top10% | 0.2750 |

These results mean that the model captures **27.5% of all churners within the top 10% highest-risk customers**, which corresponds to a **Lift@Top10% of 2.75x** compared with random targeting.

Threshold tuning was also tested. A threshold of `0.52` slightly improved validation F1-score, but on the test set the default threshold of `0.50` remained slightly better. I kept that result because it reflects a realistic modeling scenario: what looks best on validation does not always generalize perfectly.

## What the model is learning

The explainability stage confirmed that the model was learning patterns that make business sense.

The strongest explanatory themes were:

* contract commitment
* customer tenure
* billing variables
* support and security services
* internet configuration
* payment behavior

SHAP analysis showed that the most influential variables included `long_contract`, `tenure`, `total_charges`, `fiber_optic_customer`, `online_security_No`, `tech_support_No`, `monthly_charges`, and `payment_method_Electronic check`.

What matters most here is not one isolated variable, but the combination of several risk signals.

## High-risk and low-risk profiles

One of the most interesting results of the project came from the cohort comparison.

Among the **top 10% highest-risk customers**:

* 100.00% were on month-to-month contracts
* 89.62% used fiber optic internet
* 77.36% paid by electronic check
* 96.23% had no tech support
* 98.11% had no online security
* 91.51% were new customers

Their observed churn rate reached **72.64%**.

By contrast, the **bottom 10% lowest-risk customers** looked almost like the opposite profile:

* 89.62% were on two-year contracts
* 100.00% had a long contract
* 88.68% were long-term customers
* only 5.66% used fiber optic internet
* almost none used electronic check
* their observed churn rate was **0.00%**

This is probably the clearest takeaway of the whole project. The model is not just fitting noise. It is identifying a coherent and interpretable pattern of customer vulnerability versus customer stability.

## Streamlit dashboard

To make the results easier to present, I built a small Streamlit dashboard.

It includes:

* a model overview with the main metrics,
* top churn drivers based on SHAP,
* the distribution of predicted churn scores,
* the highest-risk customers to prioritize,
* a comparison between high-risk and low-risk profiles,
* and a short recommendation section.

The dashboard displays the final XGBoost model and its main results, including ROC-AUC 0.8482, PR-AUC 0.6519, Recall 0.7893, and Recall@Top10% 0.2750 

## Business recommendations

The final results suggest a few practical actions.

First, new customers and month-to-month customers should be prioritized for early retention campaigns. They appear repeatedly among the most vulnerable profiles.

Second, fiber optic customers deserve attention, especially when they also lack tech support or online security. That combination shows up very clearly in the explainability stage.

Third, customers paying by electronic check seem to represent a particularly risky segment. They could be targeted with payment migration incentives or retention offers.

More broadly, this project suggests that churn prevention should focus on **customer lifecycle stage**, **commitment level**, **service stability**, and **support configuration**, rather than only on broad demographic variables.

## How to run the project

Clone the repository, create a virtual environment, install the dependencies, and place the raw dataset in `data/raw/telco_churn.csv`.

Then load the raw data into DuckDB:

```bash
python src/load_data.py
```

Run the notebooks in this order:

1. `01_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`
4. `04_explainability.ipynb`

Finally, launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Suggested visuals

* a churn rate chart by contract type
* a SHAP feature importance chart
* a screenshot of the Streamlit dashboard

## Final noted.

In short, the project is about more than predicting churn. It is about identifying the customers most at risk, understanding why they are at risk, and presenting that information in a way that can support real retention decisions.
