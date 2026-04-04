from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "churn.duckdb"

RAW_TABLE_NAME = "raw_customers"
FEATURES_VIEW_NAME = "customer_features"