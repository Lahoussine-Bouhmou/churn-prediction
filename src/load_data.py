from pathlib import Path
import duckdb
import pandas as pd

from config import DB_PATH, RAW_DATA_PATH

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # renommage explicite depuis les colonnes originales du dataset IBM Telco
    rename_map = {
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
    }

    df = df.rename(columns=rename_map)

    # nettoyage
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce").astype("Int64")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("Int64")

    # cible binaire
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0}).astype(int)

    return df


def load_sql_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_dataframe(df)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nChurn distribution:")
    print(df["churn"].value_counts(dropna=False))

    conn = duckdb.connect(str(DB_PATH))

    schema_sql = load_sql_file(Path("sql/schema.sql"))
    conn.execute(schema_sql)

    conn.register("df_customers", df)

    conn.execute(
        """
        INSERT INTO raw_customers
        SELECT
            customer_id,
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
            churn
        FROM df_customers
        """
    )

    feature_sql = load_sql_file(Path("sql/feature_queries.sql"))
    conn.execute(feature_sql)

    print("\nraw_customers row count:")
    print(conn.execute("SELECT COUNT(*) FROM raw_customers").fetchone()[0])

    print("\nPreview customer_features:")
    preview = conn.execute("SELECT * FROM customer_features LIMIT 5").df()
    print(preview)

    conn.close()
    print(f"\nDatabase created at: {DB_PATH}")


if __name__ == "__main__":
    main()