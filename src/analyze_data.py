"""Basic exploratory analysis for COD Shield AI."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_orders.csv"
)


def main() -> None:
    """Load and analyze the synthetic dataset."""

    df = pd.read_csv(DATA_FILE)

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- LABEL DISTRIBUTION ---")
    print(df["is_simulated_abuse"].value_counts())

    print("\n--- ABUSE RATE ---")
    print(
        f"{df['is_simulated_abuse'].mean():.2%}"
    )

    print("\n--- AVERAGE ORDER VALUE BY LABEL ---")
    print(
        df.groupby("is_simulated_abuse")[
            "order_value_inr"
        ].mean().round(2)
    )

    print("\n--- AVERAGE ORDERS IN LAST 15 MINUTES ---")
    print(
        df.groupby("is_simulated_abuse")[
            "orders_last_15m"
        ].mean().round(2)
    )

    print("\n--- CONFIRMATION STATUS BY LABEL ---")
    print(
        pd.crosstab(
            df["is_simulated_abuse"],
            df["confirmation_status"],
            normalize="index",
        ).round(3)
    )


if __name__ == "__main__":
    main()
