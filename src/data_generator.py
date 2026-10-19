"""
COD Shield AI
Improved synthetic COD order dataset generator.

All records are fictional.
The generated labels are simulated ground truth,
not verified real-world fraud outcomes.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_orders.csv"


def generate_orders(
    n_orders: int = 5000,
    abuse_rate: float = 0.10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate overlapping legitimate and abuse scenarios."""

    if n_orders < 1:
        raise ValueError("n_orders must be greater than zero.")

    if not 0 <= abuse_rate <= 1:
        raise ValueError("abuse_rate must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)

    # Assign the simulated ground-truth labels.
    is_abuse = (
        rng.random(n_orders) < abuse_rate
    ).astype(int)

    # Order values overlap between classes.
    order_values = np.clip(
        rng.lognormal(
            mean=6.5,
            sigma=1.0,
            size=n_orders,
        ),
        100,
        100000,
    ).round(2)

    # Both classes can have low or high order bursts.
    # Abuse is more likely to have a high burst, but the
    # feature is not a perfect indicator of the label.
    normal_burst = rng.poisson(
        lam=2.5,
        size=n_orders,
    ) + 1

    abuse_burst = rng.poisson(
        lam=3.5,
        size=n_orders,
    ) + 1

    orders_last_15m = np.where(
        is_abuse == 1,
        abuse_burst,
        normal_burst,
    )

    orders_last_15m = np.clip(
        orders_last_15m,
        1,
        15,
    )

    # Some genuine orders may have failed confirmations.
    normal_failures = rng.binomial(
        n=4,
        p=0.12,
        size=n_orders,
    )

    abuse_failures = rng.binomial(
        n=4,
        p=0.35,
        size=n_orders,
    )

    failed_confirmations = np.where(
        is_abuse == 1,
        abuse_failures,
        normal_failures,
    )

    # Shared devices or accounts can occur legitimately.
    normal_accounts = rng.choice(
        [1, 2, 3, 4, 5],
        size=n_orders,
        p=[0.55, 0.25, 0.12, 0.06, 0.02],
    )

    abuse_accounts = rng.choice(
        [1, 2, 3, 4, 5, 6, 7],
        size=n_orders,
        p=[0.12, 0.16, 0.18, 0.18, 0.16, 0.12, 0.08],
    )

    distinct_accounts = np.where(
        is_abuse == 1,
        abuse_accounts,
        normal_accounts,
    )

    # Confirmation is informative, but not decisive.
    normal_unconfirmed_probability = 0.18
    abuse_unconfirmed_probability = 0.48

    confirmation_probability = np.where(
        is_abuse == 1,
        abuse_unconfirmed_probability,
        normal_unconfirmed_probability,
    )

    confirmation_status = np.where(
        rng.random(n_orders) < confirmation_probability,
        "unconfirmed",
        "confirmed",
    )

    # Payment method is deliberately not a direct label.
    payment_method = rng.choice(
        ["COD", "PREPAID"],
        size=n_orders,
        p=[0.70, 0.30],
    )

    # Assign campaign IDs consistently within each class.
    # These identifiers are metadata, not model features.
    campaign_ids = np.empty(
        n_orders,
        dtype=object,
    )

    for label, prefix in [
        (0, "NORMAL"),
        (1, "ABUSE"),
    ]:
        indices = np.flatnonzero(
            is_abuse == label
        )

        rng.shuffle(indices)

        if len(indices) == 0:
            continue

        # Campaigns contain multiple orders where possible.
        group_size = 10

        campaign_number = (
            np.arange(len(indices)) // group_size
        ) + 1

        campaign_ids[indices] = [
            f"{prefix}_{number:04d}"
            for number in campaign_number
        ]

    df = pd.DataFrame(
        {
            "order_id": [
                f"ORD_{i:06d}"
                for i in range(1, n_orders + 1)
            ],
            "order_value_inr": order_values,
            "orders_last_15m": orders_last_15m,
            "failed_confirmations": failed_confirmations,
            "distinct_accounts": distinct_accounts,
            "confirmation_status": confirmation_status,
            "payment_method": payment_method,
            "campaign_id": campaign_ids,
            "is_simulated_abuse": is_abuse,
        }
    )

    return df


def main() -> None:
    """Generate and save the dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_orders()

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("COD Shield AI: improved dataset created.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total orders: {len(df):,}")

    print(
        "Simulated abuse rate: "
        f"{df['is_simulated_abuse'].mean():.2%}"
    )

    print("\nFirst five orders:")
    print(df.head())

    print("\nLabel distribution:")
    print(
        df["is_simulated_abuse"]
        .value_counts()
        .sort_index()
    )

    print("\nFeature averages by simulated label:")
    print(
        df.groupby("is_simulated_abuse")[
            [
                "order_value_inr",
                "orders_last_15m",
                "failed_confirmations",
                "distinct_accounts",
            ]
        ].mean().round(2)
    )

    print("\nCampaign counts by label:")
    print(
        df.groupby("is_simulated_abuse")[
            "campaign_id"
        ].nunique()
    )


if __name__ == "__main__":
    main()