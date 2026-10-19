"""
Prepare COD Shield AI for cloud deployment.

Generates the synthetic dataset and trains the model
when the required local artifacts are missing.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_orders.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "cod_risk_model.joblib"
THRESHOLD_FILE = PROJECT_ROOT / "models" / "decision_threshold.json"


def prepare_project():
    """Generate missing data and train the model if needed."""

    # Step 1: Generate synthetic data if missing.
    if not DATA_FILE.exists():
        print("Generating synthetic demonstration data...")

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data_generator.py"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    # Step 2: Train the model if missing.
    if not MODEL_FILE.exists() or not THRESHOLD_FILE.exists():
        print("Training the COD Shield AI model...")

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    # Step 3: Confirm required files exist.
    missing_files = [
        path
        for path in [DATA_FILE, MODEL_FILE, THRESHOLD_FILE]
        if not path.exists()
    ]

    if missing_files:
        missing_names = ", ".join(
            str(path.relative_to(PROJECT_ROOT))
            for path in missing_files
        )

        raise FileNotFoundError(
            f"Cloud setup could not create: {missing_names}"
        )

    print("COD Shield AI cloud setup completed.")


if __name__ == "__main__":
    prepare_project()
