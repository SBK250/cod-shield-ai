"""Prepare synthetic data and model artifacts for cloud deployment."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "synthetic_orders.csv"
)
MODEL_FILE = PROJECT_ROOT / "models" / "cod_risk_model.joblib"
THRESHOLD_FILE = PROJECT_ROOT / "models" / "decision_threshold.json"


def prepare_cloud_files():
    """Generate missing synthetic data and model artifacts."""

    if not DATA_FILE.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data_generator.py"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    if not MODEL_FILE.exists() or not THRESHOLD_FILE.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "train_model.py"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    missing = [
        path for path in (
            DATA_FILE,
            MODEL_FILE,
            THRESHOLD_FILE,
        )
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Could not create required files: "
            + ", ".join(str(path) for path in missing)
        )