"""Load corrupted CSVs, run deterministic checks, print and save findings.

Production path:

    data/corrupted/*.csv  ->  DQ Engine  ->  outputs/dq_findings.json

injected_issues.json is never opened by this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .checks import run_all_checks
from .reporting import print_console_summary, write_findings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
CORRUPTED_DIR = PROJECT_ROOT / "data" / "corrupted"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "dq_findings.json"

CSV_DTYPES = {
    "customers": {"customer_id": "string"},
    "campaigns": {"campaign_id": "string"},
    "interactions": {
        "interaction_id": "string",
        "customer_id": "string",
        "campaign_id": "string",
    },
    "opportunities": {"opportunity_id": "string", "customer_id": "string"},
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_corrupted_datasets(data_dir: Path = CORRUPTED_DIR) -> dict[str, pd.DataFrame]:
    """Load the four corrupted CSV files. Only CSV names are read."""
    datasets = {}
    for name, dtypes in CSV_DTYPES.items():
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Generate corrupted data first: "
                "python src/inject_quality_issues.py"
            )
        datasets[name] = pd.read_csv(csv_path, dtype=dtypes)
    return datasets


def run_engine(
    data_dir: Path = CORRUPTED_DIR,
    config_path: Path = CONFIG_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list:
    config = load_config(config_path)
    datasets = load_corrupted_datasets(data_dir)
    findings = run_all_checks(datasets, config)
    write_findings(findings, output_path)
    return findings


def main() -> None:
    findings = run_engine()
    print_console_summary(findings)
    print()
    print("Wrote", OUTPUT_PATH)


if __name__ == "__main__":
    main()
