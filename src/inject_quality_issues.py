"""Copy clean datasets and inject seven known data-quality issues.

This script never writes to data/clean/. It only reads those files, modifies
copies in memory, and saves results under data/corrupted/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CORRUPTED_DIR = PROJECT_ROOT / "data" / "corrupted"
MANIFEST_PATH = CORRUPTED_DIR / "injected_issues.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML settings used by generation and corruption."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seeds(seed: int) -> np.random.Generator:
    """Fix Python and NumPy seeds so corruption patterns are stable."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def read_clean_datasets(clean_dir: Path = CLEAN_DIR) -> dict[str, pd.DataFrame]:
    """Load clean CSVs. Missing files mean generate_data.py has not been run."""
    dtypes = {
        "customers": {"customer_id": "string"},
        "campaigns": {"campaign_id": "string"},
        "interactions": {
            "interaction_id": "string",
            "customer_id": "string",
            "campaign_id": "string",
        },
        "opportunities": {"opportunity_id": "string", "customer_id": "string"},
    }
    datasets = {}
    for name in ("customers", "campaigns", "interactions", "opportunities"):
        path = clean_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Generate clean data first: python src/generate_data.py"
            )
        datasets[name] = pd.read_csv(path, dtype=dtypes[name])
    return datasets


def inject_duplicate_customers(
    customers: pd.DataFrame, rng: np.random.Generator, rate: float
) -> tuple[pd.DataFrame, int]:
    """DQ001: append exact copies so customer_id is no longer unique."""
    n_dup = int(round(len(customers) * rate))
    chosen = rng.choice(customers.index.to_numpy(), size=n_dup, replace=False)
    duplicates = customers.loc[chosen].copy()
    combined = pd.concat([customers, duplicates], ignore_index=True)
    order = rng.permutation(len(combined))
    return combined.iloc[order].reset_index(drop=True), n_dup


def inject_missing_customer_ids(
    interactions: pd.DataFrame, rng: np.random.Generator, rate: float
) -> tuple[pd.DataFrame, int]:
    """DQ002: blank customer_id on a controlled share of interactions."""
    n_missing = int(round(len(interactions) * rate))
    chosen = rng.choice(interactions.index.to_numpy(), size=n_missing, replace=False)
    interactions.loc[chosen, "customer_id"] = pd.NA
    return interactions, n_missing


def inject_broken_customer_references(
    interactions: pd.DataFrame,
    rng: np.random.Generator,
    rate: float,
    fictional_id_start: int,
) -> tuple[pd.DataFrame, int]:
    """DQ003: point interactions at customer IDs that do not exist in the master."""
    n_broken = int(round(len(interactions) * rate))
    eligible = interactions.index[interactions["customer_id"].notna()].to_numpy()
    chosen = rng.choice(eligible, size=n_broken, replace=False)
    fake_ids = [f"C{fictional_id_start + i:06d}" for i in range(n_broken)]
    interactions.loc[chosen, "customer_id"] = fake_ids
    return interactions, n_broken


def inject_missing_campaign_attribution(
    interactions: pd.DataFrame, rng: np.random.Generator, rate: float
) -> tuple[pd.DataFrame, int]:
    """DQ004: drop campaign_id on rows that currently have attribution.

    Rows that were generated without a campaign stay as legitimate nulls.
    """
    eligible = interactions.index[interactions["campaign_id"].notna()].to_numpy()
    n_remove = int(round(len(eligible) * rate))
    chosen = rng.choice(eligible, size=n_remove, replace=False)
    interactions.loc[chosen, "campaign_id"] = pd.NA
    return interactions, n_remove


def inject_industry_null_spike(
    customers: pd.DataFrame,
    rng: np.random.Generator,
    snapshot_date: str,
    window_days: int,
    null_rate: float,
) -> tuple[pd.DataFrame, int]:
    """DQ005: raise industry NULLs only inside a recent last_updated window."""
    last_updated = pd.to_datetime(customers["last_updated"])
    snapshot = pd.Timestamp(snapshot_date)
    window_start = snapshot - pd.Timedelta(days=window_days - 1)
    subgroup = customers.index[last_updated >= window_start].to_numpy()
    n_null = int(round(len(subgroup) * null_rate))
    chosen = rng.choice(subgroup, size=n_null, replace=False)
    customers.loc[chosen, "industry"] = pd.NA
    return customers, n_null


def inject_stale_opportunities(
    opportunities: pd.DataFrame, stale_days: int
) -> tuple[pd.DataFrame, int]:
    """DQ006: shift opportunity last_updated backward so the table looks older."""
    shifted = pd.to_datetime(opportunities["last_updated"]) - pd.Timedelta(days=stale_days)
    opportunities["last_updated"] = shifted.dt.strftime("%Y-%m-%d")
    return opportunities, len(opportunities)


def inject_interaction_volume_drop(
    interactions: pd.DataFrame,
    rng: np.random.Generator,
    snapshot_date: str,
    days_before_snapshot: int,
    target_volume: int,
) -> tuple[pd.DataFrame, int, str]:
    """DQ007: keep only a small number of rows on one recent day."""
    snapshot = pd.Timestamp(snapshot_date)
    affected_day = (snapshot - pd.Timedelta(days=days_before_snapshot)).strftime("%Y-%m-%d")
    on_day = interactions.index[interactions["interaction_date"] == affected_day].to_numpy()
    if len(on_day) <= target_volume:
        raise ValueError(
            f"Day {affected_day} has {len(on_day)} rows; cannot reduce to {target_volume}."
        )
    n_remove = len(on_day) - target_volume
    drop_idx = rng.choice(on_day, size=n_remove, replace=False)
    remaining = interactions.drop(index=drop_idx).reset_index(drop=True)
    return remaining, n_remove, affected_day


def build_manifest(entries: list[dict]) -> list[dict]:
    """Store ground truth for tests only. Future detection must not read this file."""
    return entries


def save_corrupted_datasets(datasets: dict[str, pd.DataFrame], output_dir: Path = CORRUPTED_DIR) -> None:
    """Write corrupted copies. The clean directory is never passed here."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in datasets.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def inject_all(config: dict | None = None) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Read clean data, corrupt copies, and return datasets plus the manifest."""
    config = config or load_config()
    rng = set_seeds(config["random_seed"])
    corruption = config["corruption"]

    clean = read_clean_datasets()
    customers = clean["customers"].copy()
    campaigns = clean["campaigns"].copy()
    interactions = clean["interactions"].copy()
    opportunities = clean["opportunities"].copy()

    customers, n_industry_nulls = inject_industry_null_spike(
        customers,
        rng,
        config["snapshot_date"],
        corruption["dq005_window_days"],
        corruption["dq005_industry_null_rate"],
    )
    customers, n_dup = inject_duplicate_customers(
        customers, rng, corruption["dq001_duplicate_customer_rate"]
    )

    interactions, n_missing_ids = inject_missing_customer_ids(
        interactions, rng, corruption["dq002_missing_customer_id_rate"]
    )
    interactions, n_broken = inject_broken_customer_references(
        interactions,
        rng,
        corruption["dq003_broken_reference_rate"],
        corruption["dq003_fictional_id_start"],
    )
    interactions, n_missing_campaigns = inject_missing_campaign_attribution(
        interactions, rng, corruption["dq004_missing_campaign_rate"]
    )
    interactions, n_dropped, affected_day = inject_interaction_volume_drop(
        interactions,
        rng,
        config["snapshot_date"],
        corruption["dq007_days_before_snapshot"],
        corruption["dq007_target_volume"],
    )

    opportunities, n_stale = inject_stale_opportunities(
        opportunities, corruption["dq006_stale_days"]
    )

    manifest = build_manifest(
        [
            {
                "rule_id": "DQ001",
                "issue_name": "Duplicate Customers",
                "affected_dataset": "customers",
                "affected_field": "customer_id",
                "intended_rate_or_condition": "approximately 1.5%",
                "number_of_records_modified": n_dup,
                "description": (
                    "Exact copies of existing customer records were appended so some "
                    "customer_id values appear more than once."
                ),
            },
            {
                "rule_id": "DQ002",
                "issue_name": "Missing Customer IDs",
                "affected_dataset": "interactions",
                "affected_field": "customer_id",
                "intended_rate_or_condition": "approximately 4%",
                "number_of_records_modified": n_missing_ids,
                "description": "Interaction records were given a NULL customer_id.",
            },
            {
                "rule_id": "DQ003",
                "issue_name": "Broken Customer References",
                "affected_dataset": "interactions",
                "affected_field": "customer_id",
                "intended_rate_or_condition": "approximately 2%",
                "number_of_records_modified": n_broken,
                "description": (
                    "Interaction records were assigned fictional customer IDs that do not "
                    "exist in the customer master."
                ),
            },
            {
                "rule_id": "DQ004",
                "issue_name": "Missing Campaign Attribution",
                "affected_dataset": "interactions",
                "affected_field": "campaign_id",
                "intended_rate_or_condition": "approximately 12% of eligible campaign-driven interactions",
                "number_of_records_modified": n_missing_campaigns,
                "description": (
                    "campaign_id was removed from interactions that already had campaign "
                    "attribution. Legitimate non-campaign interactions were left unchanged."
                ),
            },
            {
                "rule_id": "DQ005",
                "issue_name": "Unusual Null-Rate Spike",
                "affected_dataset": "customers",
                "affected_field": "industry",
                "intended_rate_or_condition": (
                    "approximately 15% NULL industry inside the recent last_updated window; "
                    "near-zero outside that subgroup"
                ),
                "number_of_records_modified": n_industry_nulls,
                "description": (
                    "Industry was blanked for a controlled share of recently updated customers "
                    "to create a measurable null-rate spike versus the baseline."
                ),
            },
            {
                "rule_id": "DQ006",
                "issue_name": "Stale Dataset",
                "affected_dataset": "opportunities",
                "affected_field": "last_updated",
                "intended_rate_or_condition": f"{corruption['dq006_stale_days']} days older than other datasets",
                "number_of_records_modified": n_stale,
                "description": (
                    "Opportunity last_updated timestamps were shifted backward so the "
                    "opportunities table appears several days stale."
                ),
            },
            {
                "rule_id": "DQ007",
                "issue_name": "Unexpected Volume Change",
                "affected_dataset": "interactions",
                "affected_field": "interaction_date",
                "intended_rate_or_condition": (
                    f"volume on {affected_day} reduced to approximately "
                    f"{corruption['dq007_target_volume']} rows"
                ),
                "number_of_records_modified": n_dropped,
                "description": (
                    f"A controlled number of interaction records were removed on {affected_day} "
                    "to simulate an unexpected drop in daily volume."
                ),
            },
        ]
    )

    return {
        "customers": customers,
        "campaigns": campaigns,
        "interactions": interactions,
        "opportunities": opportunities,
    }, manifest


def main() -> None:
    corrupted, manifest = inject_all()
    save_corrupted_datasets(corrupted)
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=4)
    print("Wrote corrupted synthetic datasets to", CORRUPTED_DIR)
    print("Wrote ground-truth manifest to", MANIFEST_PATH)
    print("Clean files were not modified.")


if __name__ == "__main__":
    main()
