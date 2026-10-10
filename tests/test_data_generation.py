"""Tests for clean synthetic data and the seven injected quality issues."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import inject_quality_issues  # noqa: E402

CLEAN_DIR = ROOT / "data" / "clean"
CORRUPTED_DIR = ROOT / "data" / "corrupted"
CONFIG_PATH = ROOT / "config" / "data_config.yaml"

CLEAN_FILES = ("customers.csv", "campaigns.csv", "interactions.csv", "opportunities.csv")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_csv(folder: Path, filename: str) -> Path:
    path = folder / filename
    if not path.exists():
        pytest.fail(
            f"Missing {path}. Run python src/generate_data.py and "
            "python src/inject_quality_issues.py before pytest."
        )
    return path


def read_dataset(folder: Path, name: str) -> pd.DataFrame:
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
    date_cols = {
        "customers": ["created_date", "last_updated"],
        "campaigns": ["start_date", "end_date", "last_updated"],
        "interactions": ["interaction_date", "last_updated"],
        "opportunities": ["created_date", "conversion_date", "last_updated"],
    }
    frame = pd.read_csv(require_csv(folder, f"{name}.csv"), dtype=dtypes[name])
    for column in date_cols[name]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def file_hashes(folder: Path) -> dict[str, str]:
    hashes = {}
    for filename in CLEAN_FILES:
        path = require_csv(folder, filename)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_clean_datasets_exist() -> None:
    for filename in CLEAN_FILES:
        require_csv(CLEAN_DIR, filename)


def test_clean_primary_ids_are_unique() -> None:
    customers = read_dataset(CLEAN_DIR, "customers")
    campaigns = read_dataset(CLEAN_DIR, "campaigns")
    interactions = read_dataset(CLEAN_DIR, "interactions")
    opportunities = read_dataset(CLEAN_DIR, "opportunities")

    assert customers["customer_id"].is_unique
    assert customers["customer_id"].notna().all()
    assert campaigns["campaign_id"].is_unique
    assert interactions["interaction_id"].is_unique
    assert opportunities["opportunity_id"].is_unique


def test_clean_interaction_customers_exist() -> None:
    customers = read_dataset(CLEAN_DIR, "customers")
    interactions = read_dataset(CLEAN_DIR, "interactions")
    valid_ids = set(customers["customer_id"])
    assert interactions["customer_id"].notna().all()
    assert interactions["customer_id"].isin(valid_ids).all()


def test_clean_opportunity_customers_exist() -> None:
    customers = read_dataset(CLEAN_DIR, "customers")
    opportunities = read_dataset(CLEAN_DIR, "opportunities")
    valid_ids = set(customers["customer_id"])
    assert opportunities["customer_id"].isin(valid_ids).all()


def test_clean_campaign_references_are_valid() -> None:
    campaigns = read_dataset(CLEAN_DIR, "campaigns")
    interactions = read_dataset(CLEAN_DIR, "interactions")
    valid_ids = set(campaigns["campaign_id"])
    attributed = interactions["campaign_id"].dropna()
    assert attributed.isin(valid_ids).all()


def test_converted_opportunities_have_conversion_date() -> None:
    opportunities = read_dataset(CLEAN_DIR, "opportunities")
    is_converted_stage = opportunities["stage"].eq("Converted")
    is_converted_flag = opportunities["converted"].astype(str).str.lower().isin(["true", "1"])
    assert (is_converted_flag == is_converted_stage).all()
    assert opportunities.loc[is_converted_stage, "conversion_date"].notna().all()


def test_non_converted_opportunities_have_no_conversion_date() -> None:
    opportunities = read_dataset(CLEAN_DIR, "opportunities")
    not_converted = opportunities["stage"].ne("Converted")
    is_converted_flag = opportunities["converted"].astype(str).str.lower().isin(["true", "1"])
    assert (~is_converted_flag.loc[not_converted]).all()
    assert opportunities.loc[not_converted, "conversion_date"].isna().all()


def test_conversion_date_not_before_created_date() -> None:
    opportunities = read_dataset(CLEAN_DIR, "opportunities")
    converted = opportunities["conversion_date"].notna()
    assert (
        opportunities.loc[converted, "conversion_date"]
        >= opportunities.loc[converted, "created_date"]
    ).all()


def test_opportunity_value_is_positive() -> None:
    opportunities = read_dataset(CLEAN_DIR, "opportunities")
    assert (opportunities["opportunity_value"] > 0).all()


def test_campaign_start_not_after_end() -> None:
    campaigns = read_dataset(CLEAN_DIR, "campaigns")
    assert (campaigns["start_date"] <= campaigns["end_date"]).all()


def test_dq001_duplicate_customers() -> None:
    clean = read_dataset(CLEAN_DIR, "customers")
    corrupted = read_dataset(CORRUPTED_DIR, "customers")
    assert clean["customer_id"].is_unique
    assert corrupted["customer_id"].duplicated().any()
    extra_rows = corrupted["customer_id"].duplicated().sum()
    rate = extra_rows / len(clean)
    assert 0.01 <= rate <= 0.02


def test_dq002_missing_interaction_customer_ids() -> None:
    clean = read_dataset(CLEAN_DIR, "interactions")
    corrupted = read_dataset(CORRUPTED_DIR, "interactions")
    assert clean["customer_id"].isna().sum() == 0
    missing_rate = corrupted["customer_id"].isna().mean()
    assert 0.03 <= missing_rate <= 0.05


def test_dq003_broken_customer_references() -> None:
    customers = read_dataset(CORRUPTED_DIR, "customers")
    interactions = read_dataset(CORRUPTED_DIR, "interactions")
    valid_ids = set(customers["customer_id"].dropna())
    referenced = interactions["customer_id"].dropna()
    broken_rate = (~referenced.isin(valid_ids)).mean()
    assert broken_rate >= 0.015


def test_dq004_missing_campaign_attribution() -> None:
    clean = read_dataset(CLEAN_DIR, "interactions")
    corrupted = read_dataset(CORRUPTED_DIR, "interactions")
    eligible_types = {"Email Open", "Email Click", "Event Attendance"}
    clean_eligible = clean[clean["interaction_type"].isin(eligible_types)]
    corrupted_eligible = corrupted[corrupted["interaction_type"].isin(eligible_types)]
    clean_missing = clean_eligible["campaign_id"].isna().mean()
    corrupted_missing = corrupted_eligible["campaign_id"].isna().mean()
    assert corrupted_missing > clean_missing
    assert corrupted["campaign_id"].isna().sum() > clean["campaign_id"].isna().sum()


def test_dq005_industry_null_rate_spike() -> None:
    config = load_config()
    customers = read_dataset(CORRUPTED_DIR, "customers")
    snapshot = pd.Timestamp(config["snapshot_date"])
    window_days = config["corruption"]["dq005_window_days"]
    window_start = snapshot - pd.Timedelta(days=window_days - 1)

    subgroup = customers[customers["last_updated"] >= window_start]
    baseline = customers[customers["last_updated"] < window_start]
    assert subgroup["industry"].isna().mean() >= 0.10
    assert baseline["industry"].isna().mean() < 0.03


def test_dq006_stale_opportunities() -> None:
    config = load_config()
    stale_days = config["corruption"]["dq006_stale_days"]
    customers = read_dataset(CORRUPTED_DIR, "customers")
    campaigns = read_dataset(CORRUPTED_DIR, "campaigns")
    interactions = read_dataset(CORRUPTED_DIR, "interactions")
    opportunities = read_dataset(CORRUPTED_DIR, "opportunities")

    other_max = max(
        customers["last_updated"].max(),
        campaigns["last_updated"].max(),
        interactions["last_updated"].max(),
    )
    opportunity_max = opportunities["last_updated"].max()
    lag = (other_max - opportunity_max).days
    assert lag >= stale_days - 1


def test_dq007_interaction_volume_drop() -> None:
    config = load_config()
    snapshot = pd.Timestamp(config["snapshot_date"])
    affected_day = snapshot - pd.Timedelta(days=config["corruption"]["dq007_days_before_snapshot"])
    target = config["corruption"]["dq007_target_volume"]

    clean = read_dataset(CLEAN_DIR, "interactions")
    corrupted = read_dataset(CORRUPTED_DIR, "interactions")
    clean_day = (clean["interaction_date"] == affected_day).sum()
    corrupted_day = (corrupted["interaction_date"] == affected_day).sum()

    assert clean_day > 1500
    assert corrupted_day == target
    assert corrupted_day < clean_day * 0.4


def test_corruption_does_not_modify_clean_files() -> None:
    before = file_hashes(CLEAN_DIR)
    inject_quality_issues.main()
    after = file_hashes(CLEAN_DIR)
    assert after == before
