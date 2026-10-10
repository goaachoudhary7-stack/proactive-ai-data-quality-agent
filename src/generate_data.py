"""Generate clean, fictional commercial-banking analytics datasets.

All customers, campaigns, interactions, and opportunities are synthetic.
This module writes only to data/clean/ and does not inject quality issues.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# Realistic pairings so channels/types are not independent noise.
CHANNEL_BY_TYPE = {
    "Email Open": ["Email"],
    "Email Click": ["Email"],
    "Website Visit": ["Web", "Paid Search"],
    "Product Page View": ["Web", "Paid Search"],
    "Form Start": ["Web", "Paid Search"],
    "Form Complete": ["Web", "Paid Search"],
    "RM Call": ["Relationship Manager"],
    "RM Meeting": ["Relationship Manager"],
    "Event Attendance": ["Event"],
}

# Campaign attribution is common for outbound/paid activity, rare for RM work.
CAMPAIGN_PROBABILITY_BY_TYPE = {
    "Email Open": 0.95,
    "Email Click": 0.95,
    "Website Visit": 0.28,
    "Product Page View": 0.35,
    "Form Start": 0.42,
    "Form Complete": 0.55,
    "RM Call": 0.12,
    "RM Meeting": 0.18,
    "Event Attendance": 0.90,
}

INTERACTION_TYPE_WEIGHTS = {
    "Email Open": 0.22,
    "Email Click": 0.10,
    "Website Visit": 0.22,
    "Product Page View": 0.12,
    "Form Start": 0.06,
    "Form Complete": 0.04,
    "RM Call": 0.10,
    "RM Meeting": 0.06,
    "Event Attendance": 0.08,
}

SOURCE_SYSTEM_BY_TYPE = {
    "Email Open": "Marketing Platform",
    "Email Click": "Marketing Platform",
    "Website Visit": "Digital Analytics",
    "Product Page View": "Digital Analytics",
    "Form Start": "Digital Analytics",
    "Form Complete": "Digital Analytics",
    "RM Call": "CRM",
    "RM Meeting": "CRM",
    "Event Attendance": "Event Platform",
}

OPPORTUNITY_STAGE_WEIGHTS = {
    "Lead": 0.25,
    "Engaged": 0.20,
    "Qualified": 0.18,
    "Opportunity": 0.15,
    "Converted": 0.12,
    "Lost": 0.10,
}

PRODUCT_VALUE_RANGES = {
    "Business Lending": (80_000, 1_800_000),
    "Payments": (5_000, 120_000),
    "Cash Management": (8_000, 180_000),
    "Trade Finance": (40_000, 900_000),
    "Working Capital": (25_000, 600_000),
}

SEGMENT_VALUE_FACTOR = {
    "SME": 0.45,
    "Mid-Market": 1.0,
    "Corporate": 2.4,
}

CAMPAIGN_NAME_THEMES = [
    "Horizon",
    "Pulse",
    "Bridge",
    "Northstar",
    "Harbor",
    "Summit",
    "Lantern",
    "Compass",
    "Anchor",
    "Vista",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML settings used by generation and corruption."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seeds(seed: int) -> tuple[np.random.Generator, Faker]:
    """Fix Python, NumPy, and Faker seeds so reruns match."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)
    return rng, fake


def _as_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


def _random_dates(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, size: int) -> pd.Series:
    """Pick inclusive calendar dates between start and end."""
    span_days = (end - start).days
    offsets = rng.integers(0, span_days + 1, size=size)
    return pd.Series(pd.to_datetime(start) + pd.to_timedelta(offsets, unit="D"))


def generate_customers(config: dict, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    """Create a unique fictional customer master."""
    n = config["dataset_sizes"]["customers"]
    settings = config["customers"]
    snapshot = _as_timestamp(config["snapshot_date"])
    created_start = _as_timestamp(config["date_ranges"]["customer_created_start"])

    created_dates = _random_dates(rng, created_start, snapshot, n)
    recent_share = settings["recent_update_share"]
    recent_window = settings["recent_update_window_days"]
    recent_start = snapshot - pd.Timedelta(days=recent_window - 1)

    last_updated = created_dates.copy()
    is_recent = rng.random(n) < recent_share
    for i in range(n):
        created = created_dates.iloc[i]
        if is_recent[i]:
            window_start = max(created, recent_start)
            last_updated.iloc[i] = _random_dates(rng, window_start, snapshot, 1).iloc[0]
        else:
            last_updated.iloc[i] = _random_dates(rng, created, snapshot, 1).iloc[0]

    status_values = np.array(settings["statuses"])
    # Inactive customers still belong in the master, but most accounts stay active.
    status = rng.choice(status_values, size=n, p=[0.88, 0.12])

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
            "customer_name": [fake.company() for _ in range(n)],
            "segment": rng.choice(settings["segments"], size=n, p=[0.55, 0.30, 0.15]),
            "region": rng.choice(settings["regions"], size=n),
            "industry": rng.choice(settings["industries"], size=n),
            "created_date": created_dates.dt.strftime("%Y-%m-%d"),
            "status": status,
            "source_system": settings["source_system"],
            "last_updated": last_updated.dt.strftime("%Y-%m-%d"),
        }
    )


def generate_campaigns(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Create a small set of overlapping fictional campaigns."""
    n = config["dataset_sizes"]["campaigns"]
    settings = config["campaigns"]
    snapshot = _as_timestamp(config["snapshot_date"])
    window_start = _as_timestamp(config["date_ranges"]["campaign_window_start"])
    interaction_days = config["date_ranges"]["interaction_window_days"]
    interaction_start = snapshot - pd.Timedelta(days=interaction_days - 1)

    rows = []
    types = settings["types"]
    channels = settings["channels"]

    for i in range(1, n + 1):
        campaign_type = str(rng.choice(types))
        channel = str(rng.choice(channels))
        theme = CAMPAIGN_NAME_THEMES[(i - 1) % len(CAMPAIGN_NAME_THEMES)]
        year = int(snapshot.year)

        if i == 1:
            # One always-on campaign keeps attribution possible on every interaction day.
            start_date = interaction_start
            end_date = snapshot
        else:
            start_date = _random_dates(rng, window_start, snapshot - pd.Timedelta(days=14), 1).iloc[0]
            duration = int(rng.integers(21, 121))
            end_date = start_date + pd.Timedelta(days=duration)
            if end_date < start_date:
                end_date = start_date

        last_updated = _random_dates(rng, start_date, snapshot, 1).iloc[0]
        status = "Active" if end_date >= snapshot else "Completed"
        budget = float(rng.integers(8_000, 250_001))

        rows.append(
            {
                "campaign_id": f"CMP{i:03d}",
                "campaign_name": f"{theme} {campaign_type} {year}",
                "campaign_type": campaign_type,
                "channel": channel,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "budget": budget,
                "status": status,
                "source_system": settings["source_system"],
                "last_updated": last_updated.strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(rows)


def _active_campaigns_by_date(campaigns: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    """Precompute valid campaign IDs for each calendar day in the campaign span."""
    campaign_dates = campaigns.copy()
    campaign_dates["start_date"] = pd.to_datetime(campaign_dates["start_date"])
    campaign_dates["end_date"] = pd.to_datetime(campaign_dates["end_date"])

    min_start = campaign_dates["start_date"].min()
    max_end = campaign_dates["end_date"].max()
    lookup: dict[pd.Timestamp, list[str]] = {}
    current = min_start
    while current <= max_end:
        active = campaign_dates[
            (campaign_dates["start_date"] <= current) & (campaign_dates["end_date"] >= current)
        ]
        lookup[current] = active["campaign_id"].tolist()
        current += pd.Timedelta(days=1)
    return lookup


def generate_interactions(
    customers: pd.DataFrame,
    campaigns: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create interaction facts with valid customer keys and optional campaigns."""
    n = config["dataset_sizes"]["interactions"]
    snapshot = _as_timestamp(config["snapshot_date"])
    window_days = config["date_ranges"]["interaction_window_days"]
    start = snapshot - pd.Timedelta(days=window_days - 1)

    types = list(INTERACTION_TYPE_WEIGHTS)
    weights = np.array([INTERACTION_TYPE_WEIGHTS[name] for name in types], dtype=float)
    interaction_type = rng.choice(types, size=n, p=weights / weights.sum())

    # Spread volume evenly so a later one-day drop is easy to see.
    day_offsets = np.arange(n) % window_days
    rng.shuffle(day_offsets)
    interaction_dates = pd.to_datetime(start) + pd.to_timedelta(day_offsets, unit="D")

    customer_ids = rng.choice(customers["customer_id"].to_numpy(), size=n)
    created_map = pd.to_datetime(customers.set_index("customer_id")["created_date"])
    customer_created = created_map.reindex(customer_ids).to_numpy()
    interaction_dates = pd.to_datetime(
        np.maximum(interaction_dates.to_numpy(), customer_created)
    )

    active_lookup = _active_campaigns_by_date(campaigns)
    fallback_campaigns = campaigns["campaign_id"].tolist()

    channels = []
    devices = []
    campaign_ids = []
    source_systems = []
    last_updated = []

    for i in range(n):
        itype = interaction_type[i]
        channel_options = CHANNEL_BY_TYPE[itype]
        channel = str(channel_options[int(rng.integers(0, len(channel_options)))])
        channels.append(channel)
        source_systems.append(SOURCE_SYSTEM_BY_TYPE[itype])

        if channel == "Relationship Manager":
            devices.append("Not Applicable")
        else:
            devices.append(str(rng.choice(["Desktop", "Mobile", "Tablet"], p=[0.46, 0.44, 0.10])))

        event_date = pd.Timestamp(interaction_dates[i]).normalize()
        assign_campaign = rng.random() < CAMPAIGN_PROBABILITY_BY_TYPE[itype]
        if assign_campaign:
            candidates = active_lookup.get(event_date, fallback_campaigns)
            if not candidates:
                candidates = fallback_campaigns
            campaign_ids.append(str(rng.choice(candidates)))
        else:
            campaign_ids.append(pd.NA)

        update_date = min(event_date + pd.Timedelta(days=int(rng.integers(0, 3))), snapshot)
        last_updated.append(update_date.strftime("%Y-%m-%d"))

    return pd.DataFrame(
        {
            "interaction_id": [f"I{i:07d}" for i in range(1, n + 1)],
            "customer_id": customer_ids,
            "campaign_id": campaign_ids,
            "interaction_date": pd.to_datetime(interaction_dates).strftime("%Y-%m-%d"),
            "channel": channels,
            "interaction_type": interaction_type,
            "device_type": devices,
            "source_system": source_systems,
            "last_updated": last_updated,
        }
    )


def generate_opportunities(
    customers: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create sales opportunities with conversion dates only for converted rows."""
    n = config["dataset_sizes"]["opportunities"]
    settings = config["opportunities"]
    snapshot = _as_timestamp(config["snapshot_date"])
    window_start = _as_timestamp(config["date_ranges"]["opportunity_created_start"])

    customer_ids = rng.choice(customers["customer_id"].to_numpy(), size=n)
    segment_map = customers.set_index("customer_id")["segment"]
    created_map = pd.to_datetime(customers.set_index("customer_id")["created_date"])

    stages = list(OPPORTUNITY_STAGE_WEIGHTS)
    stage_weights = np.array([OPPORTUNITY_STAGE_WEIGHTS[name] for name in stages], dtype=float)
    stage = rng.choice(stages, size=n, p=stage_weights / stage_weights.sum())
    products = rng.choice(settings["products"], size=n)

    created_dates = []
    conversion_dates = []
    converted_flags = []
    values = []
    last_updated_dates = []

    for i in range(n):
        customer_id = customer_ids[i]
        customer_created = created_map.loc[customer_id]
        earliest = max(customer_created, window_start)
        created = _random_dates(rng, earliest, snapshot, 1).iloc[0]
        created_dates.append(created)

        product = products[i]
        low, high = PRODUCT_VALUE_RANGES[product]
        factor = SEGMENT_VALUE_FACTOR[segment_map.loc[customer_id]]
        raw_value = rng.uniform(low, high) * factor
        values.append(round(float(raw_value), 2))

        if stage[i] == "Converted":
            converted_flags.append(True)
            conversion = _random_dates(rng, created, snapshot, 1).iloc[0]
            conversion_dates.append(conversion)
            last_updated = _random_dates(rng, conversion, snapshot, 1).iloc[0]
        else:
            converted_flags.append(False)
            conversion_dates.append(pd.NaT)
            last_updated = _random_dates(rng, created, snapshot, 1).iloc[0]
        last_updated_dates.append(last_updated)

    conversion_series = pd.to_datetime(pd.Series(conversion_dates))
    conversion_formatted = conversion_series.dt.strftime("%Y-%m-%d")
    conversion_formatted = conversion_formatted.where(conversion_series.notna(), other=pd.NA)

    return pd.DataFrame(
        {
            "opportunity_id": [f"OPP{i:06d}" for i in range(1, n + 1)],
            "customer_id": customer_ids,
            "created_date": pd.to_datetime(created_dates).strftime("%Y-%m-%d"),
            "product": products,
            "stage": stage,
            "opportunity_value": values,
            "converted": converted_flags,
            "conversion_date": conversion_formatted,
            "source_system": settings["source_system"],
            "last_updated": pd.to_datetime(last_updated_dates).strftime("%Y-%m-%d"),
        }
    )


def save_clean_datasets(datasets: dict[str, pd.DataFrame], output_dir: Path = CLEAN_DIR) -> None:
    """Write clean CSV files. Never used by the corruption script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in datasets.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def generate_all(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Build the four related clean datasets."""
    config = config or load_config()
    rng, fake = set_seeds(config["random_seed"])
    customers = generate_customers(config, rng, fake)
    campaigns = generate_campaigns(config, rng)
    interactions = generate_interactions(customers, campaigns, config, rng)
    opportunities = generate_opportunities(customers, config, rng)
    return {
        "customers": customers,
        "campaigns": campaigns,
        "interactions": interactions,
        "opportunities": opportunities,
    }


def main() -> None:
    config = load_config()
    datasets = generate_all(config)
    save_clean_datasets(datasets)
    print("Wrote clean synthetic datasets to", CLEAN_DIR)
    for name, frame in datasets.items():
        print(f"  {name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
