"""Seven deterministic data-quality checks.

Each function inspects pandas DataFrames loaded from corrupted CSVs.
None of these functions open, parse, or import injected_issues.json.
"""

from __future__ import annotations

import pandas as pd

from .models import DQFinding

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"


def _as_float(value: float | int) -> float:
    return float(value)


def _missing(series: pd.Series) -> pd.Series:
    """True where a value is null or a blank string after CSV load."""
    text = series.astype("string")
    return series.isna() | text.str.strip().eq("") | text.str.lower().eq("nan")


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _status_from_rate(metric: float, threshold: float, *, higher_is_worse: bool = True) -> tuple[str, str]:
    """Map a numeric metric onto PASS / WARNING / FAIL with a fixed severity."""
    if higher_is_worse:
        if metric > threshold:
            return FAIL, "HIGH"
        if metric > 0:
            return WARNING, "LOW"
        return PASS, "INFO"
    if metric > threshold:
        return FAIL, "HIGH"
    if metric > 0:
        return WARNING, "LOW"
    return PASS, "INFO"


def check_customer_uniqueness(customers: pd.DataFrame, thresholds: dict) -> DQFinding:
    """DQ001: customer_id must be unique in the customer master."""
    total_rows = int(len(customers))
    unique_ids = int(customers["customer_id"].nunique(dropna=True))
    duplicate_excess = total_rows - unique_ids
    duplicate_rate = _rate(duplicate_excess, total_rows)
    threshold = _as_float(thresholds["duplicate_customer_rate"])
    status, severity = _status_from_rate(duplicate_rate, threshold)

    duplicated_mask = customers["customer_id"].duplicated(keep=False)
    sample_ids = (
        customers.loc[duplicated_mask, "customer_id"]
        .dropna()
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    return DQFinding(
        rule_id="DQ001",
        check_name="Customer Uniqueness",
        dataset="customers",
        field="customer_id",
        dimension="UNIQUENESS",
        status=status,
        severity=severity,
        records_checked=total_rows,
        records_affected=int(duplicate_excess),
        metric_value=duplicate_rate,
        threshold=threshold,
        description=(
            "Duplicate customer_id values in the customer master. "
            "duplicate_excess_records = total rows minus unique IDs. "
            "duplicate_rate = excess / total rows. "
            f"FAIL when duplicate_rate > {threshold}."
        ),
        evidence={
            "total_rows": total_rows,
            "unique_customer_ids": unique_ids,
            "duplicate_excess_records": int(duplicate_excess),
            "duplicate_rate": duplicate_rate,
            "sample_duplicate_ids": [str(x) for x in sample_ids],
        },
    )


def check_customer_id_completeness(interactions: pd.DataFrame, thresholds: dict) -> DQFinding:
    """DQ002: interactions.customer_id should rarely be missing."""
    total_rows = int(len(interactions))
    missing = int(_missing(interactions["customer_id"]).sum())
    missing_rate = _rate(missing, total_rows)
    threshold = _as_float(thresholds["missing_customer_id_rate"])
    status, severity = _status_from_rate(missing_rate, threshold)

    return DQFinding(
        rule_id="DQ002",
        check_name="Customer ID Completeness",
        dataset="interactions",
        field="customer_id",
        dimension="COMPLETENESS",
        status=status,
        severity=severity,
        records_checked=total_rows,
        records_affected=missing,
        metric_value=missing_rate,
        threshold=threshold,
        description=(
            "NULL or blank customer_id on interaction records. "
            "NULL percentage = missing / total interaction records. "
            f"FAIL when the missing rate > {threshold}."
        ),
        evidence={
            "total_interaction_records": total_rows,
            "null_customer_ids": missing,
            "null_percentage": missing_rate,
        },
    )


def check_customer_referential_integrity(
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    thresholds: dict,
) -> DQFinding:
    """DQ003: non-null interaction customer_id values must exist in customers."""
    has_id = ~_missing(interactions["customer_id"])
    checked = interactions.loc[has_id]
    valid_ids = set(customers["customer_id"].dropna().astype(str))
    broken_mask = ~checked["customer_id"].astype(str).isin(valid_ids)
    broken = int(broken_mask.sum())
    records_checked = int(len(checked))
    broken_rate = _rate(broken, records_checked)
    threshold = _as_float(thresholds["broken_reference_rate"])
    status, severity = _status_from_rate(broken_rate, threshold)
    if status == FAIL:
        severity = "CRITICAL"

    sample = checked.loc[broken_mask, "customer_id"].drop_duplicates().head(5).tolist()

    return DQFinding(
        rule_id="DQ003",
        check_name="Customer Referential Integrity",
        dataset="interactions",
        field="customer_id",
        dimension="REFERENTIAL_INTEGRITY",
        status=status,
        severity=severity,
        records_checked=records_checked,
        records_affected=broken,
        metric_value=broken_rate,
        threshold=threshold,
        description=(
            "Non-null interactions.customer_id values that are absent from "
            "customers.customer_id. broken-reference percentage = broken / records checked. "
            f"FAIL when that rate > {threshold}."
        ),
        evidence={
            "records_checked": records_checked,
            "broken_references": broken,
            "broken_reference_percentage": broken_rate,
            "sample_unknown_customer_ids": [str(x) for x in sample],
        },
    )


def check_campaign_attribution_completeness(
    interactions: pd.DataFrame,
    thresholds: dict,
    campaign_driven_types: list[str],
) -> DQFinding:
    """DQ004: campaign-driven interaction types should carry campaign_id.

    Business rule (configurable):
    Only selected interaction types are treated as campaign-driven. Those types
    are outbound/event activity that the generator usually attributes to a
    campaign (Email Open, Email Click, Event Attendance by default).

    Relationship Manager calls/meetings and most organic web browsing may
    legitimately have a NULL campaign_id and are excluded from this check.
    """
    types = list(campaign_driven_types)
    eligible_mask = interactions["interaction_type"].isin(types)
    eligible = interactions.loc[eligible_mask]
    missing = int(_missing(eligible["campaign_id"]).sum())
    records_checked = int(len(eligible))
    missing_rate = _rate(missing, records_checked)
    threshold = _as_float(thresholds["campaign_attribution_missing_rate"])
    status, severity = _status_from_rate(missing_rate, threshold)

    non_eligible = int((~eligible_mask).sum())
    non_eligible_missing = int(_missing(interactions.loc[~eligible_mask, "campaign_id"]).sum())

    return DQFinding(
        rule_id="DQ004",
        check_name="Campaign Attribution Completeness",
        dataset="interactions",
        field="campaign_id",
        dimension="ATTRIBUTION",
        status=status,
        severity=severity,
        records_checked=records_checked,
        records_affected=missing,
        metric_value=missing_rate,
        threshold=threshold,
        description=(
            "Missing campaign_id among campaign-driven interaction types only. "
            f"Eligible types: {types}. "
            "Organic / RM interactions with NULL campaign_id are not counted. "
            f"FAIL when missing attribution rate > {threshold}."
        ),
        evidence={
            "eligible_interaction_types": types,
            "eligible_campaign_driven_interactions": records_checked,
            "eligible_records_missing_campaign_id": missing,
            "missing_attribution_rate": missing_rate,
            "non_eligible_interactions": non_eligible,
            "non_eligible_null_campaign_id": non_eligible_missing,
        },
    )


def check_industry_null_rate_spike(customers: pd.DataFrame, thresholds: dict) -> DQFinding:
    """DQ005: recent industry NULL rate versus an older baseline.

    Recent window:
        last_updated >= max(last_updated) minus (industry_recent_window_days - 1).
        The window is anchored on the latest timestamp in the customer file,
        not on the computer's current date.

    Baseline:
        customers whose last_updated is older than that window.

    Metric:
        recent_null_rate - baseline_null_rate (percentage-point difference).

    FAIL when that difference exceeds industry_null_rate_difference.
    """
    last_updated = pd.to_datetime(customers["last_updated"], errors="coerce")
    window_days = int(thresholds["industry_recent_window_days"])
    threshold = _as_float(thresholds["industry_null_rate_difference"])

    max_updated = last_updated.max()
    window_start = max_updated - pd.Timedelta(days=window_days - 1)
    recent_mask = last_updated >= window_start
    baseline_mask = last_updated < window_start

    recent = customers.loc[recent_mask]
    baseline = customers.loc[baseline_mask]
    recent_nulls = int(_missing(recent["industry"]).sum())
    baseline_nulls = int(_missing(baseline["industry"]).sum())
    recent_n = int(len(recent))
    baseline_n = int(len(baseline))
    recent_rate = _rate(recent_nulls, recent_n)
    baseline_rate = _rate(baseline_nulls, baseline_n)
    rate_difference = recent_rate - baseline_rate

    if rate_difference > threshold:
        status, severity = FAIL, "HIGH"
    elif rate_difference > 0:
        status, severity = WARNING, "LOW"
    else:
        status, severity = PASS, "INFO"

    return DQFinding(
        rule_id="DQ005",
        check_name="Industry Null-Rate Spike",
        dataset="customers",
        field="industry",
        dimension="COMPLETENESS",
        status=status,
        severity=severity,
        records_checked=recent_n,
        records_affected=recent_nulls,
        metric_value=rate_difference,
        threshold=threshold,
        description=(
            f"Compare industry NULL rate for customers updated in the last {window_days} "
            "days of the dataset versus older customers. "
            "Anchored on max(customers.last_updated). "
            f"FAIL when (recent_null_rate - baseline_null_rate) > {threshold}."
        ),
        evidence={
            "anchor_max_last_updated": None if pd.isna(max_updated) else str(pd.Timestamp(max_updated).date()),
            "window_start": None if pd.isna(window_start) else str(pd.Timestamp(window_start).date()),
            "window_days": window_days,
            "recent_records": recent_n,
            "recent_null_industry_count": recent_nulls,
            "recent_null_rate": recent_rate,
            "baseline_records": baseline_n,
            "baseline_null_industry_count": baseline_nulls,
            "baseline_null_rate": baseline_rate,
            "rate_difference": rate_difference,
        },
    )


def check_opportunity_freshness(
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    campaigns: pd.DataFrame,
    interactions: pd.DataFrame,
    thresholds: dict,
) -> DQFinding:
    """DQ006: opportunity last_updated lag versus other datasets.

    Reference timestamp = the latest last_updated among customers, campaigns,
    and interactions. Wall-clock / system time is not used, because the
    synthetic snapshot is a fixed historical date.
    """
    threshold_days = _as_float(thresholds["opportunity_freshness_days"])
    opp_max = pd.to_datetime(opportunities["last_updated"], errors="coerce").max()
    other_maxes = [
        pd.to_datetime(customers["last_updated"], errors="coerce").max(),
        pd.to_datetime(campaigns["last_updated"], errors="coerce").max(),
        pd.to_datetime(interactions["last_updated"], errors="coerce").max(),
    ]
    reference = max(other_maxes)
    lag_days = float((reference - opp_max).days) if pd.notna(opp_max) and pd.notna(reference) else 0.0

    if lag_days > threshold_days:
        status, severity = FAIL, "HIGH"
    elif lag_days > 0:
        status, severity = WARNING, "LOW"
    else:
        status, severity = PASS, "INFO"

    return DQFinding(
        rule_id="DQ006",
        check_name="Opportunity Freshness",
        dataset="opportunities",
        field="last_updated",
        dimension="FRESHNESS",
        status=status,
        severity=severity,
        records_checked=int(len(opportunities)),
        records_affected=int(len(opportunities)) if lag_days > threshold_days else 0,
        metric_value=lag_days,
        threshold=threshold_days,
        description=(
            "Freshness lag in days = max(other datasets' last_updated) minus "
            "max(opportunities.last_updated). Other datasets are customers, "
            "campaigns, and interactions. System date is not used. "
            f"FAIL when lag > {threshold_days} days."
        ),
        evidence={
            "opportunity_max_last_updated": None if pd.isna(opp_max) else str(pd.Timestamp(opp_max).date()),
            "reference_timestamp": None if pd.isna(reference) else str(pd.Timestamp(reference).date()),
            "customers_max_last_updated": str(pd.Timestamp(other_maxes[0]).date()),
            "campaigns_max_last_updated": str(pd.Timestamp(other_maxes[1]).date()),
            "interactions_max_last_updated": str(pd.Timestamp(other_maxes[2]).date()),
            "freshness_lag_days": lag_days,
        },
    )


def check_interaction_volume_anomaly(interactions: pd.DataFrame, thresholds: dict) -> DQFinding:
    """DQ007: unusually low daily interaction volume.

    Method:
    1. Count rows per interaction_date.
    2. For each day, baseline = median of the previous N days
       (N = volume_rolling_window_days). Days without enough history use the
       overall median of all other days.
    3. Percentage deviation = (actual - baseline) / baseline.
    4. The affected date is the day with the most negative deviation.
    5. FAIL when that drop is at least interaction_volume_drop_percentage
       below the baseline (for example 0.50 means 50% below median).
    """
    drop_threshold = _as_float(thresholds["interaction_volume_drop_percentage"])
    window = int(thresholds.get("volume_rolling_window_days", 7))

    dates = pd.to_datetime(interactions["interaction_date"], errors="coerce")
    daily = dates.groupby(dates.dt.normalize()).size().sort_index()
    daily.index = pd.DatetimeIndex(daily.index)

    prior_median = daily.shift(1).rolling(window=window, min_periods=3).median()
    overall_other = []
    for day in daily.index:
        others = daily.drop(labels=[day])
        overall_other.append(float(others.median()) if len(others) else float(daily.loc[day]))
    fallback = pd.Series(overall_other, index=daily.index)
    baseline = prior_median.fillna(fallback)

    deviation = (daily.astype(float) - baseline) / baseline.replace(0, pd.NA)
    deviation = deviation.fillna(0.0)
    affected_day = deviation.idxmin()
    actual = int(daily.loc[affected_day])
    baseline_volume = float(baseline.loc[affected_day])
    pct_deviation = float(deviation.loc[affected_day])

    if pct_deviation <= -drop_threshold:
        status, severity = FAIL, "HIGH"
    elif pct_deviation < 0:
        status, severity = WARNING, "LOW"
    else:
        status, severity = PASS, "INFO"

    return DQFinding(
        rule_id="DQ007",
        check_name="Interaction Volume Anomaly",
        dataset="interactions",
        field="interaction_date",
        dimension="VOLUME",
        status=status,
        severity=severity,
        records_checked=int(len(interactions)),
        records_affected=int(max(baseline_volume - actual, 0)),
        metric_value=pct_deviation,
        threshold=-drop_threshold,
        description=(
            f"Daily interaction counts compared with a {window}-day trailing median "
            "(fallback: median of all other days). "
            "The engine selects the date with the largest percentage drop. "
            f"FAIL when percentage deviation <= {-drop_threshold}."
        ),
        evidence={
            "method": (
                f"Trailing {window}-day median of prior daily volumes; "
                "fallback median of all other days; percentage deviation from that baseline."
            ),
            "affected_date": str(pd.Timestamp(affected_day).date()),
            "actual_volume": actual,
            "baseline_volume": baseline_volume,
            "percentage_deviation": pct_deviation,
            "rolling_window_days": window,
        },
    )


def run_all_checks(
    datasets: dict[str, pd.DataFrame],
    config: dict,
) -> list[DQFinding]:
    """Run DQ001–DQ007 in order. Does not read injected_issues.json."""
    thresholds = config["dq_thresholds"]
    campaign_types = list(config["dq_engine"]["campaign_driven_interaction_types"])
    # Rolling-window size lives with the other volume settings.
    thresholds = {
        **thresholds,
        "volume_rolling_window_days": config["dq_engine"]["volume_rolling_window_days"],
    }

    customers = datasets["customers"]
    campaigns = datasets["campaigns"]
    interactions = datasets["interactions"]
    opportunities = datasets["opportunities"]

    return [
        check_customer_uniqueness(customers, thresholds),
        check_customer_id_completeness(interactions, thresholds),
        check_customer_referential_integrity(interactions, customers, thresholds),
        check_campaign_attribution_completeness(interactions, thresholds, campaign_types),
        check_industry_null_rate_spike(customers, thresholds),
        check_opportunity_freshness(opportunities, customers, campaigns, interactions, thresholds),
        check_interaction_volume_anomaly(interactions, thresholds),
    ]
