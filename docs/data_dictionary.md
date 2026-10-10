# Data Dictionary

All datasets in this project are **100% synthetic and fictional**. They were generated for a portfolio prototype. They do not represent real bank customers, real campaigns, or any confidential commercial data.

The clean relational model lives in `sql/schema.sql`. Files in `data/corrupted/` intentionally violate some of these rules.

## How the datasets relate

```text
customers.customer_id  ----<  interactions.customer_id
customers.customer_id  ----<  opportunities.customer_id
campaigns.campaign_id  ----<  interactions.campaign_id   (nullable)
```

- One customer can have many interactions and many opportunities.
- One campaign can have many interactions.
- An interaction may have no campaign when the activity was not campaign-driven.

Snapshot date used for generation: **2026-09-30** (see `config/data_config.yaml`).

---

## customers

Business meaning: fictional commercial-customer master used as the reference table for analytics.

Approximate size: 10,000 rows.

| Column | Type | Nullable in clean data | Business meaning |
| --- | --- | --- | --- |
| `customer_id` | string (`C000001`) | No | Primary key. Unique commercial customer identifier. |
| `customer_name` | string | No | Fictional company name. |
| `segment` | string | No | Relationship size band: SME, Mid-Market, or Corporate. |
| `region` | string | No | Fictional coverage city in the Netherlands. |
| `industry` | string | No | High-level industry classification. |
| `created_date` | date | No | Date the customer record was created. |
| `status` | string | No | Active or Inactive. |
| `source_system` | string | No | Originating system. Always `CRM` in this prototype. |
| `last_updated` | date | No | Date the customer record was last maintained. Always on or after `created_date`. |

Primary key: `customer_id`.

---

## campaigns

Business meaning: fictional marketing campaigns that can be attributed to some customer interactions.

Approximate size: 25 rows.

| Column | Type | Nullable in clean data | Business meaning |
| --- | --- | --- | --- |
| `campaign_id` | string (`CMP001`) | No | Primary key. Unique campaign identifier. |
| `campaign_name` | string | No | Fictional campaign name. |
| `campaign_type` | string | No | Acquisition, Cross-sell, Engagement, or Retention. |
| `channel` | string | No | Primary delivery channel. |
| `start_date` | date | No | Campaign start. |
| `end_date` | date | No | Campaign end. Always on or after `start_date`. |
| `budget` | numeric | No | Fictional campaign budget. Always greater than 0. |
| `status` | string | No | Active or Completed relative to the snapshot date. |
| `source_system` | string | No | Always `Marketing Platform` in this prototype. |
| `last_updated` | date | No | Date the campaign record was last maintained. |

Primary key: `campaign_id`.

---

## interactions

Business meaning: fictional customer engagement events from digital, marketing, event, and relationship-manager activity.

Approximate size: 100,000 rows in the clean file. The corrupted file has fewer rows because DQ007 removes records on one day.

| Column | Type | Nullable in clean data | Business meaning |
| --- | --- | --- | --- |
| `interaction_id` | string (`I0000001`) | No | Primary key. Unique interaction identifier. |
| `customer_id` | string | No | Foreign key to `customers.customer_id`. |
| `campaign_id` | string | Yes | Foreign key to `campaigns.campaign_id` when the event is campaign-driven. Null is valid for organic or relationship activity. |
| `interaction_date` | date | No | When the interaction occurred. |
| `channel` | string | No | Channel used for the interaction. Paired realistically with `interaction_type`. |
| `interaction_type` | string | No | Event type such as Email Open or RM Meeting. |
| `device_type` | string | No | Desktop, Mobile, Tablet, or Not Applicable. Relationship-manager events use Not Applicable. |
| `source_system` | string | No | Fictional source such as Digital Analytics, CRM, Marketing Platform, or Event Platform. |
| `last_updated` | date | No | Date the interaction record was last maintained. |

Primary key: `interaction_id`.

Foreign keys:

- `customer_id` → `customers.customer_id`
- `campaign_id` → `campaigns.campaign_id` (optional)

---

## opportunities

Business meaning: fictional commercial sales opportunities used for conversion analysis.

Approximate size: 15,000 rows.

| Column | Type | Nullable in clean data | Business meaning |
| --- | --- | --- | --- |
| `opportunity_id` | string (`OPP000001`) | No | Primary key. Unique opportunity identifier. |
| `customer_id` | string | No | Foreign key to `customers.customer_id`. |
| `created_date` | date | No | Date the opportunity was opened. On or after the customer `created_date`. |
| `product` | string | No | Fictional product family. |
| `stage` | string | No | Pipeline stage. |
| `opportunity_value` | numeric | No | Fictional expected value. Always greater than 0. |
| `converted` | boolean | No | True only when `stage` is Converted. |
| `conversion_date` | date | Yes | Required when converted; otherwise null. Never before `created_date`. |
| `source_system` | string | No | Always `CRM` in this prototype. |
| `last_updated` | date | No | Date the opportunity record was last maintained. On or after `created_date` in clean data. |

Primary key: `opportunity_id`.

Foreign key: `customer_id` → `customers.customer_id`.

---

## Clean versus corrupted files

| Location | Purpose |
| --- | --- |
| `data/clean/` | Referentially valid synthetic data. Used as the baseline. |
| `data/corrupted/` | Copies with seven known quality issues. |
| `data/corrupted/injected_issues.json` | Ground truth for tests only. A future Data Quality Agent must not use this file as an input. |
