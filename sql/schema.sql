-- Clean relational model for the fictional commercial-banking analytics datasets.
-- This schema describes the intended CLEAN state. Corrupted files are expected
-- to violate some of these constraints.
-- All data is synthetic and fictional.

CREATE TABLE customers (
    customer_id   VARCHAR(7)   PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL,
    segment       VARCHAR(32)  NOT NULL,
    region        VARCHAR(64)  NOT NULL,
    industry      VARCHAR(64)  NOT NULL,
    created_date  DATE         NOT NULL,
    status        VARCHAR(16)  NOT NULL,
    source_system VARCHAR(64)  NOT NULL,
    last_updated  DATE         NOT NULL,
    CONSTRAINT chk_customers_last_updated
        CHECK (last_updated >= created_date)
);

CREATE TABLE campaigns (
    campaign_id   VARCHAR(6)     PRIMARY KEY,
    campaign_name VARCHAR(255)   NOT NULL,
    campaign_type VARCHAR(32)    NOT NULL,
    channel       VARCHAR(64)    NOT NULL,
    start_date    DATE           NOT NULL,
    end_date      DATE           NOT NULL,
    budget        DECIMAL(12, 2) NOT NULL,
    status        VARCHAR(16)    NOT NULL,
    source_system VARCHAR(64)    NOT NULL,
    last_updated  DATE           NOT NULL,
    CONSTRAINT chk_campaigns_dates
        CHECK (start_date <= end_date),
    CONSTRAINT chk_campaigns_budget
        CHECK (budget > 0)
);

CREATE TABLE interactions (
    interaction_id   VARCHAR(8)  PRIMARY KEY,
    customer_id      VARCHAR(7)  NOT NULL,
    campaign_id      VARCHAR(6)  NULL,
    interaction_date DATE        NOT NULL,
    channel          VARCHAR(64) NOT NULL,
    interaction_type VARCHAR(64) NOT NULL,
    device_type      VARCHAR(32) NOT NULL,
    source_system    VARCHAR(64) NOT NULL,
    last_updated     DATE        NOT NULL,
    CONSTRAINT fk_interactions_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    CONSTRAINT fk_interactions_campaign
        FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
);

CREATE TABLE opportunities (
    opportunity_id    VARCHAR(9)     PRIMARY KEY,
    customer_id       VARCHAR(7)     NOT NULL,
    created_date      DATE           NOT NULL,
    product           VARCHAR(64)    NOT NULL,
    stage             VARCHAR(32)    NOT NULL,
    opportunity_value DECIMAL(14, 2) NOT NULL,
    converted         BOOLEAN        NOT NULL,
    conversion_date   DATE           NULL,
    source_system     VARCHAR(64)    NOT NULL,
    last_updated      DATE           NOT NULL,
    CONSTRAINT fk_opportunities_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    CONSTRAINT chk_opportunities_value
        CHECK (opportunity_value > 0),
    CONSTRAINT chk_opportunities_last_updated
        CHECK (last_updated >= created_date),
    CONSTRAINT chk_opportunities_conversion
        CHECK (
            (converted = TRUE AND conversion_date IS NOT NULL AND conversion_date >= created_date)
            OR
            (converted = FALSE AND conversion_date IS NULL)
        )
);
