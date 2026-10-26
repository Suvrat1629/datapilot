CREATE SCHEMA IF NOT EXISTS staging;

-- Read-only role used by the SQL Analyst. Table/column grants are applied by
-- `datapilot db apply-grants` so hidden_columns in hints.yaml cannot drift from the database.
CREATE ROLE datapilot_ro LOGIN PASSWORD 'datapilot_ro';
ALTER ROLE datapilot_ro SET default_transaction_read_only = on;
ALTER ROLE datapilot_ro SET statement_timeout = '10s';
ALTER ROLE datapilot_ro SET search_path = public;

-- ETL role: writes to staging only, no access to source tables.
CREATE ROLE datapilot_etl LOGIN PASSWORD 'datapilot_etl';
ALTER ROLE datapilot_etl SET search_path = staging;
ALTER ROLE datapilot_etl SET statement_timeout = '60s';

REVOKE ALL ON DATABASE datapilot FROM PUBLIC;
GRANT CONNECT ON DATABASE datapilot TO datapilot_ro, datapilot_etl;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO datapilot_ro;

GRANT USAGE, CREATE ON SCHEMA staging TO datapilot_etl;
GRANT USAGE ON SCHEMA staging TO datapilot_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE datapilot IN SCHEMA staging GRANT SELECT ON TABLES TO datapilot_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE datapilot_etl IN SCHEMA staging GRANT SELECT ON TABLES TO datapilot_ro;

-- Load log for idempotent, auditable staging loads (Phase 4 writes to it).
CREATE TABLE IF NOT EXISTS staging._loads (
    id          bigserial PRIMARY KEY,
    table_name  text        NOT NULL,
    source      text        NOT NULL,
    row_count   integer,
    status      text        NOT NULL,
    loaded_at   timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE staging._loads OWNER TO datapilot_etl;
