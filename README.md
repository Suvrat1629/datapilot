# DataPilot

An agentic data assistant for **any PostgreSQL database**. Ask questions in plain English and it
writes and runs read-only SQL. Ask it to load data from a public API and it fetches, reshapes and
loads it into a separate `staging` schema. A router decides which one you need.

```
you> Which city had the most cancellations last week?
you> Load the last 30 days of weather for every city, then which city is hottest?
```

The bundled demo is an Ola-style ride-booking dataset. Nothing in the agents is specific to it.

## How it works

```
                    +--> SQL Analyst  (LangGraph: curate, schema, generate, safety, run, answer)
chat --> Router ----+--> ETL Analyst  (tool-calling agent: read, fetch, transform, load)
                    +--> direct reply / clarifying question
```

- **SQL Analyst**: a stateful graph. Generated SQL passes a static check, then runs as a database
  role that can only `SELECT`. Failed queries are retried with the error fed back.
- **ETL Analyst**: a tool-calling agent. It can call a fixed set of tools only (no code
  execution): read allowed reference columns, fetch JSON from allow-listed hosts, apply typed
  transforms, and atomically replace a table in `staging`.
- **Router**: one structured call that picks `sql`, `etl`, `both` (load, then analyse), `chat` or
  `clarify`. Routing is not a security layer; both agents enforce their own limits.

## Safety model

The database is the boundary, not the prompt.

| Layer | What it does |
|---|---|
| Postgres roles | The SQL role has column-level `SELECT` only, read-only sessions and a statement timeout. Hidden columns (for example emails) are not granted at all. The ETL role can write only to `staging` and read only the columns you list in `ETL_READABLE_COLUMNS`. |
| Startup guard | Every command checks the connected roles first and refuses to run if one is a superuser, can write outside its area, or can read a hidden column. |
| Static SQL check | Single `SELECT` only, no dangerous functions, allow-listed schemas, row limit. Defence in depth. |
| ETL guards | Exact HTTPS host allow-list, no redirects, response size cap, request cap, step cap, typed transforms, atomic idempotent loads. |
| LLM judge | Checks the query answers the question. It is for correctness, not security. |

## Quick start (demo)

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, and a free
[Groq](https://console.groq.com) API key.

```bash
uv sync
cp .env.example .env            # then set GROQ_API_KEY
docker compose up -d postgres   # Postgres 16 with the limited roles
uv run datapilot demo load      # synthetic rides, grants applied
uv run datapilot db check       # confirm the roles are limited
uv run datapilot chat           # talk to it
```

One-off commands:

```bash
uv run datapilot ask "Top 3 drivers by rating" --show-sql
uv run datapilot etl "Load the last 30 days of daily temperature for every city into weather_daily"
uv run datapilot chat "Which city is hottest in staging.weather_daily?"
```

Optional tracing with Langfuse: `docker compose --profile observability up -d`, UI on
<http://localhost:3001>. Without it, each request is still logged to `logs/traces.jsonl`.

## Use your own database

1. Point `ADMIN_DATABASE_URL` at your database with an admin user (used only for setup).
2. Optionally list hidden columns and table notes in a hints file and set `HINTS_PATH`
   (see `src/datapilot/demo/hints.yaml`). Hidden columns are never granted to the agent roles.
3. Set `SCHEMAS` (default `public`) and, if the ETL agent needs reference data such as a list of
   cities, `ETL_READABLE_COLUMNS` (`table.column` or `schema.table.column`).
4. Run `uv run datapilot db setup`. It creates the two limited roles and `staging`, applies the
   grants, verifies them, and prints the `DATABASE_URL` / `ETL_DATABASE_URL` lines for `.env`.
   It never changes your data and does not revoke anything from `PUBLIC`.
5. `uv run datapilot db check` any time; it also runs automatically before every command.

`demo load` refuses to run on a database that was not created by it, so it cannot drop your
tables.

## Commands

| Command | Purpose |
|---|---|
| `chat ["message"]` | Interactive chat, or one message with `--json`. Chat commands: `/sql`, `/clear`, `/quit`. |
| `ask "question"` | SQL Analyst only. `--show-sql`, `--json`. |
| `etl "request"` | ETL Analyst only. |
| `db setup` / `db check` | Create or verify the limited roles. |
| `db apply-grants` / `db schema` | Re-apply grants after editing hints; print what the agent sees. |
| `demo load` | Generate the demo dataset. |
| `eval run` / `eval list` / `eval compare` / `eval routing` | Accuracy, safety and routing evals. |

## Configuration

All settings are environment variables (see `.env.example`). The important ones:

| Variable | Meaning |
|---|---|
| `LLM_LOW`, `LLM_HIGH` | `provider:model` for the cheap and strong tiers (`groq` or `google`). |
| `SCHEMAS` | Schemas the agent may read; `staging` is always added. |
| `ETL_ALLOWED_HOSTS` | Exact hostnames `fetch_api` may call. Empty denies everything. |
| `ETL_READABLE_COLUMNS` | Reference columns the ETL role may read. |
| `MAX_ROWS`, `STATEMENT_TIMEOUT_MS` | Result cap and query timeout. |
| `RO_ROLE`, `ETL_ROLE` | Role names, if `db setup` used custom ones. |

## Development

```bash
uv run pytest            # unit tests; database tests need the Postgres container
uv run ruff check . && uv run ruff format --check .
uv run datapilot eval run --label baseline     # live accuracy and safety eval
uv run datapilot eval routing                  # router accuracy
```

The full design, trade-offs and phase history are in [`docs/DESIGN.md`](docs/DESIGN.md).

## Limitations

- Free-tier model limits (tokens per minute and per day) make long ETL runs slow or fail; the
  agents report this instead of crashing.
- The ETL agent loads JSON from allow-listed APIs only; no joins between fetched datasets yet.
- The staging load log (`staging._loads`) is readable by the SQL role.
- No web UI yet; the CLI is the interface.
