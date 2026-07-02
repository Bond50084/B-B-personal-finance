# FoerderMatch (MVP)

A modular Retrieval-Augmented-Generation (RAG) tool for German subsidy
consultants (*Fördermittelberater*). It aggregates public funding programmes,
stores client profiles, and matches clients to suitable programmes with a
short, professional German evaluation report.

This is **Step 1: a runnable single-machine MVP**. It runs fully offline by
default (no API keys) and is structured so each layer can be swapped for a
production-grade component later (see [Scaling path](#scaling-path-step-2)).

---

## What it does

1. **Ingest** subsidy programmes (mock fixture or live scrape of
   `foerderdatenbank.de`) and index them in a vector store, re-embedding only
   what actually changed — the basis for a daily refresh.
2. **Manage clients**: company name, industry, Bundesland, size, and a free-text
   project description.
3. **Match**: embed the client's project description, apply hard metadata
   filters (region, eligibility, deadline), retrieve the top-K semantically
   closest programmes, and synthesize a German consultant report.

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                   main.py (CLI)                │
                 │   demo · ingest · client · match               │
                 └───────┬───────────────┬───────────────┬───────┘
                         │               │               │
              ┌──────────▼──────┐  ┌──────▼───────┐  ┌────▼─────────────┐
   Component A │   ingest.py     │  │ database.py  │  │ matching_engine.py│ Component C
              │  scrape/parse +  │  │  client CRUD │  │  filters + RAG +  │
              │  normalize +     │  │  + run logs  │  │  LLM synthesis    │
              │  upsert (diff)   │  │ (Component B)│  │                   │
              └───────┬─────────┘  └──────┬───────┘  └────┬──────┬───────┘
                      │                   │               │      │
              ┌───────▼────────┐   ┌──────▼──────┐  ┌──────▼──┐  │
              │   ChromaDB      │   │   SQLite    │  │ ChromaDB│  │
              │ (program vectors│   │ (clients,   │  │ (query) │  │
              │  + metadata)    │   │  run logs)  │  └─────────┘  │
              └────────▲────────┘   └─────────────┘               │
                       │                                  ┌───────▼────────┐
              ┌────────┴─────────────────────────────┐   │  providers.py  │
              │            providers.py               │◄──┤  get_llm()     │
              │  get_embeddings()  (mock|openai|ollama)│  │ (mock|openai|  │
              └───────────────────────────────────────┘  │ ollama|anthropic)
                                                          └────────────────┘
```

Data flow for a match: `client.project_description → embed → Chroma query
(with where-filter) → top-K candidates → LLM (or template) → German report`
saved to `reports/` and logged in SQLite.

---

## Quickstart

Requires Python 3.10+.

```bash
cd foerdermatch
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

# Full offline showcase: ingest mock data, seed two clients, match both.
python main.py demo
```

The demo needs no API keys (mock embeddings + template report). Reports are
written to `reports/`, and state lives in `storage/`.

### Typical commands

```bash
# Refresh the programme index (the daily job)
python main.py ingest                       # mock source
python main.py ingest --source live --urls https://www.foerderdatenbank.de/FDB/Content/DE/Foerderprogramm/Bund/BMWi/fue-kooperationsprojekte-zim.html

# Manage clients
python main.py client add --name "Acme GmbH" --industry "Logistik" \
    --state NRW --size "KMU, 40 MA" \
    --project "Elektrifizierung des Fuhrparks und PV auf dem Logistikzentrum."
python main.py client list
python main.py client show 1

# Match
python main.py match --client-id 1          # one client
python main.py match --all -k 5             # every client, top 5
python main.py match --client-id 1 --no-filters   # debug: disable hard filters
```

### Using real models

Copy `.env.example` to `.env` and set, e.g.:

```ini
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Then rebuild the index so vectors come from the new embedding model:

```bash
python main.py ingest --rebuild
python main.py match --all
```

Anthropic (`LLM_PROVIDER=anthropic`, default model `claude-sonnet-4-6`) and a
local Ollama server (`EMBEDDING_PROVIDER=ollama` / `LLM_PROVIDER=ollama`) are
supported the same way.

---

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` | `mock` | `mock` \| `openai` \| `ollama` |
| `LLM_PROVIDER` | `mock` | `mock` \| `openai` \| `ollama` \| `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | – | Required for the matching provider you pick |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_LLM_MODEL` | `gpt-4o` | OpenAI chat model |
| `ANTHROPIC_LLM_MODEL` | `claude-sonnet-4-6` | Anthropic chat model |
| `OLLAMA_LLM_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `llama3.1` / `nomic-embed-text` | Local models |
| `TOP_K` | `5` | Programmes per report |
| `RETRIEVAL_POOL_FACTOR` | `3` | Oversampling before post-filtering |
| `FILTER_REGION` / `FILTER_DEADLINE` / `FILTER_APPLICANT` | `true` | Toggle hard filters |
| `MOCK_EMBEDDING_DIM` | `512` | Mock embedding dimensionality |

---

## Data model (SQLite)

`clients` — `client_id` (PK), `company_name`, `industry`, `state` (canonical
Bundesland or `bundesweit`), `company_size`, `project_description`,
`created_at`, `updated_at`.

`ingestion_runs` — audit trail of each refresh: `run_at`, `source`, and counts
(`n_total`, `n_new`, `n_updated`, `n_unchanged`, `n_errors`).

`match_runs` — audit trail of each match: `client_id` (FK, cascade), `run_at`,
`top_k`, `program_ids` (JSON), `llm_provider`, `report_path`.

Programme **content and vectors** live in ChromaDB (not SQLite); each entry
carries metadata used for filtering: `region_scope`, `regions`,
`eligible_companies`, `deadline_ts`, `content_hash`, `last_seen_at`, etc.

---

## Design decisions

- **Offline-first mock mode.** `HashingEmbeddings` (signed feature hashing over
  word n-grams and character 4-grams) approximates lexical similarity with zero
  dependencies, and the mock LLM falls back to a deterministic German template.
  The entire pipeline — and the test suite — runs without network or keys.
  *Limitation:* mock embeddings capture lexical, not semantic, similarity, so
  absolute scores are low and synonyms are missed; use a real embedding model
  for production matching quality.
- **Provider abstraction.** No module imports a vendor SDK directly; everything
  goes through `providers.py`, so changing models is two env vars. Vendor
  packages are imported lazily, so mock mode doesn't require them at runtime.
- **Change detection.** A stable `program_id` plus a `content_hash` means a
  daily re-ingest only re-embeds new or changed programmes; unchanged ones get a
  cheap `last_seen_at` metadata refresh. An embedding *signature* is stored with
  the index and re-checked on every run, so switching embedding models without
  `--rebuild` fails loudly instead of silently corrupting similarity search.
- **Hard filters before semantics.** Region, company-eligibility, and deadline
  are correctness constraints, not ranking signals, so they're enforced as a
  ChromaDB `where` filter (with a Python post-filter for multi-Land programmes)
  rather than left to the embedding.
- **Separation for scaling.** SQL is confined to `database.py`; vector-store
  access to `ingest.py` / `matching_engine.py`; scraping to `ingest.py`. Each
  can be replaced independently.

---

## Live scraping note

The live scraper fetches individual programme detail pages politely (honest
User-Agent, default 2 s delay, per-URL error isolation) and parses by visible
field labels, which is resilient to template tweaks. The Förderdatenbank is
operated by the BMWE; **review the site's terms of use and `robots.txt` before
scaling up**, and prefer official/bulk data access where available. The mock
source is the default and needs no network.

---

## Scaling path (Step 2)

- **PostgreSQL + pgvector** instead of SQLite + Chroma: one transactional store
  for client data and vectors. Only `database.py` and the vector-store helpers
  change; the function signatures stay.
- **Distributed ingestion**: per-source scrapers (Bund, 16 Länder, EU) feeding a
  queue, with the same normalize → diff → upsert contract; schedule via
  cron / systemd timers / Airflow for the daily refresh.
- **Service layer**: wrap the engine in a FastAPI app (client CRUD, on-demand
  and scheduled matching, report retrieval) for a web frontend.
- **Matching quality**: production embeddings, a re-ranking stage, and
  structured eligibility rules (headcount, turnover, de-minimis) beyond the
  current keyword/region/deadline filters.

---

## Project layout

```
foerdermatch/
├── main.py              # CLI entry point
├── config.py            # env-driven settings
├── utils.py             # region/date normalisation, IDs, hashing
├── providers.py         # embedding + LLM factories (incl. offline mock)
├── database.py          # Component B — SQLite client CRUD + run logs
├── ingest.py            # Component A — scrape/parse + vector upsert (diff)
├── matching_engine.py   # Component C — filters + RAG + report synthesis
├── data/
│   └── mock_foerderdatenbank.json   # 10 example programmes
├── requirements.txt
└── .env.example
```

*Generated runtime state (`storage/`, `reports/`) is created on first run.*
