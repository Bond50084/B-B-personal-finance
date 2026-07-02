"""Component A — subsidy data ingestion and vector indexing (MVP scale).

Two interchangeable sources produce the same raw dict shape:

    mock  Fixture file ``data/mock_foerderdatenbank.json`` (default; offline,
          deterministic — ideal for tests and the demo).
    live  Targeted scraping of individual programme detail pages on
          https://www.foerderdatenbank.de via requests + BeautifulSoup.
          The parser is label-based ("Förderart:", "Fördergebiet:", ...)
          rather than CSS-class-based, which makes it resilient to template
          tweaks. Page structure verified against the live site (June 2026).

``normalize_program()`` converts raw dicts into ``SubsidyProgram`` objects,
which are then upserted into a persistent ChromaDB collection with
change detection (content hashes): unchanged programmes are not re-embedded,
updated ones are, new ones are added — this is the "daily update" mechanism.
Schedule ``python ingest.py`` via cron/systemd for daily refreshes.

Replacing this module's source functions with a distributed crawler
(EU / Länder level) later does not touch the rest of the pipeline.

Note on live scraping: keep request volume minimal (single detail pages,
default 2 s delay), set an honest User-Agent, and check the site's terms of
use / robots.txt before scaling up. The Förderdatenbank is operated by the
BMWE; for production volumes, prefer official data access where available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import chromadb

try:  # works both as a package (Flask) and standalone (python main.py)
    from .config import Settings, settings as default_settings
    from .providers import embedding_signature, get_embeddings
    from .utils import (
        NATIONWIDE,
        clean_text,
        content_hash,
        normalize_regions,
        parse_deadline,
        stable_program_id,
        utcnow_iso,
    )
except ImportError:
    from config import Settings, settings as default_settings
    from providers import embedding_signature, get_embeddings
    from utils import (
        NATIONWIDE,
        clean_text,
        content_hash,
        normalize_regions,
        parse_deadline,
        stable_program_id,
        utcnow_iso,
    )

# Keywords that mark a programme as open to companies (Antragsberechtigte).
_COMPANY_KEYWORDS = (
    "unternehmen",
    "kmu",
    "existenzgründ",
    "freie berufe",
    "freiberuf",
    "handwerk",
    "start-up",
    "startup",
    "wirtschaft",
    "kontraktor",
)

# foerderdatenbank.de sidebar labels -> raw dict keys (live scraper).
_FIELD_LABEL_MAP = {
    "Förderart": "funding_type",
    "Förderbereich": "funding_area",
    "Fördergebiet": "regions",
    "Förderberechtigte": "eligible_applicants",
    "Fördergeber": "provider",
}


# ---------------------------------------------------------------------------
# Domain model -----------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class SubsidyProgram:
    """Normalised subsidy programme, ready for indexing."""

    program_id: str
    title: str
    provider: str                 # Fördergeber
    funding_type: str             # Art der Förderung
    funding_area: str             # Förderbereich
    regions: list[str]            # canonical Fördergebiet(e)
    eligible_applicants: str      # Antragsberechtigte (raw text)
    eligible_companies: bool      # derived flag for hard filtering
    objective: str                # Förderziel (Kurztext)
    description: str              # Beschreibung (Volltext)
    deadline_text: str            # Frist as published
    deadline_iso: str             # parsed ISO date or ""
    deadline_ts: int              # unix ts of deadline end-of-day, 0 = open-ended
    url: str
    source: str                   # 'mock' | 'live'
    content_hash: str = field(default="")

    @property
    def region_scope(self) -> str:
        """Single scalar used for the Chroma ``where`` filter.

        ``bundesweit`` | exact Bundesland | ``multi`` (several Länder; the
        engine then post-filters in Python against ``regions``).
        """
        if NATIONWIDE in self.regions:
            return NATIONWIDE
        if len(self.regions) == 1:
            return self.regions[0]
        return "multi"


# ---------------------------------------------------------------------------
# Source: mock fixture ----------------------------------------------------------
# ---------------------------------------------------------------------------


def load_mock_programs(settings: Settings) -> list[dict]:
    """Load raw programme dicts from the JSON fixture."""
    with open(settings.mock_data_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["programs"]


# ---------------------------------------------------------------------------
# Source: live scraper (foerderdatenbank.de) -------------------------------------
# ---------------------------------------------------------------------------


def scrape_foerderdatenbank(urls: list[str], settings: Settings) -> tuple[list[dict], int]:
    """Fetch and parse individual programme detail pages.

    Returns ``(raw_programs, error_count)``. Failures are isolated per URL so
    one broken page never aborts a whole ingestion run.
    """
    import requests  # local import: mock mode must not require network deps

    session = requests.Session()
    session.headers.update({"User-Agent": settings.scraper_user_agent})

    raw_programs: list[dict] = []
    errors = 0
    for index, url in enumerate(urls):
        if index > 0:
            time.sleep(settings.scraper_delay_seconds)  # polite crawling
        try:
            response = session.get(url, timeout=settings.scraper_timeout_seconds)
            response.raise_for_status()
            raw_programs.append(parse_program_page(response.text, url))
            print(f"  [live] OK      {url}")
        except Exception as exc:  # noqa: BLE001 — log and continue
            errors += 1
            print(f"  [live] FAILED  {url} ({exc})", file=sys.stderr)
    return raw_programs, errors


def parse_program_page(html: str, url: str) -> dict:
    """Parse one foerderdatenbank.de detail page into a raw dict.

    Strategy (label-based, resilient to markup changes):
      * title:    first ``<h1>``
      * key data: ``<dt>Label:</dt><dd>value</dd>`` pairs in the info box
      * texts:    sections under the headings "Kurztext", "Volltext", "Fristen"
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    raw: dict = {"url": url, "title": "", "deadline_text": ""}

    h1 = soup.find("h1")
    raw["title"] = clean_text(h1.get_text(" ")) if h1 else url

    # dt/dd key data --------------------------------------------------------
    for dt in soup.find_all("dt"):
        label = clean_text(dt.get_text(" ")).rstrip(":")
        key = _FIELD_LABEL_MAP.get(label)
        if not key:
            continue
        dd = dt.find_next_sibling("dd")
        if dd:
            raw[key] = clean_text(dd.get_text(" "))

    # Text sections ---------------------------------------------------------
    raw["objective"] = _section_text(soup, ("Kurztext",))
    raw["description"] = _section_text(soup, ("Volltext",))
    raw["deadline_text"] = _section_text(soup, ("Fristen", "Frist"))

    # Fallback: if the page had no Kurztext, use the meta description.
    if not raw["objective"]:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            raw["objective"] = clean_text(meta["content"])

    return raw


def _section_text(soup, heading_names: tuple[str, ...], max_chars: int = 4000) -> str:
    """Collect text that follows a heading until the next heading starts."""
    heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and clean_text(tag.get_text(" ")) in heading_names
    )
    if heading is None:
        return ""
    chunks: list[str] = []
    for element in heading.find_all_next():
        if element.name in ("h1", "h2", "h3", "h4"):
            break
        if element.name in ("p", "li"):
            text = clean_text(element.get_text(" "))
            if text:
                chunks.append(text)
        if sum(len(chunk) for chunk in chunks) > max_chars:
            break
    return clean_text(" ".join(chunks))[:max_chars]


# ---------------------------------------------------------------------------
# Normalisation ------------------------------------------------------------------
# ---------------------------------------------------------------------------


def normalize_program(raw: dict, source: str) -> SubsidyProgram:
    """Convert a raw source dict into a validated ``SubsidyProgram``."""
    title = clean_text(raw.get("title"))
    url = clean_text(raw.get("url"))
    if not title:
        raise ValueError(f"Programme without title (url={url!r}) — skipping.")

    eligible_raw = clean_text(raw.get("eligible_applicants"))
    deadline_text = clean_text(raw.get("deadline_text"))
    deadline_iso, deadline_ts = parse_deadline(deadline_text)

    program = SubsidyProgram(
        program_id=stable_program_id(url, title),
        title=title,
        provider=clean_text(raw.get("provider")),
        funding_type=clean_text(raw.get("funding_type")),
        funding_area=clean_text(raw.get("funding_area")),
        regions=normalize_regions(raw.get("regions", "")),
        eligible_applicants=eligible_raw,
        eligible_companies=any(k in eligible_raw.lower() for k in _COMPANY_KEYWORDS),
        objective=clean_text(raw.get("objective")),
        description=clean_text(raw.get("description")),
        deadline_text=deadline_text,
        deadline_iso=deadline_iso,
        deadline_ts=deadline_ts,
        url=url,
        source=source,
    )

    # Hash over the content fields only (not timestamps) for change detection.
    hash_payload = asdict(program)
    hash_payload.pop("content_hash")
    hash_payload.pop("source")
    program.content_hash = content_hash(hash_payload)
    return program


def build_embedding_text(program: SubsidyProgram) -> str:
    """Text that gets embedded — the semantic core of the programme."""
    return (
        f"{program.title}\n"
        f"Förderbereich: {program.funding_area}\n"
        f"Förderziel: {program.objective}\n"
        f"Beschreibung: {program.description}"
    )


def chroma_metadata(program: SubsidyProgram) -> dict:
    """Chroma-safe metadata (scalars only, no ``None``)."""
    return {
        "title": program.title,
        "provider": program.provider,
        "funding_type": program.funding_type,
        "funding_area": program.funding_area,
        "region_scope": program.region_scope,        # used by the where-filter
        "regions": ", ".join(program.regions),       # display + 'multi' post-filter
        "eligible_applicants": program.eligible_applicants,
        "eligible_companies": program.eligible_companies,
        "deadline_text": program.deadline_text,
        "deadline_iso": program.deadline_iso,
        "deadline_ts": program.deadline_ts,          # 0 = open-ended
        "url": program.url,
        "source": program.source,
        "content_hash": program.content_hash,
        "last_seen_at": utcnow_iso(),
    }


# ---------------------------------------------------------------------------
# Vector store -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def get_chroma_collection(settings: Settings) -> chromadb.api.models.Collection.Collection:
    settings.ensure_directories()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},  # distances = 1 - cosine similarity
    )


def _signature_path(settings: Settings):
    return settings.storage_dir / "embedding_signature.json"


def write_embedding_signature(settings: Settings) -> None:
    _signature_path(settings).write_text(
        json.dumps({"signature": embedding_signature(settings), "written_at": utcnow_iso()}),
        encoding="utf-8",
    )


def check_embedding_signature(settings: Settings) -> None:
    """Fail loudly if the index was built with a different embedding model."""
    path = _signature_path(settings)
    if not path.exists():
        return
    stored = json.loads(path.read_text(encoding="utf-8")).get("signature")
    current = embedding_signature(settings)
    if stored != current:
        raise RuntimeError(
            f"Vector store was built with embedding '{stored}' but the current "
            f"configuration is '{current}'. Mixing embedding spaces breaks "
            f"similarity search — re-ingest with:  python ingest.py --rebuild"
        )


# ---------------------------------------------------------------------------
# Inspecting the index ---------------------------------------------------------
# ---------------------------------------------------------------------------


def list_indexed_programs(settings: Settings) -> list[dict]:
    """Return every programme currently stored in the vector store.

    Reads the metadata of all entries in the Chroma collection (no embeddings,
    no query) and returns plain dicts sorted by title — the readable view of
    "what is currently searched through".
    """
    collection = get_chroma_collection(settings)
    stored = collection.get(include=["metadatas"])
    programs: list[dict] = []
    for program_id, metadata in zip(stored["ids"], stored["metadatas"]):
        entry = {"program_id": program_id, **(metadata or {})}
        programs.append(entry)
    programs.sort(key=lambda item: item.get("title", "").lower())
    return programs


# Columns shown in exports / listings, in display order.
_EXPORT_COLUMNS = [
    ("title", "Titel"),
    ("provider", "Fördergeber"),
    ("funding_type", "Förderart"),
    ("funding_area", "Förderbereich"),
    ("regions", "Fördergebiet"),
    ("eligible_companies", "Für Unternehmen"),
    ("deadline_text", "Frist"),
    ("source", "Quelle (Typ)"),
    ("last_seen_at", "Zuletzt gesehen"),
    ("url", "URL"),
]


def export_programs_markdown(settings: Settings, path) -> int:
    """Write all indexed programmes to a readable Markdown file. Returns count."""
    programs = list_indexed_programs(settings)
    lines = [
        "# Aktuell indexierte Förderprogramme",
        "",
        f"Stand: {utcnow_iso()}  ·  Einbettung: {embedding_signature(settings)}  ·  "
        f"Anzahl: {len(programs)}",
        "",
        "Diese Datei listet alle Programme, die derzeit im Vektor-Index liegen "
        "und beim Matching durchsucht werden. Sie wird mit "
        "`python main.py programs export` neu erzeugt.",
        "",
    ]
    for index, program in enumerate(programs, start=1):
        lines.append(f"## {index}. {program.get('title', '(ohne Titel)')}")
        lines.append("")
        for key, label in _EXPORT_COLUMNS:
            if key == "title":
                continue
            value = program.get(key, "")
            if key == "eligible_companies":
                value = "ja" if value else "nein"
            lines.append(f"- **{label}:** {value if value != '' else 'k. A.'}")
        lines.append(f"- **Programm-ID:** `{program.get('program_id', '')}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(programs)


def export_programs_csv(settings: Settings, path) -> int:
    """Write all indexed programmes to a CSV file (Excel-friendly). Returns count."""
    import csv

    programs = list_indexed_programs(settings)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")  # ';' opens cleanly in German Excel
        writer.writerow([label for _, label in _EXPORT_COLUMNS] + ["Programm-ID"])
        for program in programs:
            row = []
            for key, _ in _EXPORT_COLUMNS:
                value = program.get(key, "")
                if key == "eligible_companies":
                    value = "ja" if value else "nein"
                row.append(value)
            row.append(program.get("program_id", ""))
            writer.writerow(row)
    return len(programs)


# ---------------------------------------------------------------------------
# Upsert with change detection ---------------------------------------------------
# ---------------------------------------------------------------------------


def upsert_programs(
    programs: list[SubsidyProgram], settings: Settings, rebuild: bool = False
) -> dict:
    """Upsert programmes into Chroma; only new/changed content is re-embedded.

    Returns stats: ``{"total", "new", "updated", "unchanged"}``.
    """
    if rebuild:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            client.delete_collection(settings.collection_name)
            print(f"  [index] collection '{settings.collection_name}' dropped (--rebuild)")
        except Exception:
            pass  # collection did not exist yet
        # Drop the stale signature too: --rebuild is the sanctioned way to
        # switch embedding models, so the old signature must not block it.
        _signature_path(settings).unlink(missing_ok=True)

    if not rebuild:
        check_embedding_signature(settings)
    collection = get_chroma_collection(settings)
    embedder = get_embeddings(settings)

    ids = [program.program_id for program in programs]
    existing_hashes: dict[str, str] = {}
    if ids:
        existing = collection.get(ids=ids, include=["metadatas"])
        for existing_id, metadata in zip(existing["ids"], existing["metadatas"]):
            existing_hashes[existing_id] = (metadata or {}).get("content_hash", "")

    to_embed: list[SubsidyProgram] = []
    unchanged: list[SubsidyProgram] = []
    stats = {"total": len(programs), "new": 0, "updated": 0, "unchanged": 0}

    for program in programs:
        previous_hash = existing_hashes.get(program.program_id)
        if previous_hash is None:
            stats["new"] += 1
            to_embed.append(program)
        elif previous_hash != program.content_hash:
            stats["updated"] += 1
            to_embed.append(program)
        else:
            stats["unchanged"] += 1
            unchanged.append(program)

    if to_embed:
        documents = [build_embedding_text(program) for program in to_embed]
        embeddings = embedder.embed_documents(documents)
        collection.upsert(
            ids=[program.program_id for program in to_embed],
            documents=documents,
            embeddings=embeddings,
            metadatas=[chroma_metadata(program) for program in to_embed],
        )

    if unchanged:
        # Refresh metadata (last_seen_at) without recomputing embeddings.
        collection.update(
            ids=[program.program_id for program in unchanged],
            metadatas=[chroma_metadata(program) for program in unchanged],
        )

    write_embedding_signature(settings)
    return stats


# ---------------------------------------------------------------------------
# Orchestration -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def run_ingestion(
    settings: Settings = default_settings,
    source: str = "mock",
    urls: list[str] | None = None,
    rebuild: bool = False,
) -> dict:
    """Full ingestion run: load -> normalise -> upsert -> log. Returns stats."""
    try:  # local import to avoid a circular dependency at module load
        from . import database
    except ImportError:
        import database

    print(f"[ingest] {utcnow_iso()} | source={source} | embedding={embedding_signature(settings)}")

    errors = 0
    if source == "mock":
        raw_programs = load_mock_programs(settings)
    elif source == "live":
        if not urls:
            raise ValueError("Live ingestion needs --urls (foerderdatenbank.de detail pages).")
        raw_programs, errors = scrape_foerderdatenbank(urls, settings)
    else:
        raise ValueError(f"Unknown source {source!r} (mock | live)")

    programs: list[SubsidyProgram] = []
    for raw in raw_programs:
        try:
            programs.append(normalize_program(raw, source))
        except ValueError as exc:
            errors += 1
            print(f"  [normalize] skipped: {exc}", file=sys.stderr)

    stats = upsert_programs(programs, settings, rebuild=rebuild)
    stats["errors"] = errors

    database.init_db(settings)
    database.log_ingestion_run(settings, source, stats)

    print(
        f"[ingest] done: {stats['total']} programmes "
        f"({stats['new']} new, {stats['updated']} updated, "
        f"{stats['unchanged']} unchanged, {stats['errors']} errors)"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest subsidy programmes into the vector store.")
    parser.add_argument("--source", choices=("mock", "live"), default="mock")
    parser.add_argument("--urls", nargs="*", help="Detail-page URLs for --source live")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and re-create the collection (required after changing the embedding model).",
    )
    args = parser.parse_args()
    run_ingestion(source=args.source, urls=args.urls, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
