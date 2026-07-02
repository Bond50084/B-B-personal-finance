"""Shared helpers: Bundesland normalisation, German date parsing, stable IDs.

Kept dependency-free (standard library only) so every module can import it
without side effects.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Regions (Fördergebiet) -----------------------------------------------------
# ---------------------------------------------------------------------------

BUNDESLAENDER: tuple[str, ...] = (
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
)

NATIONWIDE = "bundesweit"

# Common abbreviations / spelling variants -> canonical name.
_REGION_ALIASES: dict[str, str] = {
    "bw": "Baden-Württemberg",
    "baden wuerttemberg": "Baden-Württemberg",
    "baden-wuerttemberg": "Baden-Württemberg",
    "by": "Bayern",
    "bavaria": "Bayern",
    "be": "Berlin",
    "bb": "Brandenburg",
    "hb": "Bremen",
    "hh": "Hamburg",
    "he": "Hessen",
    "mv": "Mecklenburg-Vorpommern",
    "mecklenburg vorpommern": "Mecklenburg-Vorpommern",
    "ni": "Niedersachsen",
    "nds": "Niedersachsen",
    "nw": "Nordrhein-Westfalen",
    "nrw": "Nordrhein-Westfalen",
    "nordrhein westfalen": "Nordrhein-Westfalen",
    "rp": "Rheinland-Pfalz",
    "rlp": "Rheinland-Pfalz",
    "rheinland pfalz": "Rheinland-Pfalz",
    "sl": "Saarland",
    "sn": "Sachsen",
    "st": "Sachsen-Anhalt",
    "sachsen anhalt": "Sachsen-Anhalt",
    "sh": "Schleswig-Holstein",
    "schleswig holstein": "Schleswig-Holstein",
    "th": "Thüringen",
    "thueringen": "Thüringen",
    "deutschland": NATIONWIDE,
    "bund": NATIONWIDE,
    "bundesgebiet": NATIONWIDE,
    "bundesweit": NATIONWIDE,
}

_CANONICAL_LOOKUP = {name.lower(): name for name in BUNDESLAENDER}


def canonical_region(value: str) -> str | None:
    """Map a single region token to its canonical form.

    Returns ``"bundesweit"``, a canonical Bundesland name, or ``None`` if the
    token cannot be resolved.
    """
    token = clean_text(value).lower().strip(" .,;")
    if not token:
        return None
    if token in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[token]
    if token in _REGION_ALIASES:
        return _REGION_ALIASES[token]
    return None


def normalize_regions(text: str) -> list[str]:
    """Parse a Fördergebiet string into canonical regions.

    ``foerderdatenbank.de`` lists values like ``"bundesweit, bundesweit"`` or
    ``"Bayern"``; this splits, maps and de-duplicates. If *anything* resolves
    to nationwide scope, the result collapses to ``["bundesweit"]``.
    Unresolvable inputs default to nationwide (recall over precision for an
    MVP: better to show a programme too many than to silently drop one).
    """
    parts = re.split(r"[,;/]|\bund\b|\bsowie\b", text or "", flags=re.IGNORECASE)
    regions: list[str] = []
    for part in parts:
        canonical = canonical_region(part)
        if canonical and canonical not in regions:
            regions.append(canonical)
    if not regions or NATIONWIDE in regions:
        return [NATIONWIDE]
    return regions


# ---------------------------------------------------------------------------
# Deadlines (Fristen) --------------------------------------------------------
# ---------------------------------------------------------------------------

_GERMAN_MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5,
    "juni": 6, "juli": 7, "august": 8, "september": 9, "oktober": 10,
    "november": 11, "dezember": 12,
}

_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_TEXTUAL_DATE_RE = re.compile(
    r"\b(\d{1,2})\.\s*(" + "|".join(_GERMAN_MONTHS) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)


def parse_deadline(text: str) -> tuple[str, int]:
    """Extract an application deadline from a German free-text Fristen field.

    Returns ``(iso_date, unix_ts)``. Open-ended programmes ("laufend", recurring
    cut-off days without a year, or no parseable date) yield ``("", 0)`` —
    metadata filters treat ``deadline_ts == 0`` as "no fixed deadline".
    If several dates appear, the latest one is used (texts often mention both
    a publication date and the final deadline).
    """
    if not text:
        return "", 0

    candidates: list[datetime] = []
    for day, month, year in _NUMERIC_DATE_RE.findall(text):
        candidates.append(_safe_date(int(year), int(month), int(day)))
    for day, month_name, year in _TEXTUAL_DATE_RE.findall(text):
        candidates.append(_safe_date(int(year), _GERMAN_MONTHS[month_name.lower()], int(day)))

    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return "", 0

    deadline = max(candidates)
    # End of day so a deadline stays valid throughout its final day.
    deadline = deadline.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    return deadline.date().isoformat(), int(deadline.timestamp())


def _safe_date(year: int, month: int, day: int) -> datetime | None:
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Misc -------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def clean_text(value: str | None) -> str:
    """Collapse whitespace and normalise unicode."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def stable_program_id(url: str, title: str) -> str:
    """Deterministic ID for a programme, stable across ingestion runs.

    The detail-page URL is the natural key on foerderdatenbank.de; the title
    is the fallback for sources without URLs (e.g. mock fixtures).
    """
    key = clean_text(url).lower() or clean_text(title).lower()
    return "fp_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def content_hash(payload: dict) -> str:
    """Hash of the normalised programme content, used for change detection."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
