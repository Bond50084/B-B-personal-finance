"""Component C — RAG matching engine.

Pipeline per client:

    1. Embed the client's Projektbeschreibung (query side of the RAG setup).
    2. Query the Chroma collection with *hard metadata filters*:
         - Fördergebiet:        bundesweit | multi | client's Bundesland
         - Antragsberechtigung: programmes open to companies
         - Frist:               open-ended or deadline still in the future
    3. Take the top-K semantic neighbours (cosine similarity).
    4. Synthesize a professional German consultant report via the configured
       LLM — or a deterministic template when ``LLM_PROVIDER=mock`` so the
       end-to-end flow works fully offline.

The engine only talks to the vector store and ``providers.py``; swapping
Chroma for pgvector or the mock LLM for GPT-4o/Claude requires no changes
here beyond configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

try:  # works both as a package (Flask) and standalone (python main.py)
    from . import database
    from .config import Settings, settings as default_settings
    from .ingest import check_embedding_signature, get_chroma_collection
    from .providers import get_embeddings, get_llm, message_content_to_text
    from .utils import NATIONWIDE, utcnow_iso
except ImportError:
    import database
    from config import Settings, settings as default_settings
    from ingest import check_embedding_signature, get_chroma_collection
    from providers import get_embeddings, get_llm, message_content_to_text
    from utils import NATIONWIDE, utcnow_iso

# ---------------------------------------------------------------------------
# Data model -----------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class MatchCandidate:
    """One retrieved programme plus its similarity to the client project."""

    program_id: str
    title: str
    provider: str
    funding_type: str
    funding_area: str
    regions: str
    eligible_applicants: str
    deadline_text: str
    deadline_iso: str
    url: str
    similarity: float            # 1 - cosine distance, in [0, 1]
    objective: str
    description: str


# ---------------------------------------------------------------------------
# German prompts ----------------------------------------------------------------
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DE = (
    "Du bist ein erfahrener Fördermittelberater in Deutschland. Du beurteilst "
    "knapp und nüchtern, wie gut öffentliche Förderprogramme zu einem Vorhaben "
    "passen. Du erfindest keine Programminhalte und stützt dich ausschließlich "
    "auf die übergebenen Programmdaten. Du antwortest ausschließlich mit gültigem "
    "JSON ohne Code-Fences und ohne weiteren Text."
)

USER_PROMPT_TEMPLATE_DE = """## Mandant
- Unternehmen: {company_name} ({industry}, {state}, {company_size})

## Projekt
{project_description}

## Programme
Die folgenden {n} Programme stammen aus einer semantischen Suche und sind \
bereits hart vorgefiltert (Fördergebiet, Antragsberechtigung, Frist).

{candidate_blocks}

## Aufgabe
Beurteile für jedes Programm, wie gut es zum Projekt des Mandanten passt.
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt (kein Markdown, keine \
Code-Fences, kein erläuternder Text). Ordne jeder Programmnummer ein Objekt zu:

{{"1": {{"passung": "Sehr hoch|Hoch|Mittel|Gering", "kommentar": "ein Satz \
auf Deutsch mit konkretem Bezug zum Projekt"}}, "2": {{...}}}}

Verwende für "passung" genau einen der vier Werte. Bewerte ausschließlich die \
gelisteten Programme und erfinde keine Fakten."""

_CANDIDATE_BLOCK_DE = """### {index}. {title}
- Förderart: {funding_type} · Fördergebiet: {regions} · Frist: {deadline}
- Förderziel: {objective}"""

# Ordering of the four verdict levels (best first) for sorting the report.
_PASSUNG_RANK = {"sehr hoch": 0, "hoch": 1, "mittel": 2, "gering": 3}


# ---------------------------------------------------------------------------
# Engine ------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class MatchingEngine:
    """Retrieval + synthesis for one configured provider setup."""

    def __init__(self, settings: Settings = default_settings, embeddings=None, llm=None):
        self.settings = settings
        check_embedding_signature(settings)
        self.collection = get_chroma_collection(settings)
        self.embeddings = embeddings or get_embeddings(settings)
        self.llm = llm if llm is not None else get_llm(settings)

    # -- Hard metadata filters ------------------------------------------------

    def build_where(self, client: dict, apply_filters: bool = True) -> dict | None:
        """Compose the Chroma ``where`` filter from the client profile."""
        if not apply_filters:
            return None
        s = self.settings
        clauses: list[dict] = []
        if s.filter_region:
            allowed_scopes = [NATIONWIDE, "multi"]
            if client["state"] != NATIONWIDE:
                allowed_scopes.append(client["state"])
            clauses.append({"region_scope": {"$in": allowed_scopes}})
        if s.filter_applicant:
            clauses.append({"eligible_companies": {"$eq": True}})
        if s.filter_deadline:
            now_ts = int(time.time())
            clauses.append(
                {"$or": [{"deadline_ts": {"$eq": 0}}, {"deadline_ts": {"$gte": now_ts}}]}
            )
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # -- Retrieval ---------------------------------------------------------------

    def retrieve(self, client: dict, k: int | None = None, apply_filters: bool = True) -> list[MatchCandidate]:
        """Top-K programmes for the client's Projektbeschreibung."""
        k = k or self.settings.top_k
        total = self.collection.count()
        if total == 0:
            raise RuntimeError("Vector store is empty — run `python ingest.py` first.")

        # Retrieve a larger pool, then post-filter in Python (multi-Land
        # programmes) and cut to k.
        pool_size = min(max(k * self.settings.retrieval_pool_factor, k), total)
        query_embedding = self.embeddings.embed_query(client["project_description"])
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=pool_size,
            where=self.build_where(client, apply_filters),
            include=["metadatas", "distances", "documents"],
        )

        candidates: list[MatchCandidate] = []
        for program_id, metadata, distance in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0]
        ):
            metadata = metadata or {}
            # Post-filter: 'multi' scope means several specific Länder; keep
            # the programme only if the client's Land is among them.
            if (
                apply_filters
                and self.settings.filter_region
                and metadata.get("region_scope") == "multi"
                and client["state"] != NATIONWIDE
                and client["state"] not in metadata.get("regions", "")
            ):
                continue
            candidates.append(
                MatchCandidate(
                    program_id=program_id,
                    title=metadata.get("title", ""),
                    provider=metadata.get("provider", ""),
                    funding_type=metadata.get("funding_type", ""),
                    funding_area=metadata.get("funding_area", ""),
                    regions=metadata.get("regions", ""),
                    eligible_applicants=metadata.get("eligible_applicants", ""),
                    deadline_text=metadata.get("deadline_text", ""),
                    deadline_iso=metadata.get("deadline_iso", ""),
                    url=metadata.get("url", ""),
                    similarity=round(1.0 - float(distance), 4),
                    objective=metadata.get("objective", "")
                    or self._objective_from_document(result, program_id),
                    description=self._description_from_document(result, program_id),
                )
            )
            if len(candidates) >= k:
                break
        return candidates

    @staticmethod
    def _document_for(result: dict, program_id: str) -> str:
        for pid, document in zip(result["ids"][0], result["documents"][0]):
            if pid == program_id:
                return document or ""
        return ""

    def _objective_from_document(self, result: dict, program_id: str) -> str:
        document = self._document_for(result, program_id)
        marker = "Förderziel:"
        if marker in document:
            tail = document.split(marker, 1)[1]
            return tail.split("Beschreibung:", 1)[0].strip()
        return ""

    def _description_from_document(self, result: dict, program_id: str) -> str:
        document = self._document_for(result, program_id)
        marker = "Beschreibung:"
        if marker in document:
            return document.split(marker, 1)[1].strip()
        return document

    # -- Synthesis ----------------------------------------------------------------

    def synthesize(self, client: dict, candidates: list[MatchCandidate]) -> str:
        """Assemble the German report: DB facts + per-programme AI verdict."""
        if not candidates:
            return (
                f"# Fördermittel-Matching: {client['company_name']}\n\n"
                f"Stand: {date.today().isoformat()}\n\n"
                "## Projektbeschreibung\n\n"
                f"{client['project_description']}\n\n"
                "Für das beschriebene Vorhaben wurden nach Anwendung der harten "
                "Filter (Fördergebiet, Antragsberechtigung, Frist) keine passenden "
                "Programme im Index gefunden. Empfehlung: Projektbeschreibung "
                "präzisieren oder Datenbestand erweitern (`python ingest.py`)."
            )

        # Ask the model only for verdict + one-sentence comment per programme.
        # The factual fields come from the database, so they are never invented.
        verdicts = self._verdicts_from_llm(client, candidates) if self.llm else {}

        # Sort by the AI verdict (best first), then by semantic similarity.
        def sort_key(item: tuple[int, MatchCandidate]) -> tuple[int, float]:
            index, candidate = item
            passung = verdicts.get(index, {}).get("passung", "")
            rank = _PASSUNG_RANK.get(passung.strip().lower(), 99)
            return (rank, -candidate.similarity)

        ordered = sorted(enumerate(candidates, start=1), key=sort_key)
        return self._render_report(client, ordered, verdicts)

    def _verdicts_from_llm(self, client: dict, candidates: list[MatchCandidate]) -> dict:
        """Return ``{index: {"passung": str, "kommentar": str}}`` from the LLM.

        Robust to fenced or chatty output; returns ``{}`` on any parse failure
        so the report still renders (just without verdicts).
        """
        blocks = "\n\n".join(
            _CANDIDATE_BLOCK_DE.format(
                index=index,
                title=candidate.title,
                funding_type=candidate.funding_type or "k. A.",
                regions=candidate.regions or "k. A.",
                deadline=candidate.deadline_text or "k. A.",
                objective=(candidate.objective or "k. A.")[:280],
            )
            for index, candidate in enumerate(candidates, start=1)
        )
        prompt = USER_PROMPT_TEMPLATE_DE.format(
            company_name=client["company_name"],
            industry=client["industry"] or "k. A.",
            state=client["state"],
            company_size=client["company_size"] or "k. A.",
            project_description=client["project_description"],
            n=len(candidates),
            candidate_blocks=blocks,
        )
        response = self.llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT_DE), HumanMessage(content=prompt)]
        )
        raw = message_content_to_text(response.content).strip()
        return self._parse_verdicts(raw, n=len(candidates))

    @staticmethod
    def _parse_verdicts(raw: str, n: int) -> dict:
        """Parse the model's JSON verdicts into ``{int_index: {...}}``."""
        import json
        import re

        text = raw.strip()
        if text.startswith("```"):  # strip ```json ... ``` fences
            text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
        # Fall back to the first {...} block if the model added stray text.
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            text = match.group(0) if match else ""
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return {}

        verdicts: dict = {}
        for key, value in data.items():
            try:
                index = int(str(key).strip())
            except ValueError:
                continue
            if 1 <= index <= n and isinstance(value, dict):
                verdicts[index] = {
                    "passung": str(value.get("passung", "")).strip(),
                    "kommentar": str(value.get("kommentar", "")).strip(),
                }
        return verdicts

    def _render_report(
        self,
        client: dict,
        ordered: list[tuple[int, MatchCandidate]],
        verdicts: dict,
    ) -> str:
        """Build the final Markdown document from facts + verdicts."""
        lines = [
            f"# Fördermittel-Matching: {client['company_name']}",
            "",
            f"Stand: {date.today().isoformat()}  ·  Standort: {client['state']}  ·  "
            f"Branche: {client['industry'] or 'k. A.'}  ·  "
            f"Unternehmensgröße: {client['company_size'] or 'k. A.'}",
            "",
            "## Projektbeschreibung",
            "",
            client["project_description"],
            "",
            f"## Passende Förderprogramme ({len(ordered)})",
            "",
        ]
        if self.llm is None:
            lines += [
                "> Hinweis: ohne LLM (`LLM_PROVIDER=mock`) — Reihenfolge nach "
                "semantischer Ähnlichkeit, ohne KI-Passungsbewertung. Für "
                "Passung und KI-Kommentar einen LLM-Provider konfigurieren "
                "(gemini | openai | anthropic | ollama).",
                "",
            ]

        for position, (orig_index, candidate) in enumerate(ordered, start=1):
            verdict = verdicts.get(orig_index, {})
            passung = verdict.get("passung") or "—"
            kommentar = verdict.get("kommentar") or (
                "Kein KI-Kommentar verfügbar." if self.llm is None
                else "Keine Bewertung erhalten."
            )
            frist = candidate.deadline_text or "k. A."
            lines += [
                f"{position}. **{candidate.title}** — Passung: {passung}",
                f"   1. Herausgeber: {candidate.provider or 'k. A.'}",
                f"   2. Förderart: {candidate.funding_type or 'k. A.'}",
                f"   3. Fördergebiet: {candidate.regions or 'k. A.'}",
                f"   4. Frist: {frist}",
                f"   5. Link: {candidate.url or 'k. A.'}",
                f"   6. Semantische Ähnlichkeit: {candidate.similarity:.0%}",
                f"   7. KI-Kommentar: {kommentar}",
                "",
            ]
        lines += [
            "_Hinweis: Anträge vor Vorhabensbeginn stellen; Angaben anhand der "
            "offiziellen Förderrichtlinien prüfen._",
        ]
        return "\n".join(lines)

    # -- Orchestration ---------------------------------------------------------------

    def match_structured(
        self,
        client_id: int,
        k: int | None = None,
        apply_filters: bool = True,
        log: bool = True,
    ) -> list[dict]:
        """Match one client and return presentation-ready result dicts.

        This is the entry point for the web frontend: it makes the single LLM
        call, applies the verdict-based ordering, and returns a list of plain
        dicts (no markdown, no file written) that a template can render
        directly. Each dict has:

            position, title, provider, funding_type, funding_area, regions,
            deadline, url, similarity (0-100 int), passung, passung_rank,
            kommentar, program_id

        Results are ordered best-first (AI verdict, then similarity).
        """
        client = database.get_client(self.settings, client_id)
        if client is None:
            raise ValueError(f"No client with client_id={client_id}.")

        k = k or self.settings.top_k
        candidates = self.retrieve(client, k=k, apply_filters=apply_filters)

        # Single LLM call for verdict + one-sentence comment per programme.
        verdicts = self._verdicts_from_llm(client, candidates) if self.llm else {}

        results: list[dict] = []
        for index, candidate in enumerate(candidates, start=1):
            verdict = verdicts.get(index, {})
            passung = verdict.get("passung", "") or ""
            results.append(
                {
                    "title": candidate.title,
                    "provider": candidate.provider,
                    "funding_type": candidate.funding_type,
                    "funding_area": candidate.funding_area,
                    "regions": candidate.regions,
                    "deadline": candidate.deadline_text,
                    "url": candidate.url,
                    "similarity": round(candidate.similarity * 100),
                    "passung": passung,
                    "passung_rank": _PASSUNG_RANK.get(passung.strip().lower(), 99),
                    "kommentar": verdict.get("kommentar", "") or "",
                    # Förderziel + first part of the official Volltext, so the
                    # consultant can judge the fit independently of the AI.
                    "description": self._program_description(candidate),
                    "objective": candidate.objective or "",
                    "program_id": candidate.program_id,
                }
            )

        # Order best-first: AI verdict, then semantic similarity.
        results.sort(key=lambda r: (r["passung_rank"], -r["similarity"]))
        for position, result in enumerate(results, start=1):
            result["position"] = position

        if log:
            database.log_match_run(
                self.settings,
                client_id=client_id,
                top_k=k,
                program_ids=[candidate.program_id for candidate in candidates],
                llm_provider=self.settings.llm_provider,
                report_path="(web)",
            )
        return results

    @staticmethod
    def _program_description(candidate: "MatchCandidate", max_chars: int = 600) -> str:
        """Förderziel plus the first few sentences of the official Volltext.

        Lets the consultant judge the fit independently of the AI comment.
        Trims at a sentence boundary near ``max_chars`` rather than mid-word.
        """
        objective = (candidate.objective or "").strip()
        description = (candidate.description or "").strip()

        # Avoid repeating the objective if the description starts with it.
        if description.startswith(objective[:60]) and objective:
            text = description
        elif objective and description:
            text = f"{objective} {description}"
        else:
            text = objective or description

        if len(text) <= max_chars:
            return text
        # Cut at the last sentence end before the limit (fall back to a word).
        window = text[: max_chars + 1]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if cut >= int(max_chars * 0.5):
            return text[: cut + 1]
        return text[:max_chars].rsplit(" ", 1)[0] + " …"

    def match(
        self,
        client_id: int,
        k: int | None = None,
        apply_filters: bool = True,
        save: bool = True,
    ) -> tuple[str, list[MatchCandidate], str]:
        """Full match for one client. Returns ``(report, candidates, report_path)``."""
        client = database.get_client(self.settings, client_id)
        if client is None:
            raise ValueError(f"No client with client_id={client_id}.")

        k = k or self.settings.top_k
        candidates = self.retrieve(client, k=k, apply_filters=apply_filters)
        report = self.synthesize(client, candidates)

        report_path = ""
        if save:
            self.settings.ensure_directories()
            timestamp = utcnow_iso().replace(":", "").replace("-", "").replace("+0000", "Z")
            filename = f"match_client{client_id}_{timestamp}.md"
            path = self.settings.reports_dir / filename
            path.write_text(report + "\n", encoding="utf-8")
            report_path = str(path)
            database.log_match_run(
                self.settings,
                client_id=client_id,
                top_k=k,
                program_ids=[candidate.program_id for candidate in candidates],
                llm_provider=self.settings.llm_provider,
                report_path=report_path,
            )
        return report, candidates, report_path