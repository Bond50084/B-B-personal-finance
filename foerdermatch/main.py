"""FoerderMatch — command-line entry point.

Subcommands:

    demo                          End-to-end showcase in offline mock mode:
                                  init DB -> ingest mock data -> seed two
                                  demo clients -> match both -> print reports.
    ingest [--source ...]         Refresh the programme index (Component A).
    client add | list | show | delete    Manage client profiles (Component B).
    match  --client-id N | --all  Run the RAG matching (Component C).

Everything works offline with the default providers (EMBEDDING_PROVIDER=mock,
LLM_PROVIDER=mock). Configure real providers via .env for production quality.
"""

from __future__ import annotations

import argparse
import sys

try:  # works both as a package and standalone (python main.py)
    from . import database
    from .config import settings
    from .ingest import (
        export_programs_csv,
        export_programs_markdown,
        list_indexed_programs,
        run_ingestion,
    )
    from .matching_engine import MatchingEngine
except ImportError:
    import database
    from config import settings
    from ingest import (
        export_programs_csv,
        export_programs_markdown,
        list_indexed_programs,
        run_ingestion,
    )
    from matching_engine import MatchingEngine

# ---------------------------------------------------------------------------
# Demo clients ------------------------------------------------------------------
# ---------------------------------------------------------------------------

_DEMO_CLIENTS = [
    {
        "company_name": "Müller Maschinenbau GmbH",
        "industry": "Metallverarbeitung / Maschinenbau",
        "state": "Nordrhein-Westfalen",
        "company_size": "KMU, 80 Mitarbeitende",
        "project_description": (
            "Wir wollen unsere Fertigung modernisieren und in energieeffiziente "
            "CNC-Fräsmaschinen investieren. Zusätzlich planen wir eine "
            "Wärmerückgewinnung an unserer Druckluftanlage sowie die Nutzung der "
            "Abwärme zur Hallenheizung. Ziel ist es, den Stromverbrauch und die "
            "CO2-Emissionen der Produktion um rund 30 Prozent zu senken."
        ),
    },
    {
        "company_name": "CodeWerk Analytics GmbH",
        "industry": "Softwareentwicklung / Künstliche Intelligenz",
        "state": "Bayern",
        "company_size": "Start-up, 12 Mitarbeitende",
        "project_description": (
            "Wir entwickeln eine KI-gestützte Software für Predictive Maintenance, "
            "die Sensordaten aus Industrieanlagen auswertet und Wartungsbedarf "
            "vorhersagt. Das Vorhaben umfasst angewandte Forschung an neuen "
            "Machine-Learning-Verfahren und die Digitalisierung unserer eigenen "
            "Entwicklungs- und Vertriebsprozesse."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers ------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _seed_client(client: dict) -> int:
    """Insert a demo client, reusing an existing row with the same name."""
    existing = database.find_client_by_name(settings, client["company_name"])
    if existing:
        print(f"  client #{existing['client_id']} '{client['company_name']}' exists — reusing")
        return existing["client_id"]
    client_id = database.add_client(settings, **client)
    print(f"  client #{client_id} '{client['company_name']}' created")
    return client_id


def _print_client(client: dict) -> None:
    print(
        f"#{client['client_id']:<3} {client['company_name']}\n"
        f"      Branche : {client['industry']}\n"
        f"      Standort: {client['state']}\n"
        f"      Größe   : {client['company_size']}\n"
        f"      Projekt : {client['project_description'][:140]}"
        f"{'…' if len(client['project_description']) > 140 else ''}"
    )


def _run_match(client_id: int, k: int, apply_filters: bool) -> None:
    engine = MatchingEngine(settings)
    report, candidates, report_path = engine.match(
        client_id, k=k, apply_filters=apply_filters, save=True
    )
    print("\n" + "=" * 78)
    print(report)
    print("=" * 78)
    print(f"[match] {len(candidates)} programme(s); report saved to {report_path}\n")


# ---------------------------------------------------------------------------
# Commands ------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> None:
    print("FoerderMatch — Demo (offline mock mode)")
    print(f"  embedding provider: {settings.embedding_provider}")
    print(f"  llm provider      : {settings.llm_provider}\n")

    print("[1/4] Initialising database …")
    database.init_db(settings)

    print("[2/4] Ingesting subsidy programmes (mock source) …")
    run_ingestion(settings, source="mock", rebuild=args.rebuild)

    print("\n[3/4] Seeding demo clients …")
    client_ids = [_seed_client(client) for client in _DEMO_CLIENTS]

    print("\n[4/4] Matching each client against the programme index …")
    for client_id in client_ids:
        _run_match(client_id, k=settings.top_k, apply_filters=True)


def cmd_ingest(args: argparse.Namespace) -> None:
    run_ingestion(settings, source=args.source, urls=args.urls, rebuild=args.rebuild)


def cmd_client(args: argparse.Namespace) -> None:
    database.init_db(settings)
    if args.client_action == "add":
        client_id = database.add_client(
            settings,
            company_name=args.name,
            industry=args.industry,
            state=args.state,
            project_description=args.project,
            company_size=args.size,
        )
        print(f"Created client #{client_id}.")
    elif args.client_action == "list":
        clients = database.list_clients(settings)
        if not clients:
            print("No clients yet. Add one with: python main.py client add …")
            return
        for client in clients:
            _print_client(client)
    elif args.client_action == "show":
        client = database.get_client(settings, args.client_id)
        if client is None:
            print(f"No client with id {args.client_id}.", file=sys.stderr)
            sys.exit(1)
        _print_client(client)
    elif args.client_action == "delete":
        ok = database.delete_client(settings, args.client_id)
        print(f"Deleted client #{args.client_id}." if ok else f"No client #{args.client_id}.")


def cmd_match(args: argparse.Namespace) -> None:
    database.init_db(settings)
    if args.all:
        clients = database.list_clients(settings)
        if not clients:
            print("No clients to match. Add one first.", file=sys.stderr)
            sys.exit(1)
        for client in clients:
            _run_match(client["client_id"], k=args.top_k, apply_filters=not args.no_filters)
    else:
        _run_match(args.client_id, k=args.top_k, apply_filters=not args.no_filters)


def cmd_programs(args: argparse.Namespace) -> None:
    """Inspect the programmes currently held in the vector index."""
    programs = list_indexed_programs(settings)
    if not programs:
        print("The index is empty. Build it first with: python main.py ingest", file=sys.stderr)
        sys.exit(1)

    if args.programs_action == "list":
        print(f"{len(programs)} programme(s) currently indexed:\n")
        for index, program in enumerate(programs, start=1):
            companies = "Unternehmen: ja" if program.get("eligible_companies") else "Unternehmen: nein"
            print(
                f"{index:>2}. {program.get('title', '(ohne Titel)')}\n"
                f"      Gebiet: {program.get('regions', 'k. A.')}  ·  {companies}  ·  "
                f"Frist: {program.get('deadline_text', 'k. A.')}\n"
                f"      Quelle: {program.get('url', 'k. A.')}"
            )
        return

    # export
    settings.ensure_directories()
    fmt = args.format
    if fmt in ("markdown", "both"):
        md_path = settings.reports_dir / "indexed_programs.md"
        count = export_programs_markdown(settings, md_path)
        print(f"Wrote {count} programme(s) to {md_path}")
    if fmt in ("csv", "both"):
        csv_path = settings.reports_dir / "indexed_programs.csv"
        count = export_programs_csv(settings, csv_path)
        print(f"Wrote {count} programme(s) to {csv_path}")


# ---------------------------------------------------------------------------
# Argument parser ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foerdermatch",
        description="Match clients to German public subsidy programmes (RAG MVP).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # demo
    demo = subparsers.add_parser("demo", help="Run the full offline showcase.")
    demo.add_argument("--rebuild", action="store_true", help="Rebuild the vector index first.")
    demo.set_defaults(func=cmd_demo)

    # ingest
    ingest = subparsers.add_parser("ingest", help="Refresh the programme index.")
    ingest.add_argument("--source", choices=("mock", "live"), default="mock")
    ingest.add_argument("--urls", nargs="*", help="Detail-page URLs for --source live.")
    ingest.add_argument("--rebuild", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    # client
    client = subparsers.add_parser("client", help="Manage client profiles.")
    client_sub = client.add_subparsers(dest="client_action", required=True)

    client_add = client_sub.add_parser("add", help="Add a client profile.")
    client_add.add_argument("--name", required=True)
    client_add.add_argument("--industry", default="")
    client_add.add_argument("--state", required=True, help="Bundesland or 'bundesweit'.")
    client_add.add_argument("--project", required=True, help="Project description (RAG query).")
    client_add.add_argument("--size", default="KMU")
    client_add.set_defaults(func=cmd_client)

    client_list = client_sub.add_parser("list", help="List all clients.")
    client_list.set_defaults(func=cmd_client)

    client_show = client_sub.add_parser("show", help="Show one client.")
    client_show.add_argument("client_id", type=int)
    client_show.set_defaults(func=cmd_client)

    client_delete = client_sub.add_parser("delete", help="Delete a client.")
    client_delete.add_argument("client_id", type=int)
    client_delete.set_defaults(func=cmd_client)

    # match
    match = subparsers.add_parser("match", help="Run RAG matching for client(s).")
    target = match.add_mutually_exclusive_group(required=True)
    target.add_argument("--client-id", type=int, dest="client_id")
    target.add_argument("--all", action="store_true", help="Match every client.")
    match.add_argument("-k", "--top-k", type=int, default=settings.top_k, dest="top_k")
    match.add_argument("--no-filters", action="store_true", help="Disable hard metadata filters.")
    match.set_defaults(func=cmd_match)

    # programs
    programs = subparsers.add_parser(
        "programs", help="Inspect the subsidy programmes currently in the index."
    )
    programs_sub = programs.add_subparsers(dest="programs_action", required=True)

    programs_list = programs_sub.add_parser("list", help="Print all indexed programmes.")
    programs_list.set_defaults(func=cmd_programs)

    programs_export = programs_sub.add_parser(
        "export", help="Write all indexed programmes to a readable file in reports/."
    )
    programs_export.add_argument(
        "--format", choices=("markdown", "csv", "both"), default="both"
    )
    programs_export.set_defaults(func=cmd_programs)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
