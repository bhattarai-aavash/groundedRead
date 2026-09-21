from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from dsqa.config import CONFIG


def _cmd_ingest(directory: str) -> int:
    from dsqa.ingest import ingest_dir
    from dsqa.store import upsert

    chunks = ingest_dir(directory)
    n = upsert(chunks)
    print(f"upserted {n} chunks into {CONFIG.collection} ({CONFIG.chroma_dir})")
    return 0


def _cmd_ask(question: str, verbose: bool) -> int:
    from dsqa.graph import ask

    state = ask(question)
    print(state.get("answer") or "")
    docs = state.get("docs") or []
    if docs:
        print("\nSources:")
        seen: set[tuple[str, int]] = set()
        for doc in docs:
            key = (str(doc.get("source", "")), int(doc.get("page", 0)))
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {key[0]} p.{key[1]}")
    if verbose:
        print("\nTrace:")
        for event in state.get("trace") or []:
            print(f"  {event}")
        spans = state.get("spans") or []
        if spans:
            print("\nSpans:")
            for span in spans:
                print(
                    f"  {span.get('node')}: {span.get('ms')}ms "
                    f"in={span.get('tokens_in')} out={span.get('tokens_out')} "
                    f"${span.get('cost_usd')}"
                )
    return 0


def _cmd_stats() -> int:
    from dsqa.store import count

    print(json.dumps({"chunks": count(), "config": asdict(CONFIG)}, indent=2))
    return 0


def _cmd_reset(yes: bool = False) -> int:
    from dsqa.store import reset

    if not yes:
        reply = input(f"Drop collection {CONFIG.collection!r}? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("aborted")
            return 1
    reset()
    print(f"reset collection {CONFIG.collection!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsqa", description="Distributed systems paper Q&A")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest PDFs into Chroma")
    p_ingest.add_argument("dir", nargs="?", default="data/papers")

    p_ask = sub.add_parser("ask", help="ask a question over ingested papers")
    p_ask.add_argument("question")
    p_ask.add_argument("--verbose", action="store_true")

    sub.add_parser("stats", help="chunk count and active config")
    p_reset = sub.add_parser("reset", help="drop the Chroma collection")
    p_reset.add_argument("--yes", action="store_true", help="skip confirmation")

    p_ui = sub.add_parser("ui", help="open a local web UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        return _cmd_ingest(args.dir)
    if args.cmd == "ask":
        return _cmd_ask(args.question, args.verbose)
    if args.cmd == "stats":
        return _cmd_stats()
    if args.cmd == "reset":
        return _cmd_reset(yes=args.yes)
    if args.cmd == "ui":
        from dsqa.ui import serve

        serve(args.host, args.port)
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
