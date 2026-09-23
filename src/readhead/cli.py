"""Command line for a single read or the local server."""

from __future__ import annotations

import argparse
import json
import os

from readhead.compile import DecideIn
from readhead.engine import Readhead
from readhead.trunk import Trunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a readout head and run it on a frozen causal model.")
    parser.add_argument("--model", default=os.environ.get("READHEAD_MODEL", "Qwen/Qwen3.5-2B"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=43124)

    ask = sub.add_parser("ask", help="Read one question")
    ask.add_argument("--context", required=True)
    ask.add_argument("--question")
    ask.add_argument("--options", nargs="*")
    ask.add_argument("--levels", nargs="*")
    ask.add_argument("--schema")
    ask.add_argument("--strategy", default="auto")
    ask.add_argument("--type", default="auto")

    args = parser.parse_args()
    if args.cmd == "serve":
        os.environ["READHEAD_MODEL"] = args.model
        os.environ["READHEAD_HOST"] = args.host
        os.environ["READHEAD_PORT"] = str(args.port)
        from readhead.server import main as serve_main

        serve_main()
        return

    schema = None
    if args.schema:
        with open(args.schema, encoding="utf-8") as handle:
            schema = json.load(handle)
    trunk = Trunk.load(args.model)
    result = Readhead(trunk).decide(
        DecideIn.model_validate(
            {
                "context": args.context,
                "question": args.question,
                "type": args.type,
                "options": args.options,
                "levels": args.levels,
                "schema": schema,
                "strategy": args.strategy,
            }
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
