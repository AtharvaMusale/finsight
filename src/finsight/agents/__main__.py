"""Run one FinSight A2A agent.

Run from the finsight/ folder, one terminal per agent:
  uv run python -m finsight.agents retrieval     # http://127.0.0.1:9101
  uv run python -m finsight.agents facts         # http://127.0.0.1:9102
  uv run python -m finsight.agents verifier      # http://127.0.0.1:9103
  uv run python -m finsight.agents analyst       # http://127.0.0.1:9100

To make the analyst delegate, set these in .env (unset = that capability stays in-process):
  FINSIGHT_A2A_RETRIEVAL_URL=http://127.0.0.1:9101
  FINSIGHT_A2A_FACTS_URL=http://127.0.0.1:9102
  FINSIGHT_A2A_VERIFIER_URL=http://127.0.0.1:9103

Agents bind to 127.0.0.1 and have no authentication: local use only.
"""

import argparse
import logging

import uvicorn

from finsight.agents.services import BUILDERS, DEFAULT_PORTS
from finsight.api.app import _build_deps
from finsight.config import get_settings
from finsight.observability import configure_langsmith
from finsight.tracing import configure as configure_tracing


def main() -> None:
    ap = argparse.ArgumentParser(prog="finsight-agent")
    ap.add_argument("agent", choices=sorted(BUILDERS))
    ap.add_argument("--host", default="127.0.0.1", help="keep local: there is no authentication")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()
    configure_tracing(args.agent, settings.trace_dir, settings.tracing)
    configure_langsmith(settings)
    port = args.port or DEFAULT_PORTS[args.agent]
    # Workers must stay local; only the analyst may delegate.
    remote = args.agent == "analyst"
    app = BUILDERS[args.agent](
        settings, lambda: _build_deps(settings, remote=remote), f"http://{args.host}:{port}"
    )
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
