"""``triage.cli`` with a fixed plan in place of Ollama, run as its own process
by ``test_durable.py`` so a kill is a real process death, not a simulation.

    python -m tests.cli_child run <csv> --thread <id>

Paths come from ``TRIAGE_CHECKPOINT_DB`` and ``TRIAGE_RUNS_DIR``.
"""

import functools
import sys

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from triage import cli
from triage.ops import CleaningPlan, DropColumn, StripWhitespace

# The drop needs approval under the default policy; the strip does not.
PLAN = CleaningPlan(ops=[StripWhitespace(column="customer", reason="r"),
                         DropColumn(column="channel", reason="r")])
PLANNER = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": PLAN, "parsing_error": None})

if __name__ == "__main__":
    cli.build_graph = functools.partial(cli.build_graph, planner=PLANNER)
    sys.exit(cli.main())
