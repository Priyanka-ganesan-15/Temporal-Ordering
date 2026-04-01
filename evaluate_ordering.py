"""CLI entrypoint for ordering baseline evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronologic.evaluation.runner import parse_evaluation_args, run_evaluation_cli


def main() -> None:
    run_evaluation_cli(parse_evaluation_args())


if __name__ == "__main__":
    main()