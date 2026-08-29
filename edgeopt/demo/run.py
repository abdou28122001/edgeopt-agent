from __future__ import annotations

import argparse
from pathlib import Path

from ..runtime import run_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path(".edgeopt-demo/input/run-spec.json"))
    parser.add_argument("--output", type=Path, default=Path(".edgeopt-demo/run"))
    args = parser.parse_args()
    manifest = run_demo(args.spec, args.output)
    print(f"decision={manifest['decision']['decision']} output={args.output}")


if __name__ == "__main__":
    main()
