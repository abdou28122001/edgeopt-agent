from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import run_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_demo(args.spec, args.output)
    print(f"sandbox_job_complete={args.output}")


if __name__ == "__main__":
    main()
