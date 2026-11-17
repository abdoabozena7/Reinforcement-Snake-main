from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Greedily evaluate a frozen six-direction 3D checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--seeds", nargs="+", help="Held-out seeds, separated by spaces or commas."
    )
    parser.add_argument("--episodes", type=int, default=8, help="Default suite size without --seeds.")
    parser.add_argument("--level", type=int, choices=range(1, 7), help="Defaults to checkpoint difficulty.")
    parser.add_argument("--full-survival", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 32:
        parser.error("--episodes must be 1..32.")
    seeds = (
        [int(value) for group in args.seeds for value in group.split(",") if value]
        if args.seeds
        else None
    )
    checkpoint = args.checkpoint.resolve()
    isolated_storage = checkpoint.parent / f"{checkpoint.stem}.evaluation"
    os.environ["SNAKE_STORAGE_DIR"] = str(isolated_storage)
    from backend.session import TrainingSession

    session = TrainingSession(storage_dir=isolated_storage, seed=0, load_saved=False)
    if not session.load_checkpoint(checkpoint):
        raise ValueError(session.log_message)
    if args.level is not None or args.full_survival:
        session.control(
            {
                "mode": "eval",
                "level": 6 if args.full_survival else args.level,
                "fullSurvival": args.full_survival,
                "autoCurriculum": False,
                "paused": True,
            }
        )
    before = session.agent.parameter_hash()
    record = session.evaluate(episodes=args.episodes, seeds=seeds)["evaluation"]
    record["checkpoint"] = str(checkpoint)
    record["policyUnchanged"] = before == session.agent.parameter_hash()
    record["seedsDisjoint"] = not bool(session.training_seeds.intersection(record["seeds"]))
    print(json.dumps(record, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
