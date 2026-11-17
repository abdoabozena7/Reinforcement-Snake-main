from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the real six-direction 3D learner without rendering.")
    parser.add_argument("--steps", type=int, default=10_000, help="Additional training transitions.")
    parser.add_argument("--seed", type=int, help="Local seed; defaults to 0 or the resumed session seed.")
    parser.add_argument("--level", type=int, choices=range(1, 7), help="Defaults to 1 or resumed difficulty.")
    parser.add_argument("--full-survival", action="store_true")
    parser.add_argument("--curriculum", action="store_true", help="Start level 1; evaluation gates promotion.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints") / "arena-3d-v3.npz")
    parser.add_argument("--resume", type=Path, help="Resume a compatible full session checkpoint.")
    parser.add_argument("--batch-steps", type=int, default=256, help="1..2048 transitions per batch.")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N training transitions.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="1..32 held-out episodes per suite.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", help="Distinct held-out seeds (never training seeds).")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive.")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be 0..4294967295.")
    if not 1 <= args.batch_steps <= 2048:
        parser.error("--batch-steps must be 1..2048.")
    if not 1 <= args.eval_episodes <= 32 or args.eval_every < 0:
        parser.error("--eval-episodes must be 1..32 and --eval-every cannot be negative.")
    if args.curriculum and args.full_survival:
        parser.error("Choose curriculum or full survival, not both.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    output = args.output.resolve()
    isolated_storage = output.parent / f"{output.stem}.training"
    # Importing the API-compatible global session must not inspect a user's storage.
    os.environ["SNAKE_STORAGE_DIR"] = str(isolated_storage)
    seed = args.seed if args.seed is not None else 0
    os.environ["SNAKE_SEED"] = str(seed)
    from backend.config import MODEL_VERSION
    from backend.session import TrainingSession

    started = time.perf_counter()
    session = TrainingSession(storage_dir=isolated_storage, seed=seed, load_saved=False)
    if args.resume is not None and not session.load_checkpoint(args.resume):
        raise ValueError(session.log_message)
    preserve_context = args.resume is not None and args.level is None
    full = args.full_survival or (
        preserve_context and session.full_survival and not args.curriculum
    )
    auto = args.curriculum or (
        preserve_context and session.auto_curriculum and not args.full_survival
    )
    initial_level = (
        1 if args.curriculum else
        6 if full else
        args.level if args.level is not None else
        session.level if preserve_context else 1
    )
    controls = {
        "mode": "train",
        "level": initial_level,
        "fullSurvival": full,
        "autoCurriculum": auto,
        "paused": False,
    }
    if args.resume is not None and args.seed is not None:
        controls["seed"] = args.seed
    session.control(controls)
    suite = args.eval_seeds or session.default_evaluation_seeds(args.eval_episodes)
    baseline = session.evaluate(episodes=args.eval_episodes, seeds=suite)["evaluation"]
    before_hash = session.agent.parameter_hash()
    starting_steps = session.training_steps
    starting_gradients = session.agent.train_steps
    starting_episodes = session.completed_episodes
    interval = args.eval_every or (5_000 if args.curriculum else 0)
    next_evaluation = interval
    training_seconds = 0.0
    intermediate: list[dict[str, object]] = []
    while session.training_steps - starting_steps < args.steps:
        completed = session.training_steps - starting_steps
        batch = min(args.batch_steps, args.steps - completed)
        if interval:
            batch = min(batch, max(1, next_evaluation - completed))
        batch_started = time.perf_counter()
        session.train_steps(batch)
        training_seconds += time.perf_counter() - batch_started
        if interval and session.training_steps - starting_steps >= next_evaluation:
            intermediate.append(session.evaluate(episodes=len(suite), seeds=suite)["evaluation"])
            del intermediate[:-40]
            next_evaluation += interval
    final = session.evaluate(episodes=len(suite), seeds=suite)["evaluation"]
    session.save_checkpoint(output)
    actual = session.training_steps - starting_steps
    after_hash = session.agent.parameter_hash()
    payload = {
        "modelVersion": MODEL_VERSION,
        "seed": session.seed,
        "level": session.level,
        "fullSurvival": session.full_survival,
        "stepsRequested": args.steps,
        "transitions": actual,
        "trainingSteps": session.training_steps,
        "gradientSteps": session.agent.train_steps,
        "gradientUpdates": session.agent.train_steps - starting_gradients,
        "episodesCompleted": session.completed_episodes - starting_episodes,
        "bufferSize": session.buffer.size(),
        "epsilon": session.agent.epsilon,
        "loss": session.agent.last_loss,
        "trainingSeconds": training_seconds,
        "elapsedSeconds": time.perf_counter() - started,
        "transitionsPerSecond": actual / max(training_seconds, 1e-9),
        "checkpoint": str(output),
        "trainingSeeds": sorted(session.training_seeds),
        "evaluationSeeds": list(suite),
        "seedsDisjoint": not bool(session.training_seeds.intersection(suite)),
        "weightHashBefore": before_hash,
        "weightHashAfter": after_hash,
        "weightChanged": before_hash != after_hash,
        "beforeEvaluation": baseline,
        "afterEvaluation": final,
        "intermediateEvaluations": intermediate,
        "curriculum": session.curriculum.snapshot(),
        "migration": "Older checkpoints remain untouched. Ready 3D weights are explicitly converted and fine-tuned; planar models require retraining.",
    }
    print(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
