from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a real survival policy with a recorded curriculum and frozen evaluation.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17073)
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--max-level", type=int, choices=(4, 6), default=6)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args(argv)
    if args.steps < 1024 or not 0 <= args.seed < 2**31:
        parser.error("--steps must be at least1024 and --seed must be0..2147483647.")
    minimum_cycle = 8 if args.max_level == 4 else 16
    if not 1 <= args.cycles <= args.steps // minimum_cycle:
        parser.error(f"--cycles must be positive and allow at least{minimum_cycle} transitions per cycle.")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01:
        parser.error("--learning-rate must be finite and in (0,0.01].")
    protected = Path(__file__).resolve().parents[1] / "storage" / "pretrained_3d.npz"
    if args.output.resolve() in {args.source.resolve(), protected}:
        parser.error("Choose a new output path; the source and original ready asset must remain unchanged.")
    if args.report.resolve() in {args.source.resolve(), args.output.resolve(), protected}:
        parser.error("The report must not overwrite a model asset.")
    os.environ["SNAKE_STORAGE_DIR"] = str(args.output.resolve().parent / "isolated-training")
    from backend.agent import DQNAgent, Experience, PARAMETERS, ReplayBuffer
    from backend.config import ENVIRONMENT_VERSION, REWARD_CONFIG, REWARD_VERSION, SHAPING_GAMMA
    from backend.environment import SnakeEnvironment

    import numpy as np

    started = time.perf_counter()
    source = DQNAgent(args.source, seed=args.seed)
    if not source.load_from(args.source):
        raise ValueError(source.last_load_error)
    source_metadata, _ = source.read_checkpoint(args.source)
    source_provenance = source_metadata["trainingMetadata"]
    inherited_seeds = set(source_provenance.get("trainingSeeds", []))
    inherited_steps = (
        source_provenance.get("survivalTransitions", source_provenance.get("transitions", 0))
        if source_provenance.get("rewardVersion") == REWARD_VERSION else 0
    )
    source_bytes = args.source.read_bytes()
    original_hash = hashlib.sha256(source_bytes).hexdigest()
    policy = DQNAgent(args.output, seed=args.seed, epsilon=0.2, learning_rate=args.learning_rate)
    for name in PARAMETERS:
        setattr(policy, name, getattr(source, name).copy())
    policy._sync_target_network()
    policy.gamma = SHAPING_GAMMA
    policy.epsilon_decay = math.exp(math.log(policy.epsilon_min / policy.epsilon) / args.steps)
    replay = ReplayBuffer(seed=args.seed + 202)
    rng = np.random.default_rng(args.seed + 101)
    seeds = [3100000701, 3100000702, 3100000703, 3100000704]
    training_seeds: set[int] = set()

    def evaluate(agent: DQNAgent) -> dict[str, object]:
        before = agent.parameter_hash()
        rows = []
        for level in (1, 4, 6):
            for seed in seeds:
                env = SnakeEnvironment(seed=seed, level=level)
                state = env.get_state()
                reward = 0.0
                while not env.done:
                    action = agent.act(state, env.action_mask(), explore=False, capture_decision=False)
                    transition = env.step(action)
                    state = transition["next_state"]
                    reward += transition["reward"]
                rows.append({
                    "seed": seed, "level": level, "score": env.score, "food": env.food_collected,
                    "gold": env.gold_collected, "steps": env.steps, "escapes": env.enemy_escapes,
                    "rivalFood": env.rival_food_collected, "cause": env.last_death,
                    "dangerExposure": env.danger_exposure / max(1, env.steps),
                    "foodPer100Ticks": env.food_collected * 100 / max(1, env.steps), "reward": reward,
                })
        assert before == agent.parameter_hash()
        return {
            "rows": rows,
            "summary": {
                str(level): {
                    "meanScore": sum(row["score"] for row in rows if row["level"] == level) / len(seeds),
                    "gold": sum(row["gold"] for row in rows if row["level"] == level),
                    "meanSteps": sum(row["steps"] for row in rows if row["level"] == level) / len(seeds),
                    "deaths": sum(row["cause"] != "time_limit" for row in rows if row["level"] == level),
                } for level in (1, 4, 6)
            },
        }

    baseline = evaluate(source)
    schedule = []
    for cycle in range(args.cycles):
        budget = args.steps // args.cycles + int(cycle < args.steps % args.cycles)
        if args.max_level == 4:
            stages = [(1, budget // 8), (2, budget // 8), (3, budget // 4)]
        else:
            stages = [(1, budget // 16), (2, budget // 16), (3, budget // 8),
                      (4, budget // 4), (5, budget // 4)]
        stages.append((args.max_level, budget - sum(steps for _, steps in stages)))
        schedule.extend((cycle + 1, level, count) for level, count in stages)
    transition_count = 0
    episode_count = 0
    reward_totals: Counter[str] = Counter()
    training_events: Counter[str] = Counter()
    stages = []
    snapshot_interval = max(1024, args.steps // 4)
    next_snapshot = snapshot_interval

    def provenance() -> dict[str, object]:
        all_seeds = inherited_seeds | training_seeds
        if all_seeds.intersection(seeds):
            raise ValueError("Warm-start or current training overlapped the frozen evaluation split.")
        return {
            "rewardVersion": REWARD_VERSION, "environmentVersion": ENVIRONMENT_VERSION,
            "rewardConfig": REWARD_CONFIG, "source": str(args.source.resolve()), "sourceSHA256": original_hash,
            "sourceParameters": source.parameter_hash(), "policyParameters": policy.parameter_hash(),
            "sourceTrainingSteps": source.environment_steps, "sourceGradientUpdates": source.train_steps,
            "seed": args.seed, "transitions": transition_count, "gradientUpdates": policy.train_steps,
            "survivalTransitions": inherited_steps + transition_count,
            "episodes": episode_count, "trainingSeeds": sorted(all_seeds), "evaluationSeeds": seeds,
            "seedsDisjoint": True, "stages": list(stages), "curriculumCycles": args.cycles,
            "maxLevel": args.max_level,
            "learningRate": args.learning_rate, "trainingEvents": dict(training_events),
            "rewardTotals": dict(reward_totals), "beforeEvaluation": baseline,
            "seconds": time.perf_counter() - started,
            "controller": "masked Double DQN only; no planner chooses player actions",
        }

    for cycle, level, count in schedule:
        env = None
        state = None
        stage_episodes = 0
        stage_started = time.perf_counter()
        for _ in range(count):
            if env is None or env.done:
                seed = int(rng.integers(0, 2**31))
                if seed in seeds:
                    raise ValueError("Training seed overlapped the frozen evaluation split.")
                training_seeds.add(seed)
                env = SnakeEnvironment(seed=seed, level=level)
                state = env.get_state()
                episode_count += 1
                stage_episodes += 1
            mask = env.action_mask()
            action = policy.act(state, mask, explore=True, capture_decision=False)
            transition = env.step(action)
            replay.add(Experience(state, action, transition["reward"], transition["next_state"],
                                  transition["done"], mask, env.action_mask()))
            state = transition["next_state"]
            transition_count += 1
            policy.record_training_step()
            reward_totals.update(transition["info"]["rewardComponents"])
            training_events.update(f"{level}:{event['type']}" for event in transition["info"]["events"])
            if replay.size() >= 128 and transition_count % 4 == 0:
                policy.train_batch(replay.sample(64))
        stages.append({"cycle": cycle, "level": level, "steps": count, "episodes": stage_episodes,
                       "seconds": time.perf_counter() - stage_started})
        if level == args.max_level:
            print(json.dumps({"cycle": cycle, "completed": transition_count,
                              "gradientUpdates": policy.train_steps, "loss": policy.last_loss,
                              "goldCollections": sum(training_events[f"{number}:gold_food"] for number in (4, 5, 6))}),
                  flush=True)
            if transition_count >= next_snapshot and transition_count < args.steps:
                checkpoint = args.output.with_name(f"{args.output.stem}-step{transition_count}.npz")
                policy.save_to(checkpoint, provenance())
                next_snapshot = transition_count + snapshot_interval
    final = evaluate(policy)
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != original_hash:
        raise ValueError("The original ready asset changed during training.")
    payload = provenance()
    payload["afterEvaluation"] = final
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    policy.save_to(args.output, payload)
    print(json.dumps({"report": str(args.report), "checkpoint": str(args.output),
                      "before": baseline["summary"], "after": final["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
