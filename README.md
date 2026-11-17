# Classic Reference Arena + Live Training HUD

The classic arena and live training HUD are the canonical application on `main`.

The supplied `Snake_3D_AI.html` remains the visual authority: outlined
cubic snakes, cyan wireframe volume, simple cyan/black controls and the
original lighting. The requested changes are a **black backdrop**, clearer
glowing food, subtle YOU glow, a bottom indicator dock and miniature real
neurons at its lower left. The dock reserves real arena space; it does not
stretch or cover the WebGL canvas.

**Original Orbit** remains the default: FOV60, radius50, height30,
look-at(0,5,0), angle increment0.002 per rendered frame. Its formula,
grid geometry and original0.9-unit cubes/1.2-unit outlines are preserved.
Two additional views are optional, not automatic cinematic cuts.
There are no new faces, goal lines or large side-by-side dashboard.

The current Python
`survival-balanced-v8` environment and `arena-3d-v3` Double DQN retain
six-axis movement, actual learning/replay, A* enemies, current
enemy counts, physical obstacles/gates/moving walls, food and abilities.
The old reference's browser-only search and random-food simulation do not run.

Red is the RL player. Hunter/rival actors use the original orange renderer;
the patrol is a grey moving obstacle, matching the obstacle colour.
Each actor now has its own labelled length bar and numeric length; YOU
and the rival additionally show their actual food Score, not unexplained
slash-separated numbers.
Additional obstacles, gold and hazards draw the real current board cells.
Original1.2-unit black outlines and0.9-unit cubes are preserved exactly;
these visual hulls do not change Python collision occupancy.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 19101
```

Open **http://127.0.0.1:19101/**.
The preview needs the real Python API, not `http.server`.
Three.js0.172.0 and its MIT license are served locally.
The reference's original Google Fonts link is unchanged.

## HUD and controls

The bottom dock shows the six real session indicators: **Episode, High
Score, Epsilon, Current Score, Average Score and Replay Buffer**. A small
left-hand neural canvas uses the same real sampled activations, signed
weights and selected action as the modern view, with weak connections
omitted for readability. It is not a decorative or independently simulated
network. The footer also reports actual training steps, optimizer updates,
loss and the last transition Reward. Before any optimizer update, loss
explicitly reads **awaiting update**, not a fabricated zero.

Open **Controls** for Policy, Scenario (including Auto curriculum and Full
Survival), View, Pause/Resume, New Run, Load Best / Ready, Load Saved / Best,
Save Weights, Reset Weights and Evaluate & Keep Best. Controls start folded
on small screens. Errors have a dedicated visible status and **Reconnect
paused** action; they never replace numeric Score values.

The visible **Speed** selector changes actual simulation/learning cadence,
not the camera or video speed. **20 steps/second** is the default, matching
the original reference's50ms move interval;40steps/second and fast training
are also available. All actors/hazard timers share this simulation clock.

The small head-tracked captions read **YOU**, **wants to kill you**,
**wants to eat your food**, and **Moving obstacle** for the grey patrol.

Keyboard shortcuts remain available with the arena focused:

| Key | Action |
|---|---|
| Space | Pause/resume actual simulation and learning |
| R | Load the retained Best, or initial Ready, and run greedy Test |
| Shift+R | Confirm Reset and automatically resume fresh-weight Train |
| S / L | Save / load the current full session |
| N | New run with the current policy |
| T / E | Train / Test |
| [ / ] | Previous/next difficulty |
| F | Toggle Full Survival and its complete hazards/enemy counts |
| 1 / 2 / 4 / 6 / 8 | Actual simulation steps/second |
| - / + | Slower/faster, up to40steps/second and fast training |
| 0 | Fast batched training |
| C | Reconnect paused after an explicitly reported error |

Camera/render timing remains independent of simulation speed, including
while paused. The **View** selector adds:

- **Tail to head:** a perspective camera near the actual tail that continually
  targets the actual head, including vertical movement and bent bodies. It
  does not point along a stale tail velocity.
- **Food focus:** the selected food is the focal point in front of the camera,
  with YOU beyond it. Position responds to YOU's location; it uses the current
  reward food objective when available, otherwise normal food. This is only
  framing, never an action controller. Missing food is reported explicitly.

Normal and Gold food retain their base colours but are larger, self-lit and
ringed. Real forward food collection and length increase trigger a650ms
YOU/grid light pulse. Loading a saved event, repeated snapshots, rival
collection or changing episodes cannot fake another pulse or learning step.

## Retained Best and continued learning

**Load Best / Ready** uses a retained improved champion when one exists;
otherwise the bundled Ready is the initial Best. It starts greedy **Test**.
Choose **Policy: Train** to continue actual Double DQN learning from those
parameters. This is fine-tuning, not blind arithmetic averaging of two
independently trained parameter sets.

After4096 new training transitions with new optimizer updates, a background
worker compares a frozen candidate against the current Best. **Evaluate &
Keep Best** can request a check sooner after training has begun. Gameplay
continues, and even a paused HUD polls for completion. Promotion stores a
policy; it never replaces the live learner midway through a run.

Every check uses all12 pinned cases: seeds3600008001-3600008004 at levels1,
4 and6. They are reserved from training. Each level must have no increase
in deaths or decrease in mean survival; basic level1 food Score and overall
food Score must not decrease. At least one aggregate Score/survival/death
metric must strictly improve. Equal or regressed policies keep the previous
Best, with the reason visible. These finite checks are safeguards, not proof
of universal optimal play.

The retained policy is `champion.npz` under the current environment save
directory, with prior versions and `champion-evaluations.jsonl` receipts.
**Reset still starts fresh random Train**, but excludes the champion from
its session archive. Restarting the process also retains it. Bundled Ready,
original weights and older user sessions are never overwritten. Invalid
retained files report an error rather than silently reverting to Ready.
Use one authoritative Python process per storage directory.

## Random obstacle appearances

During each round a seeded random obstacle cluster disappears and appears
at a new location. Amber wireframes and **Obstacles changing soon** warn
about the destination for16simulation ticks before it becomes solid.
The next warning starts48–88ticks after a successful change. These warning
durations scale with the selected simulation rate.

Changes wait if they would overlap any snake or food, disconnect free space
or remove the player's last legal exit. Broad axial passages remain open.
Pending warning cells enter the policy's existing next-tick hazard inputs
and food cannot spawn there. RNG/layout/timer state is reconstructed from
the same seed/action journal for exact checkpoint replay.

Current saves are environment-namespaced under
`storage\arena-3d-v3\survival-balanced-v8\`.
Older v6/v7 sessions are preserved rather than silently overwritten.
`SNAKE_STORAGE_DIR` can isolate a preview's mutable sessions. Normal runs
use the local `storage` directory. The classic HUD, optional cameras and
retained Best service are the main application, not a separate experiment.
Local sessions, legacy user model/state files and runtime logs are ignored
by Git; the two bundled pretrained assets remain versioned.
The original `storage\pretrained_3d.npz` and user saves remain unchanged.

## Balanced training

**Visible Score is food points only:** normal food adds1, Gold adds3.
All deductions affect **training Reward**, not that scoreboard:

- Food is **+6 / +18** for normal / Gold; real death is **-30**.
  A Gold-then-fatal dash stays negative.
- Each tick costs **-0.05**, with **-0.08 x local risk** for lingering near
  danger. Risk reads actual enemies, blocked exits and physical hazard
  warnings. Existing loop/ability costs and sustained-escape rewards remain.
- Food value, actual obstacle-aware A* route distance, destination danger
  and the rival's estimated arrival form bounded resource/safety/race
  potentials. Feedback is **0.99 x potentialAfter - potentialBefore**;
  terminal potential is zero and reward-farming proximity loops are tested.
  Safe reachable Gold can win, but unsafe, expired, unreachable or unwinnable
  Gold must not override safer normal food.
- Only actual consumption of a plausibly contestable reward target by the
  rival adds the small **-0.5 x food value** competition penalty.

Search measures routes and controls enemies; **Double DQN alone chooses
the player action**. No planner drives the player and no hazard is removed
from its legal action mask. These are learned tradeoffs, not guaranteed
shortest paths or immunity. A newly reset random policy can still score zero.

The common reward profile is **`survival-objectives-v1`**.
`/api/state` includes `board.trainingReward`, with the actual total,
components, shaping and before/after objectives; that total enters replay.
Evaluation also records Gold, rival food, danger exposure and food per100ticks.
**R** loads retained Best or initial `storage\pretrained_survival.npz`,
requiring current arena/reward provenance in either case. It starts greedy
Test without rewriting the distributed asset or saved user sessions.

### Initial bundled Ready policy audit

The bundled policy's lineage contains **327,680 actual transitions** under
these rewards:262,144 mixed level1-4, then a selected65,536 from an all-six-
difficulty continuation. All12 validation cases were used for selection;
the arbitrary final checkpoint and failed candidates were not substituted.
The selected parameters were frozen before24 additional untouched cases:

| Level | Mean Score, old -> new | Mean survival ticks, old -> new | Deaths /8, old -> new | Gold total, old -> new |
|---|---|---|---|---|
| 1 |7.625 ->8.625|260 ->260|0 ->0|0 ->0|
| 4 |3.875 ->6.625|162.75 ->409.75|6 ->2|0 ->0|
| 6 |1.625 ->1.000|37 ->89.625|8 ->8|0 ->1|

Master Mode is **not solved**: all8 died and mean Score fell, despite longer
survival. Gold remains rare. Food per100ticks is lower on danger levels,
consistent with a safety-first tradeoff, not proof of optimal routes.
The separate12-case validation had food-score regressions at levels4/6.
All rows, seeds, preserved source hashes, actual optimizer counts and those
tradeoffs are embedded in `pretrained_survival.npz`.

The actual training stages are reproducible into separate files:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -B -m backend.train_ready --source storage\pretrained_3d.npz --output .runtime-proof\ready-retrain\warm.npz --report .runtime-proof\ready-retrain\warm.json --steps 262144 --cycles 32 --max-level 4 --seed 17073 --learning-rate 0.0001
.\.venv\Scripts\python.exe -B -m backend.train_ready --source .runtime-proof\ready-retrain\warm.npz --output .runtime-proof\ready-retrain\six.npz --report .runtime-proof\ready-retrain\six.json --steps 131072 --cycles 32 --max-level 6 --seed 17073 --learning-rate 0.0001
```

The selected continuation is `six-step65536.npz`. Compare full fixed
evaluation suites before promoting a new candidate; changed hardware or
numerical libraries can change exact parameters.

## Validation

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
npm.cmd ci --ignore-scripts
npm.cmd run check
npm.cmd run test:reference
node --test tests\camera.mjs tests\presentation.mjs
npm.cmd run test:browser
```

The original read-only reference remains a fixture for the preserved Orbit
formula, grid geometry and outlined-cube functions. Focused tests additionally
cover vertical tail/food framing, real-growth-only pulses and glow cleanup.
Native browser tests exercise the bottom dock, actual rendered neural data,
all visible controls/cameras, food collection, background Best checking,
error/reconnect, checkpoint replay, real20/40step rates and narrow viewports.
Champion unit tests distinguish real evaluation from synthetic gate/storage
fixtures; a rejected candidate is a valid result, never a claimed improvement.
