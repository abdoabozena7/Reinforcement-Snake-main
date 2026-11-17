const clamp = (n, low = 0, high = 1) => Math.min(high, Math.max(low, n));
const ease = (n) => { const t = clamp(n); return t * t * (3 - 2 * t); };
const point = (p) => ({ x: p.x, y: p.y, z: p.z });
const distance = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
const mix = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t });
const same = (a, b) => distance(a, b) < 0.00001;
const body = (points = []) => points.map(point);

export const SIMULATION_RATES = Object.freeze({
    "rate-1": 1, slow: 2, "rate-4": 4, "rate-6": 6, realtime: 8, "rate-12": 12,
    "rate-16": 16, "rate-20": 20, "rate-30": 30, "rate-40": 40, fast: 0,
});

export class SimulationPacer {
    constructor(mode = "rate-4") {
        this.active = false;
        this.inFlight = false;
        this.nextAt = null;
        this.lastStartedAt = null;
        this.setMode(mode, 0);
    }

    get stepsPerSecond() { return SIMULATION_RATES[this.mode] || null; }
    get intervalMs() { return this.stepsPerSecond ? 1000 / this.stepsPerSecond : 0; }

    setMode(mode, now) {
        if (!Object.hasOwn(SIMULATION_RATES, mode)) throw new Error(`Unknown simulation speed: ${mode}`);
        this.mode = mode;
        this.nextAt = this.active ? now + this.intervalMs : null;
    }

    suspend() {
        this.active = false;
        this.nextAt = null;
    }

    resume(now) {
        if (this.active) return;
        this.active = true;
        this.nextAt = now + this.intervalMs;
    }

    waitMs(now) {
        if (!this.active || this.inFlight) return Infinity;
        return Math.max(0, this.nextAt - now);
    }

    tryStart(now) {
        if (this.waitMs(now) > 0) return false;
        this.inFlight = true;
        this.lastStartedAt = now;
        this.nextAt = now + this.intervalMs;
        return true;
    }

    complete(now) {
        this.inFlight = false;
        // Missed deadlines are discarded; a slow request never creates a catch-up queue.
        if (this.active && now >= this.nextAt) this.nextAt = now + this.intervalMs;
    }

    debug() {
        return {
            mode: this.mode, stepsPerSecond: this.stepsPerSecond, intervalMs: this.intervalMs, batched: this.mode === "fast",
            active: this.active, inFlight: this.inFlight, nextAt: this.nextAt, lastStartedAt: this.lastStartedAt,
        };
    }
}

export function episodeKey(board, episode = "") {
    return `${episode}:${board.seed}:${board.level}:${Boolean(board.fullSurvival)}:${board.gridSize}`;
}

export function describeObjectives(board, decision = {}) {
    const food = (board.foods || []).find((item) => item.type === "normal");
    const hasDecision = decision.lastAction != null;
    const source = !hasDecision ? "waiting" : decision.mode === "eval" ? "test"
        : decision.isExploring ? "exploration" : "policy";
    const activity = source === "exploration" ? `Random: ${decision.actionLabel}`
        : source === "test" ? `Test: ${decision.actionLabel}`
        : source === "policy" ? `Policy: ${decision.actionLabel}` : "Awaiting a decision";
    const player = {
        id: "player", role: "player", objective: "Get food and stay alive", source, activity,
        label: source === "exploration" ? "YOU · EXPLORING" : source === "test" ? "YOU · TEST"
            : source === "policy" ? "YOU · POLICY" : "YOU",
        target: food?.position || null, targetId: food?.id || null,
        targetKind: "objective", moveKind: "last-action",
        moveDirection: hasDecision ? board.direction : null,
        gridDistance: food ? distance(board.snake[0], food.position) : null,
    };
    const enemies = (board.enemies || []).map((enemy) => {
        const intent = enemy.intent;
        const targetFood = (board.foods || []).find((item) => item.id === intent?.targetId);
        const foodName = targetFood?.type === "gold" ? "GOLD" : "FOOD";
        const behavior = intent?.behavior;
        const activity = behavior === "seek-food" ? `Racing to ${foodName}`
            : behavior === "intercept" ? "Cutting YOU off"
            : behavior === "chase" ? "Chasing YOU"
            : behavior === "search" ? "Searching for a target"
            : behavior === "yield" ? "Waiting for space"
            : behavior === "stunned" ? "Shield-stunned"
            : behavior === "leave-zone" ? "Leaving the danger zone"
            : behavior === "patrol" ? "Following its patrol"
            : behavior === "detour" ? "Patrol detour"
            : behavior === "waiting" ? `Waiting ${enemy.nextMoveIn} ticks`
            : behavior === "recover" ? "Recovering"
            : intent ? "Waiting to ambush" : "Target not reported";
        const name = enemy.type.toUpperCase();
        return {
            id: enemy.id, role: enemy.type, source: "search-planner", activity,
            objective: enemy.type === "rival" ? "Take food before YOU"
                : enemy.type === "patrol" ? "Guard its patrol route" : "Catch YOU",
            label: behavior === "seek-food" ? `${name} > ${foodName}`
                : behavior === "intercept" ? `${name} · CUT OFF`
                : behavior === "chase" ? `${name} > YOU`
                : behavior === "search" ? `${name} · SEARCH`
                : enemy.type === "patrol" ? "PATROL > ROUTE"
                : enemy.type === "ambush" && ["waiting", "recover"].includes(behavior) ? `AMBUSH · ${behavior.toUpperCase()}` : name,
            target: intent?.target || null, targetId: intent?.targetId || null,
            targetKind: "planner-target", moveKind: "next-planned-step",
            nextPosition: intent?.nextPosition || enemy.nextPosition || null,
            head: enemy.snake[0],
        };
    });
    return { player, enemies, food, linksAreRoutes: false };
}

export function poseOf(board) {
    return {
        player: body(board.snake),
        enemies: (board.enemies || []).map((enemy) => ({
            id: enemy.id, type: enemy.type, points: body(enemy.snake), direction: point(enemy.direction || board.direction),
        })),
        direction: point(board.direction || { x: 0, y: 0, z: -1 }),
        step: board.steps,
    };
}

function pathsFor(previous, next) {
    if (!previous.length || !next.length) return null;
    const shift = Math.round(distance(previous[0], next[0]));
    const trail = [...next.slice(0, shift), ...previous];
    const paths = [];
    for (let i = 0; i < next.length; i += 1) {
        const from = previous[Math.min(i, previous.length - 1)];
        const to = next[i];
        if (distance(from, to) <= 1.00001) {
            paths.push([from, to]);
        } else {
            // Follow the recorded body trail through every dash cell, including bends.
            if (shift < 2 || shift > 3 || !trail[i] || !same(trail[i], to)) return null;
            const path = [from, ...trail.slice(i, Math.min(i + shift, trail.length)).reverse()];
            if (path.some((p, index) => index > 0 && distance(path[index - 1], p) > 1.00001)) return null;
            paths.push(path);
        }
    }
    return paths;
}

function transition(previous, next) {
    const player = pathsFor(previous.player, next.player);
    if (!player) return null;
    const enemies = next.enemies.map((enemy) => {
        const old = previous.enemies.find((item) => item.id === enemy.id && item.type === enemy.type);
        const paths = old && pathsFor(old.points, enemy.points);
        return { ...enemy, paths: paths || enemy.points.map((p) => [p, p]) };
    });
    return { player, enemies, next };
}

function along(path, t) {
    if (path.length === 2) return mix(path[0], path[1], t);
    const lengths = path.slice(1).map((p, index) => distance(path[index], p));
    const total = lengths.reduce((sum, n) => sum + n, 0);
    let remaining = t * total;
    for (let i = 0; i < lengths.length; i += 1) {
        if (remaining <= lengths[i] || i === lengths.length - 1) {
            return mix(path[i], path[i + 1], lengths[i] ? clamp(remaining / lengths[i]) : 1);
        }
        remaining -= lengths[i];
    }
    return point(path[path.length - 1]);
}

function sampleTransition(motion, t) {
    return {
        ...motion.next,
        player: motion.player.map((path) => along(path, t)),
        enemies: motion.enemies.map(({ paths, ...enemy }) => ({ ...enemy, points: paths.map((path) => along(path, t)) })),
    };
}

export function contiguousBoards(previous, next) {
    return Boolean(previous && next && episodeKey(previous) === episodeKey(next)
        && !previous.done && next.steps === previous.steps + 1);
}

export class PoseInterpolator {
    constructor() {
        this.board = null;
        this.pose = null;
        this.motion = null;
        this.elapsed = 0;
        this.duration = 0;
        this.sampled = false;
    }

    setBoard(board, { sampled = false, interval = 120, snap = false } = {}) {
        const duration = Math.max(0.06, Math.min(0.96, interval / 1000 * 0.88));
        const duplicate = this.board && episodeKey(this.board) === episodeKey(board) && this.board.steps === board.steps;
        if (duplicate && !snap && sampled === this.sampled) {
            const progress = this.duration ? this.elapsed / this.duration : 0;
            this.duration = duration;
            this.elapsed = progress * duration;
            this.board = board;
            return;
        }
        const next = poseOf(board);
        const canMove = !snap && !sampled && !this.sampled && contiguousBoards(this.board, board);
        this.motion = canMove ? transition(poseOf(this.board), next) : null;
        this.elapsed = 0;
        this.duration = duration;
        this.pose = this.motion ? sampleTransition(this.motion, 0) : next;
        this.board = board;
        this.sampled = sampled;
    }

    advance(dt, paused = false) {
        if (this.motion && !paused) {
            this.elapsed += Math.max(0, dt);
            const t = clamp(this.elapsed / this.duration);
            this.pose = sampleTransition(this.motion, t);
            if (t === 1) this.motion = null;
        }
        return this.pose;
    }

    snap() {
        if (this.board) this.pose = poseOf(this.board);
        this.motion = null;
    }

    debug() {
        return { interpolating: Boolean(this.motion), progress: this.motion ? clamp(this.elapsed / this.duration) : 1, sampled: this.sampled };
    }
}

export class PoseHistory {
    constructor(limit = 32) {
        this.limit = Math.max(2, limit);
        this.frames = [];
        this.key = null;
    }

    record(board, { sampled = false, episode = "" } = {}) {
        const key = episodeKey(board, episode);
        const last = this.frames[this.frames.length - 1];
        if (sampled) {
            this.clear();
            return;
        }
        if (key !== this.key || (last && board.steps !== last.board.steps && !contiguousBoards(last.board, board))) this.clear();
        this.key = key;
        if (this.frames[this.frames.length - 1]?.board.steps === board.steps) return;
        this.frames.push({ board, pose: poseOf(board) });
        if (this.frames.length > this.limit) this.frames.shift();
    }

    clear() {
        this.frames.length = 0;
        this.key = null;
    }
}

const actorName = (type) => ({ hunter: "Hunter", rival: "Rival", patrol: "Patrol", ambush: "Ambusher", ambusher: "Ambusher" })[type] || "Enemy";

export function describeContact(contact, cause = "") {
    if (contact?.outcome === "blocked") return {
        title: "Shield blocked the hit", detail: "The shield absorbed this contact. Your snake is still alive.", lethal: false,
    };
    const name = actorName(contact?.attacker?.type);
    const kind = contact?.kind;
    if (kind === "enemy-body") return {
        title: `Hit ${name.toLowerCase()} body`, detail: "Your head collided with an occupied body cell. This was not a bite.", lethal: true,
    };
    if (kind === "body-strike") return {
        title: `${name} struck your body`, detail: `Contact on body segment ${(contact.victim?.segmentIndex ?? 0) + 1} ended the round.`, lethal: true,
    };
    if (kind === "head-on" || kind === "crossing") return {
        title: `${name} ${["hunter", "ambush", "ambusher"].includes(contact.attacker?.type) ? "caught you" : "collision"}`,
        detail: kind === "crossing" ? "The two heads crossed on this recorded move." : "The two heads met on this recorded move.",
        lethal: true,
    };
    const labels = {
        wall: ["Hit the arena boundary", "There was no free cell beyond the arena edge."],
        obstacle: ["Hit an obstacle", "Your attempted move met solid geometry."],
        moving_wall: ["Hit a moving wall", "The moving wall occupied the contact cell."],
        maze_wall: ["Hit a maze wall", "Your attempted move met a solid maze barrier."],
        laser: ["Laser contact", "An active gate intersected your snake."],
        zone: ["Outside the safe zone", "The shrinking boundary reached your snake."],
        self: ["Hit your own body", "Your head reached one of your occupied body cells."],
        time_limit: ["Time limit reached", "The round ended on its step limit. No enemy bite occurred."],
        starvation: ["Out of time to find food", "The food timer ended the round. No enemy bite occurred."],
    };
    const [title, detail] = labels[cause] || labels[kind] || ["Round ended", cause ? `Cause: ${cause.replaceAll("_", " ")}.` : "The simulation ended this round."];
    return { title, detail, lethal: !["time_limit", "starvation"].includes(cause) };
}

export function isPredatorAttack(contact) {
    return Boolean(contact && ["head-on", "crossing", "body-strike"].includes(contact.kind)
        && ["hunter", "ambush", "ambusher"].includes(contact.attacker?.type)
        && contact.attacker.from && contact.attacker.to && !same(contact.attacker.from, contact.attacker.to));
}

function contactPose(base, contact, progress) {
    if (!contact) return base;
    const t = clamp(contact.timeFraction ?? 1) * clamp(progress);
    const actorPoints = (actor) => {
        if (!actor?.before?.length || !actor.attempted?.length) return null;
        return actor.before.map((from, index) => mix(from, actor.attempted[Math.min(index, actor.attempted.length - 1)], t));
    };
    const player = contact.victim?.id === "player" ? actorPoints(contact.victim) : null;
    const attacker = actorPoints(contact.attacker);
    const heading = (actor, fallback) => actor?.from && actor?.to && !same(actor.from, actor.to) ? {
        x: actor.to.x - actor.from.x, y: actor.to.y - actor.from.y, z: actor.to.z - actor.from.z,
    } : fallback;
    return {
        ...base,
        player: player || body(base.player),
        direction: heading(contact.victim, base.direction),
        enemies: base.enemies.map((enemy) => enemy.id === contact.attacker?.id && attacker
            ? { ...enemy, points: attacker, direction: heading(contact.attacker, enemy.direction) } : { ...enemy, points: body(enemy.points) }),
    };
}

function poseTravel(a, b) {
    let length = 0;
    for (let i = 0; i < Math.min(a.player.length, b.player.length); i += 1) length = Math.max(length, distance(a.player[i], b.player[i]));
    for (const enemy of b.enemies) {
        const old = a.enemies.find((item) => item.id === enemy.id);
        if (old?.points[0] && enemy.points[0]) length = Math.max(length, distance(old.points[0], enemy.points[0]));
    }
    return length;
}

function approachPose(run, progress) {
    if (run.prefix && progress < run.prefixShare) return sampleTransition(run.prefix, ease(progress / run.prefixShare));
    return contactPose(run.base, run.contact, ease((progress - run.prefixShare) / (1 - run.prefixShare)));
}

function reactionPose(base, contact, recoil, defeat, reducedMotion, gridSize) {
    const pose = { ...base, player: body(base.player), enemies: base.enemies.map((enemy) => ({ ...enemy, points: body(enemy.points) })) };
    const attackerDriven = ["head-on", "crossing", "body-strike"].includes(contact?.kind)
        && contact.attacker?.from && contact.attacker?.to && !same(contact.attacker.from, contact.attacker.to);
    let normal = attackerDriven && contact?.position ? {
        x: contact.position.x - contact.attacker.from.x,
        y: contact.position.y - contact.attacker.from.y,
        z: contact.position.z - contact.attacker.from.z,
    } : contact?.victim?.from && contact?.victim?.to ? {
        x: contact.victim.from.x - contact.victim.to.x,
        y: contact.victim.from.y - contact.victim.to.y,
        z: contact.victim.from.z - contact.victim.to.z,
    } : { x: 0, y: 0, z: 1 };
    const length = Math.hypot(normal.x, normal.y, normal.z) || 1;
    normal = { x: normal.x / length, y: normal.y / length, z: normal.z / length };
    const sideLength = Math.hypot(normal.x, normal.z);
    const side = sideLength > 0.1 ? { x: -normal.z / sideLength, z: normal.x / sideLength }
        : { x: 1, z: 0 };
    const hit = contact?.victim?.segmentIndex ?? 0;
    const scale = reducedMotion ? 0.2 : 1;
    pose.player.forEach((p, index) => {
        const offset = index - hit;
        const wave = 0.6 + 0.4 * Math.exp(-Math.abs(offset) * 0.22);
        const kick = recoil * 1.25 * wave * scale;
        const curl = Math.sin(offset * 0.95) * defeat * 0.52 * scale;
        p.x += normal.x * kick + side.x * curl;
        p.y += normal.y * kick - defeat * (1.25 + 0.5 * (1 - Math.exp(-Math.abs(offset) * 0.8))) * scale
            + Math.sin(offset * 1.1) * recoil * 0.16 * scale;
        p.z += normal.z * kick + side.z * curl;
        if (gridSize) {
            p.x = clamp(p.x, -0.02, gridSize - 0.98);
            p.y = clamp(p.y, -0.02, gridSize - 0.98);
            p.z = clamp(p.z, -0.02, gridSize - 0.98);
        }
    });
    return pose;
}

export class ContactPresentation {
    constructor({ onFinish = () => {}, historyLimit = 32, reducedMotion = false } = {}) {
        this.history = new PoseHistory(historyLimit);
        this.onFinish = onFinish;
        this.reducedMotion = reducedMotion;
        this.serial = 0;
        this.run = null;
        this.lastTerminal = null;
        this.lastPhase = "idle";
        this.cancelReason = null;
        this.lastFrame = null;
    }

    get active() { return Boolean(this.run); }
    get id() { return this.run?.id ?? null; }

    observe(board, { sampled = false, episode = "", waitForLock = true } = {}) {
        this.history.record(board, { sampled, episode });
        if (sampled || !board.done || this.run) return false;
        const terminal = `${episodeKey(board, episode)}:${board.steps}:${board.contact?.id || board.lastDeath || "end"}`;
        if (this.lastTerminal === terminal) return false;
        this.lastTerminal = terminal;
        const contact = board.contact?.outcome === "death" ? board.contact : null;
        const description = describeContact(contact, board.lastDeath);
        const frames = this.history.frames.slice(-10);
        const base = poseOf(board);
        const before = contactPose(base, contact, 0);
        const previous = frames[frames.length - 2];
        const prefixLength = previous && contiguousBoards(previous.board, board) ? poseTravel(previous.pose, before) : 0;
        const prefix = prefixLength > 0.00001 ? transition(previous.pose, before) : null;
        const contactLength = poseTravel(before, contactPose(base, contact, 1));
        this.run = {
            id: ++this.serial, board, contact, description, frames, elapsed: 0, ready: !waitForLock,
            base, terminal, prefix,
            prefixShare: prefix ? Math.min(0.8, prefixLength / (prefixLength + Math.max(0.1, contactLength))) : 0,
        };
        this.lastPhase = waitForLock ? "locking" : contact ? "approach" : "summary";
        this.cancelReason = null;
        this.lastFrame = null;
        return true;
    }

    unlock(id) {
        if (this.run?.id === id) this.run.ready = true;
    }

    setReducedMotion(value) {
        const reduced = Boolean(value);
        if (reduced === this.reducedMotion) return;
        let current = null;
        if (this.run) {
            let time = this.run.elapsed;
            for (const stage of this.stages()) {
                if (time < stage.duration) { current = { phase: stage.phase, progress: time / stage.duration }; break; }
                time -= stage.duration;
            }
        }
        this.reducedMotion = reduced;
        if (reduced && this.lastFrame && ["impact", "defeat", "aftermath", "rewind"].includes(current?.phase)) {
            this.run.motionHold = this.lastFrame.pose;
        }
        if (current) {
            let elapsed = 0;
            for (const stage of this.stages()) {
                if (stage.phase === current.phase) { elapsed += stage.duration * current.progress; break; }
                elapsed += stage.duration;
            }
            this.run.elapsed = elapsed;
        }
    }

    stages() {
        const reduced = this.reducedMotion;
        const contact = this.run?.contact;
        return [
            ...(contact ? [{ phase: "approach", duration: reduced ? 0.5 : 1 }] : []),
            ...(contact ? [
                { phase: "impact", duration: reduced ? 0.4 : 0.76 },
                { phase: "defeat", duration: reduced ? 0.6 : 1.05 },
                { phase: "aftermath", duration: reduced ? 1.1 : 1.45 },
                { phase: "rewind", duration: reduced ? 0.25 : 0.45 },
            ] : [{ phase: "summary", duration: reduced ? 0.5 : 0.8 }]),
            { phase: "recap", duration: reduced ? 1.1 : Math.max(1.15, (this.run?.frames.length || 1) * 0.16 + 0.4) },
            { phase: "recap-contact", duration: contact ? reduced ? 0.6 : 0.95 : reduced ? 0.35 : 0.55 },
        ];
    }

    advance(dt) {
        const run = this.run;
        if (!run) return null;
        if (run.ready) run.elapsed += Math.max(0, dt);
        let time = run.elapsed;
        let stage = null;
        for (const candidate of this.stages()) {
            if (time < candidate.duration) { stage = candidate; break; }
            time -= candidate.duration;
        }
        if (!stage) {
            this.finish("complete", run.id);
            return null;
        }
        const phase = run.ready ? stage.phase : "locking";
        const progress = run.ready ? clamp(time / stage.duration) : 0;
        const replay = ["rewind", "recap", "recap-contact"].includes(phase);
        const predator = isPredatorAttack(run.contact);
        const travel = clamp((progress - 0.42) / 0.58);
        let board = run.board;
        let pose = phase === "approach" || phase === "locking" ? approachPose(run, travel * travel) : contactPose(run.base, run.contact, 1);
        let impact = 0, recoil = 0, defeat = 0, death = 0, jawOpen = 0;
        let hitStop = false;
        if (phase === "approach" && predator) jawOpen = ease(clamp(progress / 0.38));
        if (phase === "impact") {
            const stop = this.reducedMotion ? 0.12 : 0.18;
            hitStop = time <= stop;
            const released = clamp((time - stop) / (stage.duration - stop));
            recoil = hitStop ? 0 : released < 0.66 ? Math.sin(released / 0.66 * Math.PI / 2)
                : 1 - 0.22 * ease((released - 0.66) / 0.34);
            death = 0.5 * ease(released);
            impact = hitStop ? 1 : 0.85 * (1 - released);
            if (predator) jawOpen = 0.06 + 0.94 * (1 - ease(clamp(time / 0.11)));
        }
        if (phase === "defeat") {
            defeat = ease(progress);
            recoil = 0.78 * (1 - defeat) + 0.2 * defeat;
            death = 0.5 + 0.5 * ease(clamp(progress * 2));
        }
        if (["aftermath", "rewind"].includes(phase)) { defeat = 1; recoil = 0.2; death = 1; }
        if (phase === "summary" && run.prefix) pose = sampleTransition(run.prefix, ease(clamp(run.elapsed / 0.12)));
        if (phase === "recap") {
            run.motionHold = null;
            const recordedShare = run.contact ? 0.76 : 1;
            const replayT = clamp(progress / recordedShare);
            const value = replayT * Math.max(0, run.frames.length - 1);
            const index = Math.min(Math.floor(value), Math.max(0, run.frames.length - 1));
            const from = run.frames[index];
            const to = run.frames[Math.min(index + 1, run.frames.length - 1)];
            if (from && to) {
                board = from.board;
                const motion = contiguousBoards(from.board, to.board) && transition(from.pose, to.pose);
                pose = motion ? sampleTransition(motion, value - index) : from.pose;
            }
            if (run.contact && progress >= recordedShare) {
                board = run.board;
                pose = contactPose(run.base, run.contact, ease((progress - recordedShare) / (1 - recordedShare)));
            }
            impact = 0;
            defeat = 0;
            death = 0;
            if (predator && progress >= recordedShare) jawOpen = 1;
        }
        if (phase === "recap-contact" && run.contact) {
            const stop = this.reducedMotion ? 0.1 : 0.14;
            hitStop = time <= stop;
            const released = clamp((time - stop) / (stage.duration - stop));
            defeat = ease(clamp(released / 0.7));
            recoil = (1 - defeat) * Math.sin(clamp(released / 0.65) * Math.PI / 2) + 0.2 * defeat;
            death = ease(clamp(released / 0.45));
            impact = hitStop ? 1 : 0.5 * (1 - released);
            if (predator) jawOpen = 0.06 + 0.94 * (1 - ease(clamp(time / 0.1)));
        }
        const truthPose = pose;
        if (recoil || defeat) pose = reactionPose(pose, run.contact, recoil, defeat, this.reducedMotion, run.board.gridSize);
        if (run.motionHold && ["impact", "defeat", "aftermath", "rewind"].includes(phase)) pose = run.motionHold;
        const eliminated = run.description.lethal && death >= 0.75;
        const labels = {
            locking: "PREPARING CONTACT VIEW",
            approach: predator ? progress < 0.42 ? "RECORDED ATTACK · WIND-UP" : "RECORDED ATTACK · LUNGE" : "RECORDED COLLISION",
            impact: hitStop ? "CONTACT · HIT-STOP" : "IMPACT · BODY REACTION",
            defeat: "ELIMINATED",
            aftermath: "ELIMINATED · ROUND OVER",
            rewind: "REWIND · RECORDED POSES",
            recap: "RECAP · RECORDED POSES",
            "recap-contact": "RECAP · RECORDED CONTACT & OUTCOME",
            summary: "ROUND ENDED",
        };
        this.lastPhase = phase;
        const frame = {
            active: true, id: run.id, phase, progress, contact: run.contact, board, pose, truthPose,
            replay, impact, recoil, defeat, death, eliminated, hitStop, jawOpen,
            attack: predator && (phase === "approach" || phase === "impact" || phase === "recap-contact" || (phase === "recap" && progress > 0.76)),
            ...run.description,
            label: labels[phase],
        };
        this.lastFrame = frame;
        return frame;
    }

    finish(reason = "skip", id = this.run?.id) {
        if (!this.run || this.run.id !== id) return false;
        const finished = { id, reason, terminal: this.run.terminal, ...this.run.description };
        this.run = null;
        this.lastFrame = null;
        this.lastPhase = reason === "complete" ? "complete" : "skipped";
        this.onFinish(finished);
        return true;
    }

    cancel(reason = "cancelled", { clearHistory = false } = {}) {
        this.serial += 1;
        this.run = null;
        this.lastFrame = null;
        this.lastPhase = "idle";
        this.cancelReason = reason;
        if (clearHistory) {
            this.history.clear();
            this.lastTerminal = null;
        }
    }

    debug() {
        return {
            active: this.active, id: this.id, phase: this.lastPhase,
            ready: Boolean(this.run?.ready), elapsed: this.run?.elapsed ?? 0,
            contactId: this.run?.contact?.id ?? null, kind: this.run?.contact?.kind ?? null,
            message: this.run?.description.title ?? null,
            bufferLength: this.history.frames.length, bufferSize: this.history.frames.length,
            bufferedSteps: this.history.frames.map((frame) => frame.board.steps),
            episodeKey: this.history.key, reducedMotion: this.reducedMotion, cancelReason: this.cancelReason,
        };
    }
}

export class PlaybackLifecycle {
    constructor() {
        this.generation = 0;
        this.desiredPaused = false;
        this.terminal = null;
        this.tail = Promise.resolve();
        this.running = 0;
        this.maxConcurrency = 0;
    }

    invalidate() {
        this.generation += 1;
        this.terminal = null;
        return this.generation;
    }

    enqueue(work, generation = this.generation) {
        const current = () => generation === this.generation;
        const result = this.tail.then(async () => {
            if (!current()) return null;
            this.running += 1;
            this.maxConcurrency = Math.max(this.maxConcurrency, this.running);
            try { return await work({ current, generation }); }
            finally { this.running -= 1; }
        });
        this.tail = result.catch(() => {});
        return result;
    }

    lock(id) {
        this.terminal = { id, generation: this.generation };
        return this.terminal;
    }

    matches(token) {
        return Boolean(token && this.terminal === token && token.generation === this.generation);
    }

    release(token) {
        if (!this.matches(token)) return null;
        this.terminal = null;
        return { paused: this.desiredPaused };
    }
}
