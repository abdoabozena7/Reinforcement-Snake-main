import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "../static/vendor/three.module.js";
import { BlockSnakeCharacter, SnakeCharacter } from "../static/character.js";
import {
    ContactPresentation, PlaybackLifecycle, PoseHistory, PoseInterpolator, SimulationPacer,
    describeContact, describeObjectives, isPredatorAttack,
} from "../static/presentation.js";

const p = (x, y = 11, z = 11) => ({ x, y, z });
const copy = (value) => structuredClone(value);
const near = (a, b, epsilon = 0.00001) => assert(Math.abs(a - b) < epsilon, `${a} should equal ${b}`);
const board = (step = 0, x = 9) => ({
    seed: 73, gridSize: 32, level: 4, fullSurvival: false, steps: step,
    snake: [p(x), p(x - 1), p(x - 2)], direction: p(1, 0, 0),
    enemies: [{ id: "hunter-1", type: "hunter", direction: p(-1, 0, 0), snake: [p(12), p(13), p(14)], alert: true }],
    foods: [{ position: p(16), type: "normal" }], obstacles: [], movingWalls: [], gates: [], events: [],
    abilities: { dash: { active: false }, shield: { active: false } }, zone: { inset: 0 },
    done: false, lastDeath: "", threat: 0.8, emotion: "terrified", lookTarget: p(12), contact: null,
});
const crossing = () => ({
    schemaVersion: 1, id: "73:3:1:death", step: 3, substep: 1, kind: "crossing", cause: "hunter", outcome: "death",
    position: p(11.5), timeFraction: 0.5, sourceId: "hunter-1",
    attacker: { id: "hunter-1", type: "hunter", from: p(12), to: p(11), before: [p(12), p(13), p(14)], attempted: [p(11), p(12), p(13)] },
    victim: { id: "player", type: "player", from: p(11), to: p(12), before: [p(11), p(10), p(9)], attempted: [p(12), p(11), p(10)], segmentIndex: 0 },
});
function terminal() {
    return { ...board(3, 11), done: true, lastDeath: "hunter", contact: crossing() };
}
function prepared(options = {}) {
    const controller = new ContactPresentation(options);
    for (let step = 0; step < 3; step += 1) controller.observe(board(step, 9 + step), { episode: 1 });
    controller.observe(terminal(), { episode: 1, waitForLock: false });
    return controller;
}

function atPhase(controller, phase, progress = 0.5) {
    let target = 0;
    for (const stage of controller.stages()) {
        if (stage.phase === phase) {
            target += stage.duration * progress + 0.000001;
            return controller.advance(Math.max(0, target - controller.debug().elapsed));
        }
        target += stage.duration;
    }
    assert.fail(`Missing required phase: ${phase}`);
}

test("normal and vertical poses interpolate once and finish on the actual cells", () => {
    const motion = new PoseInterpolator();
    const a = board();
    a.direction = p(0, 1, 0);
    const b = copy(a);
    b.steps += 1;
    b.snake = [p(9, 12), p(9, 11), p(8, 11)];
    motion.setBoard(a);
    motion.setBoard(b, { interval: 120 });
    const halfway = motion.advance(0.0528);
    near(halfway.player[0].y, 11.5);
    near(halfway.player[0].x, 9);
    near(halfway.player[1].x, 8.5);
    motion.advance(1);
    assert.deepEqual(motion.pose.player, b.snake);
    assert.deepEqual(a.snake, [p(9), p(8), p(7)]);
    motion.setBoard(copy(b), { paused: true });
    assert.equal(motion.debug().interpolating, false, "A control reread must not restart movement");
});

test("three-cell dash follows real intermediate cells instead of cutting a body corner", () => {
    const a = board();
    a.snake = [p(5, 5, 5), p(5, 5, 6), p(5, 5, 7)];
    const b = { ...copy(a), steps: 1, snake: [p(8, 5, 5), p(7, 5, 5), p(6, 5, 5)] };
    const motion = new PoseInterpolator();
    motion.setBoard(a);
    motion.setBoard(b, { interval: 120 });
    const firstCell = motion.advance(0.1056 / 3);
    assert.deepEqual(firstCell.player, [p(6, 5, 5), p(5, 5, 5), p(5, 5, 6)]);
    const secondCell = motion.advance(0.1056 / 3);
    assert.deepEqual(secondCell.player, [p(7, 5, 5), p(6, 5, 5), p(5, 5, 5)]);
    motion.advance(1);
    assert.deepEqual(motion.pose.player, b.snake);
});

test("batches, gaps, resets and unrelated seeds never manufacture a journey", () => {
    const motion = new PoseInterpolator();
    motion.setBoard(board());
    const batched = { ...board(150, 25), seed: 1001 };
    motion.setBoard(batched, { sampled: true });
    assert.deepEqual(motion.pose.player, batched.snake);
    assert.equal(motion.debug().interpolating, false);
    const normal = { ...board(151, 24), seed: 1001 };
    motion.setBoard(normal);
    assert.deepEqual(motion.pose.player, normal.snake, "Leaving fast mode must also snap");
    const gap = { ...board(154, 26), seed: 1001 };
    motion.setBoard(gap);
    assert.deepEqual(motion.pose.player, gap.snake);
    const repeatedSeed = { ...board(0, 4), seed: 1001 };
    motion.setBoard(repeatedSeed, { snap: true });
    assert.deepEqual(motion.pose.player, repeatedSeed.snake);
    const impossible = { ...board(1, 19), seed: 1001 };
    motion.setBoard(impossible);
    assert.equal(motion.debug().interpolating, false);
});

test("history is bounded, deduplicated and scoped to the real episode", () => {
    const history = new PoseHistory(4);
    for (let i = 0; i < 10; i += 1) history.record(board(i, 9 + i), { episode: 1 });
    assert.deepEqual(history.frames.map((frame) => frame.board.steps), [6, 7, 8, 9]);
    history.record(board(9, 18), { episode: 1 });
    assert.equal(history.frames.length, 4);
    history.record(board(0), { episode: 2 });
    assert.equal(history.frames.length, 1, "The same seed in a new episode is not the same trajectory");
    history.record(board(5), { episode: 2 });
    assert.equal(history.frames.length, 1, "A missing tick invalidates the earlier replay");
    history.record(board(6), { sampled: true, episode: 2 });
    assert.equal(history.frames.length, 0);
});

test("true crossing waits for the server lock, reaches contact and moves the whole victim body", () => {
    const finished = [];
    const controller = new ContactPresentation({ onFinish: (result) => finished.push(result) });
    controller.observe(board(2, 11), { episode: 1 });
    const dead = terminal();
    const untouched = JSON.stringify(dead);
    assert(controller.observe(dead, { episode: 1 }));
    const id = controller.id;
    const locked = controller.advance(20);
    assert.equal(locked.phase, "locking");
    assert.equal(finished.length, 0);
    assert.deepEqual(locked.pose.player, dead.contact.victim.before);
    controller.unlock(id);
    const windup = atPhase(controller, "approach", 0.3);
    assert.deepEqual(windup.pose.player, dead.contact.victim.before, "The camera gets a held preparation beat");
    assert(windup.jawOpen > 0.8, "Actual predator attack visibly opens before its recorded lunge");
    const approach = atPhase(controller, "approach", 0.8);
    assert.equal(approach.phase, "approach");
    assert(approach.truthPose.player[0].x > 11 && approach.truthPose.player[0].x < 11.5);
    assert(approach.truthPose.enemies[0].points[0].x > 11.5 && approach.truthPose.enemies[0].points[0].x < 12);
    assert.equal(approach.attack, true);
    const stopped = atPhase(controller, "impact", 0.2);
    assert(stopped.hitStop);
    assert(stopped.jawOpen < 0.15, "The jaw clamps at actual contact instead of hanging open");
    assert.deepEqual(stopped.pose, stopped.truthPose);
    const impact = atPhase(controller, "impact", 0.65);
    assert.equal(impact.phase, "impact");
    near(impact.truthPose.player[0].x, 11.5);
    near(impact.truthPose.enemies[0].points[0].x, 11.5);
    assert(impact.pose.player.every((point, index) => Math.hypot(
        point.x - impact.truthPose.player[index].x, point.y - impact.truthPose.player[index].y,
        point.z - impact.truthPose.player[index].z,
    ) > 0.75), "The recoil must read as whole-body impact, not the rejected sub-cell twitch");
    assert.equal(JSON.stringify(dead), untouched, "Rendered recoil must never overwrite committed Python positions");
    const phases = new Set([locked.phase, approach.phase, impact.phase]);
    let replayed = false;
    for (let i = 0; i < 250; i += 1) {
        const frame = controller.advance(0.04);
        if (!frame) continue;
        phases.add(frame.phase);
        if (frame.replay) {
            replayed = true;
            assert.match(frame.label, /(?:RECAP|REWIND).*RECORDED/);
        }
    }
    assert(replayed);
    for (const phase of ["defeat", "aftermath", "rewind", "recap", "recap-contact"]) assert(phases.has(phase));
    assert.equal(finished.length, 1);
    assert.equal(controller.observe(dead, { episode: 1 }), false, "Rereading terminal state must not repeat its lifecycle");
    controller.finish();
    assert.equal(finished.length, 1);
});

test("a terminal dash first plays its committed cells before the recorded failed substep", () => {
    const before = board(0, 5);
    const after = { ...board(1, 6), done: true, lastDeath: "obstacle" };
    after.contact = {
        ...crossing(), kind: "obstacle", cause: "obstacle", substep: 2, position: p(6.5), attacker: null, sourceId: null,
        victim: { id: "player", type: "player", from: p(6), to: p(7), before: [p(6), p(5), p(4)], attempted: [p(7), p(6), p(5)], segmentIndex: 0 },
    };
    const controller = new ContactPresentation();
    controller.observe(before);
    controller.observe(after, { waitForLock: false });
    const start = controller.advance(0);
    assert.deepEqual(start.pose.player, before.snake, "Do not jump directly to the second dash cell");
    const middle = atPhase(controller, "approach", 0.8);
    assert(middle.pose.player[0].x > 5 && middle.pose.player[0].x < 6);
    const contact = atPhase(controller, "impact", 0.1);
    near(contact.truthPose.player[0].x, 6.5);
    assert.equal(contact.attack, false);
});

test("a real body strike aims along its recorded intent at the contacted segment, not the victim head", () => {
    const contact = crossing();
    contact.kind = "body-strike";
    contact.timeFraction = 1;
    contact.position = p(11);
    contact.victim.segmentIndex = 1;
    contact.attacker.from = p(11, 10);
    contact.attacker.to = p(11, 11);
    contact.attacker.before = [p(11, 10), p(12, 10), p(13, 10)];
    contact.attacker.attempted = [p(11, 11), p(11, 10), p(12, 10)];
    const before = board(2, 11);
    before.enemies[0].snake = copy(contact.attacker.before);
    before.enemies[0].direction = p(0, 0, -1);
    const after = { ...copy(before), steps: 3, done: true, lastDeath: "hunter", contact };
    const controller = new ContactPresentation();
    controller.observe(before);
    controller.observe(after, { waitForLock: false });
    const frame = atPhase(controller, "impact", 0.1);
    assert.deepEqual(frame.truthPose.player[1], contact.position);
    assert.notDeepEqual(frame.truthPose.player[0], contact.position);
    assert.deepEqual(frame.truthPose.enemies[0].direction, p(0, 1, 0));
    assert.equal(frame.contact.victim.segmentIndex, 1);
    assert(frame.attack);
});

test("recap contains only the recorded contiguous approach and never advances a simulation", () => {
    const controller = prepared();
    const recorded = controller.history.frames.map((frame) => copy(frame.pose));
    const samples = [];
    for (let i = 0; i < 400; i += 1) {
        const frame = controller.advance(0.025);
        if (frame?.phase === "recap") samples.push(frame);
    }
    assert(samples.length > 10);
    assert(samples.some((frame) => frame.pose.player[0].x < 10));
    for (const frame of samples) {
        assert(frame.pose.player[0].x >= 9 && frame.pose.player[0].x <= 11.5);
        assert.equal(frame.pose.player[0].y, 11);
        assert.equal(frame.pose.player[0].z, 11);
    }
    assert.deepEqual(controller.history.frames.map((frame) => frame.pose), recorded);
    assert.equal(controller.history.frames.at(-1).board.steps, 3);
});

test("an unmistakable eliminated aftermath is held before the labelled rewind can show living poses", () => {
    const controller = prepared();
    const aftermath = atPhase(controller, "aftermath", 0.01);
    assert(aftermath.eliminated && aftermath.death === 1);
    assert.equal(aftermath.replay, false);
    assert.match(aftermath.label, /ELIMINATED/);
    const still = copy(aftermath.pose);
    const held = controller.advance(1.05);
    assert.equal(held.phase, "aftermath", "The end state needs a readable held beat, not an instant rewind");
    assert.deepEqual(held.pose, still);
    assert(held.pose.player.every((point, index) => held.truthPose.player[index].y - point.y > 1),
        "The defeated body must slump as a whole, not remain a live-looking neighbour");
    const rewind = atPhase(controller, "rewind", 0.2);
    assert(rewind.eliminated && rewind.replay);
    assert.deepEqual(rewind.pose, still);
    assert.match(rewind.label, /REWIND.*RECORDED/);
    const recordedLife = atPhase(controller, "recap", 0.2);
    assert.equal(recordedLife.death, 0);
    assert.equal(recordedLife.eliminated, false);
    assert.match(recordedLife.label, /RECAP.*RECORDED/);
});

test("live reduced-motion changes preserve the aftermath phase and do not undo its dead pose", () => {
    const controller = prepared();
    const before = atPhase(controller, "aftermath", 0.35);
    controller.setReducedMotion(true);
    const after = controller.advance(0);
    assert.equal(after.phase, "aftermath");
    near(before.progress, after.progress);
    assert.deepEqual(after.pose, before.pose);
    assert(after.eliminated);
    controller.setReducedMotion(false);
    const restored = controller.advance(0);
    assert.equal(restored.phase, "aftermath");
    assert.deepEqual(restored.pose, before.pose);
});

test("body, wall, shield and timeout causes cannot turn into fabricated predator bites", () => {
    for (const kind of ["enemy-body", "wall", "obstacle", "laser", "zone", "self"]) {
        const contact = { ...crossing(), kind, cause: kind };
        assert.equal(isPredatorAttack(contact), false);
        const controller = new ContactPresentation();
        controller.observe({ ...terminal(), lastDeath: kind, contact }, { waitForLock: false });
        for (let i = 0; i < 80; i += 1) {
            const frame = controller.advance(0.1);
            if (frame) { assert.equal(frame.attack, false); assert.equal(frame.jawOpen, 0); }
        }
    }
    assert.match(describeContact({ ...crossing(), kind: "enemy-body", attacker: { type: "rival" } }).title, /rival body/);
    assert.match(describeContact({ ...crossing(), kind: "body-strike", victim: { segmentIndex: 2 } }).detail, /segment 3/);
    assert.equal(describeContact({ ...crossing(), outcome: "blocked" }).lethal, false);
    const controller = new ContactPresentation();
    const shield = { ...board(1), contact: { ...crossing(), outcome: "blocked" } };
    assert.equal(controller.observe(shield), false, "Shield blocks are nonterminal, not a defeat timeline");
    const timeout = { ...board(2), done: true, lastDeath: "time_limit", contact: shield.contact };
    assert(controller.observe(timeout, { waitForLock: false }));
    for (let i = 0; i < 90; i += 1) {
        const frame = controller.advance(0.03);
        if (!frame) continue;
        assert.equal(frame.contact, null, "A stale shield event cannot become the timeout's contact");
        assert.equal(frame.attack, false);
        assert.equal(frame.impact, 0);
        assert.equal(frame.death, 0);
        assert.deepEqual(frame.pose.player, frame.truthPose.player);
        assert.match(frame.title, /Time limit/);
    }
});

test("skip is exact-once, cancellation invalidates callbacks and fast mode bypasses presentation", () => {
    const finished = [];
    const controller = prepared({ onFinish: (result) => finished.push(result) });
    const old = controller.id;
    controller.cancel("reset", { clearHistory: true });
    controller.advance(99);
    assert.equal(finished.length, 0);
    assert.equal(controller.history.frames.length, 0);
    const another = { ...terminal(), seed: 101 };
    controller.observe(another, { waitForLock: false, episode: 2 });
    assert.equal(controller.finish("stale", old), false);
    assert(controller.finish("skip"));
    assert.equal(controller.finish("skip"), false);
    assert.equal(finished.length, 1);
    controller.observe({ ...terminal(), seed: 102 }, { sampled: true, episode: 3 });
    assert.equal(controller.active, false);
    assert.equal(controller.history.frames.length, 0);
});

test("reduced motion retains the cause and labelled recap with much smaller body displacement", () => {
    const normal = prepared();
    const reduced = prepared({ reducedMotion: true });
    const a = atPhase(normal, "impact", 0.65);
    const b = atPhase(reduced, "impact", 0.65);
    const movement = (frame) => frame.pose.player.reduce((sum, point, index) => sum + Math.hypot(
        point.x - frame.truthPose.player[index].x, point.y - frame.truthPose.player[index].y,
        point.z - frame.truthPose.player[index].z,
    ), 0);
    assert(movement(b) < movement(a) * 0.25);
    assert.equal(a.title, b.title);
    let labelled = false;
    for (let i = 0; i < 180; i += 1) if (reduced.advance(0.03)?.label.includes("RECAP")) labelled = true;
    assert(labelled);
});

test("serial lifecycle prevents a cancelled terminal callback from resuming a fresh reset", async () => {
    const lifecycle = new PlaybackLifecycle();
    let release;
    let began;
    const started = new Promise((resolve) => { began = resolve; });
    const block = new Promise((resolve) => { release = resolve; });
    const calls = [];
    const tick = lifecycle.enqueue(async ({ current }) => {
        began();
        await block;
        if (current()) calls.push("old-tick-state");
    });
    await started;
    const token = lifecycle.lock("old-contact");
    const staleResume = lifecycle.enqueue(async () => { if (lifecycle.matches(token)) calls.push("stale-resume"); });
    lifecycle.invalidate();
    lifecycle.desiredPaused = false;
    const reset = lifecycle.enqueue(async () => { calls.push("fresh-train-reset"); });
    release();
    await Promise.all([tick, staleResume, reset]);
    assert.deepEqual(calls, ["fresh-train-reset"]);
    assert.equal(lifecycle.maxConcurrency, 1);
    assert.equal(lifecycle.release(token), null);
    const next = lifecycle.lock("next-contact");
    lifecycle.desiredPaused = true;
    assert.deepEqual(lifecycle.release(next), { paused: true }, "Restore the user's latest preference, never a temporary forced pause");
});

test("character uses all three real axes, exact supplied render poses and a cell-sized head", () => {
    const character = new SnakeCharacter("player");
    try {
        const state = board();
        state.direction = p(0, 1, 0);
        state.snake = [p(16, 20, 16), p(16, 19, 16), p(16, 18, 16)];
        character.setState(state, state, true);
        assert.deepEqual(character.headPosition.toArray(), [0.5, 4.5, 0.5]);
        near(character.forward.y, 1);
        assert(character._bodyGeometry.attributes.position.array.some((value, i) => i % 3 === 1 && value > 3));
        character.setRenderPose([p(0, 2, 0), p(0, 1, 0), p(0, 0, 0)], { direction: p(0, 1, 0), snap: true });
        character.update(1 / 60, 0.5);
        assert.deepEqual(character.headPosition.toArray(), [0, 2, 0]);
        character.setRenderPose([p(0, 0, 0), p(0, 0, -1), p(0, 0, -2)], { direction: p(0, 0, 1), snap: true });
        character.setPresentation({ death: 0, reducedMotion: true });
        character.update(0, 0.5);
        const bounds = new THREE.Box3();
        character.head.updateMatrixWorld(true);
        character.head.traverseVisible((mesh) => {
            if (mesh.isMesh) bounds.union(new THREE.Box3().setFromObject(mesh));
        });
        const extent = bounds.getSize(new THREE.Vector3());
        assert(extent.x < 1 && extent.y < 1 && extent.z < 1, `Physical head must fit a cell: ${extent.toArray()}`);
    } finally { character.dispose(); }
});

        test("all snake roles use separate grid-aligned cubes through real vertical and corner poses", () => {
            const points = [p(4, 7, 2), p(4, 6, 2), p(3, 6, 2), p(3, 6, 1)];
            for (const kind of ["player", "hunter", "rival", "patrol", "ambush"]) {
                const character = new BlockSnakeCharacter(kind);
                try {
                    character.setState({ snake: points, direction: p(0, 1, 0) }, board());
                    character.setRenderPose(points, { direction: p(0, 1, 0) });
                    character.setPresentation({ death: 0 });
                    character.update();
                    assert.deepEqual(character.headPosition.toArray(), [4, 7, 2]);
                    near(character.forward.y, 1);
                    const geometry = character._bodyGeometry;
                    const edges = character._edgeGeometry;
                    const vertices = character._cellBox.attributes.position.count;
                    for (let cell = 1; cell < points.length; cell++) {
                        const bounds = new THREE.Box3();
                        for (let i = 0; i < vertices; i++) {
                            bounds.expandByPoint(new THREE.Vector3().fromBufferAttribute(geometry.attributes.position,
                                (cell - 1) * vertices + i));
                        }
                        const center = bounds.getCenter(new THREE.Vector3());
                        const size = bounds.getSize(new THREE.Vector3());
                        near(center.x, points[cell].x);
                        near(center.y, points[cell].y);
                        near(center.z, points[cell].z);
                        for (const side of size.toArray()) near(side, 0.9);
                        const edgeBounds = new THREE.Box3();
                        const edgeCount = character._edgeTemplate.attributes.position.count;
                        for (let i = 0; i < edgeCount; i++) edgeBounds.expandByPoint(
                            new THREE.Vector3().fromBufferAttribute(edges.attributes.position, (cell - 1) * edgeCount + i));
                        assert(edgeBounds.min.distanceTo(bounds.min) < 0.00001
                            && edgeBounds.max.distanceTo(bounds.max) < 0.00001,
                        "Black boundaries must follow each real cube, not form oversized overlapping hulls.");
                    }
                    for (const direction of [p(1, 0, 0), p(-1, 0, 0), p(0, 1, 0), p(0, -1, 0), p(0, 0, 1), p(0, 0, -1)]) {
                        character.setRenderPose(points, { direction });
                        character.update();
                        const headBounds = new THREE.Box3();
                        character.head.updateMatrixWorld(true);
                        character.head.traverseVisible((mesh) => {
                            if (mesh.isMesh) headBounds.union(new THREE.Box3().setFromObject(mesh));
                        });
                        const size = headBounds.getSize(new THREE.Vector3());
                        assert(size.x < 1 && size.y < 1 && size.z < 1,
                            `${kind} block head/face must remain inside one cell on every axis: ${size.toArray()}`);
                    }
                    character.setRenderPose(points.slice(0, 1), { direction: p(0, 1, 0) });
                    character.update();
                    assert.equal(geometry.drawRange.count, 0, "A reset to one cell must not draw old tail cubes.");
                    assert.equal(edges.drawRange.count, 0, "Reset must also remove the old black tail boundaries.");
                    character.setRenderPose(points, { direction: p(0, 1, 0) });
                    character.update();
                    assert.equal(geometry.drawRange.count, (points.length - 1) * character._cellBox.index.count);
                    assert.equal(edges.drawRange.count, (points.length - 1) * character._edgeTemplate.attributes.position.count);
                } finally { character.dispose(); }
            }
        });

        test("block actors keep real attack-only jaw motion, gray dead cells and protection without a fake defeat", () => {
            const player = new BlockSnakeCharacter("player");
            const hunter = new BlockSnakeCharacter("hunter");
            try {
                for (const character of [player, hunter]) {
                    character.setState({ snake: board().snake, direction: p(0, 0, 1) }, board());
                    character.setRenderPose([p(0, 0, 0), p(0, 0, -1), p(0, 0, -2)], { direction: p(0, 0, 1) });
                    character.setPresentation({ death: 0, attack: 0 });
                    character.update();
                }
                const alive = Array.from(player._bodyGeometry.attributes.color.array.slice(0, 3));
                hunter.setPresentation({ attack: 1, death: 0 });
                hunter.update();
                assert(hunter._jawPivot.rotation.x > 0.6, "The moving predator still opens a physical block jaw.");
                player.setPresentation({ attack: 1, death: 0, shieldBlock: 1 });
                player.update();
                assert.equal(player._jawPivot.rotation.x, 0, "A player/shield contact cannot invent a predator bite.");
                assert(player._shieldBubble.visible);
                assert.equal(player._crosses.visible, false);
                player.setPresentation({ death: 1 });
                player.update();
                const dead = Array.from(player._bodyGeometry.attributes.color.array.slice(0, 3));
                const chroma = (rgb) => Math.max(...rgb) - Math.min(...rgb);
                assert(chroma(dead) < chroma(alive) * 0.3);
                assert(player._crosses.visible);
                assert.equal(player._skinMaterial.emissiveIntensity, 0);
                assert.equal(player._shieldBubble.visible, false);
                player.setPresentation({ death: 0 });
                player.update();
                assert.equal(player._crosses.visible, false, "Reset/next episode restores a living block face.");
            } finally { player.dispose(); hunter.dispose(); }
        });

test("objectives separate random exploration from goals and use only authoritative enemy targets", () => {
    const state = board();
    state.foods = [
        { id: "food-1", type: "normal", position: p(14), value: 1 },
        { id: "gold-1", type: "gold", position: p(18), value: 3 },
    ];
    state.enemies[0].intent = {
        behavior: "intercept", target: p(11), targetId: "player", nextPosition: p(11),
    };
    state.enemies.push({
        id: "rival-1", type: "rival", snake: [p(20), p(21)], direction: p(-1, 0, 0),
        intent: { behavior: "seek-food", target: p(18), targetId: "gold-1", nextPosition: p(19) },
    });
    const original = copy(state);
    const view = describeObjectives(state, { mode: "train", lastAction: 4, actionLabel: "Up", isExploring: true });
    assert.match(view.player.label, /EXPLORING/);
    assert.equal(view.player.activity, "Random: Up");
    assert.equal(view.player.targetId, "food-1", "The food objective is not an invented selected policy route.");
    assert.equal(view.player.moveKind, "last-action");
    assert.equal(view.player.nextPosition, undefined, "Never invent the RL player's next decision.");
    assert.match(view.enemies[0].activity, /Cutting YOU off/);
    assert.deepEqual(view.enemies[0].target, state.enemies[0].intent.target);
    assert.equal(view.enemies[1].targetId, "gold-1");
    assert.deepEqual(view.enemies[1].nextPosition, state.enemies[1].intent.nextPosition);
    assert.match(view.enemies[1].label, /GOLD/);
    assert.deepEqual(state, original);
});

test("objective changes clear stale food/enemy links and distinguish policy, test and searching", () => {
    const state = board();
    state.foods = [];
    state.enemies[0].intent = { behavior: "search", target: null, targetId: null, nextPosition: p(12) };
    const policy = describeObjectives(state, { mode: "train", lastAction: 1, actionLabel: "E", isExploring: false });
    assert.match(policy.player.label, /POLICY/);
    assert.equal(policy.player.target, null);
    assert.equal(policy.player.gridDistance, null);
    assert.equal(policy.enemies[0].target, null);
    assert.match(policy.enemies[0].label, /SEARCH/);
    const testView = describeObjectives({ ...state, enemies: [] },
        { mode: "eval", lastAction: 1, actionLabel: "E", isExploring: false });
    assert.match(testView.player.label, /TEST/);
    assert.deepEqual(testView.enemies, []);
});

test("authored role expressions remain different and only actual attack presentation opens the jaw", () => {
    const faces = ["player", "hunter", "rival"].map((kind) => new SnakeCharacter(kind, { headOnly: true }));
    try {
        faces.forEach((face) => { face.setPortraitState({ reset: true }); face.update(0, 0.5); });
        const [player, hunter, rival] = faces;
        const matrix = new THREE.Matrix4();
        hunter._brows.getMatrixAt(0, matrix);
        const rotation = new THREE.Quaternion();
        matrix.decompose(new THREE.Vector3(), rotation, new THREE.Vector3());
        assert(new THREE.Euler().setFromQuaternion(rotation).z < -0.1);
        assert(rival._eyeRoots[1].scale.y < rival._eyeRoots[0].scale.y);
        assert.equal(hunter._mouthCavity.visible, false);
        hunter.setPresentation({ attack: 1, death: 0 });
        hunter.update(0.016, 0.6);
        assert(hunter._jaw.position.y < -0.4);
        assert(hunter._mouthCavity.visible);
        assert(hunter._attackJaw.visible);
        assert(hunter._attackJaw.rotation.x > 0.9, "The jaw must create a visible open silhouette, not a hidden mouth decal");
        assert(hunter._upperMuzzle.scale.y < 0.6);
        player.setPortraitState({ dead: true, reset: true });
        player.update(0, 0.6);
        assert(player._crosses.visible);
        player.setPortraitState({ reset: true });
        player.update(0, 0.6);
        assert.equal(player._crosses.visible, false);
    } finally { faces.forEach((face) => face.dispose()); }
});

test("enemy silhouettes fit physical cells as well as the player", () => {
    for (const kind of ["hunter", "rival", "patrol", "ambush"]) {
        const character = new SnakeCharacter(kind);
        try {
            const state = board();
            character.setState({ ...state.enemies[0], type: kind }, state, true);
            character.setRenderPose([p(0, 0, 0), p(0, 0, -1), p(0, 0, -2)], { direction: p(0, 0, 1), snap: true });
            character.setPresentation({ death: 0, reducedMotion: true });
            character.update(0, 0.5);
            const bounds = new THREE.Box3();
            character.head.updateMatrixWorld(true);
            character.head.traverseVisible((mesh) => { if (mesh.isMesh) bounds.union(new THREE.Box3().setFromObject(mesh)); });
            const size = bounds.getSize(new THREE.Vector3());
            assert(size.x <= 1 && size.y <= 1 && size.z <= 1, `${kind} must not look colliding in an adjacent cell: ${size.toArray()}`);
        } finally { character.dispose(); }
    }
});

test("defeat desaturates actual body vertices and skin, uses dead eyes and stops idle motion", () => {
    const character = new SnakeCharacter("player");
    try {
        const state = board();
        character.setState(state, state, true);
        character.setRenderPose([p(0, 0, 0), p(0, 0, -1), p(0, 0, -2)], { direction: p(0, 0, 1), snap: true });
        character.setPresentation({ death: 0 });
        character.update(0, 0.5);
        const chroma = (rgb) => Math.max(...rgb) - Math.min(...rgb);
        const alive = chroma(Array.from(character._bodyGeometry.attributes.color.array.slice(0, 3)));
        character.setPresentation({ death: 1, hitSegment: 1 });
        character.update(0.016, 1);
        assert(character._crosses.visible);
        assert(character._visual.rotation.z > 1.2, "The head visibly flops instead of retaining a live upright pose");
        assert(chroma(character._skinMaterial.color.toArray()) < 0.12);
        assert(chroma(Array.from(character._bodyGeometry.attributes.color.array.slice(0, 3))) < alive * 0.3,
            "Multiplying bright-green body material by a tint is not an eliminated body");
        const facePose = character._visual.position.toArray();
        const bodyPose = character._bodyGeometry.attributes.position.array.slice();
        character.update(0.05, 4);
        character._visual.position.toArray().forEach((value, index) => near(value, facePose[index]));
        assert(bodyPose.every((value, index) => Math.abs(value - character._bodyGeometry.attributes.position.array[index]) < 0.00000001));
    } finally { character.dispose(); }
});

test("live pacing dispatches real step work at the selected one, two and default four-step cadence", () => {
    function startsAt(mode) {
        const pacer = mode ? new SimulationPacer(mode) : new SimulationPacer();
        const starts = [];
        let completesAt = null;
        pacer.resume(0);
        for (let now = 0; now <= 5000; now += 1) {
            if (completesAt !== null && now >= completesAt) { pacer.complete(now); completesAt = null; }
            if (pacer.tryStart(now)) {
                assert.equal(completesAt, null, "A new Python step must not overlap the previous request");
                starts.push(now);
                completesAt = now + 20;
            }
        }
        return starts;
    }
    const one = startsAt("rate-1"), two = startsAt("slow"), defaultRate = startsAt();
    assert.equal(one.length, 5);
    assert.equal(two.length, 10);
    assert.equal(defaultRate.length, 20);
    assert(one.every((time, index) => !index || time - one[index - 1] >= 1000));
    assert(two.every((time, index) => !index || time - two[index - 1] >= 500));
});

test("slow requests and long browser stalls discard missed steps instead of bursting to catch up", () => {
    const pacer = new SimulationPacer();
    pacer.resume(0);
    assert(pacer.tryStart(250));
    assert.equal(pacer.tryStart(1000), false);
    pacer.complete(3000);
    assert.equal(pacer.tryStart(3000), false);
    assert.equal(pacer.tryStart(3249), false);
    assert(pacer.tryStart(3250));
    pacer.complete(3260);
    assert(pacer.tryStart(10000), "A late browser frame may issue one current step");
    pacer.complete(10010);
    assert.equal(pacer.tryStart(10010), false, "It must not issue a backlog of missed steps");
    assert.equal(pacer.tryStart(10249), false);
    assert(pacer.tryStart(10250));
});

test("pause, intro, terminal or tab suspension preserves rate and resumes from a fresh deadline", () => {
    const pacer = new SimulationPacer("slow");
    pacer.resume(0);
    assert(pacer.tryStart(500));
    pacer.suspend();
    pacer.complete(550);
    assert.equal(pacer.tryStart(9000), false);
    pacer.resume(10000);
    assert.equal(pacer.tryStart(10000), false);
    assert.equal(pacer.tryStart(10499), false);
    assert(pacer.tryStart(10500), "Resume retains the chosen two-step cadence, not the default or an old deadline");
});

test("an in-flight old rate cannot re-arm itself after a selected rate change", () => {
    const pacer = new SimulationPacer("rate-12");
    pacer.resume(0);
    assert(pacer.tryStart(84));
    pacer.suspend();
    pacer.setMode("rate-1", 90);
    pacer.complete(110);
    pacer.resume(120);
    assert.equal(pacer.tryStart(203), false);
    assert.equal(pacer.tryStart(1119), false);
    assert(pacer.tryStart(1120));
});

test("fast batching stays uncapped but still serial, with no inherited fast deadline on exit", () => {
    const pacer = new SimulationPacer("fast");
    pacer.resume(0);
    assert(pacer.tryStart(0));
    assert.equal(pacer.tryStart(100), false);
    pacer.complete(200);
    assert(pacer.tryStart(200), "A completed fast batch must not acquire a normal-mode rate cap");
    pacer.suspend();
    pacer.setMode("slow", 250);
    pacer.complete(300);
    pacer.resume(350);
    assert.equal(pacer.tryStart(350), false);
    assert.equal(pacer.tryStart(849), false);
    assert(pacer.tryStart(850));
});

test("slow live interpolation spans the chosen interval and retimes without jumping or inventing cells", () => {
    const motion = new PoseInterpolator();
    const a = board(0, 9), b = board(1, 10), c = board(2, 11);
    motion.setBoard(a, { interval: 1000 });
    motion.setBoard(b, { interval: 1000 });
    const halfway = copy(motion.advance(0.44));
    near(halfway.player[0].x, 9.5);
    motion.setBoard(copy(b), { interval: 250 });
    assert.deepEqual(motion.advance(0), halfway, "Changing training rate preserves the current displayed pose");
    motion.advance(0.11);
    assert.deepEqual(motion.pose.player, b.snake);
    motion.setBoard(c, { interval: 1000 });
    near(motion.advance(0.44).player[0].x, 10.5);
    motion.advance(0.44);
    assert.deepEqual(motion.pose.player, c.snake);
});
