import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "../static/vendor/three.module.js";
import { CinematicDirector } from "../static/director.js";
import { ContactPresentation } from "../static/presentation.js";

const p = (x, y, z) => ({ x, y, z });
const cellToScene = ({ x, y, z }) => new THREE.Vector3(x + 0.5 - 11, y + 0.5 - 11, z + 0.5 - 11);
const sceneToCell = ({ x, y, z }) => p(x - 0.5 + 11, y - 0.5 + 11, z - 0.5 + 11);
const sceneFor = (size) => ({ x, y, z }) => new THREE.Vector3(x + 0.5 - size / 2, y + 0.5 - size / 2, z + 0.5 - size / 2);
const board = (changes = {}) => ({
    gridSize: 22, seed: 871, level: 3, fullSurvival: false, steps: 0, done: false,
    snake: [p(11, 11, 11), p(10, 11, 11), p(9, 11, 11), p(8, 11, 11), p(7, 11, 11)],
    direction: p(1, 0, 0), enemies: [], obstacles: [], movingWalls: [], gates: [], foods: [], events: [],
    threat: 0, abilities: { dash: { active: false } }, ...changes,
});
const enemy = (points, id = "hunter-1") => ({ id, type: "hunter", direction: p(-1, 0, 0), snake: points });
const create = (state = board(), playback = {}, size = [1440, 900]) => {
    const director = new CinematicDirector({
        camera: new THREE.PerspectiveCamera(50, size[0] / size[1], 0.1, 300), cellToScene: sceneFor(state.gridSize),
    });
    director.setViewport(...size);
    director.setBoard(state, playback);
    director.update(1 / 60, 0);
    return director;
};
const pose = (director) => JSON.stringify({
    position: director.activeCamera.position.toArray(), quaternion: director.activeCamera.quaternion.toArray(),
    projection: director.activeCamera.projectionMatrix.elements,
});
const advance = (director, seconds, renderState) => {
    for (let i = 0; i < Math.round(seconds * 60); i += 1) director.update(1 / 60, director.now + 1 / 60, renderState);
};
const assertUpright = (director) => {
    assert.deepEqual(director.activeCamera.up.toArray(), [0, 1, 0]);
    const right = new THREE.Vector3().setFromMatrixColumn(director.activeCamera.matrixWorld, 0);
    assert(Math.abs(right.y) < 1e-10, "Vertical movement may pitch the view, never roll the world");
};
const assertFramed = (director, points, radius = 0.62, viewport = null) => {
    const camera = director.activeCamera;
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
    for (const point of points) {
        for (const offset of [new THREE.Vector3(), right.clone().multiplyScalar(radius), right.clone().multiplyScalar(-radius),
            up.clone().multiplyScalar(radius), up.clone().multiplyScalar(-radius)]) {
            const q = point.clone().add(offset).project(camera);
            assert(q.x >= -0.84001 && q.x <= 0.84001, `Body clipped horizontally: ${q.toArray()}`);
            assert(q.y >= -0.58001 && q.y <= 0.70001, `Body clipped behind top/bottom HUD: ${q.toArray()}`);
            assert(q.z > -1 && q.z < 1, "Body must remain between near and far planes");
            if (viewport) {
                const margin = viewport[0] <= 320 ? 196 : viewport[0] <= 520 ? 184 : 140;
                assert((q.y + 1) * viewport[1] / 2 >= margin - 1e-5, "Actual participant bodies must clear the terminal caption");
            }
        }
    }
};
const bodyPoints = (state) => [
    ...state.snake.map(cellToScene), ...state.enemies.flatMap((actor) => actor.snake.map(cellToScene)),
];
const contact = (changes = {}) => {
    const victim = {
        id: "player", type: "player", from: p(11, 11, 11), to: p(12, 11, 11), segmentIndex: 0,
        before: [p(11, 11, 11), p(10, 11, 11), p(9, 11, 11), p(8, 11, 11)],
        attempted: [p(12, 11, 11), p(11, 11, 11), p(10, 11, 11), p(9, 11, 11)],
    };
    const attacker = {
        id: "hunter-1", type: "hunter", from: p(12, 11, 11), to: p(11, 11, 11),
        before: [p(12, 11, 11), p(13, 11, 11), p(14, 11, 11), p(15, 11, 11)],
        attempted: [p(11, 11, 11), p(12, 11, 11), p(13, 11, 11), p(14, 11, 11)],
    };
    return {
        schemaVersion: 1, id: "871:19:1:death", step: 19, substep: 1, kind: "crossing", cause: "hunter",
        outcome: "death", position: p(11.5, 11, 11), timeFraction: 0.5, sourceId: attacker.id, victim, attacker, ...changes,
    };
};

    const verticalBodyStrike = () => contact({
        id: "73:20:1:3:contact", seed: 73, step: 20, kind: "body-strike", position: p(14, 11, 18),
        timeFraction: 1, sourceId: "hunter-0",
        attacker: {
            id: "hunter-0", type: "hunter", from: p(14, 12, 18), to: p(14, 11, 18), segmentIndex: 0,
            before: [p(14, 12, 18), p(15, 12, 18), p(15, 12, 19)],
            attempted: [p(14, 11, 18), p(14, 12, 18), p(15, 12, 18)],
        },
        victim: {
            id: "player", type: "player", from: p(14, 11, 18), to: p(14, 11, 19), segmentIndex: 1,
            before: [p(14, 11, 18), p(14, 10, 18), p(15, 10, 18)],
            attempted: [p(14, 11, 19), p(14, 11, 18), p(14, 10, 18)],
        },
    });
    const pairCoverage = (director, points, radius = 0.62) => {
        const camera = director.activeCamera;
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
        const projected = points.flatMap((point) => [right.clone().multiplyScalar(radius), right.clone().multiplyScalar(-radius),
            up.clone().multiplyScalar(radius), up.clone().multiplyScalar(-radius)].map((offset) => point.clone().add(offset).project(camera)));
        const width = Math.max(...projected.map((q) => q.x)) - Math.min(...projected.map((q) => q.x));
        const height = Math.max(...projected.map((q) => q.y)) - Math.min(...projected.map((q) => q.y));
        const safe = director.debug().safeFrame;
        return Math.max(width / (safe.x * 2), height / (safe.top + safe.bottom));
    };

    test("Rejected seed73 Y-down strike gets a close profile shot before its first impact frame", () => {
        const hit = verticalBodyStrike(), convert = sceneFor(32);
        for (const size of [[816, 657], [340, 422], [230, 422]]) {
            const state = board({ gridSize: 32, seed: 73, snake: hit.victim.before, enemies: [enemy(hit.attacker.before, hit.sourceId)] });
            const d = create(state, {}, size);
            d.setBoard({ ...state, steps: 20, done: true, contact: hit }, { paused: true });
            d.setPresentation({ phase: "approach", progress: 0, contact: hit });
            d.update(1 / 60, d.now + 1 / 60);
            const direction = d.activeCamera.position.clone().sub(d.look).normalize();
            assert(Math.abs(direction.y) <= 0.3, "A Y-down bite must be seen in profile, not from behind/above");
            const before = [...hit.victim.before, ...hit.attacker.before].map(convert);
            assert(pairCoverage(d, before) >= 0.5, "The actual presented pair must fill a substantial fraction of usable stage before the strike");
            const points = [...hit.victim.attempted, ...hit.attacker.attempted].map(convert);
            d.setPresentation({ phase: "impact", progress: 0, contact: hit });
            d.update(1 / 60, d.now + 1 / 60, {
                player: hit.victim.attempted.map(convert),
                enemies: [{ id: hit.sourceId, type: "hunter", points: hit.attacker.attempted.map(convert) }],
            });
            assert(pairCoverage(d, points) >= 0.5, "The actual contact bodies, not particles or recorded-history bounds, must remain large enough");
            const head = convert(hit.victim.to).project(d.activeCamera), target = convert(hit.position).project(d.activeCamera);
            assert(Math.hypot(head.x - target.x, head.y - target.y) >= 0.14, "The struck segment must not disappear behind the player's head");
            assertFramed(d, [...hit.victim.before, ...hit.attacker.before, ...hit.victim.attempted, ...hit.attacker.attempted].map(convert));
            assertUpright(d);
        }
    });

    test("The established contact shot holds through windup, bite and defeat instead of finishing its zoom after impact", () => {
        const hit = verticalBodyStrike(), convert = sceneFor(32);
        const state = board({ gridSize: 32, seed: 73, snake: hit.victim.before, enemies: [enemy(hit.attacker.before, hit.sourceId)] });
        const d = create(state, {}, [816, 657]);
        d.setBoard({ ...state, steps: 20, done: true, contact: hit }, { paused: true });
        d.setPresentation({ phase: "establish", progress: 0, contact: hit });
        d.update(1 / 60, d.now + 1 / 60);
        const established = pose(d);
        for (const phase of ["establish", "windup", "approach", "impact", "defeat", "aftermath", "rewind", "complete"]) {
            for (const progress of [0, 0.25, 0.5, 0.75, 1]) {
                d.setPresentation({ phase, progress, contact: hit });
                d.update(1 / 60, d.now + 1 / 60, {
                    player: hit.victim.attempted.map(convert),
                    enemies: [{ id: hit.sourceId, type: "hunter", points: hit.attacker.attempted.map(convert) }],
                });
                assert.equal(pose(d), established, "The camera must already be composed when the bite happens");
            }
        }
        const limp = hit.victim.attempted.map(convert).map((point, index) =>
            point.add(new THREE.Vector3(Math.cos(index) * 0.52, -2.2, Math.sin(index) * 0.52)));
        d.setPresentation({ phase: "aftermath", progress: 1, contact: hit });
        d.update(1 / 60, d.now + 1 / 60, {
            player: limp, enemies: [{ id: hit.sourceId, type: "hunter", points: hit.attacker.attempted.map(convert) }],
        });
        assertFramed(d, limp);
        assert.equal(pose(d), established, "The held aftermath must already fit the agreed whole-body limp, without belated dollying");
        assert.equal(d.debug().event.safetyExpanded, false);
    });

test("Long contact bodies remain complete above phone HUDs without reusing a blind compact zoom", () => {
    const hit = verticalBodyStrike(), convert = sceneFor(32);
    for (let x = 16; x <= 29; x += 1) hit.victim.before.push(p(x, 10, 18));
    hit.victim.attempted = [hit.victim.to, ...hit.victim.before.slice(0, -1)];
    for (let y = 13; y <= 24; y += 1) hit.attacker.before.push(p(15, y, 19));
    hit.attacker.attempted = [hit.attacker.to, ...hit.attacker.before.slice(0, -1)];
    const points = [...hit.victim.before, ...hit.victim.attempted, ...hit.attacker.before, ...hit.attacker.attempted].map(convert);
    for (const size of [[816, 657], [340, 422], [230, 422]]) {
        const state = board({ gridSize: 32, seed: 73, snake: hit.victim.before, enemies: [enemy(hit.attacker.before, hit.sourceId)] });
        const d = create(state, {}, size);
        d.setBoard({ ...state, done: true, steps: 20, contact: hit }, { paused: true });
        for (const phase of ["approach", "impact", "defeat", "aftermath", "rewind", "recap-contact"]) {
            d.setPresentation({ phase, progress: 0.5, contact: hit });
            d.update(1 / 60, d.now + 1 / 60);
            assertFramed(d, points, 0.62, size);
            assertUpright(d);
            assert(d.debug().visibility.clearance >= 0.7);
            if (size[0] < 520) {
                const up = new THREE.Vector3().setFromMatrixColumn(d.activeCamera.matrixWorld, 1);
                for (const point of points) {
                    const q = point.clone().addScaledVector(up, -0.62).project(d.activeCamera);
                    assert((q.y + 1) * size[1] / 2 >= (size[0] <= 320 ? 196 : 184) - 1e-5, "A long body must not enter the phone's terminal caption");
                }
            }
        }
    }
});
test("Actual frontend reaction poses fit the prepared seed73 shot through the held dead aftermath", () => {
    const hit = verticalBodyStrike(), convert = sceneFor(32);
    for (const size of [[816, 657], [340, 422], [230, 422]]) for (const reducedMotion of [false, true]) {
        const before = board({
            gridSize: 32, seed: 73, steps: 19, snake: hit.victim.before,
            enemies: [enemy(hit.attacker.before, hit.sourceId), enemy([p(29, 27, 29)], "unrelated")],
        });
        const terminal = { ...before, steps: 20, done: true, contact: hit, lastDeath: "hunter" };
        const d = create(before, { reducedMotion }, size);
        const timeline = new ContactPresentation({ reducedMotion });
        timeline.observe(before, { episode: 1 });
        timeline.observe(terminal, { episode: 1, waitForLock: false });
        d.setBoard(terminal, { paused: true, reducedMotion });
        const phases = new Set();
        let established = null, measuredImpact = false;
        for (let frameNumber = 0; frameNumber < 1200; frameNumber += 1) {
            const frame = timeline.advance(1 / 60);
            if (!frame) break;
            phases.add(frame.phase);
            const rendered = {
                player: frame.pose.player.map(convert),
                enemies: frame.pose.enemies.map((actor) => ({ id: actor.id, type: actor.type, points: actor.points.map(convert) })),
            };
            d.setPresentation({ phase: frame.phase, progress: frame.progress, contact: frame.contact });
            d.update(1 / 60, d.now + 1 / 60, rendered);
            const points = [...rendered.player, ...rendered.enemies.find(({ id }) => id === hit.sourceId).points];
            assertFramed(d, points, 0.62, size);
            assertUpright(d);
            if (frame.phase !== "recap") {
                established ??= pose(d);
                assert.equal(pose(d), established, "The real reaction must not force a belated camera adjustment during impact or elimination");
                assert.equal(d.debug().event.safetyExpanded, false);
            }
            if (frame.phase === "impact" && !measuredImpact) {
                assert(pairCoverage(d, points) >= 0.5, "Actual first-contact actor poses must visibly occupy the stage");
                measuredImpact = true;
            }
        }
        assert(measuredImpact);
        assert(phases.has("aftermath") && phases.has("rewind") && phases.has("recap"));
    }
});

test("Game View is orthographic by default; both lenses and existing aliases remain available", () => {
    const d = create();
    assert(d.activeCamera.isOrthographicCamera);
    d.setViewport(390, 844);
    assert.equal(d.camera.aspect, 390 / 844);
    assert(Math.abs(d.gameplayCamera.right / d.gameplayCamera.top - 390 / 844) < 1e-12);
    d.mode = "smart";
    assert.equal(d.mode, "game");
    d.mode = "overview";
    d.update(1 / 60, 1);
    assert(d.activeCamera.isPerspectiveCamera);
    d.setCameraMode("game");
    d.update(1 / 60, 2);
    assert(d.activeCamera.isOrthographicCamera);
    const provided = new THREE.OrthographicCamera();
    const custom = new CinematicDirector({ camera: new THREE.PerspectiveCamera(), gameplayCamera: provided, cellToScene });
    assert.equal(custom.activeCamera, provided);
    assert.throws(() => d.setCameraMode("unknown"));
});

test("Full bent bodies, not just heads, respect horizontal and HUD-safe extents at native/narrow viewports", () => {
    const state = board({
        snake: [p(11, 11, 11), p(10, 11, 11), p(9, 11, 11), p(8, 11, 11), p(8, 12, 11),
            p(8, 13, 11), p(8, 14, 11), p(8, 14, 12), p(8, 14, 13), p(8, 14, 14), p(8, 14, 15)],
        enemies: [enemy([p(15, 13, 9), p(16, 13, 9), p(17, 13, 9), p(18, 13, 9), p(18, 12, 9)])],
    });
    for (const size of [[1440, 900], [390, 844], [320, 560], [280, 560]]) {
        const d = create(state, {}, size);
        assertFramed(d, bodyPoints(state));
        assertUpright(d);
        advance(d, 3);
        assertFramed(d, bodyPoints(state));
        assert(d.debug().visibility.clearance >= 0.7);
    }
});

test("Busy pursuit meaningfully pans, turns and widens without an indefinite hold or abrupt orbit", () => {
    let state = board();
    const d = create(state);
    advance(d, 4);
    const initial = { center: d.smart.center.clone(), azimuth: d.smart.azimuth, halfHeight: d.smart.halfHeight };
    state = board({
        steps: 1, threat: 0.9,
        snake: state.snake.map(({ x, y, z }) => p(x + 3, y + 2, z - 2)),
        enemies: [enemy([p(20, 13, 14), p(20, 13, 15), p(20, 13, 16), p(20, 13, 17), p(20, 13, 18)])],
    });
    d.setBoard(state);
    let maxTurn = 0, widest = initial.halfHeight;
    for (let i = 0; i < 720; i += 1) {
        if (i % 15 === 0) {
            state = { ...state, steps: state.steps + 1 };
            d.setBoard(state);
        }
        const previous = d.smart.azimuth;
        d.update(1 / 60, d.now + 1 / 60);
        maxTurn = Math.max(maxTurn, Math.abs(d.smart.azimuth - previous) * 60);
        widest = Math.max(widest, d.smart.halfHeight);
        assertFramed(d, bodyPoints(state));
    }
    assert(d.smart.center.distanceTo(initial.center) > 2, "Pursuit must produce a real recenter");
    assert(Math.abs(d.smart.azimuth - initial.azimuth) > 0.08, "A justified new pursuit axis must change the viewing angle");
    assert(widest > initial.halfHeight + 0.8, "Include pursuit and escape context by actually widening");
    assert(d.smart.halfHeight < widest - 0.6, "A better viewing angle should gently recover scale after the safety widening");
    assert(maxTurn <= 0.100001, `Busy azimuth velocity must be bounded, got ${maxTurn}`);
    const settled = pose(d);
    advance(d, 10);
    assert.equal(pose(d), settled, "An unchanged quiet scene must settle rather than constantly orbit");
});

test("Threats changing sides can improve angle; vertical pursuit preserves a world-up horizon", () => {
    let state = board({ enemies: [enemy([p(16, 11, 16), p(17, 11, 16), p(18, 11, 16)])], threat: 0.8 });
    const d = create(state);
    advance(d, 5);
    const first = d.smart.azimuth;
    state = board({ steps: 1, direction: p(0, 1, 0), snake: [p(11, 16, 11), p(11, 15, 11), p(11, 14, 11), p(11, 13, 11)],
        enemies: [enemy([p(7, 13, 17), p(7, 12, 17), p(7, 11, 17)])], threat: 0.9 });
    d.setBoard(state);
    for (let i = 0; i < 600; i += 1) {
        d.update(1 / 60, d.now + 1 / 60);
        assertUpright(d);
        assertFramed(d, bodyPoints(state));
    }
    assert(Math.abs(d.smart.azimuth - first) > 0.08);
});

test("Interpolated scene-coordinate bodies take precedence without mutating authoritative snapshots", () => {
    const state = board({ enemies: [enemy([p(12, 11, 11), p(13, 11, 11)])] });
    const original = JSON.stringify(state);
    const renderState = {
        player: [new THREE.Vector3(-3.25, 4.3, -1.2), p(-4.25, 4.3, -1.2), p(-5.25, 4.3, -1.2)],
        enemies: [{ id: "hunter-1", type: "hunter", points: [p(-2.2, 4.3, -1.2), p(-1.2, 4.3, -1.2)] }],
    };
    const d = create(state);
    d.update(1 / 60, 1, renderState);
    const subjects = d.smartSubjects();
    assert.deepEqual(subjects.find((s) => s.id === "player").point.toArray(), [-3.25, 4.3, -1.2]);
    assert.deepEqual(subjects.find((s) => s.id === "hunter-1").point.toArray(), [-2.2, 4.3, -1.2]);
    assertFramed(d, [...renderState.player, ...renderState.enemies[0].points].map(({ x, y, z }) => new THREE.Vector3(x, y, z)));
    assert.equal(JSON.stringify(state), original);
});

test("Occlusion considers the tail as well as head and exposes only genuine local cutaway cells", () => {
    const d = create();
    const initial = d.activeCamera.position.clone();
    const tail = cellToScene(d.board.snake.at(-1));
    const midpoint = tail.clone().lerp(initial, 0.19);
    const obstacle = sceneToCell(midpoint);
    const cell = p(Math.round(obstacle.x), Math.round(obstacle.y), Math.round(obstacle.z));
    const state = board({ steps: 1, obstacles: [cell] });
    d.setBoard(state);
    assert(!d.clearView(tail, initial), "Fixture must actually obstruct the tail");
    advance(d, 10);
    assert(d.clearView(tail, d.activeCamera.position) || d.debug().cutaway.some((point) => JSON.stringify(point) === JSON.stringify(cell)));
    assert(d.debug().cutaway.every((point) => d.solids.has(`${point.x},${point.y},${point.z}`)));
    assert(d.cameraClearance(d.activeCamera.position) >= 0.7);
    const solidCenter = cellToScene(cell);
    const safe = d.safePosition(cellToScene(state.snake[0]), solidCenter);
    assert(d.cameraClearance(safe) >= 0.7, "Physical camera placement must not intersect an expanded obstacle");
    assert(safe.distanceTo(cellToScene(state.snake[0])) >= 5.99, "Obstruction must not cause a face-only fallback");
});

test("Pause freezes the full pose; sampled batches use a stable full-volume overview", () => {
    const d = create();
    advance(d, 2);
    d.setBoard(d.board, { paused: true });
    const frozen = pose(d);
    for (let i = 0; i < 100; i += 1) d.update(1 / 60, 500 + i);
    assert.equal(pose(d), frozen);
    d.setBoard(d.board, { sampled: true });
    d.update(1 / 60, 800);
    const overview = pose(d);
    assertFramed(d, d.overviewSubjects().map(({ point }) => point), 0.6);
    for (let i = 0; i < 8; i += 1) {
        d.setBoard(board({ steps: 200 + i * 128, snake: [p(1 + i * 2, 19, 3), p(1 + i * 2, 18, 3)] }), { sampled: true });
        d.update(1 / 60, 801 + i);
        assert.equal(pose(d), overview, "Fast batches must not invent a camera pursuit through unsampled routes");
    }
    d.setBoard(board({ steps: 1300 }));
    advance(d, 2);
    assertFramed(d, d.board.snake.map(cellToScene));
});

test("Every impact/defeat/recap phase frames both real contact paths, full bodies and contact in perspective", () => {
    const hit = contact();
    for (const size of [[1440, 900], [390, 844], [390, 440], [320, 560], [280, 560], [280, 440]]) {
        const state = board({ done: true, contact: hit, enemies: [enemy(hit.attacker.before)], snake: hit.victim.before });
        const d = create(state, { paused: true }, size);
        const points = [hit.position, ...hit.attacker.before, ...hit.attacker.attempted, ...hit.victim.before, ...hit.victim.attempted].map(cellToScene);
        for (const phase of ["locking", "approach", "impact", "defeat", "aftermath", "rewind", "recap", "recap-contact", "complete"]) {
            d.setPresentation({ phase, progress: 0.5, contact: hit });
            advance(d, 0.3);
            assert(d.activeCamera.isPerspectiveCamera);
            assertFramed(d, points);
            assert.equal(d.debug().event.contactId, hit.id);
            assert.equal(d.debug().event.actors, 2);
            assert(d.debug().event.pairFramed);
            assertUpright(d);
            assert(d.camera.position.distanceTo(cellToScene(hit.victim.from)) > 6);
            assert(points.every((point) => d.camera.position.distanceTo(point) >= 6), "Neither actor may become a near-lens face fallback");
            if (size[0] <= 480) {
                const up = new THREE.Vector3().setFromMatrixColumn(d.activeCamera.matrixWorld, 1);
                for (const point of points) {
                    const q = point.clone().addScaledVector(up, -0.62).project(d.activeCamera);
                    const margin = size[0] <= 320 ? 200 : 184;
                    assert((q.y + 1) * size[1] / 2 >= margin - 1e-5, "Contact bodies must sit above the mobile terminal overlay");
                }
            }
        }
        d.setPresentation(null);
        d.update(1 / 60, d.now + 1 / 60);
        assert(d.activeCamera.isOrthographicCamera, "Clearing presentation must not resurrect the same terminal event");
    }
});

test("Body strikes, geometry impacts and timeout do not invent an attacker or discard the struck tail", () => {
    for (const kind of ["body-strike", "enemy-body", "obstacle", "wall", "laser", "zone", "self"]) {
        const hit = contact({ kind, id: `hit:${kind}` });
        hit.victim.before.push(p(8, 12, 11), p(8, 13, 11), p(8, 14, 11), p(8, 15, 11));
        hit.victim.segmentIndex = kind === "body-strike" ? 7 : 0;
        if (["obstacle", "wall", "laser", "zone", "self"].includes(kind)) {
            hit.attacker = null; hit.sourceId = null;
        }
        const d = create(board({ done: true, contact: hit }), { paused: true });
        assertFramed(d, [...hit.victim.before, ...hit.victim.attempted, hit.position].map(cellToScene));
        assert.equal(d.debug().event.actors, hit.attacker ? 2 : 1);
    }
    const d = create(board({ done: true }), { paused: true });
    d.setPresentation({ phase: "defeat", progress: 0.5, contact: null });
    advance(d, 0.2);
    assert.equal(d.debug().event.contactId, null);
    assert.equal(d.debug().event.actors, 1);
});

test("Reduced motion removes elective angular travel but retains safe moving/body framing", () => {
    const d = create(board(), { reducedMotion: true });
    const initial = d.smart.azimuth;
    const state = board({ steps: 1, threat: 0.9, enemies: [enemy([p(18, 15, 18), p(19, 15, 18), p(20, 15, 18)])] });
    d.setBoard(state, { reducedMotion: true });
    advance(d, 10);
    assert.equal(d.smart.azimuth, initial);
    assertFramed(d, bodyPoints(state));
    const hit = contact();
    d.setPresentation({ phase: "impact", contact: hit, progress: 1 });
    d.update(1 / 60, d.now + 1 / 60);
    const eventPose = pose(d);
    advance(d, 1);
    assert.equal(pose(d), eventPose, "Reduced motion gets one readable matched cut, never a spatial fly-through");
    d.setPresentation({ phase: "aftermath", contact: hit, progress: 1 });
    advance(d, 1);
    assert.equal(pose(d), eventPose);
    assertFramed(d, [hit.position, ...hit.victim.before, ...hit.attacker.before].map(cellToScene));
});

test("A live reduced-motion change cancels a pending turn, including during contact presentation", () => {
    const state = board({ enemies: [enemy([p(16, 11, 16), p(17, 11, 16), p(18, 11, 16)])] });
    const d = create();
    d.setBoard({ ...state, steps: 1 });
    advance(d, 1);
    d.setBoard({ ...state, steps: 2 }, { reducedMotion: true });
    const frozenAngle = d.smart.azimuth;
    advance(d, 4);
    assert.equal(d.smart.azimuth, frozenAngle);
    d.setBoard({ ...state, steps: 3 }, { reducedMotion: false });
    d.setPresentation({ phase: "impact", contact: contact(), progress: 0.4 });
    advance(d, 0.2);
    d.setBoard({ ...state, steps: 3 }, { reducedMotion: true });
    const eventAngle = d.eventRig.azimuth;
    advance(d, 1);
    assert.equal(d.eventRig.azimuth, eventAngle);
    assert(d.debug().event.pairFramed);
    assertUpright(d);
});

test("Static remains exactly fixed; legacy advanced modes remain bounded and upright through vertical travel", () => {
    const d = create();
    d.setCameraMode("static");
    d.update(1 / 60, 1);
    const staticPose = pose(d);
    d.setBoard(board({ steps: 1, snake: [p(1, 20, 19), p(1, 19, 19)], direction: p(0, 1, 0) }));
    advance(d, 3);
    assert.equal(pose(d), staticPose);
    assertFramed(d, d.overviewSubjects().map(({ point }) => point), 0.6);
    for (const mode of ["film", "follow", "chase", "cine-chase", "cinema", "orbit"]) {
        d.setCameraMode(mode);
        for (let i = 0; i < 90; i += 1) {
            d.update(1 / 60, d.now + 1 / 60);
            assertUpright(d);
            assertFramed(d, d.board.snake.map(cellToScene));
            assert(d.cameraClearance(d.activeCamera.position) >= 0.7);
            assert(d.activeCamera.fov >= 40 && d.activeCamera.fov <= 55);
        }
        d.setBoard(d.board, { paused: true });
        const frozen = pose(d);
        advance(d, 1);
        assert.equal(pose(d), frozen, `${mode} must honor pause`);
        d.setBoard(d.board);
    }
});

test("Intro finishes exactly once, stays upright and cancellation returns camera ownership safely", () => {
    const d = create(board({ enemies: [enemy([p(15, 11, 15), p(16, 11, 15)])], foods: [{ position: p(14, 11, 13) }] }));
    let completed = 0;
    d.playIntro(() => { completed += 1; });
    for (let i = 0; i < 900; i += 1) {
        d.update(1 / 60, d.now + 1 / 60);
        assertUpright(d);
    }
    assert.equal(completed, 1);
    assert(d.activeCamera.isOrthographicCamera);
    d.playIntro(() => { completed += 1; });
    advance(d, 0.5);
    d.cancelIntro();
    advance(d, 1);
    assert.equal(completed, 1);
    assert(d.activeCamera.isOrthographicCamera);
});
