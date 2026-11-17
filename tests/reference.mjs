import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { test } from "node:test";
import * as THREE from "../static/vendor/three.module.js";
import { ReferenceWorld } from "../static/reference-world.js";
import { SimulationPacer } from "../static/presentation.js";

const original = await fs.readFile(new URL("./fixtures/original-reference.html", import.meta.url), "utf8");
const page = await fs.readFile(new URL("../static/index.html", import.meta.url), "utf8");
const view = await fs.readFile(new URL("../static/reference-view.js", import.meta.url), "utf8");
const normalize = (source) => source.replace(/\s+/g, "");

function functionSource(source, name) {
    const start = source.search(new RegExp(`function\\s+${name}\\s*\\(`));
    assert(start >= 0, `Missing original function ${name}`);
    const opening = source.indexOf("{", start);
    let depth = 1;
    let end = opening + 1;
    while (depth && end < source.length) {
        if (source[end] === "{") depth += 1;
        if (source[end] === "}") depth -= 1;
        end += 1;
    }
    assert.equal(depth, 0, `${name} must be a complete function`);
    return source.slice(start, end);
}

test("the requested bottom HUD keeps the lightweight reference page rather than importing the modern dashboard", () => {
    assert.equal(page.match(/<title>(.*?)<\/title>/)[1], original.match(/<title>(.*?)<\/title>/)[1]);
    assert(page.indexOf('id="scoreboard"') > page.indexOf('id="gameCanvas"'));
    assert(!page.includes('class="workspace"') && !page.includes('class="top-grid"'));
    assert(!page.includes('href="/static/styles.css'),
        "The old reference must not accidentally load the unrelated dashboard theme.");
});

test("the original Orbit, grid geometry and outlined cube renderer are preserved beneath optional views and pulses", () => {
    for (const name of ["gridToWorld", "create3DGrid", "updateSnakeMesh", "updateCamera", "removeApple"]) {
        const actual = functionSource(view, name).replace("return gridGroup;", "");
        assert.equal(normalize(actual), normalize(functionSource(original, name)),
            `${name} must remain the reference implementation, not an approximation.`);
    }
    const setup = (source) => source.slice(source.indexOf("const scene ="), source.indexOf("function gridToWorld"));
    const stripComments = (source) => source.replace(/\/\/[^\n]*/g, "");
    const expectedSetup = setup(original).replace(/const GRID_SIZE = 32;|const CELL_SIZE = 1;/g, "")
        .replace("0x30122d", "0x000000");
    assert.equal(normalize(stripComments(setup(view))), normalize(stripComments(expectedSetup)));
    assert(!view.includes("CinematicDirector") && !view.includes("Snake3DRenderer") && !view.includes("setCameraMode"),
        "The new application's camera must not take ownership of the original orbit.");
});

test("only genuine forward food growth pulses the grid/player; snapshot loads and repeated events cannot retrigger it", () => {
    const grid = new THREE.Group();
    grid.add(new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ opacity: 0.1 })));
    const world = new ReferenceWorld(new THREE.Scene(), (p) => new THREE.Vector3(p.x, p.y, p.z), grid);
    const base = { seed: 7, steps: 0, gridSize: 32, snake: [{ x: 1, y: 1, z: 1 }],
        foods: [], obstacles: [], movingWalls: [], gates: [], events: [], zone: { inset: 0, warning: false } };
    world.setBoard(base, { episode: "first", now: 1000 });
    world.update(1000);
    assert.equal(world.growthPulse, 0);
    const ate = { ...base, steps: 1, snake: [...base.snake, { x: 2, y: 1, z: 1 }],
        events: [{ type: "food", step: 1 }] };
    world.setBoard(ate, { advanced: true, episode: "first", now: 1100 });
    world.update(1100);
    assert(world.growthPulse > 0.9 && grid.children[0].material.opacity > 0.4);
    world.setBoard(ate, { advanced: true, episode: "first", now: 1200 });
    world.update(1300);
    assert(world.growthPulse < 0.6, "Repeating the same event must not restart the pulse.");
    world.update(1800);
    assert.equal(world.growthPulse, 0);
    world.setBoard({ ...ate, steps: 5 }, { advanced: false, episode: "first", now: 1900 });
    world.update(1900);
    assert.equal(world.growthPulse, 0, "Restoring a saved food event is not a new collection.");
    world.setBoard(ate, { advanced: true, episode: "new", now: 2000 });
    world.update(2000);
    assert.equal(world.growthPulse, 0, "A new episode must not borrow the previous episode's growth.");
});
test("the reference apple keeps its original geometry and base material beneath the requested glow", () => {
    const appleStyle = (source) => source.slice(source.indexOf("const geometry = new THREE.SphereGeometry(0.4, 16, 16);"),
        source.indexOf("apple = { pos: applePos, mesh: appleMesh };") + "apple = { pos: applePos, mesh: appleMesh };".length);
    assert.equal(normalize(appleStyle(view)), normalize(appleStyle(original)));
    assert(!view.includes("function moveSnake") && !view.includes("function findPath") && !view.includes("Math.random"),
        "The reference's local simulation/random food cannot remain behind a supposedly real RL view.");
});

test("current physical obstacles, gate states and hazard removals update without modifying the reference actors", () => {
    const scene = new THREE.Scene();
    const convert = (p) => new THREE.Vector3(p.x - 15.5, p.y - 15.5, p.z - 15.5);
    const world = new ReferenceWorld(scene, convert);
    const board = {
        gridSize: 32, obstacles: [{ x: 8, y: 12, z: 17 }, { x: 8, y: 13, z: 17 }],
        movingWalls: [{ cells: [{ x: 6, y: 19, z: 9 }], nextCells: [{ x: 7, y: 19, z: 9 }], warning: true }],
        foods: [{ type: "gold", position: { x: 21, y: 18, z: 8 } }],
        gates: [{ id: "gate-0", state: "warning", cells: [{ x: 4, y: 6, z: 8 }] }],
        zone: { inset: 0, nextInset: 2, warning: true },
        obstacleChange: { addedCells: [{ x: 16, y: 8, z: 12 }] },
    };
    world.setBoard(board);
    assert.deepEqual(world.debug().positions.obstacles, board.obstacles.map((p) => convert(p).toArray()));
    assert.equal(world.pools.get("gate:gate-0")[0].material, world.materials.warning);
    assert.equal(world.debug().cells["wall-preview"], 1);
    assert.equal(world.debug().cells["obstacle-warning"], 1);
    assert.deepEqual(world.debug().positions["obstacle-warning"], [[0.5, -7.5, -3.5]]);
    assert.equal(world.debug().zone.extent, 28);
    world.setBoard({ ...board, obstacles: [], movingWalls: [], foods: [], gates: [], obstacleChange: null,
        zone: { inset: 0, warning: false } });
    assert.equal(world.debug().cells.obstacles, 0);
    assert.equal(world.debug().cells.gold, 0);
    assert.equal(world.debug().cells["wall-preview"], 0);
    assert.equal(world.debug().cells["gate:gate-0"], undefined);
    assert.equal(world.debug().cells["obstacle-warning"], 0);
    assert.equal(world.debug().zone.visible, false);
});

test("soft player and stronger food glows follow real cells and remove obsolete sprites without touching physics", () => {
    const scene = new THREE.Scene();
    const convert = (p) => new THREE.Vector3(p.x - 15.5, p.y - 15.5, p.z - 15.5);
    const world = new ReferenceWorld(scene, convert);
    const board = {
        gridSize: 32, snake: [{ x: 11, y: 12, z: 13 }, { x: 11, y: 11, z: 13 }],
        foods: [{ type: "normal", position: { x: 15, y: 12, z: 13 } },
            { type: "gold", position: { x: 18, y: 14, z: 15 } }],
        obstacles: [], movingWalls: [], gates: [], zone: { inset: 0, warning: false },
    };
    const before = structuredClone(board);
    world.setBoard(board);
    const originalGlow = world.pools.get("player-glow")[0];
    assert.deepEqual(world.debug().positions["player-glow"], [[-4.5, -3.5, -2.5], [-4.5, -4.5, -2.5]]);
    assert.deepEqual(world.debug().positions["normal-glow"], [[-0.5, -3.5, -2.5]]);
    assert.deepEqual(world.debug().positions["gold-glow"], [[2.5, -1.5, -0.5]]);
    assert(world.pools.get("player-glow").every((sprite) => sprite.isSprite));
    world.setBoard({ ...board, snake: [{ x: 12, y: 12, z: 13 }], foods: [] });
    assert.equal(world.pools.get("player-glow")[0], originalGlow, "Live poses must reuse the glow, not rebuild it.");
    assert.deepEqual(world.debug().positions["player-glow"], [[-3.5, -3.5, -2.5]]);
    assert.equal(world.debug().cells["normal-glow"], 0);
    assert.equal(world.debug().cells["gold-glow"], 0);
    assert.deepEqual(board, before, "Presentation may not move, spawn or remove physical food or snake cells.");
});

test("reference-speed and faster playback dispatch single real steps at 20 and 40 steps per second", () => {
    for (const [mode, expected] of [["rate-20", 20], ["rate-40", 40]]) {
        const pacer = new SimulationPacer(mode);
        pacer.resume(0);
        let steps = 0;
        for (let now = 1; now <= 1000; now++) if (pacer.tryStart(now)) {
            steps++;
            pacer.complete(now + 3);
        }
        assert.equal(steps, expected);
        pacer.suspend();
        assert.equal(pacer.tryStart(5000), false);
        pacer.resume(5000);
        assert.equal(pacer.tryStart(5000), false, "Pause must not accumulate a catch-up burst.");
        assert(pacer.tryStart(5000 + 1000 / expected));
    }
});
