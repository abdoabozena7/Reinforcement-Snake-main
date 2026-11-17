import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import * as THREE from "../static/vendor/three.module.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const evidence = path.join(root, ".runtime-proof", "exact-reference", new Date().toISOString().replaceAll(/[:.]/g, "-"));
const python = process.env.ARENA_PYTHON || path.join(root, ".venv", "Scripts", "python.exe");
const expectedEnvironment = "survival-balanced-v8";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const report = { status: "RUNNING", checks: [], rates: [], errors: [], expectedError: false, motion: [] };
await fs.mkdir(evidence, { recursive: true });

const listener = net.createServer();
await new Promise((resolve) => listener.listen(0, "127.0.0.1", resolve));
const port = listener.address().port;
await new Promise((resolve) => listener.close(resolve));
const url = `http://127.0.0.1:${port}`;
let server;
let browser;
let context;
let page;
let log = "";

async function until(work, description, timeout = 20000) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
        if (await work()) return;
        await sleep(40);
    }
    throw new Error(`Timed out: ${description}`);
}

async function api(endpoint) {
    const response = await fetch(url + endpoint);
    assert(response.ok, `Actual server read failed: ${endpoint}`);
    return response.json();
}

async function debug() { return page.evaluate(() => window.__snakeDebug?.()); }
async function idle() { await until(async () => (await debug()).lifecycle.pendingControls === 0, "serialized controls settle"); }
async function press(key) {
    await idle();
    const canvas = await page.locator("#gameCanvas").boundingBox();
    await page.mouse.click(canvas.x + canvas.width / 2, canvas.y + canvas.height / 2);
    await page.keyboard.press(key);
    await idle();
}
async function clickControl(id) {
    await idle();
    if (!(await page.locator("#referenceGameControls").evaluate((details) => details.open))) {
        await page.locator("#referenceGameControls > summary").click();
    }
    await page.locator(id).click();
    await idle();
}
async function scenario(value) {
    await idle();
    if (!(await page.locator("#referenceGameControls").evaluate((details) => details.open))) {
        await page.locator("#referenceGameControls > summary").click();
    }
    await page.locator("#referenceScenario").selectOption(value);
    await idle();
}
async function cameraView(value) {
    await idle();
    if (!(await page.locator("#referenceGameControls").evaluate((details) => details.open))) {
        await page.locator("#referenceGameControls > summary").click();
    }
    await page.locator("#referenceCamera").selectOption(value);
    await until(async () => {
        const renderer = (await debug()).renderer;
        return renderer.cameraView.mode === value && (value === "orbit" ? renderer.cameraNear === 1
            : renderer.cameraNear === 0.1 && renderer.cameraView.target !== null);
    }, "actual rendered camera changes, not only its selected option");
}
async function statisticsMatch() {
    await until(async () => page.evaluate(() => {
        const state = window.__snakeDebug().state;
        return [...document.querySelectorAll("#referenceStats dd")].every((node) => {
            const value = Number(node.textContent.replaceAll(",", ""));
            const tolerance = node.dataset.stat === "epsilon" ? 0.00051 : node.dataset.stat === "avgScore" ? 0.0051 : 0;
            return Number.isFinite(value) && Math.abs(value - state[node.dataset.stat]) <= tolerance;
        });
    }), "all six visible statistics match the actual live learner");
}
async function pause() {
    await idle();
    if (!(await debug()).state.paused) await clickControl("#referencePause");
    await until(async () => (await debug()).state.paused && (await debug()).lifecycle.desiredPaused, "acknowledged pause");
}
async function resume() {
    await idle();
    if ((await debug()).state.paused) await clickControl("#referencePause");
    await until(async () => !(await debug()).state.paused, "acknowledged resume");
}
function check(name, values = {}) {
    report.checks.push({ name, ...values });
    console.log(`PASS ${name}`);
}
function parameters(file) {
    const script = "import hashlib,numpy as np,sys; h=hashlib.sha256(); d=np.load(sys.argv[1],allow_pickle=False); [h.update(d[k].tobytes()) for k in ('w1','b1','w2','b2','w3','b3')]; print(h.hexdigest())";
    return execFileSync(python, ["-B", "-c", script, file], { encoding: "utf8" }).trim();
}
function cameraMatchesReference(snapshot) {
    const { angle, cameraPosition, cameraQuaternion, cameraFov, cameraNear, cameraFar } = snapshot.renderer;
    const expected = new THREE.PerspectiveCamera(60, 1, 1, 144);
    expected.position.set(Math.cos(angle) * 50, 30, Math.sin(angle) * 50);
    expected.lookAt(0, 5, 0);
    assert(expected.position.distanceTo(new THREE.Vector3().fromArray(cameraPosition)) < 0.000001);
    assert(expected.quaternion.angleTo(new THREE.Quaternion().fromArray(cameraQuaternion)) < 0.000001);
    assert.deepEqual([cameraFov, cameraNear, cameraFar], [60, 1, 144]);
}
function actorsMatchServer(snapshot) {
    const entities = [{ id: "player", snake: snapshot.state.board.snake }, ...snapshot.state.board.enemies];
    assert.equal(snapshot.renderer.actors.length, entities.length);
    for (const actor of snapshot.renderer.actors) {
        const entity = entities.find((item) => item.id === actor.id);
        assert.deepEqual(actor.cells, entity.snake);
        assert.equal(actor.meshes.length, entity.snake.length * 2);
        actor.meshes.forEach((mesh, index) => {
            const point = entity.snake[Math.floor(index / 2)];
            assert.deepEqual(mesh.position, [point.x - 15.5, point.y - 15.5, point.z - 15.5]);
            assert.equal(mesh.size, index % 2 ? 0.9 : 1.2);
            assert.equal(mesh.color, index % 2 ? actor.type === "patrol" ? snapshot.renderer.obstacleColor
                : actor.id === "player" ? 0x870734 : 0xef473a : 0);
        });
    }
}
async function capture(name) { await page.screenshot({ path: path.join(evidence, `${name}.png`) }); }
async function rate(mode, expected) {
    await pause();
    await page.locator("#referenceSpeed").selectOption(mode);
    await resume();
    const start = performance.now();
    const before = await api("/api/state");
    await sleep(2400);
    const after = await api("/api/state");
    const seconds = (performance.now() - start) / 1000;
    const steps = after.totalSteps - before.totalSteps;
    report.rates.push({ selected: expected, steps, learned: after.trainingSteps - before.trainingSteps, seconds });
    assert(Math.abs(steps - expected * seconds) <= Math.max(1.5, expected * seconds * 0.2),
        `Selected ${expected} steps/s, measured ${steps} steps in ${seconds.toFixed(3)}s; real progress must track the selected speed.`);
    if (expected === 40) assert(steps / seconds > 25,
        "The faster setting must actually exceed the original reference's20steps/s.");
    assert.equal(after.trainingSteps - before.trainingSteps, steps);
    await pause();
    return { selected: expected, steps, learned: after.trainingSteps - before.trainingSteps, seconds };
}

try {
    server = spawn(python, ["-B", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", String(port), "--no-access-log"], {
        cwd: root, env: { ...process.env, SNAKE_STORAGE_DIR: path.join(evidence, "storage"), SNAKE_SEED: "73" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    server.stdout.on("data", (chunk) => { log += chunk; });
    server.stderr.on("data", (chunk) => { log += chunk; });
    await until(async () => {
        if (server.exitCode !== null) throw new Error(log);
        try { return (await api("/api/health")).environmentVersion === expectedEnvironment; }
        catch { return false; }
    }, "actual current Python service");
    assert.equal((await api("/api/health")).rewardVersion, "survival-objectives-v1");
    report.source = {};
    for (const file of ["main.py", "backend\\agent.py", "backend\\config.py", "backend\\environment.py", "backend\\session.py", "backend\\champion.py",
        "static\\index.html", "static\\reference-view.js", "static\\reference-client.js", "static\\reference-world.js",
        "static\\reference-overlay.js", "static\\reference-controls.css", "static\\reference-cameras.js",
        "static\\network-view.js", "static\\presentation.js",
        "storage\\pretrained_3d.npz", "storage\\pretrained_survival.npz"]) report.source[file] = hash(await fs.readFile(path.join(root, file)));
    browser = await chromium.launch({ channel: "msedge", headless: true });
    context = await browser.newContext({ viewport: { width: 1115, height: 750 }, deviceScaleFactor: 1,
        recordVideo: { dir: path.join(evidence, "raw"), size: { width: 1115, height: 750 } } });
    page = await context.newPage();
    const video = page.video();
    page.on("pageerror", (error) => report.errors.push({ message: error.message, expected: report.expectedError }));
    page.on("console", (message) => {
        if (message.type() === "error") report.errors.push({ message: message.text(), expected: report.expectedError });
    });
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await until(async () => (await debug())?.state && (await debug()).renderer.angle > 0, "original scene and real game");
    await pause();
    const opening = await debug();
    assert.equal(opening.state.rewardVersion, "survival-objectives-v1");
    assert.equal(await page.title(), "3D Futuristic Tron Snake Battle");
    assert.equal(await page.locator("body > canvas").count(), 1);
    assert.equal(await page.locator("#networkCanvas,.dashboard-panel,.stat-card,aside").count(), 0,
        "The miniature requested neurons must not import the large modern dashboard.");
    assert.equal(await page.locator("#referenceNeurons").count(), 1);
    assert.equal(await page.locator("#referenceStats dd").count(), 6);
    assert.equal(await page.locator("select").count(), 4);
    await statisticsMatch();
    assert.equal(await page.locator("#referenceSpeed").isVisible(), true);
    assert.equal(await page.locator("#referenceSpeed").inputValue(), "rate-20");
    const expectedCaptions = { player: "YOU", hunter: "wants to kill you", rival: "wants to eat your food",
        patrol: "Moving obstacle", ambush: "wants to kill you" };
    await until(async () => (await debug()).renderer.labels.length === (await debug()).renderer.actors.length,
        "all initial actor popups are visibly attached");
    const measured = await page.evaluate(() => {
        const snapshot = window.__snakeDebug();
        return {
            snapshot,
            labels: [...document.querySelectorAll(".snake-caption")].map((label) => {
                const rect = label.getBoundingClientRect();
                return { id: label.dataset.actor, text: label.textContent, hidden: label.hidden, x: rect.x, y: rect.y };
            }),
        };
    });
    const labelled = measured.snapshot;
    for (const actor of labelled.renderer.actors) {
        const label = measured.labels.find((item) => item.id === actor.id);
        assert.equal(label.text, expectedCaptions[actor.type]);
        assert.equal(label.hidden, false);
        const position = labelled.renderer.labels.find((item) => item.id === actor.id);
        assert(Math.abs(label.x - position.box.left) < 1 && Math.abs(label.y - position.box.top) < 1,
            "The popup must be an actual projected DOM element, not only a debug description.");
    }
    const grey = labelled.renderer.actors.find((actor) => actor.type === "patrol").color;
    const channels = [grey >> 16 & 255, grey >> 8 & 255, grey & 255];
    assert(Math.max(...channels) - Math.min(...channels) <= 12, "The patrol is a grey moving obstacle, not another orange competitor.");
    assert.equal(opening.renderer.background, "000000");
    assert.deepEqual(opening.renderer.world.positions["player-glow"],
        opening.state.board.snake.map((p) => [p.x - 15.5, p.y - 15.5, p.z - 15.5]));
    for (const type of ["normal", "gold"]) assert.deepEqual(opening.renderer.world.positions[`${type}-glow`],
        opening.state.board.foods.filter((food) => food.type === type).map((food) =>
            [food.position.x - 15.5, food.position.y - 15.5, food.position.z - 15.5]));
    cameraMatchesReference(opening);
    actorsMatchServer(opening);
    assert.equal(opening.renderer.world.cells.obstacles, opening.state.board.obstacles.length);
    assert.equal(opening.state.board.obstacles.length, 128);
    assert(opening.state.board.enemies.some((enemy) => enemy.type === "hunter"));
    assert(opening.state.board.enemies.some((enemy) => enemy.type === "rival"));
    assert.match(await page.locator('.reference-actor-meter[data-actor="player"]').getAttribute("aria-label"),
        new RegExp(`length ${opening.state.board.snake.length} cells`));
    const layout = await page.evaluate(() => {
        const box = (id) => {
            const rect = document.getElementById(id).getBoundingClientRect();
            return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom, width: rect.width, height: rect.height };
        };
        const canvas = document.getElementById("referenceNeurons");
        const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let colored = 0;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 50) colored++;
        return { hud: box("scoreboard"), neurons: box("referenceNeurons"), game: box("gameCanvas"), colored };
    });
    assert(layout.hud.top > 400 && layout.hud.bottom === 750);
    assert(Math.abs(layout.game.bottom - layout.hud.top) < 1);
    assert(layout.neurons.left < 30 && layout.neurons.width < 250 && layout.neurons.height < 160);
    assert(layout.colored > 50, "The miniature network must actually render real activations and signed edges.");
    assert.deepEqual((await debug()).renderer.neurons.dimensions, opening.state.network.dimensions);
    await capture("01-exact-reference-with-current-world");
    const frozen = JSON.stringify(await api("/api/state"));
    await sleep(350);
    const still = await debug();
    assert(still.renderer.angle > opening.renderer.angle, "The original camera keeps its same orbit even when simulation is paused.");
    assert.equal(JSON.stringify(await api("/api/state")), frozen);
    cameraMatchesReference(still);
    check("Literal original camera, cube renderer and scoreboard show the current real 3D world");
    check("Visible speed control defaults to reference20steps/s; exact role popups and grey moving obstacles are present");
    const beforeMissing = JSON.stringify((await debug()).state.board);
    await clickControl("#referenceLoad");
    await until(async () => /No compatible/.test(await page.locator("#referenceStatus").innerText()),
        "a missing saved policy is reported instead of disappearing behind resume status");
    assert.equal(JSON.stringify((await debug()).state.board), beforeMissing);
    const cameraFrozen = JSON.stringify((await debug()).state.board);
    for (const mode of ["tail", "food"]) {
        await cameraView(mode);
        const shot = await debug();
        const position = new THREE.Vector3().fromArray(shot.renderer.cameraPosition);
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(new THREE.Quaternion().fromArray(shot.renderer.cameraQuaternion));
        const target = new THREE.Vector3().fromArray(shot.renderer.cameraView.target);
        assert(forward.dot(target.clone().sub(position).normalize()) > 0.99999);
        if (mode === "tail") {
            const tail = shot.state.board.snake.at(-1);
            assert(position.distanceTo(new THREE.Vector3(tail.x - 15.5, tail.y - 15.5, tail.z - 15.5)) <= 7.001);
        } else {
            const head = shot.state.board.snake[0];
            assert(position.clone().sub(target).dot(new THREE.Vector3(head.x - 15.5, head.y - 15.5, head.z - 15.5).sub(target)) < 0);
        }
        assert.equal(JSON.stringify(shot.state.board), cameraFrozen, "Changing viewpoint cannot move the player or train.");
        await capture(`camera-${mode}`);
    }
    await cameraView("orbit");
    cameraMatchesReference(await debug());
    check("Bottom indicators, compact real neurons and optional tail/head and food/YOU cameras preserve the simulation");

    await page.locator("#referenceSpeed").selectOption("rate-20");
    await resume();
    let ended = false;
    const initialEpisode = opening.state.episode;
    await until(async () => {
        const current = await debug();
        cameraMatchesReference(current);
        actorsMatchServer(current);
        report.motion.push({ episode: current.state.episode, steps: current.state.totalSteps,
            learned: current.state.trainingSteps, head: current.state.board.snake[0],
            done: current.state.board.done, cause: current.state.board.lastDeath, angle: current.renderer.angle });
        if (current.state.board.done) ended = true;
        return current.state.episode > initialEpisode
            && (ended || current.state.completedEpisodes > opening.state.completedEpisodes);
    }, "real RL terminal and next round at reference speed with unchanged orbit", 40000);
    await pause();
    await capture("02-real-collision-recovery");
    check("Only Python controls the player, contacts, scores and next round; the reference orbit never cuts");

    await scenario("full");
    const full = await debug();
    assert.equal(full.state.fullSurvival, true);
    for (const role of ["hunter", "rival", "patrol", "ambush"]) {
        assert(full.state.board.enemies.some((enemy) => enemy.type === role));
    }
    actorsMatchServer(full);
    assert(full.state.board.gates.length && full.state.board.movingWalls.length);
    for (const gate of full.state.board.gates) assert.equal(full.renderer.world.cells[`gate:${gate.id}`], gate.cells.length);
    assert.equal(full.renderer.world.cells.obstacles, full.state.board.obstacles.length);
    await capture("03-all-current-enemies-and-hazards");
    await scenario("curriculum");
    assert.equal((await debug()).state.autoCurriculum, true);
    assert.equal((await debug()).state.level, 1);
    await scenario("4");
    check("Current full-survival enemy counts, real obstacles, gates and walls are retained");

    await scenario("1");
    await clickControl("#referenceReady");
    assert.equal((await debug()).state.mode, "eval");
    assert.equal((await debug()).state.epsilon, 0);
    await until(async () => /Loaded ready 3D weights/.test(await page.locator("#referenceStatus").innerText()),
        "the visible Ready control confirms the actual loaded policy");
    await until(async () => (await debug()).state.currentScore > 0,
        "ready DQN collects real food and produces visible score", 20000);
    await until(async () => (await debug()).renderer.world.growthPulse > 0.05,
        "a real collection pulses YOU and the outer grid");
    const growth = await debug();
    assert(growth.renderer.world.gridOpacity > 0.1);
    await capture("real-collection-pulse");
    await pause();
    await statisticsMatch();
    await capture("ready-policy-real-food-score");
    const checkpoint = path.join(evidence, "storage", "arena-3d-v3", expectedEnvironment, "manual.npz");
    await clickControl("#referenceSave");
    const readyParameters = parameters(checkpoint);
    await clickControl("#referenceLoad");
    assert.equal((await debug()).state.mode, "eval");
    assert.equal(parameters(checkpoint), readyParameters);
    check("Ready and saved weights use the current real policy without replacing the original visual layer");

    await clickControl("#referenceNewRun");
    const beforeChange = await debug();
    const pending = beforeChange.state.board.obstacleChange;
    assert(pending?.warning && pending.ticksUntilChange > 0);
    await until(async () => await page.locator("#obstacleWarning").isVisible()
        && (await debug()).renderer.world.cells["obstacle-warning"] === pending.addedCells.length,
    "real warning wireframes and visible status precede physical obstacle appearance");
    await press("s");
    const pendingBoard = JSON.stringify((await debug()).state.board);
    await press("l");
    assert.equal(JSON.stringify((await debug()).state.board), pendingBoard,
        "A warned random layout and its countdown must replay exactly after save/load.");
    await capture("obstacle-warning-before-appearance");
    await page.locator("#referenceSpeed").selectOption("rate-20");
    await resume();
    const changeStartedAt = performance.now();
    let changeEvent;
    await until(async () => {
        const current = await debug();
        changeEvent = current.state.board.events.find((event) => event.type === "obstacles_changed");
        return Boolean(changeEvent);
    }, "actual public policy play commits a previously warned random obstacle change", 15000);
    await pause();
    const changed = await debug();
    const key = (point) => `${point.x},${point.y},${point.z}`;
    const present = new Set(changed.state.board.obstacles.map(key));
    assert(changeEvent.removed.every((point) => !present.has(key(point))), "Warned old obstacles must physically disappear.");
    assert(changeEvent.added.every((point) => present.has(key(point))), "Warned new obstacles must become physical solids.");
    assert(changeEvent.added.every((point) => !changed.state.board.snake.some((body) => key(point) === key(body))),
        "Random appearance must not spawn into the actual player.");
    assert.equal(changed.renderer.world.cells.obstacles, changed.state.board.obstacles.length);
    await capture("obstacle-after-real-appearance");
    check("A real warning precedes random disappearance/appearance, and pending layouts survive checkpoint replay",
        { ticksWarned: pending.ticksUntilChange, added: changeEvent.added, removed: changeEvent.removed,
            wallSeconds: (performance.now() - changeStartedAt) / 1000 });

    const beforeCancel = JSON.stringify((await debug()).state.board);
    page.once("dialog", (dialog) => dialog.dismiss());
    await clickControl("#referenceReset");
    assert.equal(JSON.stringify((await debug()).state.board), beforeCancel, "Cancelled Reset must leave the run untouched.");
    page.once("dialog", (dialog) => dialog.accept());
    await clickControl("#referenceReset");
    await until(async () => (await debug()).state.trainingSteps > 0, "fresh random-weight Train resumes");
    assert.equal((await debug()).state.mode, "train");
    await pause();
    await statisticsMatch();
    await page.locator("#referenceMode").selectOption("eval");
    await idle();
    assert.equal((await debug()).state.mode, "eval");
    assert.equal((await debug()).state.epsilon, 0);
    await page.locator("#referenceMode").selectOption("train");
    await idle();
    assert.equal((await debug()).state.mode, "train");
    await press("["); await press("["); await press("[");
    assert.equal((await debug()).state.level, 1);
    const samples = [await rate("rate-1", 1), await rate("rate-6", 6),
        await rate("rate-20", 20), await rate("rate-40", 40)];
    await press("s");
    const beforeParameters = parameters(checkpoint);
    const beforeGradients = (await debug()).state.gradientSteps;
    const beforeLearning = (await debug()).state.trainingSteps;
    await press("0");
    await resume();
    await until(async () => (await debug()).state.trainingSteps > beforeLearning + 384, "actual batched learning");
    await pause();
    await press("s");
    const afterParameters = parameters(checkpoint);
    assert.notEqual(afterParameters, beforeParameters);
    assert((await debug()).state.gradientSteps > beforeGradients);
    await capture("04-real-learned-policy");
    check("Actual simulation rate is controllable and Python training changes real learned parameters",
        { samples, beforeParameters, afterParameters, gradients: (await debug()).state.gradientSteps });

    await clickControl("#referenceCheckBest");
    await until(async () => ["checking", "retained", "promoted"].includes((await debug()).state.bestPolicy.status),
        "actual best-policy evaluation is queued or completed");
    if ((await debug()).state.bestPolicy.status === "checking") {
        await page.locator("#referenceSpeed").selectOption("rate-20");
        const beforeBackgroundPlay = (await debug()).state.totalSteps;
        await resume();
        await sleep(1600);
        await pause();
        assert((await debug()).state.totalSteps - beforeBackgroundPlay > 15,
            "The live game must keep progressing during best-policy evaluation.");
    }
    await clickControl("#referenceSave");
    const duringBest = parameters(checkpoint);
    const beforeBestBoard = JSON.stringify((await debug()).state.board);
    await until(async () => ["retained", "promoted"].includes((await debug()).state.bestPolicy.status),
        "paused HUD polls for the completed real held-out comparison", 120000);
    await statisticsMatch();
    assert.equal(JSON.stringify((await debug()).state.board), beforeBestBoard,
        "Checking Best while paused must not simulate on the live board.");
    await clickControl("#referenceSave");
    assert.equal(parameters(checkpoint), duringBest, "Promotion may not replace the currently trained live policy.");
    await capture("retained-best-result");
    check("Best is evaluated on held-out cases without freezing play or replacing live weights; completion appears while paused");

    await press("4");
    await page.route("**/api/load-ready-weights", (route) => route.fulfill({
        status: 400, contentType: "application/json", body: JSON.stringify({ detail: "Deliberate reference-UI failure check" }),
    }));
    report.expectedError = true;
    await clickControl("#referenceReady");
    await until(async () => (await debug()).lifecycle.offline, "explicit failure state");
    assert.match(await page.locator("#referenceStatus").innerText(), /Python error/);
    assert.equal(await page.locator("#referenceStatus").getAttribute("role"), "alert");
    assert.equal(await page.locator("#referenceReconnect").isVisible(), true);
    await page.unroute("**/api/load-ready-weights");
    await page.locator("#referenceReconnect").click();
    await idle();
    await until(async () => !(await debug()).lifecycle.offline && (await debug()).state.paused, "safe reconnect");
    report.expectedError = false;
    assert.equal((await debug()).lifecycle.maxRequestConcurrency, 1);
    assert.equal(hash(await fs.readFile(path.join(root, "storage", "pretrained_3d.npz"))), report.source["storage\\pretrained_3d.npz"]);
    assert.equal(hash(await fs.readFile(path.join(root, "storage", "pretrained_survival.npz"))), report.source["storage\\pretrained_survival.npz"]);
    assert.equal(report.errors.filter((error) => !error.expected).length, 0, JSON.stringify(report.errors));
    check("Errors are explicit in the HUD; visible reconnect is paused and API requests stay serialized");
    await context.close();
    await video.saveAs(path.join(evidence, "exact-reference-real-rl.webm"));
    context = null;
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }, { width: 280, height: 844 }]) {
        const small = await browser.newContext({ viewport, deviceScaleFactor: 1 });
        page = await small.newPage();
        await page.goto(url, { waitUntil: "domcontentloaded" });
        await until(async () => {
            const snapshot = await debug();
            return snapshot?.state && snapshot.renderer.angle > 0;
        }, "real reference game and rendered camera at native dimensions");
        await pause();
        const size = await page.locator("#gameCanvas").evaluate((canvas) => [canvas.width, canvas.height]);
        const dockHeight = await page.locator("#scoreboard").evaluate((dock) => dock.getBoundingClientRect().height);
        assert.equal(size[0], viewport.width);
        assert(Math.abs(size[1] + dockHeight - viewport.height) < 1.1,
            "Bottom indicators reserve real arena space instead of covering or stretching the game.");
        cameraMatchesReference(await debug());
        await statisticsMatch();
        const panels = await page.locator("#scoreboard,#reference-speed-control").evaluateAll((nodes) =>
            nodes.map((node) => {
                const rect = node.getBoundingClientRect();
                return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
            }));
        for (const panel of panels) {
            assert(panel.left >= 0 && panel.right <= viewport.width && panel.top >= 0 && panel.bottom <= viewport.height);
        }
        if (viewport.width <= 650) {
            await page.locator("#referenceGameControls").evaluate((details) => {
                if (details.open) details.querySelector("summary").click();
            });
        }
        await capture(`reference-${viewport.width}`);
        if (viewport.width <= 650) {
            await page.locator("#referenceGameControls > summary").click();
            assert.equal(await page.locator("#referenceReady").isVisible(), true);
            assert.equal(await page.locator("#referenceLoad").isVisible(), true);
            await capture(`reference-${viewport.width}-controls`);
        }
        await small.close();
    }
    const hidden = await browser.newContext({ viewport: { width: 1115, height: 750 } });
    const hiddenPage = await hidden.newPage();
    await hiddenPage.addInitScript(() => {
        window.requestAnimationFrame = () => 0;
        Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
    });
    await hiddenPage.goto(url, { waitUntil: "domcontentloaded" });
    await until(async () => hiddenPage.evaluate(() => {
        const state = window.__snakeDebug?.().state;
        return state && !document.getElementById("referenceReady").disabled
            && [...document.querySelectorAll("#referenceStats dd")].every((node) => node.textContent !== "...")
            && window.__snakeDebug().renderer.neurons.dimensions.length === state.network.dimensions.length
            && document.getElementById("referenceNeurons").getAttribute("aria-label").includes("Double DQN");
    }), "hidden-page state, statistics and controls initialize without waiting for a rendered frame");
    const hiddenSteps = (await api("/api/state")).totalSteps;
    await sleep(250);
    assert.equal((await api("/api/state")).totalSteps, hiddenSteps);
    await hidden.close();
    check("Hidden integrated tabs initialize HUD and controls without RAF; hidden simulation stays suspended");
    report.status = "PASS";
    check("The reference renderer retains its original native viewport sizing without alternate cameras or styling");
} catch (error) {
    report.status = "FAIL";
    report.failure = error.stack;
    console.error(error.stack);
    process.exitCode = 1;
    if (page && !page.isClosed()) await page.screenshot({ path: path.join(evidence, "failure.png") }).catch(() => {});
} finally {
    if (context) await context.close();
    if (browser) await browser.close();
    if (server && server.exitCode === null) {
        execFileSync("powershell.exe", ["-NoProfile", "-Command", `Stop-Process -Id ${server.pid} -Force -ErrorAction Stop`]);
        await Promise.race([new Promise((resolve) => server.once("exit", resolve)), sleep(3000)]);
    }
    await fs.writeFile(path.join(evidence, "server.log"), log);
    await fs.writeFile(path.join(evidence, "report.json"), JSON.stringify(report, null, 2));
    console.log(`Evidence: ${evidence}`);
}
