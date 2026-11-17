import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import * as THREE from "../static/vendor/three.module.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replaceAll(/[:.]/g, "-");
const evidence = path.resolve(process.env.ARENA_EVIDENCE || path.join(root, ".runtime-proof", "readable-browser", stamp));
const python = process.env.ARENA_PYTHON || path.join(root, ".venv", "Scripts", "python.exe");
const expectedEnvironment = "survival-readable-v6";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const report = { status: "RUNNING", assertions: [], captures: [], motion: [], terminal: [], recordings: [], errors: [], expectedError: false };
let server;
let browser;
let context;
let page;
let serverLog = "";
let processFailure;
await fs.mkdir(evidence, { recursive: true });

async function unusedPort() {
    const listener = net.createServer();
    await new Promise((resolve, reject) => listener.once("error", reject).listen(0, "127.0.0.1", resolve));
    const port = listener.address().port;
    await new Promise((resolve) => listener.close(resolve));
    return port;
}

const port = await unusedPort();
const baseURL = `http://127.0.0.1:${port}`;

async function until(work, description, timeout = 20_000, interval = 60) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
        const result = await work();
        if (result) return result;
        await sleep(interval);
    }
    throw new Error(`Timed out: ${description}`);
}

async function api(endpoint, body) {
    const response = await fetch(`${baseURL}${endpoint}`, body === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    assert(response.ok, `${endpoint}: ${response.status}: ${await response.clone().text()}`);
    return response.json();
}

async function debug() { return page.evaluate(() => window.__snakeDebug?.()); }
async function state() { return (await debug()).state; }
function presentation(snapshot) {
    assert(snapshot.renderer.presentation, "Renderer must expose its actual presentation lifecycle.");
    return snapshot.renderer.presentation;
}
async function idle() {
    await until(async () => (await debug()).lifecycle.pendingControls === 0, "serialized controls settle");
}
function check(name, values = {}) {
    report.assertions.push({ name, ...values });
    console.log(`PASS ${name}`);
}
async function openMenu() {
    if (await page.locator("#cameraMenuToggle").getAttribute("aria-expanded") !== "true") {
        await page.locator("#cameraMenuToggle").click();
    }
}
async function closeMenu() {
    if (await page.locator("#cameraMenuToggle").getAttribute("aria-expanded") === "true") {
        await page.locator("#cameraMenuToggle").click();
    }
}
async function skipRecap() {
    if (presentation(await debug()).active) {
        await page.locator("#skipRecapBtn").click();
        await until(async () => !presentation(await debug()).active, "Skip releases terminal presentation");
        await idle();
    }
}
async function pause() {
    for (let attempt = 0; attempt < 4; attempt += 1) {
        await idle();
        await skipRecap();
        await idle();
        if ((await state()).paused && !presentation(await debug()).active) {
            await closeMenu();
            return;
        }
        await openMenu();
        await page.locator("#pauseBtn").click();
        await idle();
    }
    throw new Error("Could not establish an acknowledged paused non-recap state through the public controls.");
}
async function resume() {
    await idle();
    if ((await state()).paused) {
        await openMenu();
        await page.locator("#pauseBtn").click();
        await until(async () => !(await state()).paused || presentation(await debug()).active, "resume acknowledged");
    }
    await idle();
    await closeMenu();
}
async function select(id, value) {
    await idle();
    await openMenu();
    await page.locator(id).selectOption(value);
    await idle();
    await closeMenu();
}
async function resetLearning() {
    page.once("dialog", (dialog) => dialog.accept());
    await page.locator("#resetBtn").click();
    await idle();
}
async function measureTrainingRate(value, expectedStepsPerSecond) {
    await pause();
    assert.equal(await page.locator("#speedSelect").isVisible(), true,
        "Training speed must be discoverable without opening the camera menu.");
    await page.locator("#speedSelect").selectOption(value);
    await idle();
    const initialSteps = (await state()).totalSteps;
    await resume();
    await until(async () => (await state()).totalSteps > initialSteps,
        "the selected live rate starts actual simulation", 5000);
    const started = performance.now();
    const before = await api("/api/state");
    await sleep(2400);
    const after = await api("/api/state");
    const elapsedSeconds = (performance.now() - started) / 1000;
    assert.equal(after.paused, false, "A live-rate sample cannot be a paused terminal/recap interval.");
    const steps = after.totalSteps - before.totalSteps;
    const learnedSteps = after.trainingSteps - before.trainingSteps;
    assert.equal(learnedSteps, steps, "The selected rate must govern actual learning, not just displayed motion.");
    assert(Math.abs(steps - elapsedSeconds * expectedStepsPerSecond) <= 1.5,
        `${value}: ${steps} actual Python steps in ${elapsedSeconds.toFixed(2)}s must track ${expectedStepsPerSecond} steps/s.`);
    await pause();
    return { value, expectedStepsPerSecond, steps, learnedSteps, elapsedSeconds,
        observedStepsPerSecond: steps / elapsedSeconds };
}
async function camera(mode) {
    await openMenu();
    if (["film", "orbit", "chase", "cine-chase", "cinema"].includes(mode)) {
        await page.locator("#advancedCameraOptions").evaluate((element) => { element.open = true; });
    }
    await page.locator(`[data-camera-view="${mode}"]`).click();
    await closeMenu();
    await sleep(100);
}
async function capture(name) {
    if (await page.locator("#cameraMenuToggle").getAttribute("aria-expanded") === "false") {
        await page.waitForFunction(() => getComputedStyle(document.getElementById("cameraMenu")).opacity === "0", null, { polling: 50, timeout: 1500 });
    }
    const before = await debug();
    const file = `${name}.png`;
    await page.screenshot({ path: path.join(evidence, file), fullPage: true });
    const after = await debug();
    report.captures.push({
        file, viewport: page.viewportSize(),
        before: { episode: before.state.episode, steps: before.state.totalSteps, phase: presentation(before).phase, board: before.state.board },
        after: { episode: after.state.episode, steps: after.state.totalSteps, phase: presentation(after).phase },
        renderer: before.renderer,
    });
}
function frozenState(snapshot) {
    return JSON.stringify(["board", "totalSteps", "trainingSteps", "gradientSteps", "bufferSize", "policyEpsilon"].map((key) => snapshot[key]));
}
function contactCoverage(snapshot, aspect) {
    const d = snapshot.renderer;
    const contact = snapshot.state.board.contact;
    if (!contact?.attacker) return null;
    const actors = d.renderedActors;
    const attacker = actors.enemies.find((actor) => actor.id === contact.attacker.id);
    assert(attacker, "The recorded attacker must be part of the actual rendered scene.");
    assert.equal(d.cameraType, "perspective");
    const lens = new THREE.PerspectiveCamera(d.cameraFov, aspect, 0.1, 1000);
    lens.position.fromArray(d.cameraPosition);
    lens.quaternion.fromArray(d.cameraQuaternion);
    lens.zoom = d.cameraZoom;
    lens.updateProjectionMatrix();
    lens.updateMatrixWorld();
    const projected = [];
    for (const point of [...actors.player, ...attacker.points]) {
        for (const x of [-0.45, 0.45]) for (const y of [-0.45, 0.45]) for (const z of [-0.45, 0.45]) {
            projected.push(new THREE.Vector3(point[0] + x, point[1] + y, point[2] + z).project(lens));
        }
    }
    const minX = Math.min(...projected.map((p) => p.x));
    const maxX = Math.max(...projected.map((p) => p.x));
    const minY = Math.min(...projected.map((p) => p.y));
    const maxY = Math.max(...projected.map((p) => p.y));
    return { width: (maxX - minX) / 2, height: (maxY - minY) / 2, minX, maxX, minY, maxY };
}
async function startRecording() {
    await page.evaluate(() => {
        const source = document.getElementById("gameCanvas");
        const canvas = document.createElement("canvas");
        canvas.width = source.width;
        canvas.height = source.height;
        const ctx = canvas.getContext("2d");
        const stream = canvas.captureStream(20);
        const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8" });
        const chunks = [];
        recorder.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
        const draw = () => {
            ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
            const overlay = document.getElementById("terminalOverlay");
            if (overlay && !overlay.hidden) {
                const title = document.getElementById("terminalTitle").textContent;
                const detail = document.getElementById("terminalDetail").textContent;
                const phase = document.getElementById("terminalPhase").textContent;
                const size = Math.max(14, Math.round(canvas.width / 48));
                ctx.fillStyle = "rgba(5,10,14,.94)";
                ctx.fillRect(12, canvas.height - size * 5, canvas.width - 24, size * 4.5);
                ctx.font = `bold ${size}px Trebuchet MS`;
                ctx.fillStyle = "#b8efd1";
                ctx.fillText(`${phase} / ${title}`, 24, canvas.height - size * 3);
                ctx.font = `${Math.max(12, size - 2)}px Trebuchet MS`;
                ctx.fillStyle = "#f3f1ec";
                ctx.fillText(detail, 24, canvas.height - size * 1.5, canvas.width - 48);
            }
        };
        const timer = setInterval(draw, 50);
        draw();
        recorder.start(250);
        window.__readableRecording = { recorder, stream, chunks, timer };
    });
}
async function stopRecording(name) {
    const recording = await page.evaluate(async () => {
        const current = window.__readableRecording;
        if (!current) throw new Error("No recording was started.");
        await new Promise((resolve) => {
            current.recorder.addEventListener("stop", resolve, { once: true });
            current.recorder.stop();
        });
        clearInterval(current.timer);
        current.stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(current.chunks, { type: "video/webm" });
        const data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(",")[1]);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(blob);
        });
        delete window.__readableRecording;
        return { data, bytes: blob.size };
    });
    assert(recording.bytes > 5000, "The retained clip must contain actual encoded rendered frames.");
    await fs.writeFile(path.join(evidence, name), Buffer.from(recording.data, "base64"));
    const metadata = { file: name, bytes: recording.bytes, source: "Actual WebGL scene plus the actual public terminal captions; real policy playback, 20fps capture" };
    report.recordings.push(metadata);
    if (!report.recording) report.recording = metadata;
}
async function waitForRealTerminal(timeout = 40_000) {
    return until(async () => {
        const snapshot = await debug();
        report.motion.push({
            observedAt: Date.now(), episode: snapshot.state.episode, totalSteps: snapshot.state.totalSteps,
            boardSteps: snapshot.state.board.steps, head: snapshot.state.board.snake[0],
            paused: snapshot.state.paused, frames: snapshot.renderer.frames,
            camera: snapshot.renderer.cameraPosition, quaternion: snapshot.renderer.cameraQuaternion,
            lens: snapshot.renderer.cameraType, phase: presentation(snapshot).phase,
            objectives: snapshot.renderer.objectiveView,
        });
        return presentation(snapshot).active ? snapshot : null;
    }, "a real policy episode reaches a rendered terminal event", timeout, 80);
}

try {
    server = spawn(python, ["-B", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", String(port), "--no-access-log"], {
        cwd: root, env: { ...process.env, SNAKE_STORAGE_DIR: path.join(evidence, "storage"), SNAKE_SEED: "73" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    server.on("error", (error) => { processFailure = error; });
    server.stdout.on("data", (chunk) => { serverLog += chunk; });
    server.stderr.on("data", (chunk) => { serverLog += chunk; });
    await until(async () => {
        if (processFailure) throw processFailure;
        if (server.exitCode !== null) throw new Error(serverLog);
        try { return (await api("/api/health")).environmentVersion === expectedEnvironment; }
        catch { return false; }
    }, "owned current server readiness");
    assert.equal((await api("/api/health")).contactSchema, 1, "The current server must advertise the rich contact protocol.");
    report.baseURL = baseURL;
    report.source = {};
    for (const relative of ["main.py", "backend\\config.py", "backend\\environment.py", "backend\\session.py",
        "static\\index.html", "static\\styles.css", "static\\app.js", "static\\arena.js",
        "static\\director.js", "static\\character.js", "static\\presentation.js", "storage\\pretrained_3d.npz"]) {
        report.source[relative] = hash(await fs.readFile(path.join(root, relative)));
    }
    const readyHash = report.source["storage\\pretrained_3d.npz"];
    browser = await chromium.launch({ channel: process.env.ARENA_BROWSER || "msedge", headless: process.env.HEADLESS !== "0" });
    context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    page = await context.newPage();
    page.on("pageerror", (error) => report.errors.push({ kind: "page", message: error.message, expected: report.expectedError }));
    page.on("console", (message) => {
        if (message.type() === "error") report.errors.push({ kind: "console", message: message.text(), expected: report.expectedError });
    });
    page.on("requestfailed", (request) => report.errors.push({ kind: "network", message: `${request.url()} ${request.failure()?.errorText}`, expected: report.expectedError }));
    await page.goto(baseURL);
    await until(async () => (await debug())?.state && (await debug()).renderer.presentation, "public scene and presentation controller load");
    await pause();
    assert.equal(await page.title(), "AI Snake Training Room");
    assert.equal(await page.locator(".stat-card").count(), 6);
    assert.equal((await state()).network.qValues.length, 18);
    assert.equal((await debug()).renderer.cameraMode, "game");
    assert.equal((await debug()).renderer.cameraType, "orthographic");
    assert.equal((await debug()).renderer.actorStyle, "grid-blocks");
    assert(Object.values((await debug()).renderer.actorStyles).every((style) => style === "grid-blocks"),
        "Every live snake role must use the requested block representation.");
    const opening = await debug();
    assert.equal(opening.renderer.outlinedActors.length, Object.keys(opening.renderer.actorStyles).length);
    const goal = opening.state.board.foods.find((food) => food.type === "normal");
    assert.equal(opening.renderer.objectiveView.player.targetId, goal.id);
    const playerGuide = opening.renderer.objectiveView.guides.find((guide) => guide.id === "player");
    assert(playerGuide?.linkVisible, "A real visible food-goal link must be drawn, not merely described in a legend.");
    assert.equal(opening.renderer.objectiveView.guides.length, opening.state.board.enemies.length + 1,
        "Patrols/ambushers must also expose their real movement instead of remaining unexplained actors.");
    const half = opening.state.board.gridSize / 2;
    assert(playerGuide.to.every((value, i) => Math.abs(value - [goal.position.x, goal.position.y, goal.position.z][i] - 0.5 + half) < 0.0001),
        "The drawn goal guide must end at actual food, not a decorative unrelated position.");
    const center = opening.state.board.gridSize / 2;
    const centralCells = opening.state.board.obstacles.filter((point) =>
        [point.x, point.y, point.z].every((value) => Math.abs(value - center) <= 6));
    assert(centralCells.length >= 9, "Real central structures must be near the opening action, not only distant corners.");
    const rival = opening.state.board.enemies.find((enemy) => enemy.type === "rival");
    assert.equal(opening.renderer.objectiveView.enemies.find((enemy) => enemy.id === rival.id).targetId,
        rival.intent.targetId);
    assert.equal(await page.locator('[data-objective="player"]').isVisible(), true);
    assert.match(await page.locator("#roleLegend").innerText(), /YOU > FOOD/);
    assert.match(await page.locator("#roleLegend").innerText(), /stay alive/);
    assert.equal(await page.locator("#introToggle").isChecked(), false);
    assert.equal(await page.locator("#speedSelect").inputValue(), "rate-4");
    assert.equal(await page.locator("#speedSelect").isVisible(), true,
        "The slower live default must be available outside the camera menu.");
    await capture("01-open-chambers-game-view");
    check("Original Training Room opens a real six-axis arena with a composed orthographic Game View");
    check("Outlined snakes show a real food objective, authoritative rival target and actual central obstacles",
        { centralCells: centralCells.length, playerGoal: goal, rivalGoal: rival.intent });

    await startRecording();
    const stage = await page.locator("#gameCanvas").boundingBox();
    await resume();
    const terminal = await waitForRealTerminal();
    const exploring = report.motion.find((sample) => sample.objectives?.player.source === "exploration");
    assert(exploring && /Random:/.test(exploring.objectives.player.activity),
        "Actual random learning must be identified as exploration, not falsely described as a food-seeking plan.");
    const contact = terminal.state.board.contact;
    assert(contact || terminal.state.board.lastDeath === "time_limit", "Physical deaths must carry true contact data.");
    if (contact) {
        assert.equal(contact.outcome, "death");
        assert(contact.victim.before.length && contact.victim.attempted.length);
        assert(contact.position && Number.isFinite(contact.position.x));
        report.realContact = contact;
    }
    assert.equal(await page.locator("#terminalOverlay").isVisible(), true, "The cause must be visible in the actual page, not only in debug state.");
    const caption = `${await page.locator("#terminalTitle").innerText()} ${await page.locator("#terminalDetail").innerText()}`;
    const causeWords = {
        hunter: /hunter/i, patrol: /patrol/i, rival: /rival/i, ambush: /ambush/i,
        obstacle: /obstacle|structure|barrier/i, moving_wall: /wall|barrier/i, maze_wall: /wall|maze/i,
        wall: /wall|boundary/i, self: /self|own body/i, laser: /laser|gate/i,
        zone: /zone|safe|boundary/i, time_limit: /time|limit/i,
    };
    assert(causeWords[terminal.state.board.lastDeath]?.test(caption), `Caption must identify the actual terminal cause: ${caption}`);
    const observedPhases = new Set();
    let frozenChecks = 0;
    const terminalEpisode = terminal.state.episode;
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
        const before = await debug();
        const current = presentation(before);
        if (!current.active) break;
        if (!observedPhases.has(current.phase)) {
            observedPhases.add(current.phase);
            await capture(`terminal-${current.phase}`);
        }
        const live = await api("/api/state");
        report.terminal.push({
            observedAt: Date.now(), phase: current.phase, message: current.message,
            camera: before.renderer.cameraPosition, renderer: before.renderer,
            episode: live.episode, totalSteps: live.totalSteps, trainingSteps: live.trainingSteps,
            contact: live.board.contact, paused: live.paused,
            coverage: contactCoverage(before, stage.width / stage.height),
        });
        if (live.paused) {
            await sleep(80);
            const after = await debug();
            if (presentation(after).active && after.state.episode === terminalEpisode) {
                const persisted = await api("/api/state");
                if (persisted.episode === live.episode && persisted.paused) {
                    assert.equal(frozenState(persisted), frozenState(live), "Terminal presentation must freeze actual Python state and training");
                    frozenChecks += 1;
                }
            }
        }
        await sleep(40);
    }
    assert.equal(presentation(await debug()).active, false, "Automatic recap must finish and release its latch.");
    assert(observedPhases.has("recap"), `A clearly labelled real recap must be observable; saw ${[...observedPhases]}`);
    assert(observedPhases.has("aftermath"), "An unmistakable eliminated aftermath must remain visible before the rewind.");
    assert(observedPhases.has("impact") || observedPhases.has("defeat"), "The sequence must include a visible physical result, not only a recap label.");
    assert(frozenChecks > 0, "Measure an actual frozen presentation interval, not only its UI flag.");
    const renderedPoses = new Set(report.terminal.map((sample) => JSON.stringify(sample.renderer.playerPositions)));
    assert(renderedPoses.size > 1, "The recorded body must actually move during the contact/recap, not only change its face or caption.");
    const firstImpact = report.terminal.find((sample) => sample.phase === "impact" && sample.coverage);
    if (firstImpact && contact?.attacker && contact.attacker.before.length <= 4 && contact.victim.before.length <= 4) {
        assert(Math.max(firstImpact.coverage.width, firstImpact.coverage.height) >= 0.24,
            "A compact impact pair must be established at a readable scale, not merely pass an in-frame boolean.");
    }
    await until(async () => (await state()).episode > terminalEpisode || presentation(await debug()).active, "real playback resumes after the recap");
    await stopRecording("real-contact-and-recap.webm");
    await pause();
    await capture("02-after-impact-recovery");
    check("Actual collision has a readable physical sequence, labelled recap, frozen Python state and automatic recovery", {
        phases: [...observedPhases], frozenChecks, cause: terminal.state.board.lastDeath, kind: contact?.kind,
    });

    const beforeCamera = frozenState(await state());
    for (const mode of ["static", "game"]) {
        await camera(mode);
        assert.equal(frozenState(await state()), beforeCamera);
    }
    const normalSize = await page.locator("#gameCanvas").boundingBox();
    await page.locator("#expandArenaBtn").click();
    await until(async () => (await page.locator("#gameCanvas").boundingBox()).width > normalSize.width * 1.35, "expanded scene gains usable width");
    assert.equal(frozenState(await state()), beforeCamera);
    await capture("03-expanded-game-view");
    await page.locator("#expandArenaBtn").click();
    await until(async () => !(await debug()).lifecycle.arenaExpanded, "dashboard restores");
    check("View and size changes preserve the actual paused game, not just displayed counters");

    await page.locator("#readyWeightsBtn").click();
    await idle();
    await until(async () => (await state()).mode === "eval", "ready policy loads");
    assert.equal((await state()).epsilon, 0);
    const trainedSteps = (await state()).trainingSteps;
    await sleep(500);
    await pause();
    assert.equal((await state()).trainingSteps, trainedSteps);
    assert.equal((await state()).bufferSize, 0);
    await page.locator("#saveWeightsBtn").click();
    await idle();
    const checkpoint = path.join(evidence, "storage", "arena-3d-v3", expectedEnvironment, "manual.npz");
    const savedHash = hash(await fs.readFile(checkpoint));
    await page.locator("#readyWeightsBtn").click();
    await idle();
    await pause();
    assert.equal(hash(await fs.readFile(checkpoint)), savedHash);
    await page.locator("#loadWeightsBtn").click();
    await idle();
    assert.equal(await page.locator("#speedSelect").inputValue(), "rate-4",
        "Loading ready or saved weights must preserve the chosen live speed.");
    assert.match(await page.locator("#operationStatus").innerText(), /loaded/i);
    assert.equal(hash(await fs.readFile(path.join(root, "storage", "pretrained_3d.npz"))), readyHash);
    check("Ready policy, real saved checkpoint and legacy-compatible policy semantics are preserved");

    await resetLearning();
    await waitForRealTerminal();
    await idle();
    const oldEpisode = (await state()).episode;
    await page.locator("#skipRecapBtn").click();
    await until(async () => !presentation(await debug()).active || (await state()).episode > oldEpisode, "Skip recap resumes only the current generation");
    await pause();
    check("Skip ends the current recap without leaving playback locked");

    await select("#difficultySelect", "1");
    await select("#modeSelect", "train");
    await select("#speedSelect", "fast");
    const beforeFast = (await state()).trainingSteps;
    await resume();
    await until(async () => (await state()).trainingSteps >= beforeFast + 384, "real fast training progresses");
    assert.equal(presentation(await debug()).active, false, "Fast batches must not invent or wait for per-frame replays");
    await pause();
    await select("#speedSelect", "realtime");
    await select("#difficultySelect", "4");
    await resetLearning();
    await waitForRealTerminal();
    await idle();
    await resetLearning();
    await until(async () => {
        const snapshot = await debug();
        return snapshot.state.mode === "train" && snapshot.state.trainingSteps > 0
            && snapshot.state.trainingSteps < 4096 && snapshot.state.epsilon > 0.9;
    }, "Reset during recap resumes fresh-weight training");
    await pause();
    await capture("04-reset-recap-recovery");
    assert.equal(hash(await fs.readFile(path.join(root, "storage", "pretrained_3d.npz"))), readyHash);
    check("Fast training bypasses spectacle; Reset during recap cancels stale effects and resumes real basic-weight learning");

    await openMenu();
    await page.locator("#replayIntroBtn").click();
    await until(async () => (await debug()).lifecycle.introActive, "optional intro starts");
    await idle();
    const frozenIntro = frozenState(await api("/api/state"));
    await sleep(180);
    assert.equal(frozenState(await api("/api/state")), frozenIntro);
    await page.locator("#skipIntroBtn").click();
    await until(async () => !(await debug()).lifecycle.introActive, "intro skip releases its own generation");
    await pause();
    check("Optional intro remains separate from contact recap and preserves the exact game state");

    await select("#difficultySelect", "1");
    await resetLearning();
    await pause();
    const oneStepRate = await measureTrainingRate("rate-1", 1);
    const sixStepRate = await measureTrainingRate("rate-6", 6);
    assert(sixStepRate.observedStepsPerSecond > oneStepRate.observedStepsPerSecond * 3,
        "A faster selected rate must materially increase actual Python learning progress.");
    await resetLearning();
    await pause();
    assert.equal(await page.locator("#speedSelect").inputValue(), "rate-6");
    await openMenu();
    await page.locator("#newRunBtn").click();
    await idle();
    await pause();
    assert.equal(await page.locator("#speedSelect").inputValue(), "rate-6");
    await page.locator("#speedSelect").selectOption("rate-4");
    await idle();
    await select("#difficultySelect", "4");
    check("Visible speed control governs actual Python training at slow user-selected rates and survives Reset/New Run",
        { samples: [oneStepRate, sixStepRate] });

    for (const viewport of [{ width: 390, height: 844 }, { width: 280, height: 780 }]) {
        await resetLearning();
        await pause();
        await page.setViewportSize(viewport);
        await sleep(150);
        const layout = await page.evaluate(() => ({
            width: innerWidth, contentWidth: document.documentElement.scrollWidth,
            canvasWidth: document.getElementById("gameCanvas").clientWidth,
            canvasHeight: document.getElementById("gameCanvas").clientHeight,
        }));
        assert(layout.contentWidth <= layout.width + 1, "Narrow view must not hide the scene behind horizontal overflow.");
        assert(layout.canvasWidth >= viewport.width - 70 && layout.canvasHeight >= 300);
        await capture(`narrow-${viewport.width}`);
        const liveGoalView = await debug();
        const liveStage = await page.locator("#gameCanvas").boundingBox();
        const goalLegend = await page.locator("#roleLegend").boundingBox();
        const liveBody = liveGoalView.renderer.presentation.participantScreenBounds.find((actor) => actor.id === "player");
        assert(liveBody.bottom < goalLegend.y - liveStage.y,
            "The live objective HUD must not cover the player's actual head/body cells on a narrow view.");
        await page.locator("#expandArenaBtn").click();
        await sleep(150);
        assert.equal(await page.locator("#expandArenaBtn").getAttribute("aria-pressed"), "true");
        await capture(`narrow-expanded-${viewport.width}`);
        await page.keyboard.press("Escape");
        await until(async () => !(await debug()).lifecycle.arenaExpanded, "narrow Escape restores dashboard");
        check(`Readable controls and scene are available at ${viewport.width}px`, layout);
        if (viewport.width === 280) await startRecording();
        await resume();
        await waitForRealTerminal();
        await until(async () => {
            const current = presentation(await debug());
            return current.active && current.phase === "aftermath";
        }, "actual narrow-view impact reaches the readable eliminated aftermath", 15_000, 35);
        const narrow = await debug();
        const narrowStage = await page.locator("#gameCanvas").boundingBox();
        const overlay = await page.locator("#terminalOverlay").boundingBox();
        const pair = contactCoverage(narrow, narrowStage.width / narrowStage.height);
        if (pair) {
            assert(pair.minX > -1 && pair.maxX < 1, "The actual narrow contact pair must fit horizontally.");
            const lowestBodyPixel = (1 - pair.minY) * 0.5 * narrowStage.height;
            assert(lowestBodyPixel < overlay.y - narrowStage.y,
                "The cause overlay must not cover the actual struck/eliminated body in the narrow view.");
        }
        await capture(`narrow-aftermath-${viewport.width}`);
        await until(async () => !presentation(await debug()).active, "narrow recap finishes normally", 15_000);
        if (viewport.width === 280) await stopRecording("narrow-contact-and-recap.webm");
        await pause();
        check(`Actual impact, eliminated aftermath and recap remain visible above the ${viewport.width}px HUD`, { pair });
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    await resume();
    await sleep(250);
    await pause();
    assert.equal((await debug()).lifecycle.offline, false);
    check("Reduced-motion playback remains usable without removing outcome feedback");

    report.expectedError = true;
    await page.route("**/api/load-ready-weights", (route) => route.fulfill({
        status: 400, contentType: "application/json", body: JSON.stringify({ detail: "Expected negative UI test: ready weights unavailable" }),
    }));
    await page.locator("#readyWeightsBtn").click();
    await until(async () => (await debug()).lifecycle.offline, "explicit error state");
    assert.match(await page.locator("#operationStatus").innerText(), /unavailable/i);
    await page.unroute("**/api/load-ready-weights");
    await page.locator("#retryBtn").click();
    await idle();
    assert.equal((await debug()).lifecycle.offline, false);
    assert.equal((await state()).paused, true);
    report.expectedError = false;
    assert.equal((await debug()).lifecycle.maxRequestConcurrency, 1);
    assert.deepEqual(report.errors.filter((error) => !error.expected), []);
    report.status = "PASS";
    check("Errors recover paused and API/control requests stay serialized");
} catch (error) {
    report.status = "FAIL";
    report.failure = error.stack;
    console.error(error);
    process.exitCode = 1;
    if (page) {
        const recordingActive = await page.evaluate(() => window.__readableRecording?.recorder.state === "recording").catch(() => false);
        if (recordingActive) {
            try { await stopRecording("failed-contact-attempt.webm"); }
            catch (captureError) { report.recordingFailure = captureError.stack; }
        }
        await page.screenshot({ path: path.join(evidence, "failure.png"), fullPage: true }).catch(() => {});
        report.lastState = await debug().catch(() => null);
    }
} finally {
    if (browser) await browser.close();
    if (server && server.exitCode === null) {
        const stopped = new Promise((resolve) => server.once("exit", resolve));
        server.kill();
        await Promise.race([stopped, sleep(5000)]);
    }
    await fs.writeFile(path.join(evidence, "report.json"), JSON.stringify(report, null, 2));
    await fs.writeFile(path.join(evidence, "server.log"), serverLog);
    console.log(`Evidence: ${evidence}`);
}
