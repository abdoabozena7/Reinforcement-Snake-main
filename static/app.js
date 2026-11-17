import { Snake3DRenderer } from "./arena.js?v=readable-v6";
import { ContactPresentation, PlaybackLifecycle, SimulationPacer, SIMULATION_RATES } from "./presentation.js?v=readable-v6";
import { drawPolicyNetwork } from "./network-view.js";

const MODEL_VERSION = "arena-3d-v3";
const ENVIRONMENT_VERSION = "survival-readable-v6";
const networkCanvas = document.getElementById("networkCanvas");
const networkCtx = networkCanvas.getContext("2d");
const cameraControl = document.getElementById("cameraControl");
const cameraMenu = document.getElementById("cameraMenu");
const cameraMenuToggle = document.getElementById("cameraMenuToggle");
const cameraOptions = [...document.querySelectorAll(".camera-option")];
const slowModeToggle = document.getElementById("slowModeToggle");
const status = document.getElementById("operationStatus");
const serverControls = [...document.querySelectorAll(".action-row button, #pauseBtn, #newRunBtn, #difficultySelect, #modeSelect, #replayIntroBtn, #introToggle")];
const pacer = new SimulationPacer();
const lifecycle = new PlaybackLifecycle();
const presentation = new ContactPresentation({ onFinish: (result) => { void finishTerminal(result); } });
let lastState = null;
let game3D = null;
let pendingControls = 0;
let active = true;
let offline = false;
let notice = "";
let speed = pacer.mode;
let previousRegularSpeed = speed;
let requestConcurrency = 0;
let maxRequestConcurrency = 0;
let cinematic = null;
let introGeneration = 0;
let introEnabled = false;
let arenaExpanded = false;
let terminalSummary = null;

class SupersededOperation extends Error {}

function ensureTerminalOverlay() {
    let root = document.getElementById("terminalOverlay");
    if (!root) {
        root = document.createElement("section");
        root.id = "terminalOverlay";
        root.className = "terminal-overlay";
        root.hidden = true;
        root.setAttribute("role", "region");
        root.setAttribute("aria-labelledby", "terminalTitle");
        document.querySelector(".game-body").append(root);
    }
    let copy = root.querySelector("div");
    if (!copy) { copy = document.createElement("div"); root.prepend(copy); }
    for (const [id, tag] of [["terminalPhase", "span"], ["terminalTitle", "strong"], ["terminalDetail", "p"], ["contactStatus", "span"], ["skipRecapBtn", "button"]]) {
        if (document.getElementById(id)) continue;
        const element = document.createElement(tag);
        element.id = id;
        if (tag === "button") {
            element.type = "button";
            element.className = "btn btn-secondary";
            element.textContent = "Skip recap";
            root.append(element);
        } else copy.append(element);
    }
    document.getElementById("terminalTitle").setAttribute("aria-live", "polite");
    return root;
}

const terminalOverlay = ensureTerminalOverlay();

function flashGreen(element) {
    element.classList.remove("flash-green");
    void element.offsetWidth;
    element.classList.add("flash-green");
}

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(message, error = false) {
    status.textContent = message;
    status.parentElement.classList.toggle("has-error", error);
    status.setAttribute("role", error ? "alert" : "status");
}

function showError(error) {
    if (error instanceof SupersededOperation) return;
    offline = true;
    pacer.suspend();
    lifecycle.invalidate();
    cancelPresentations("offline", true);
    console.error(error);
    setStatus(`${error.message} Reconnect to continue safely.`, true);
    document.getElementById("retryBtn").hidden = false;
}

function syncControls() {
    const disabled = pendingControls > 0 || !lastState || offline;
    serverControls.forEach((control) => { control.disabled = disabled; });
    document.getElementById("speedSelect").value = speed;
    const suffix = cinematic ? " after intro" : presentation.active || lifecycle.terminal ? " after recap" : "";
    document.getElementById("pauseBtn").textContent = `${lifecycle.desiredPaused ? "Resume" : "Pause"}${suffix}`;
    document.getElementById("roleLegend").hidden = Boolean(cinematic || lastState?.introActive || presentation.active || lifecycle.terminal || terminalSummary);
    if (lastState) {
        document.getElementById("modeSelect").value = lastState.mode;
        document.getElementById("difficultySelect").value = lastState.fullSurvival ? "full"
            : lastState.autoCurriculum ? "curriculum" : String(lastState.level);
    }
}

function resizeNetworkCanvas() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    networkCanvas.width = Math.floor(networkCanvas.clientWidth * ratio);
    networkCanvas.height = Math.floor(networkCanvas.clientHeight * ratio);
    networkCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (lastState) drawNetwork(lastState.network);
}

function drawNetwork(network) {
    drawPolicyNetwork(networkCanvas, network);
}

function updateUI(state, { observe = false, snap = false } = {}) {
    const previous = lastState;
    lastState = state;
    if (!state.board.done) terminalSummary = null;
    const started = observe && !cinematic && !state.introActive && presentation.observe(state.board, {
        sampled: speed === "fast", episode: state.episode, waitForLock: true,
    });
    if (!observe && !presentation.active && !state.board.done && !cinematic && !state.introActive) {
        presentation.history.record(state.board, { sampled: speed === "fast", episode: state.episode });
    }
    const values = {
        episodes: state.episode, highScore: state.highScore, epsilon: Number(state.epsilon).toFixed(3),
        currentScore: state.currentScore, avgScore: Number(state.avgScore).toFixed(2), bufferSize: state.bufferSize,
    };
    for (const [id, value] of Object.entries(values)) document.getElementById(id).textContent = value;
    if (state.network) {
        const dimensions = state.network.dimensions.join(" / ");
        document.getElementById("networkSummary").textContent = dimensions;
        networkCanvas.setAttribute("aria-label", `${dimensions} policy. Sampled real activations and weights; all 18 action values. Last action: ${state.actionLabel}.`);
        drawNetwork(state.network);
    }
    game3D?.applyBoard(state.board, {
        paused: state.paused || state.introActive || presentation.active || Boolean(lifecycle.terminal),
        sampled: speed === "fast", interval: pacer.intervalMs || 250,
        snap: snap || Boolean(previous && state.episode !== previous.episode),
        decision: { mode: state.mode, isExploring: state.isExploring,
            actionLabel: state.actionLabel, lastAction: state.lastAction, epsilon: state.epsilon },
    });
    if (previous?.episode === state.episode && state.currentScore > previous.currentScore) {
        flashGreen(document.getElementById("gamePanel"));
        flashGreen(document.getElementById("currentScoreCard"));
    }
    if (previous && state.avgScore > previous.avgScore) flashGreen(document.getElementById("avgScoreCard"));
    if (!offline) setStatus(notice || `${state.mode === "eval" ? "Test" : "Train"} | ${state.levelName} | ${state.actionLabel} | ${state.board.contact?.outcome === "blocked" ? "Shield blocked the hit" : state.deathCause || "3D: X / Y / Z"}`);
    if (started) lockTerminal();
    syncControls();
}

async function requestJson(url, body) {
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 120000);
    requestConcurrency += 1;
    maxRequestConcurrency = Math.max(maxRequestConcurrency, requestConcurrency);
    try {
        const response = await fetch(url, {
            cache: "no-store",
            ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
            signal: abort.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(`${url}: ${typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail)} (${response.status})`);
        if (payload.modelVersion !== MODEL_VERSION || payload.environmentVersion !== ENVIRONMENT_VERSION) {
            throw new Error("The Python server is running an older game version. Restart the server, then reload this page");
        }
        return payload;
    } finally {
        clearTimeout(timeout);
        requestConcurrency -= 1;
    }
}

function cancelIntroPresentation() {
    introGeneration += 1;
    cinematic = null;
    game3D?.cancelIntro();
    document.getElementById("introOverlay").hidden = true;
}

function cancelPresentations(reason, clearHistory = false) {
    cancelIntroPresentation();
    presentation.cancel(reason, { clearHistory });
    game3D?.clearPresentation();
    terminalSummary = null;
    terminalOverlay.hidden = true;
}

async function api(context, url, body) {
    if (!context.current()) throw new SupersededOperation();
    let state;
    try { state = await requestJson(url, body); }
    catch (error) { if (!context.current()) throw new SupersededOperation(); throw error; }
    if (!context.current()) throw new SupersededOperation();
    return state;
}

function operation(work, { cancel = false, clearHistory = false, invalidate = true } = {}) {
    pacer.suspend();
    const epoch = invalidate ? lifecycle.invalidate() : lifecycle.generation;
    if (cancel) cancelPresentations("control", clearHistory);
    pendingControls += 1;
    notice = "";
    syncControls();
    return lifecycle.enqueue(work, epoch).catch(showError).finally(() => {
        pendingControls -= 1;
        syncControls();
    });
}

function playbackPreference() {
    return !lifecycle.desiredPaused;
}

function lockTerminal() {
    pacer.suspend();
    const token = lifecycle.lock(presentation.id);
    setCameraMenuOpen(false);
    void lifecycle.enqueue(async (context) => {
        if (!lifecycle.matches(token)) return;
        const state = await api(context, "/api/control", { intro: false, paused: true });
        if (!lifecycle.matches(token)) return;
        updateUI(state, { observe: false });
        presentation.unlock(token.id);
        syncControls();
    }, token.generation).catch(showError);
}

async function restorePreference(context, intro = false) {
    let state = await api(context, "/api/control", { intro, paused: lifecycle.desiredPaused });
    while (state.paused !== lifecycle.desiredPaused) {
        state = await api(context, "/api/control", { intro, paused: lifecycle.desiredPaused });
    }
    return state;
}

function finishTerminal(result) {
    const token = lifecycle.terminal;
    if (!token || token.id !== result.id || !lifecycle.matches(token)) return Promise.resolve();
    terminalSummary = { title: result.title, detail: result.detail };
    return lifecycle.enqueue(async (context) => {
        if (!lifecycle.matches(token)) return;
        const state = await restorePreference(context);
        if (!lifecycle.matches(token)) return;
        lifecycle.release(token);
        updateUI(state, { observe: false });
        renderPresentation(null);
    }, token.generation).catch(showError);
}

async function beginCinematic(context) {
    const id = ++introGeneration;
    updateUI(await api(context, "/api/control", { intro: true, paused: true }), { observe: false, snap: true });
    if (id !== introGeneration) return;
    setCameraMenuOpen(false);
    cinematic = { id, generation: context.generation, completing: false };
    game3D.playIntro(() => { void finishCinematic(id); });
    document.getElementById("introOverlay").hidden = false;
    document.getElementById("introTitle").textContent = "Your snake";
    syncControls();
}

function finishCinematic(id = cinematic?.id) {
    if (!cinematic || cinematic.id !== id || cinematic.completing) return Promise.resolve();
    cinematic.completing = true;
    game3D.cancelIntro();
    document.getElementById("introOverlay").hidden = true;
    return lifecycle.enqueue(async (context) => {
        if (!cinematic || cinematic.id !== id || introGeneration !== id) return;
        const state = await restorePreference(context);
        if (cinematic?.id === id) cinematic = null;
        updateUI(state, { observe: false });
    }, cinematic.generation).catch(showError);
}

function renderPresentation(frame, context = {}) {
    const showSummary = !frame && terminalSummary && lastState?.board.done && lifecycle.desiredPaused && !cinematic;
    terminalOverlay.hidden = !(frame || showSummary);
    const panel = document.getElementById("gamePanel");
    panel.classList.toggle("has-presentation", !terminalOverlay.hidden);
    panel.classList.toggle("is-replay", Boolean(frame?.replay));
    panel.classList.toggle("is-eliminated", Boolean(frame?.eliminated || showSummary));
    if (frame) {
        terminalOverlay.dataset.phase = frame.phase;
        document.getElementById("terminalPhase").textContent = frame.label;
        document.getElementById("terminalTitle").textContent = frame.title;
        document.getElementById("terminalDetail").textContent = frame.phase === "rewind"
            ? "Replaying the recorded approach. This round has already ended."
            : frame.phase === "aftermath" ? `${frame.detail} Your snake is eliminated.` : frame.detail;
        document.getElementById("contactStatus").textContent = frame.phase === "locking" ? "Stopping playback before the contact view…" : frame.replay
            ? `Recorded step ${frame.board.steps} · round ended at ${lastState.board.steps} · simulation paused` : frame.contact
            ? `Step ${frame.contact.step} · ${frame.contact.kind.replaceAll("-", " ")} · simulation paused`
            : "No physical contact · simulation paused";
        document.getElementById("skipRecapBtn").textContent = "Skip recap";
    } else if (showSummary) {
        terminalOverlay.dataset.phase = "summary";
        document.getElementById("terminalPhase").textContent = "ROUND ENDED · PAUSED";
        document.getElementById("terminalTitle").textContent = terminalSummary.title;
        document.getElementById("terminalDetail").textContent = terminalSummary.detail;
        document.getElementById("contactStatus").textContent = "Paused by your preference. Resume starts the next round.";
        document.getElementById("skipRecapBtn").textContent = "Resume next round";
    }
    const cutaway = document.getElementById("cutawayNotice");
    if (cutaway) cutaway.hidden = !context.cutaway || Boolean(cinematic);
    const viewStatus = document.getElementById("viewStatus");
    const mode = game3D?.cameraMode || "game";
    const phase = frame?.phase === "rewind" ? "REWIND" : frame?.eliminated ? "ELIMINATED" : frame?.replay ? "RECAP"
        : frame && !frame.contact ? "ROUND ENDED" : frame ? "CONTACT" : context.shieldBlocked ? "SHIELD BLOCK" : null;
    const height = lastState?.board.snake[0]?.y;
    viewStatus.textContent = phase ? `${frame ? "EVENT" : mode === "game" ? "GAME" : mode.toUpperCase()} VIEW · ${phase}`
        : speed === "fast" ? "FAST TRAINING · SAMPLED" : `${mode === "game" ? "GAME" : mode.toUpperCase()} VIEW${height == null ? "" : ` · Y ${height + 1}/${lastState.board.gridSize}`}`;
    viewStatus.hidden = Boolean(cinematic);
    if (cinematic) document.getElementById("introTitle").textContent = context.director?.title || "The opening";
}

function setCameraMenuOpen(open) {
    cameraControl.classList.toggle("is-open", open);
    cameraMenuToggle.setAttribute("aria-expanded", String(open));
    cameraMenu.setAttribute("aria-hidden", String(!open));
    cameraMenu.inert = !open;
}

function setSpeedValue(value) {
    speed = value;
    if (value !== "slow" && value !== "fast") previousRegularSpeed = value;
    pacer.suspend();
    pacer.setMode(value, performance.now());
    document.getElementById("speedSelect").value = value;
    slowModeToggle.setAttribute("aria-pressed", String(value === "slow"));
    slowModeToggle.textContent = "2 steps / second";
    slowModeToggle.title = value === "slow"
        ? `Restore ${SIMULATION_RATES[previousRegularSpeed]} training steps / second`
        : "Set actual simulation and learning to 2 steps / second";
}

function setSpeed(value) {
    if (!Object.hasOwn(SIMULATION_RATES, value) || value === speed) return;
    const wasFast = speed === "fast";
    setSpeedValue(value);
    if (wasFast || value === "fast") {
        void operation(async (context) => {
            updateUI(await restorePreference(context), { observe: false, snap: true });
        }, { cancel: true, clearHistory: true });
    } else {
        if (lastState && !cinematic && !presentation.active && !lifecycle.terminal && !lastState.introActive) {
            game3D?.applyBoard(lastState.board, { paused: lastState.paused, sampled: false, interval: pacer.intervalMs });
        }
        syncControls();
    }
}

function setArenaExpanded(expanded) {
    arenaExpanded = expanded;
    document.body.classList.toggle("arena-expanded", expanded);
    const button = document.getElementById("expandArenaBtn");
    const label = expanded ? "Restore dashboard" : "Expand arena";
    button.setAttribute("aria-label", label);
    button.setAttribute("aria-pressed", String(expanded));
    button.title = label;
}

function steppingBlocked() {
    return !active || !lastState || pendingControls || offline || cinematic || presentation.active || lifecycle.terminal
        || lastState.introActive || lastState.paused || lifecycle.desiredPaused || document.hidden;
}

async function loop() {
    while (active) {
        if (steppingBlocked()) {
            pacer.suspend();
            await delay(40);
            continue;
        }
        pacer.resume(performance.now());
        const wait = pacer.waitMs(performance.now());
        if (wait > 0) {
            await delay(Math.min(wait, 40));
            continue;
        }
        const token = lifecycle.generation;
        try {
            await lifecycle.enqueue(async (context) => {
                if (steppingBlocked()) { pacer.suspend(); return; }
                if (!pacer.tryStart(performance.now())) return;
                try {
                    updateUI(await api(context, speed === "fast" && lastState.mode === "train" ? "/api/train" : "/api/tick",
                        { steps: speed === "fast" ? 128 : 1 }), { observe: true });
                } finally { pacer.complete(performance.now()); }
            }, token);
        } catch (error) {
            showError(error);
        }
        await delay(0);
    }
}

document.getElementById("saveWeightsBtn").addEventListener("click", () => operation(async (context) => {
    const state = await api(context, "/api/save-weights", {});
    notice = state.logMessage;
    updateUI(state);
}, { invalidate: false }));
document.getElementById("loadWeightsBtn").addEventListener("click", () => {
    void operation(async (context) => {
        await api(context, "/api/control", { intro: false, paused: true });
        const state = await api(context, "/api/load-weights", {});
        notice = state.logMessage;
        updateUI(await restorePreference(context), { snap: true });
    }, { cancel: true, clearHistory: true });
});
document.getElementById("readyWeightsBtn").addEventListener("click", () => {
    lifecycle.desiredPaused = false;
    void operation(async (context) => {
        await api(context, "/api/control", { intro: false, paused: true });
        const state = await api(context, "/api/load-ready-weights", {});
        notice = state.logMessage;
        updateUI(state, { snap: true });
        if (introEnabled) await beginCinematic(context);
    }, { cancel: true, clearHistory: true });
});
document.getElementById("resetBtn").addEventListener("click", () => {
    if (!window.confirm("Reset training? Current-format checkpoints will be archived. Older checkpoints and ready weights will be kept.")) return;
    lifecycle.desiredPaused = false;
    void operation(async (context) => {
        await api(context, "/api/control", { intro: false, paused: true });
        const state = await api(context, "/api/reset", {});
        notice = state.logMessage;
        updateUI(state, { snap: true });
    }, { cancel: true, clearHistory: true });
});
document.getElementById("pauseBtn").addEventListener("click", () => {
    lifecycle.desiredPaused = !lifecycle.desiredPaused;
    if (cinematic || presentation.active || lifecycle.terminal) {
        syncControls();
        return;
    }
    void operation(async (context) => updateUI(await restorePreference(context), { observe: false }));
});
document.getElementById("newRunBtn").addEventListener("click", () => {
    void operation(async (context) => {
        updateUI(await api(context, "/api/control", { intro: false, newEpisode: true, paused: true }), { snap: true });
        if (introEnabled && speed !== "fast") await beginCinematic(context);
        else updateUI(await restorePreference(context));
    }, { cancel: true, clearHistory: true });
});
document.getElementById("replayIntroBtn").addEventListener("click", () => {
    void operation((context) => beginCinematic(context), { cancel: true });
});
document.getElementById("skipIntroBtn").addEventListener("click", () => { void finishCinematic(); });
document.getElementById("skipRecapBtn").addEventListener("click", () => {
    if (presentation.active) presentation.finish("skip");
    else if (terminalSummary && lifecycle.desiredPaused) {
        lifecycle.desiredPaused = false;
        void operation(async (context) => updateUI(await restorePreference(context), { observe: false }));
    }
});
document.getElementById("introToggle").addEventListener("change", (event) => {
    introEnabled = event.target.checked;
    if (!introEnabled && cinematic) void finishCinematic();
});
document.getElementById("difficultySelect").addEventListener("change", (event) => {
    const value = event.target.value;
    void operation(async (context) => {
        updateUI(await api(context, "/api/control", {
            level: value === "full" ? 6 : value === "curriculum" ? 1 : Number(value),
            autoCurriculum: value === "curriculum", fullSurvival: value === "full",
            intro: false, paused: !playbackPreference(),
        }), { snap: true });
    }, { cancel: true, clearHistory: true });
});
document.getElementById("modeSelect").addEventListener("change", (event) => {
    const mode = event.target.value;
    void operation(async (context) => updateUI(await api(context, "/api/control", {
        intro: false, mode, paused: !playbackPreference(),
    })), { cancel: true, clearHistory: true });
});
document.getElementById("speedSelect").addEventListener("change", (event) => {
    setSpeed(event.target.value);
});
slowModeToggle.addEventListener("click", () => setSpeed(speed === "slow" ? previousRegularSpeed : "slow"));
cameraMenuToggle.addEventListener("click", () => setCameraMenuOpen(!cameraControl.classList.contains("is-open")));
document.getElementById("expandArenaBtn").addEventListener("click", () => setArenaExpanded(!arenaExpanded));
cameraOptions.forEach((option) => option.addEventListener("click", () => {
    try {
        game3D?.setCameraMode(option.dataset.cameraView);
        if (cinematic) void finishCinematic();
        cameraOptions.forEach((button) => {
            const selected = button === option;
            button.classList.toggle("is-active", selected);
            button.setAttribute("aria-pressed", String(selected));
        });
        setCameraMenuOpen(false);
    } catch (error) {
        showError(error);
    }
}));
document.addEventListener("click", (event) => {
    if (!cameraControl.contains(event.target)) setCameraMenuOpen(false);
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        if (cinematic) void finishCinematic();
        else if (!cameraControl.classList.contains("is-open") && arenaExpanded) setArenaExpanded(false);
        setCameraMenuOpen(false);
        cameraMenuToggle.focus();
    }
});
document.addEventListener("visibilitychange", () => pacer.suspend());
document.getElementById("retryBtn").addEventListener("click", () => operation(async (context) => {
    offline = false;
    lifecycle.desiredPaused = true;
    notice = "";
    updateUI(await api(context, "/api/control", { intro: false, paused: true }), { observe: false, snap: true });
    document.getElementById("retryBtn").hidden = true;
}, { cancel: true, clearHistory: true }));

const networkObserver = new ResizeObserver(resizeNetworkCanvas);
networkObserver.observe(networkCanvas);
window.__snakeDebug = () => ({
    state: lastState, renderer: game3D?.debug(), lifecycle: {
        pendingControls, offline, speed, requestConcurrency, maxRequestConcurrency,
        simulationStepsPerSecond: pacer.stepsPerSecond, tickIntervalMs: pacer.intervalMs, pacing: pacer.debug(),
        introActive: Boolean(cinematic), introGeneration, arenaExpanded,
        presentationActive: presentation.active || Boolean(lifecycle.terminal), presentation: presentation.debug(),
        desiredPaused: lifecycle.desiredPaused, generation: lifecycle.generation,
    },
});
window.addEventListener("pagehide", () => {
    active = false;
    pacer.suspend();
    lifecycle.invalidate();
    presentation.cancel("pagehide", { clearHistory: true });
    game3D?.dispose();
    networkObserver.disconnect();
}, { once: true });

async function boot() {
    try {
        game3D = new Snake3DRenderer(document.getElementById("gameCanvas"), {
            labelLayer: document.getElementById("actorLabels"), presentation, onPresentationFrame: renderPresentation,
        });
    } catch (error) {
        showError(new Error(`3D rendering could not start: ${error.message}`));
        syncControls();
        return;
    }
    await operation(async (context) => {
        updateUI(await api(context, "/api/state"), { observe: false, snap: true });
        updateUI(await api(context, "/api/control", { intro: false, paused: true }), { observe: false });
        if (introEnabled) await beginCinematic(context);
        else updateUI(await restorePreference(context));
    });
    void loop();
}

setSpeedValue(speed);
void boot();
