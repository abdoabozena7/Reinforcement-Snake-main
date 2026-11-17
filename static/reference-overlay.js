import * as THREE from "./vendor/three.module.js";
import { SIMULATION_RATES } from "./presentation.js?v=reference-live-2";
import { drawPolicyNetwork } from "./network-view.js";

const CAPTIONS = {
    player: "YOU",
    hunter: "wants to kill you",
    rival: "wants to eat your food",
    patrol: "Moving obstacle",
    ambush: "wants to kill you",
};
const clamp = THREE.MathUtils.clamp;
const overlaps = (a, b) => a.left < b.right + 5 && a.right > b.left - 5
    && a.top < b.bottom + 5 && a.bottom > b.top - 5;

export class ReferenceOverlay {
    constructor(canvas, client) {
        this.canvas = canvas;
        this.client = client;
        this.labels = new Map();
        this.sizes = new WeakMap();
        this.positions = [];
        this.hud = document.getElementById("scoreboard");
        this.hud.innerHTML = `
            <section class="reference-network" aria-label="Live neural network">
                <header>Neurons <span id="referenceNetworkSize">Connecting...</span></header>
                <canvas id="referenceNeurons" role="img" aria-label="Waiting for a real neural sample"></canvas>
                <small>Real activations and signed weights</small>
            </section>
            <div class="reference-telemetry">
                <div id="referenceActorBars" role="list" aria-label="Snake lengths and food scores"></div>
                <div id="referencePrimaryStats"></div>
                <p id="referenceModelStats">Waiting for training data...</p>
            </div>`;
        this.neurons = this.hud.querySelector("#referenceNeurons");
        this.networkSize = this.hud.querySelector("#referenceNetworkSize");
        this.modelStats = this.hud.querySelector("#referenceModelStats");
        this.actorList = this.hud.querySelector("#referenceActorBars");
        this.actorBars = new Map();
        this.lastNetwork = null;
        this.lastNetworkDraw = -Infinity;
        this.layer = document.createElement("div");
        this.layer.id = "reference-actor-labels";
        this.layer.setAttribute("aria-hidden", "true");
        this.controls = document.createElement("div");
        this.controls.id = "reference-speed-control";
        const label = document.createElement("label");
        label.htmlFor = "referenceSpeed";
        label.textContent = "Speed";
        this.speed = document.createElement("select");
        this.speed.id = "referenceSpeed";
        this.speed.title = "Actual simulation and learning speed. The reference camera is unchanged.";
        for (const [mode, rate] of Object.entries(SIMULATION_RATES)) {
            const option = document.createElement("option");
            option.value = mode;
            option.textContent = rate ? `${rate} steps / s${rate === 20 ? " (reference)" : ""}` : "Fast training";
            this.speed.append(option);
        }
        this.speed.value = client.pacer.mode;
        this.speed.addEventListener("change", () => client.setSpeed(this.speed.value));
        this.warning = document.createElement("span");
        this.warning.id = "obstacleWarning";
        this.warning.setAttribute("role", "status");
        this.warning.hidden = true;
        this.warning.textContent = "Obstacles changing soon";
        this.controls.append(label, this.speed, this.warning);
        this.statistics = document.createElement("dl");
        this.statistics.id = "referenceStats";
        this.statistics.setAttribute("aria-label", "Training statistics");
        this.values = new Map();
        for (const [key, title] of [
            ["episode", "Episode"], ["highScore", "High Score"], ["epsilon", "Epsilon"],
            ["currentScore", "Current Score"], ["avgScore", "Average Score"], ["bufferSize", "Replay Buffer"],
        ]) {
            const term = document.createElement("dt");
            term.textContent = title;
            const value = document.createElement("dd");
            value.dataset.stat = key;
            value.textContent = "...";
            this.values.set(key, value);
            const cell = document.createElement("div");
            cell.className = "reference-stat";
            cell.dataset.metric = key;
            cell.append(term, value);
            this.statistics.append(cell);
        }
        this.hud.querySelector("#referencePrimaryStats").append(this.statistics);
        this.gameControls = document.createElement("details");
        this.gameControls.id = "referenceGameControls";
        this.gameControls.open = !window.matchMedia("(max-width: 650px)").matches;
        this.gameControls.innerHTML = `
            <summary>Controls</summary>
            <div class="reference-control-fields">
                <label for="referenceCamera">View</label>
                <select id="referenceCamera">
                    <option value="orbit">Original Orbit</option>
                    <option value="tail">Tail to head</option>
                    <option value="food">Food focus</option>
                </select>
                <label for="referenceMode">Policy</label>
                <select id="referenceMode"><option value="train">Train</option><option value="eval">Test - no learning</option></select>
                <label for="referenceScenario">Scenario</label>
                <select id="referenceScenario">
                    <option value="curriculum">Auto curriculum</option>
                    <option value="1">1 - Collect &amp; Survive</option>
                    <option value="2">2 - Moving Danger</option>
                    <option value="3">3 - The Hunter</option>
                    <option value="4">4 - Risk &amp; Reward</option>
                    <option value="5">5 - Survival Arena</option>
                    <option value="6">6 - Master Mode</option>
                    <option value="full">Full Survival</option>
                </select>
                <div class="reference-actions">
                    <button id="referencePause" type="button">Pause</button>
                    <button id="referenceNewRun" type="button">New Run</button>
                    <button id="referenceReady" class="reference-wide" type="button">Load Best / Ready</button>
                    <button id="referenceLoad" class="reference-wide" type="button">Load Saved / Best</button>
                    <button id="referenceSave" type="button">Save Weights</button>
                    <button id="referenceReset" type="button">Reset Weights</button>
                    <button id="referenceCheckBest" class="reference-wide" type="button">Evaluate &amp; Keep Best</button>
                </div>
            </div>`;
        this.camera = this.gameControls.querySelector("#referenceCamera");
        this.mode = this.gameControls.querySelector("#referenceMode");
        this.scenario = this.gameControls.querySelector("#referenceScenario");
        this.mode.addEventListener("change", () => void client.configure({ mode: this.mode.value }));
        this.scenario.addEventListener("change", () => {
            const value = this.scenario.value;
            void client.configure({
                level: value === "full" ? 6 : value === "curriculum" ? 1 : Number(value),
                autoCurriculum: value === "curriculum", fullSurvival: value === "full",
            });
        });
        for (const [id, action] of [
            ["referencePause", () => client.togglePause()], ["referenceNewRun", () => client.configure({ newEpisode: true })],
            ["referenceReady", () => client.ready()], ["referenceLoad", () => client.load()],
            ["referenceSave", () => client.save()], ["referenceReset", () => this.reset()],
            ["referenceCheckBest", () => client.checkBest()],
        ]) this.gameControls.querySelector(`#${id}`).addEventListener("click", () => void action());
        this.viewStatus = document.createElement("span");
        this.viewStatus.id = "referenceViewStatus";
        this.viewStatus.textContent = "Original Orbit";
        this.bestStatus = document.createElement("p");
        this.bestStatus.id = "referenceBestStatus";
        this.bestStatus.setAttribute("role", "status");
        this.status = document.createElement("p");
        this.status.id = "referenceStatus";
        this.status.setAttribute("role", "status");
        this.status.textContent = "Connecting to Python...";
        this.reconnect = document.createElement("button");
        this.reconnect.id = "referenceReconnect";
        this.reconnect.type = "button";
        this.reconnect.textContent = "Reconnect paused";
        this.reconnect.hidden = true;
        this.reconnect.addEventListener("click", () => void client.reconnect());
        this.controls.append(this.viewStatus, this.gameControls, this.bestStatus, this.status, this.reconnect);
        document.body.append(this.layer, this.controls);
        this.syncControls();
    }

    reset() {
        if (window.confirm("Reset training? Current-format checkpoints will be archived. Older checkpoints and ready weights will be kept.")) {
            return this.client.reset();
        }
    }

    syncControls() {
        const { state, offline, pendingControls, error } = this.client;
        const busy = !state || pendingControls > 0;
        this.controls.setAttribute("aria-busy", String(busy));
        for (const control of this.gameControls.querySelectorAll("button, select")) {
            if (control === this.camera) continue;
            if (control.disabled !== Boolean(busy || offline)) control.disabled = Boolean(busy || offline);
        }
        this.reconnect.hidden = !offline;
        this.reconnect.disabled = pendingControls > 0;
        const message = offline ? `Python error: ${error}. Reconnect paused to continue.`
            : busy ? "Applying control..." : this.client.notice
                ?? `${state.mode === "eval" ? "Test - no learning" : "Training"} | ${state.levelName}${state.paused ? " | Paused" : ""}`;
        if (this.status.textContent !== message) this.status.textContent = message;
        const role = offline ? "alert" : "status";
        if (this.status.getAttribute("role") !== role) this.status.setAttribute("role", role);
        if (!state) return;
        const best = state.bestPolicy;
        const bestMessage = best ? `Best: ${best.source === "trained" ? "improved policy" : "Ready"} | ${best.message}`
            : "Best service unavailable: restart this preview.";
        if (this.bestStatus.textContent !== bestMessage) this.bestStatus.textContent = bestMessage;
        this.bestStatus.dataset.status = best?.status || "error";
        const checkBest = this.gameControls.querySelector("#referenceCheckBest");
        checkBest.disabled = Boolean(busy || offline || !best || best.status === "checking" || state.gradientSteps < 1);
        const checkText = best?.status === "checking" ? "Checking Best..." : "Evaluate & Keep Best";
        if (checkBest.textContent !== checkText) checkBest.textContent = checkText;
        const pause = this.gameControls.querySelector("#referencePause");
        const pauseText = this.client.lifecycle.desiredPaused ? "Resume" : "Pause";
        if (pause.textContent !== pauseText) pause.textContent = pauseText;
        if (!busy) {
            const scenario = state.fullSurvival ? "full" : state.autoCurriculum ? "curriculum" : String(state.level);
            if (document.activeElement !== this.mode) this.mode.value = state.mode;
            if (document.activeElement !== this.scenario) this.scenario.value = scenario;
        }
        if (state === this.lastState) return;
        this.lastState = state;
        this.updateActors(state);
        const reward = state.board.trainingReward?.total;
        const loss = state.loss === null ? "awaiting update" : state.loss.toFixed(3);
        this.modelStats.textContent = `Steps ${state.trainingSteps.toLocaleString("en-US")} | Updates ${state.gradientSteps.toLocaleString("en-US")} | Loss ${loss} | Reward ${typeof reward === "number" ? reward.toFixed(2) : "..."}`;
        const dimensions = `${state.network.dimensions.join(" / ")} - sampled`;
        if (this.networkSize.textContent !== dimensions) this.networkSize.textContent = dimensions;
        for (const [key, node] of this.values) {
            const value = key === "epsilon" ? state[key].toFixed(3) : key === "avgScore" ? state[key].toFixed(2)
                : state[key].toLocaleString("en-US");
            if (node.textContent !== value) node.textContent = value;
        }
    }

    updateActors(state) {
        const actors = [{ id: "player", type: "player", snake: state.board.snake, resources: state.currentScore },
            ...state.board.enemies];
        const names = { player: "YOU", hunter: "HUNTER", rival: "RIVAL", patrol: "OBSTACLE", ambush: "AMBUSH" };
        const active = new Set(actors.map((actor) => actor.id));
        for (const [id, bar] of this.actorBars) if (!active.has(id)) {
            bar.root.remove();
            this.actorBars.delete(id);
        }
        for (const actor of actors) {
            let bar = this.actorBars.get(actor.id);
            if (!bar) {
                const root = document.createElement("div");
                root.className = "reference-actor-meter";
                root.dataset.actor = actor.id;
                root.dataset.role = actor.type;
                root.setAttribute("role", "listitem");
                root.innerHTML = `<div><strong></strong><span class="reference-length"></span><small></small></div>
                    <div class="reference-length-track"><span></span></div>`;
                bar = { root, name: root.querySelector("strong"), length: root.querySelector(".reference-length"),
                    score: root.querySelector("small"), fill: root.querySelector(".reference-length-track span") };
                this.actorBars.set(actor.id, bar);
                this.actorList.append(root);
            }
            const length = actor.snake.length;
            const signature = `${actor.type}:${length}:${actor.resources}`;
            if (bar.signature === signature) continue;
            bar.signature = signature;
            bar.name.textContent = names[actor.type];
            bar.length.textContent = `${length} cells`;
            bar.score.textContent = actor.type === "player" || actor.type === "rival" ? `Score ${actor.resources}` : "";
            bar.fill.style.width = `${Math.min(100, length / 32 * 100)}%`;
            bar.root.setAttribute("aria-label", `${names[actor.type]}, length ${length} cells${bar.score.textContent ? `, ${bar.score.textContent}` : ""}. Bar scale32cells.`);
        }
    }

    drawNeurons(now = performance.now()) {
        const network = this.client.state?.network;
        const size = `${this.neurons.clientWidth}:${this.neurons.clientHeight}`;
        const resized = size !== this.networkDrawSize;
        if (!resized && (network === this.lastNetwork
            || this.lastNetwork !== null && !document.hidden && now - this.lastNetworkDraw < 100)) return;
        drawPolicyNetwork(this.neurons, network, true);
        this.lastNetwork = network;
        this.lastNetworkDraw = now;
        this.networkDrawSize = size;
        if (network) this.neurons.setAttribute("aria-label",
            `${network.dimensions.join(" by ")} Double DQN. Real sampled activations and signed weights; last action ${network.lastAction ?? "not selected"}.`);
    }

    update(camera, actors) {
        this.syncControls();
        this.drawNeurons();
        if (document.activeElement !== this.speed && this.speed.value !== this.client.pacer.mode) {
            this.speed.value = this.client.pacer.mode;
        }
        const change = this.client.state?.board.obstacleChange;
        this.warning.hidden = !change;
        this.warning.title = change ? `${change.ticksUntilChange} simulation ticks remaining` : "";
        const viewport = this.canvas.getBoundingClientRect();
        const occupied = [this.hud.getBoundingClientRect(), this.controls.getBoundingClientRect()];
        const active = new Set(actors.map((actor) => actor.id));
        this.positions = [];
        for (const [id, label] of this.labels) if (!active.has(id)) {
            label.remove();
            this.labels.delete(id);
        }
        const ordered = [...actors].sort((a, b) => ["player", "hunter", "rival", "ambush", "patrol"].indexOf(a.type)
            - ["player", "hunter", "rival", "ambush", "patrol"].indexOf(b.type));
        for (const actor of ordered) {
            let label = this.labels.get(actor.id);
            if (!label) {
                label = document.createElement("span");
                label.className = "snake-caption";
                label.dataset.actor = actor.id;
                label.dataset.role = actor.type;
                label.textContent = CAPTIONS[actor.type];
                this.layer.append(label);
                this.labels.set(actor.id, label);
            }
            const anchor = actor.meshes[1].position.clone().project(camera);
            const visible = anchor.z >= -1 && anchor.z <= 1 && Math.abs(anchor.x) <= 1 && Math.abs(anchor.y) <= 1;
            label.hidden = !visible;
            if (!visible) continue;
            if (!this.sizes.has(label)) this.sizes.set(label, { width: label.offsetWidth, height: label.offsetHeight });
            const { width, height } = this.sizes.get(label);
            const anchorX = viewport.left + (anchor.x + 1) * viewport.width / 2;
            const anchorY = viewport.top + (1 - anchor.y) * viewport.height / 2;
            const x = clamp(anchorX, width / 2 + 5, innerWidth - width / 2 - 5);
            const candidates = [-18, -46, 28, -74, 56, -102, 84].map((offset) => {
                const bottom = clamp(anchorY + offset, viewport.top + height + 5, viewport.bottom - 8);
                return { left: x - width / 2, right: x + width / 2, top: bottom - height, bottom };
            });
            let box = candidates.find((candidate) => !occupied.some((other) => overlaps(candidate, other)));
            if (!box) {
                for (let bottom = viewport.top + height + 6; bottom < viewport.bottom; bottom += height + 8) {
                    const candidate = { left: x - width / 2, right: x + width / 2, top: bottom - height, bottom };
                    if (!occupied.some((other) => overlaps(candidate, other))) { box = candidate; break; }
                }
            }
            box ||= candidates[0];
            label.style.transform = `translate(${x.toFixed(2)}px, ${box.bottom.toFixed(2)}px) translate(-50%, -100%)`;
            occupied.push(box);
            this.positions.push({ id: actor.id, text: label.textContent, anchor: [anchorX, anchorY], box });
        }
    }

    debug() { return this.positions; }

    neuralDebug() {
        return { dimensions: this.lastNetwork?.dimensions || [],
            sampledNodes: this.lastNetwork?.layers.map((layer) => layer.length) || [],
            lastAction: this.lastNetwork?.lastAction ?? null, drawnAt: this.lastNetworkDraw };
    }
}
