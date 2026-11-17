const DIRECTIONS = ["N ↑", "E →", "S ↓", "W ←"];
const ABILITIES = ["MOVE", "DASH", "SHIELD"];
const HISTORY_LIMIT = 240;
const VISIBLE_EPISODES = 80;
const MEAN_WINDOW = 20;
const ESCAPE_DEFINITION = "An alerted hunter or ambusher came within 4 Manhattan cells. The player then stayed at least 8 cells from that same pursuer for 6 consecutive simulation ticks (0.72 seconds), with no other alerted hunter or ambusher within 4 cells. Counted once per encounter, followed by a 60-tick re-arm cooldown (7.2 seconds). Actual escape reward: 0.75.";
const ABILITY_DEFINITIONS = {
    DASH: "3 collision-checked cells; 24-tick cooldown (2.88 seconds).",
    SHIELD: "Protects one supported dangerous contact within 4 ticks (0.48 seconds); 60-tick cooldown (7.2 seconds). Never passes through solid walls.",
};

function numeric(value) {
    if (value === null || value === undefined || value === "") return null;
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
}

function fixed(value, places = 0, suffix = "") {
    const result = numeric(value);
    return result === null ? "—" : `${result.toLocaleString("en-US", { minimumFractionDigits: places, maximumFractionDigits: places })}${suffix}`;
}

function compact(value) {
    const result = numeric(value);
    if (result === null) return "—";
    if (result >= 1000000) return `${(result / 1000000).toFixed(1)}m`;
    if (result >= 10000) return `${(result / 1000).toFixed(1)}k`;
    return fixed(result);
}

export function contextKey(value, fallback = {}) {
    return `${value.mode ?? fallback.mode ?? "train"}:${value.level ?? fallback.level ?? 1}:${Boolean(value.fullSurvival ?? fallback.fullSurvival)}`;
}

export function episodeSeries(history, context, limit = VISIBLE_EPISODES) {
    const points = history.filter((point) => !point.mixedMode && contextKey(point) === contextKey(context) && numeric(point.score) !== null && numeric(point.episode) !== null);
    return points.map((point, index) => {
        const window = points.slice(Math.max(0, index - MEAN_WINDOW + 1), index + 1);
        return { ...point, mean: window.reduce((sum, item) => sum + Number(item.score), 0) / window.length };
    }).slice(-limit);
}

export function evaluationSeries(history, context) {
    return history.filter((point) => Number(point.level) === Number(context.level)
        && Boolean(point.fullSurvival) === Boolean(context.fullSurvival)
        && numeric(point.episodes) > 0
        && (numeric(point.successRate) !== null || numeric(point.successes) !== null)).slice(-40);
}

function sizeCanvas(canvas) {
    const width = Math.max(1, canvas.clientWidth || 1);
    const height = Math.max(1, canvas.clientHeight || 1);
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 1.5);
    const targetWidth = Math.round(width * ratio);
    const targetHeight = Math.round(height * ratio);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
    }
    const ctx = canvas.getContext("2d");
    if (ctx) {
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);
    }
    return { ctx, width, height };
}

function graphAxes(canvas, maxY, xLabelLeft, xLabelRight, percent = false) {
    const { ctx, width, height } = sizeCanvas(canvas);
    if (!ctx) return null;
    const bounds = { left: 36, top: 10, right: Math.max(37, width - 8), bottom: Math.max(12, height - 22) };
    ctx.font = "10px Consolas, monospace";
    ctx.lineWidth = 0.6;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let index = 0; index < 3; index += 1) {
        const fraction = index / 2;
        const y = bounds.bottom - fraction * (bounds.bottom - bounds.top);
        ctx.beginPath();
        ctx.moveTo(bounds.left, y);
        ctx.lineTo(bounds.right, y);
        ctx.strokeStyle = "#253247";
        ctx.stroke();
        ctx.fillStyle = "#9aafc3";
        const label = maxY * fraction;
        ctx.fillText(`${label >= 10 ? Math.round(label) : Number(label.toFixed(1))}${percent ? "%" : ""}`, bounds.left - 6, y);
    }
    ctx.fillStyle = "#9aafc3";
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillText(String(xLabelLeft ?? ""), bounds.left, height - 3);
    ctx.textAlign = "right";
    ctx.fillText(String(xLabelRight ?? ""), bounds.right, height - 3);
    return { ctx, ...bounds };
}

function drawSeries(graph, values, minX, maxX, maxY, color, lineWidth, dots = false) {
    if (!graph || !values.length) return;
    const { ctx, left, right, top, bottom } = graph;
    const x = (value) => maxX === minX ? (left + right) / 2 : left + (value - minX) / (maxX - minX) * (right - left);
    const y = (value) => bottom - Math.max(0, value) / maxY * (bottom - top);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    values.forEach((value, index) => {
        if (index === 0) ctx.moveTo(x(value.x), y(value.y));
        else ctx.lineTo(x(value.x), y(value.y));
    });
    ctx.stroke();
    if (dots || values.length === 1) {
        ctx.fillStyle = color;
        for (const point of values) {
            ctx.beginPath();
            ctx.arc(x(point.x), y(point.y), 2.2, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

export class Dashboard {
    constructor(root = document) {
        this.root = root;
        this.state = null;
        this.history = [];
        this.evaluations = [];
        this.actions = [];
        this.lastDraw = -Infinity;
        this.drawTimer = null;
        this.visibleScores = [];
        this.visibleEvaluations = [];
        this.elements = new Map();
        for (const element of root.querySelectorAll("[id]")) this.elements.set(element.id, element);
        const grid = this.element("actionGrid");
        for (let index = 0; index < 12; index += 1) {
            const cell = root.createElement("div");
            cell.className = "action-cell";
            cell.id = `qAction${index}`;
            cell.setAttribute("role", "listitem");
            const name = root.createElement("span");
            name.className = "action-name";
            name.textContent = `${ABILITIES[Math.floor(index / 4)]} ${DIRECTIONS[index % 4]}`;
            const q = root.createElement("span");
            q.className = "action-q";
            q.textContent = "—";
            cell.append(name, q);
            grid.append(cell);
            this.actions.push({ cell, name, q });
        }
        const escapeLabel = this.element("escapeLabel");
        escapeLabel.title = ESCAPE_DEFINITION;
        escapeLabel.setAttribute("aria-label", `Enemy escapes. ${ESCAPE_DEFINITION}`);
        if (typeof ResizeObserver !== "undefined") {
            this.resizeObserver = new ResizeObserver(() => this.scheduleDraw());
            for (const id of ["networkCanvas", "scoreChart", "evalChart"]) {
                this.resizeObserver.observe(this.element(id).parentElement);
            }
        }
        this.draw();
    }

    element(id) { return this.elements.get(id); }
    text(id, value) {
        const element = this.element(id);
        if (element && element.textContent !== String(value)) element.textContent = String(value);
    }

    update(state) {
        this.state = state;
        const board = state.board || {};
        if (Array.isArray(state.chartPoints)) this.history = state.chartPoints.slice(-HISTORY_LIMIT);
        else if (state.chartPoint) {
            const key = `${contextKey(state.chartPoint)}:${state.chartPoint.episode}`;
            this.history = this.history.filter((point) => `${contextKey(point)}:${point.episode}` !== key);
            this.history.push(state.chartPoint);
            this.history = this.history.slice(-HISTORY_LIMIT);
        }
        if (Array.isArray(state.evaluationHistory)) this.evaluations = state.evaluationHistory.slice(-40);
        else if (state.evaluation) {
            const record = state.evaluation;
            const signature = JSON.stringify([record.policyHash, record.level, record.fullSurvival, record.seeds]);
            if (!this.evaluations.some((item) => JSON.stringify([item.policyHash, item.level, item.fullSurvival, item.seeds]) === signature)) {
                this.evaluations.push(record);
                this.evaluations = this.evaluations.slice(-40);
            }
        }
        this.text("episodes", fixed(state.episode));
        this.text("currentScore", fixed(state.currentScore ?? board.score));
        this.text("survivalSeconds", fixed(state.survivalSeconds, 1));
        this.text("foodCollected", fixed(board.foodCollected));
        this.text("foodAverage", fixed(state.foodAverage, 1));
        this.text("avgSurvival", fixed(state.avgSurvival, 1, "s"));
        this.text("enemyEscapes", fixed(state.enemyEscapes ?? board.enemyEscapes));
        this.text("deathCause", state.deathCause || board.lastDeath || "None yet");
        this.text("highScore", fixed(state.highScore));
        this.text("epsilon", fixed(state.epsilon, 3));
        this.text("trainingLoss", fixed(state.loss, 4));
        this.text("bufferSize", compact(state.bufferSize));
        this.text("transitionsPerSecond", fixed(state.transitionsPerSecond, 0));
        this.text("trainingSteps", fixed(state.trainingSteps));
        this.text("levelName", state.fullSurvival ? "Full Survival" : state.levelName || board.levelName || "—");
        this.text("modelVersion", state.modelVersion || "—");
        this.text("hudMode", state.mode === "eval" ? "TEST · GREEDY" : "TRAIN");
        this.text("sessionLog", state.logMessage || "Python-authoritative simulation. Genuine policy decisions.");
        const mode = state.mode === "eval" ? "TEST" : "TRAIN";
        const level = state.fullSurvival ? "FULL SURVIVAL" : `L${state.level ?? board.level ?? "?"} · ${state.levelName || board.levelName || ""}`;
        this.text("performanceContext", `${mode} · ${level}`);
        this.text("curriculumStatus", state.autoCurriculum ? `Auto · level ${state.level}${state.curriculum?.pendingLevel ? ` → ${state.curriculum.pendingLevel} pending` : ""}` : `Manual · ${state.fullSurvival ? "full survival" : `level ${state.level}`}`);
        this.renderDecision(state.network);
        this.visibleScores = episodeSeries(this.history, state);
        this.visibleEvaluations = evaluationSeries(this.evaluations, state);
        this.renderEvaluation();
        this.element("scoreEmpty").hidden = this.visibleScores.length > 0;
        const mixed = this.history.filter((point) => point.mixedMode && point.level === state.level && Boolean(point.fullSurvival) === Boolean(state.fullSurvival)).length;
        this.text("chartWindow", `${mode === "TEST" ? "Test" : "Train"} context · ${this.visibleScores.length} episodes · mean up to ${MEAN_WINDOW}${mixed ? ` · ${mixed} mixed excluded` : ""}`);
        this.element("highScore").title = "Session high score reported by Python; chart lines are filtered to the displayed mode, difficulty and full-survival context.";
        this.renderTerminal(state);
        this.scheduleDraw();
    }

    renderDecision(network) {
        const dimensions = Array.isArray(network?.dimensions) ? network.dimensions : [];
        const layers = Array.isArray(network?.layers) ? network.layers : [];
        this.text("networkDimensions", dimensions.length
            ? `${dimensions.join(" › ")} · ${layers.map((layer) => layer.length).join("/")} sampled`
            : "Waiting for policy dimensions");
        const chosen = Number.isInteger(network?.lastAction) ? network.lastAction : null;
        const exploring = network?.isExploring === true;
        const values = Array.isArray(network?.qValues) ? network.qValues : [];
        const mask = Array.isArray(network?.actionMask) ? network.actionMask : [];
        this.actions.forEach(({ cell, q }, index) => {
            const label = network?.outputLabels?.[index] || `${ABILITIES[Math.floor(index / 4)]} ${DIRECTIONS[index % 4]}`;
            const value = numeric(values[index]);
            const available = mask[index];
            const selected = chosen === index;
            const ability = ABILITIES[Math.floor(index / 4)];
            q.textContent = value === null ? "—" : value.toFixed(2);
            cell.classList.toggle("is-selected", selected);
            cell.classList.toggle("is-random", selected && exploring);
            cell.classList.toggle("is-masked", available === false);
            cell.dataset.masked = available === false ? "true" : available === true ? "false" : "unknown";
            cell.dataset.selected = String(selected);
            cell.title = `${label} · Q ${value === null ? "not sampled yet" : value.toFixed(4)} · ${available === false ? "masked: reverse direction or ability unavailable" : available === true ? "available; not safety-filtered" : "awaiting mask"}${selected ? exploring ? " · chosen by exploration" : " · chosen by greedy policy" : ""}${ABILITY_DEFINITIONS[ability] ? ` · ${ABILITY_DEFINITIONS[ability]}` : ""}`;
            cell.setAttribute("aria-label", cell.title);
        });
        const policyStatus = this.element("policyStatus");
        const status = chosen === null ? "No decision yet" : exploring ? "Random exploration" : "Greedy selection";
        this.text("policyStatus", status);
        policyStatus.classList.toggle("is-random", chosen !== null && exploring);
        policyStatus.classList.toggle("is-greedy", chosen !== null && !exploring);
        this.text("lastActionLabel", chosen === null ? "Awaiting first decision" : network?.outputLabels?.[chosen] || this.state?.actionLabel || `Action ${chosen}`);
        policyStatus.title = chosen === null ? "No simulation decision has been made in this episode." : `Last decision: episode ${network.decisionEpisode ?? "?"}, step ${network.decisionStep ?? "?"}. Activations, Q-values, mask and selected action are from this same decision; the arena shows its resulting state.`;
        this.element("networkCanvas").setAttribute("aria-label", `${dimensions.join(" by ")} policy. Real sampled activations and signed sampled weights. ${policyStatus.title}`);
        this.element("networkCanvas").title = `Actual architecture: ${dimensions.join(" → ")}. Sampled inputs: ${(network?.inputLabels || []).join(", ")}. Connections with a sampled magnitude below 27% of their layer maximum are omitted for readability; no decorative pulses.`;
    }

    renderEvaluation() {
        const latest = this.visibleEvaluations.at(-1);
        this.element("evalEmpty").hidden = Boolean(latest);
        if (!latest) {
            this.text("evaluationRate", "—");
            this.text("evaluationMeta", "Greedy policy · seeded independent episodes");
            this.element("evaluationMeta").title = "No evaluation exists for the current difficulty and full-survival context.";
            return;
        }
        const rate = numeric(latest.successRate) ?? Number(latest.successes) / Number(latest.episodes);
        this.text("evaluationRate", `${(rate * 100).toFixed(0)}%`);
        this.text("evaluationMeta", `${latest.successes} / ${latest.episodes} successes · ${fixed(latest.trainingSteps)} training steps · ${String(latest.policyHash || "").slice(0, 8)}`);
        this.element("evaluationMeta").title = `Frozen policy ${latest.policyHash || "not reported"}; level ${latest.level}${latest.fullSurvival ? " full survival" : ""}; seeds: ${(latest.seeds || []).join(", ")}. Mean food ${fixed(latest.meanFood, 2)}, mean survival ${fixed(latest.meanSurvivalSeconds, 1)} seconds. Failures ${Array.isArray(latest.failures) ? latest.failures.length : latest.failures ?? "not reported"}. Goals: ${JSON.stringify(latest.goals || {})}.`;
    }

    renderTerminal(state) {
        const terminal = state.episodeEnded || state.board?.done;
        this.element("terminalOverlay").hidden = !terminal || Boolean(state.introActive);
        if (!terminal) return;
        this.text("terminalTitle", `Episode ${state.episode} · ${state.deathCause || state.board?.lastDeath || "complete"}`);
        this.text("terminalDetail", `${fixed(state.currentScore)} score · ${fixed(state.board?.foodCollected)} food · ${fixed(state.survivalSeconds, 1)}s survived · level ${state.level}${state.fullSurvival ? " full" : ""}`);
    }

    scheduleDraw() {
        if (this.drawTimer !== null) return;
        const wait = Math.max(0, 140 - (performance.now() - this.lastDraw));
        this.drawTimer = setTimeout(() => {
            this.drawTimer = null;
            this.draw();
        }, wait);
    }

    draw() {
        this.lastDraw = performance.now();
        this.drawNetwork();
        this.drawScore();
        this.drawEvaluation();
    }

    drawNetwork() {
        const { ctx, width, height } = sizeCanvas(this.element("networkCanvas"));
        if (!ctx) return;
        const network = this.state?.network;
        const layers = network?.layers;
        if (!Array.isArray(layers) || layers.length < 2) {
            ctx.fillStyle = "#73879a";
            ctx.font = "10px Segoe UI, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("Awaiting a real policy sample", width / 2, height / 2);
            return;
        }
        const positions = layers.map((layer, layerIndex) => layer.map((_, index) => ({
            x: 21 + layerIndex / (layers.length - 1) * (width - 42),
            y: layer.length <= 1 ? height / 2 : 23 + index / (layer.length - 1) * (height - 33),
        })));
        const maxima = layers.map((layer) => Math.max(0.001, ...layer.map((value) => Math.abs(numeric(value) || 0))));
        ctx.font = "9px Consolas, monospace";
        ctx.textAlign = "center";
        network.dimensions?.forEach((dimension, index) => {
            const x = positions[index]?.[0]?.x;
            if (x === undefined) return;
            ctx.fillStyle = "#a3b6c8";
            ctx.fillText(`${index === 0 ? "IN" : index === layers.length - 1 ? "Q" : "H"} ${dimension}`, x, 12);
        });
        (network.weights || []).forEach((matrix, layerIndex) => {
            if (!positions[layerIndex + 1]) return;
            const maxWeight = Math.max(0.001, ...matrix.flat().map((value) => Math.abs(numeric(value) || 0)));
            matrix.forEach((row, source) => row.forEach((weight, target) => {
                const from = positions[layerIndex]?.[source];
                const to = positions[layerIndex + 1]?.[target];
                const value = numeric(weight);
                if (!from || !to || value === null) return;
                const magnitude = Math.abs(value) / maxWeight;
                if (magnitude < 0.27) return;
                const activation = Math.abs(numeric(layers[layerIndex][source]) || 0) / maxima[layerIndex];
                const alpha = 0.035 + magnitude * (0.08 + activation * 0.18);
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = value >= 0 ? `rgba(157, 215, 136, ${alpha})` : `rgba(224, 172, 103, ${alpha})`;
                ctx.lineWidth = 0.4 + magnitude * 0.4;
                ctx.stroke();
            }));
        });
        layers.forEach((layer, layerIndex) => layer.forEach((value, index) => {
            const position = positions[layerIndex][index];
            const activation = Math.abs(numeric(value) || 0) / maxima[layerIndex];
            const chosen = layerIndex === layers.length - 1 && index === network.lastAction;
            ctx.beginPath();
            ctx.arc(position.x, position.y, chosen ? 3.5 : 1.4 + activation * 1.2, 0, Math.PI * 2);
            ctx.fillStyle = chosen ? network.isExploring ? "#e7b66a" : "#b0f18a"
                : value < 0 ? `rgba(220, 173, 113, ${0.22 + activation * 0.7})`
                    : `rgba(156, 207, 151, ${0.22 + activation * 0.7})`;
            ctx.fill();
            if (layerIndex === layers.length - 1 && network.actionMask?.[index] === false) {
                ctx.beginPath();
                ctx.moveTo(position.x - 3, position.y + 3);
                ctx.lineTo(position.x + 3, position.y - 3);
                ctx.strokeStyle = "#71849b";
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }));
    }

    drawScore() {
        const points = this.visibleScores;
        const maxY = Math.max(1, ...points.map((point) => Number(point.score))) * 1.08;
        const first = points[0]?.episode;
        const last = points.at(-1)?.episode;
        const graph = graphAxes(this.element("scoreChart"), maxY, first === undefined ? "" : `ep ${first}`, last === undefined ? "" : `ep ${last}`);
        drawSeries(graph, points.map((point) => ({ x: Number(point.episode), y: Number(point.score) })), first, last, maxY, "#5e7691", 1);
        drawSeries(graph, points.map((point) => ({ x: Number(point.episode), y: point.mean })), first, last, maxY, "#b0f18a", 1.8);
    }

    drawEvaluation() {
        const points = this.visibleEvaluations;
        const graph = graphAxes(this.element("evalChart"), 100, points.length ? "suite 1" : "", points.length > 1 ? `suite ${points.length}` : "", true);
        drawSeries(graph, points.map((point, index) => ({
            x: index,
            y: (numeric(point.successRate) ?? Number(point.successes) / Number(point.episodes)) * 100,
        })), 0, points.length - 1, 100, "#e7b66a", 1.5, true);
    }

    getDebugState() {
        const network = this.state?.network;
        return {
            dimensions: network?.dimensions || [],
            sampledNodes: network?.layers?.map((layer) => layer.length) || [],
            decisionEpisode: network?.decisionEpisode ?? null,
            decisionStep: network?.decisionStep ?? null,
            historyCount: this.history.length,
            visibleContext: this.state ? contextKey(this.state) : null,
            scorePoints: this.visibleScores.map(({ episode, score, mean, mode, level, fullSurvival }) => ({ episode, score, mean, mode, level, fullSurvival })),
            evaluationPoints: this.visibleEvaluations.map(({ episodes, successes, successRate, level, fullSurvival, trainingSteps, policyHash, seeds }) => ({ episodes, successes, successRate, level, fullSurvival, trainingSteps, policyHash, seeds })),
            actions: this.actions.map((_, index) => ({
                action: index,
                label: network?.outputLabels?.[index] || `${ABILITIES[Math.floor(index / 4)]} ${DIRECTIONS[index % 4]}`,
                q: network?.qValues?.[index] ?? null,
                available: network?.actionMask?.[index] ?? null,
                selected: network?.lastAction === index,
            })),
        };
    }

    dispose() {
        clearTimeout(this.drawTimer);
        this.resizeObserver?.disconnect();
    }
}
