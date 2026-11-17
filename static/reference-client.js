import { PlaybackLifecycle, SimulationPacer, SIMULATION_RATES } from "./presentation.js?v=reference-live-2";

class SupersededRequest extends Error {}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const rates = Object.keys(SIMULATION_RATES);

export class ReferenceGameClient {
    constructor({ onState, onError, onStatus }) {
        this.onState = onState;
        this.onError = onError;
        this.onStatus = onStatus;
        this.lifecycle = new PlaybackLifecycle();
        this.pacer = new SimulationPacer("rate-20");
        this.state = null;
        this.active = true;
        this.offline = false;
        this.error = null;
        this.notice = null;
        this.pendingControls = 0;
        this.requests = 0;
        this.maxRequests = 0;
        this.lastBestPoll = -Infinity;
    }

    async request(context, url, body) {
        if (!context.current()) throw new SupersededRequest();
        const abort = new AbortController();
        const timer = setTimeout(() => abort.abort(), 120000);
        this.requests += 1;
        this.maxRequests = Math.max(this.maxRequests, this.requests);
        try {
            const response = await fetch(url, {
                method: body === undefined ? "GET" : "POST",
                headers: body === undefined ? undefined : { "Content-Type": "application/json" },
                body: body === undefined ? undefined : JSON.stringify(body),
                signal: abort.signal,
            });
            const state = await response.json();
            if (!context.current()) throw new SupersededRequest();
            if (!response.ok) throw new Error(`${url}: ${JSON.stringify(state.detail)} (${response.status})`);
            if (state.modelVersion !== "arena-3d-v3" || state.environmentVersion !== "survival-balanced-v8") {
                throw new Error("The Python game is out of date. Restart the server.");
            }
            return state;
        } finally {
            clearTimeout(timer);
            this.requests -= 1;
        }
    }

    commit(context, state, advanced = false) {
        if (!context.current()) throw new SupersededRequest();
        this.state = state;
        this.onState(state, { advanced });
    }

    fail(error) {
        if (error instanceof SupersededRequest) return;
        this.offline = true;
        this.error = error.message;
        this.pacer.suspend();
        this.lifecycle.invalidate();
        console.error(error);
        this.onError(error);
    }

    operation(work, invalidate = true) {
        this.pacer.suspend();
        this.notice = null;
        const generation = invalidate ? this.lifecycle.invalidate() : this.lifecycle.generation;
        this.pendingControls += 1;
        this.onStatus?.();
        return this.lifecycle.enqueue(work, generation).catch((error) => this.fail(error)).finally(() => {
            this.pendingControls -= 1;
            this.onStatus?.();
        });
    }

    async restorePreference(context) {
        let state;
        do {
            state = await this.request(context, "/api/control", {
                intro: false, paused: this.lifecycle.desiredPaused,
            });
        } while (state.paused !== this.lifecycle.desiredPaused);
        return state;
    }

    async start() {
        await this.operation(async (context) => {
            this.commit(context, await this.request(context, "/api/state"));
            this.commit(context, await this.restorePreference(context));
        });
        void this.loop();
    }

    blocked() {
        return !this.active || !this.state || this.offline || this.pendingControls
            || this.state.paused || this.lifecycle.desiredPaused || document.hidden;
    }

    async loop() {
        while (this.active) {
            if (this.blocked()) {
                this.pacer.suspend();
                if (this.state?.bestPolicy?.status === "checking" && !this.offline && !this.pendingControls
                    && !document.hidden && performance.now() - this.lastBestPoll >= 500) {
                    this.lastBestPoll = performance.now();
                    try {
                        await this.lifecycle.enqueue(async (context) =>
                            this.commit(context, await this.request(context, "/api/state")));
                    } catch (error) { this.fail(error); }
                }
                await delay(25);
                continue;
            }
            this.pacer.resume(performance.now());
            const wait = this.pacer.waitMs(performance.now());
            if (wait > 0) { await delay(Math.min(wait, 25)); continue; }
            if (!this.pacer.tryStart(performance.now())) continue;
            try {
                await this.lifecycle.enqueue(async (context) => {
                    if (this.blocked()) return;
                    const fast = this.pacer.mode === "fast";
                    const endpoint = fast && this.state.mode === "train" ? "/api/train" : "/api/tick";
                    this.commit(context, await this.request(context, endpoint, { steps: fast ? 128 : 1 }), true);
                });
            } catch (error) {
                this.fail(error);
            } finally {
                this.pacer.complete(performance.now());
            }
        }
    }

    togglePause() {
        this.lifecycle.desiredPaused = !this.lifecycle.desiredPaused;
        if (this.pendingControls) return Promise.resolve();
        return this.operation(async (context) => this.commit(context, await this.restorePreference(context)), false);
    }

    setSpeed(mode) {
        this.pacer.setMode(mode, performance.now());
    }

    changeSpeed(delta) {
        const index = rates.indexOf(this.pacer.mode);
        this.setSpeed(rates[Math.max(0, Math.min(rates.length - 1, index + delta))]);
    }

    configure(settings) {
        return this.operation(async (context) => {
            this.commit(context, await this.request(context, "/api/control", {
                ...settings, intro: false, paused: this.lifecycle.desiredPaused,
            }));
        });
    }

    ready() {
        this.lifecycle.desiredPaused = false;
        return this.operation(async (context) => {
            await this.request(context, "/api/control", { intro: false, paused: true });
            const loaded = await this.request(context, "/api/load-ready-weights", {});
            this.notice = loaded.logMessage;
            this.commit(context, loaded);
            this.commit(context, await this.restorePreference(context));
        });
    }

    reset() {
        this.lifecycle.desiredPaused = false;
        return this.operation(async (context) => {
            await this.request(context, "/api/control", { intro: false, paused: true });
            const reset = await this.request(context, "/api/reset", {});
            this.notice = reset.logMessage;
            this.commit(context, reset);
            this.commit(context, await this.restorePreference(context));
        });
    }

    save() {
        return this.operation(async (context) => {
            const state = await this.request(context, "/api/save-weights", {});
            console.info(state.logMessage);
            this.notice = state.logMessage;
            this.commit(context, state);
        }, false);
    }

    load() {
        return this.operation(async (context) => {
            await this.request(context, "/api/control", { intro: false, paused: true });
            const loaded = await this.request(context, "/api/load-weights", {});
            this.notice = loaded.logMessage;
            this.commit(context, loaded);
            this.commit(context, await this.restorePreference(context));
        });
    }

    checkBest() {
        return this.operation(async (context) => {
            const state = await this.request(context, "/api/check-best", {});
            this.commit(context, state);
        }, false);
    }

    reconnect() {
        this.offline = false;
        this.error = null;
        this.lifecycle.desiredPaused = true;
        return this.operation(async (context) => this.commit(context, await this.restorePreference(context)));
    }

    dispose() {
        this.active = false;
        this.pacer.suspend();
        this.lifecycle.invalidate();
    }

    debug() {
        return {
            offline: this.offline, error: this.error,
            pendingControls: this.pendingControls, desiredPaused: this.lifecycle.desiredPaused,
            speed: this.pacer.mode, simulationStepsPerSecond: this.pacer.stepsPerSecond,
            maxRequestConcurrency: this.maxRequests, generation: this.lifecycle.generation,
        };
    }
}
