const FPS = 30;
const FORMATS = {
    landscape: { width: 1280, height: 720 },
    portrait: { width: 720, height: 1280 },
};

export function chooseRecordingMime(Recorder = globalThis.MediaRecorder) {
    if (!Recorder || typeof Recorder.isTypeSupported !== "function") return null;
    return ["video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm"].find((mime) => Recorder.isTypeSupported(mime)) || null;
}

function value(value, places = 0) {
    return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "—";
}

function elapsedLabel(milliseconds) {
    const seconds = Math.max(0, Math.floor(milliseconds / 1000));
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function byteLabel(bytes) {
    return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export class CreatorRecorder {
    constructor({ canvas, stage, renderer, onFilm = () => {}, onError = () => {}, onChange = () => {} }) {
        this.canvas = canvas;
        this.stage = stage;
        this.renderer = renderer;
        this.onFilm = onFilm;
        this.onError = onError;
        this.onChange = onChange;
        this.format = "landscape";
        this.overlay = true;
        this.clean = false;
        this.state = null;
        this.summary = null;
        this.summaryUntil = 0;
        this.lastCompletedKey = null;
        this.haveSeenState = false;
        this.recording = false;
        this.stopping = false;
        this.disposed = false;
        this.framesComposited = 0;
        this.bytes = 0;
        this.chunks = [];
        this.startedAt = null;
        this.durationMs = 0;
        this.lastFrameAt = -Infinity;
        this.lastStatusAt = -Infinity;
        this.animationFrame = null;
        this.lastDownload = null;
        this.downloadRequests = 0;
        this.downloadUrl = null;
        this.failure = null;
        this.mime = chooseRecordingMime();
        this.supported = Boolean(this.mime && renderer && typeof canvas.captureStream === "function");
        this.buttons = [document.getElementById("recordBtn"), document.getElementById("cleanRecordBtn")];
        this.frameSelect = document.getElementById("frameSelect");
        this.overlayToggle = document.getElementById("overlayToggle");
        this.cleanButton = document.getElementById("cleanViewBtn");
        this.restoreTray = document.getElementById("creatorRestoreTray");
        this.restoreButton = document.getElementById("restoreViewBtn");
        this.status = document.getElementById("recordStatus");
        this.fallback = document.getElementById("recordFallback");
        this.downloadLink = document.getElementById("recordDownload");
        this.frameSelect.addEventListener("change", () => this.setFormat(this.frameSelect.value));
        this.overlayToggle.addEventListener("change", () => {
            this.overlay = this.overlayToggle.checked;
            this.syncHud();
        });
        this.cleanButton.addEventListener("click", () => this.setClean(!this.clean));
        this.restoreButton.addEventListener("click", () => this.setClean(false));
        for (const button of this.buttons) button.addEventListener("click", () => { void this.toggle(); });
        this.downloadLink.addEventListener("click", () => { this.downloadRequests += 1; });
        this.fallback.hidden = this.supported;
        this.renderControls();
        this.setFormat("landscape");
    }

    setFormat(format) {
        if (!(format in FORMATS)) return;
        if (this.recording || this.stopping) {
            this.frameSelect.value = this.format;
            return;
        }
        this.format = format;
        this.frameSelect.value = format;
        this.stage.dataset.frame = format;
        this.resizeScene();
        this.renderControls();
    }

    resizeScene() {
        requestAnimationFrame(() => {
            if (this.disposed) return;
            try { this.renderer?.resize(); }
            catch (error) { this.onError(error); }
        });
    }

    setClean(clean) {
        this.clean = Boolean(clean);
        document.body.classList.toggle("clean-view", this.clean);
        this.cleanButton.setAttribute("aria-pressed", String(this.clean));
        this.restoreTray.hidden = !this.clean;
        this.resizeScene();
        const focusTarget = this.clean ? this.restoreButton : this.cleanButton;
        focusTarget.focus({ preventScroll: true });
        this.onChange();
    }

    update(state) {
        this.state = state;
        const board = state.board || {};
        const terminal = state.episodeEnded || board.done;
        const completed = Array.isArray(state.chartPoints) ? state.chartPoints.at(-1) : state.chartPoint;
        const key = completed ? `${completed.episode}:${completed.level}:${Boolean(completed.fullSurvival)}:${completed.trainingSteps}:${completed.mode}` : null;
        if (terminal) {
            const terminalKey = `live:${state.episode}:${board.seed}:${board.steps}:${state.level}`;
            if (this.summary?.key !== terminalKey) {
                this.summary = {
                    key: terminalKey,
                    episode: state.episode,
                    score: state.currentScore,
                    food: board.foodCollected,
                    survivalSeconds: state.survivalSeconds,
                    deathCause: state.deathCause || board.lastDeath,
                    level: state.level,
                    fullSurvival: state.fullSurvival,
                    mode: state.mode,
                };
                this.summaryUntil = performance.now() + 4200;
            }
        } else if (this.haveSeenState && key && key !== this.lastCompletedKey) {
            this.summary = { key, ...completed };
            this.summaryUntil = performance.now() + 4200;
        }
        this.lastCompletedKey = key;
        this.haveSeenState = true;
        this.syncHud();
    }

    syncHud() {
        document.getElementById("sceneHud").hidden = !this.overlay;
        const terminal = this.state?.episodeEnded || this.state?.board?.done;
        document.getElementById("terminalOverlay").hidden = !this.overlay || !terminal || Boolean(this.state?.introActive);
    }

    async toggle() {
        if (this.stopping) return;
        if (this.recording) await this.stop();
        else await this.start();
    }

    async start() {
        if (!this.supported || this.recording || this.stopping) return false;
        this.failure = null;
        this.framesComposited = 0;
        this.bytes = 0;
        this.chunks = [];
        this.durationMs = 0;
        this.lastFrameAt = -Infinity;
        this.lastStatusAt = -Infinity;
        this.captureFormat = this.format;
        this.captureSize = { ...FORMATS[this.format] };
        try {
            this.onFilm();
            this.outputCanvas = document.createElement("canvas");
            this.outputCanvas.width = this.captureSize.width;
            this.outputCanvas.height = this.captureSize.height;
            this.outputContext = this.outputCanvas.getContext("2d", { alpha: false });
            if (!this.outputContext) throw new Error("The browser could not create the video compositing canvas.");
            this.drawFrame(performance.now());
            this.stream = this.outputCanvas.captureStream(FPS);
            if (!this.stream.getVideoTracks().length) throw new Error("The canvas did not supply a recordable video track.");
            this.recorder = new MediaRecorder(this.stream, { mimeType: this.mime, videoBitsPerSecond: 5000000 });
            this.stopPromise = new Promise((resolve) => { this.resolveStop = resolve; });
            this.recorder.ondataavailable = (event) => {
                if (!event.data?.size) return;
                this.chunks.push(event.data);
                this.bytes += event.data.size;
            };
            this.recorder.onerror = (event) => this.captureFailed(event.error || new Error("The browser stopped the WebM encoder."));
            this.recorder.onstop = () => this.finishRecording();
            this.startedAt = performance.now();
            this.recording = true;
            this.recorder.start(1000);
            this.renderControls();
            this.composite(this.startedAt);
            this.onChange();
            return true;
        } catch (error) {
            this.recording = false;
            this.stopTracks();
            this.resolveStop?.(null);
            this.resolveStop = null;
            this.failure = error.message || String(error);
            this.status.textContent = `Recording unavailable: ${this.failure} Use system screen recording.`;
            this.fallback.hidden = false;
            this.renderControls(true);
            this.onError(error);
            this.onChange();
            return false;
        }
    }

    composite(now) {
        if (!this.recording || this.stopping || this.disposed) return;
        if (now - this.lastFrameAt >= 1000 / FPS - 1) {
            try {
                this.drawFrame(now);
                this.framesComposited += 1;
                this.lastFrameAt = now;
            } catch (error) {
                this.captureFailed(error);
                return;
            }
        }
        if (now - this.lastStatusAt >= 250) {
            this.durationMs = now - this.startedAt;
            this.status.textContent = `Recording ${elapsedLabel(this.durationMs)} · ${this.captureSize.width} × ${this.captureSize.height} · WebM`;
            this.lastStatusAt = now;
        }
        this.animationFrame = requestAnimationFrame((time) => this.composite(time));
    }

    drawFrame(now) {
        const { width, height } = this.captureSize;
        const ctx = this.outputContext;
        ctx.fillStyle = "#050b14";
        ctx.fillRect(0, 0, width, height);
        const sourceWidth = this.canvas.width;
        const sourceHeight = this.canvas.height;
        if (!sourceWidth || !sourceHeight) throw new Error("The live arena canvas has no image to record.");
        const scale = Math.min(width / sourceWidth, height / sourceHeight);
        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        ctx.drawImage(this.canvas, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
        if (this.overlay && this.state) this.drawOverlay(ctx, width, height, now);
    }

    drawOverlay(ctx, width, height, now) {
        const portrait = height > width;
        const inset = portrait ? 36 : 38;
        const panelWidth = portrait ? width - inset * 2 : 490;
        ctx.save();
        ctx.fillStyle = "rgba(5, 15, 25, 0.79)";
        ctx.fillRect(inset, inset, panelWidth, portrait ? 124 : 108);
        ctx.fillStyle = "#b0f18a";
        ctx.fillRect(inset, inset, 3, portrait ? 124 : 108);
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = "#b0f18a";
        ctx.font = "600 15px Segoe UI, sans-serif";
        ctx.fillText("VERDANT / SURVIVAL LAB", inset + 20, inset + 27);
        ctx.fillStyle = "#e9f0e5";
        ctx.font = "500 24px Segoe UI, sans-serif";
        ctx.fillText(`EP ${value(this.state.episode)}  ·  ${this.state.mode === "eval" ? "GREEDY TEST" : "TRAIN"}`, inset + 20, inset + 59);
        ctx.font = "18px Consolas, monospace";
        ctx.fillStyle = "#adbfca";
        ctx.fillText(`SCORE ${value(this.state.currentScore)}   FOOD ${value(this.state.board?.foodCollected)}   ${value(this.state.survivalSeconds, 1)}s`, inset + 20, inset + 89);
        if (portrait) {
            ctx.font = "12px Segoe UI, sans-serif";
            ctx.fillStyle = "#8097aa";
            ctx.fillText(this.state.fullSurvival ? "FULL SURVIVAL" : `LEVEL ${value(this.state.level)} / ${this.state.levelName || ""}`, inset + 20, inset + 110);
        }
        if (this.summary && now < this.summaryUntil) {
            const summary = this.summary;
            const panelHeight = portrait ? 146 : 128;
            const y = height - inset - panelHeight;
            const summaryWidth = portrait ? panelWidth : 580;
            ctx.fillStyle = "rgba(5, 15, 25, 0.85)";
            ctx.fillRect(inset, y, summaryWidth, panelHeight);
            ctx.fillStyle = "#e7b66a";
            ctx.fillRect(inset, y, 3, panelHeight);
            ctx.font = "600 13px Segoe UI, sans-serif";
            ctx.fillText(`EPISODE ${value(summary.episode)} COMPLETE · ${summary.fullSurvival ? "FULL SURVIVAL" : `LEVEL ${value(summary.level)}`}`, inset + 20, y + 28);
            ctx.fillStyle = "#edf2e8";
            ctx.font = "500 25px Segoe UI, sans-serif";
            const cause = String(summary.deathCause || "Run complete").replaceAll("_", " ");
            ctx.fillText(cause.charAt(0).toUpperCase() + cause.slice(1), inset + 20, y + 63, summaryWidth - 40);
            ctx.fillStyle = "#b3c2cd";
            ctx.font = "18px Consolas, monospace";
            ctx.fillText(`${value(summary.score)} score · ${value(summary.food)} food · ${value(summary.survivalSeconds, 1)}s survived`, inset + 20, y + 96, summaryWidth - 40);
            if (portrait) {
                ctx.fillStyle = "#8097aa";
                ctx.font = "12px Segoe UI, sans-serif";
                ctx.fillText("ACTUAL POLICY DECISIONS / NO SCRIPTED OUTCOME", inset + 20, y + 124);
            }
        }
        ctx.restore();
    }

    async stop() {
        if (!this.recording || this.stopping) return this.stopPromise;
        this.stopping = true;
        this.durationMs = performance.now() - this.startedAt;
        cancelAnimationFrame(this.animationFrame);
        this.renderControls();
        try {
            if (this.recorder.state !== "inactive") this.recorder.stop();
            else this.finishRecording();
        } catch (error) {
            this.failure = error.message || String(error);
            this.finishRecording();
            this.onError(error);
        }
        return this.stopPromise;
    }

    captureFailed(error) {
        if (this.failure) return;
        this.failure = error.message || String(error);
        this.onError(new Error(`Canvas recording failed: ${this.failure}`));
        if (this.recording) void this.stop();
    }

    finishRecording() {
        if (this.finalizing || (!this.recording && !this.stopping)) return;
        this.finalizing = true;
        cancelAnimationFrame(this.animationFrame);
        this.stopTracks();
        this.recording = false;
        this.stopping = false;
        try {
            const mime = this.recorder?.mimeType || this.mime;
            const blob = new Blob(this.chunks, { type: mime });
            if (this.failure || blob.size === 0 || this.framesComposited === 0) {
                this.status.textContent = `No usable clip exported. ${this.failure || "The browser returned no video frames."} Try again or use system screen recording.`;
                this.fallback.hidden = false;
            } else if (!this.disposed) {
                if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
                this.downloadUrl = URL.createObjectURL(blob);
                const timestamp = new Date().toISOString().replaceAll(":", "").replaceAll("-", "").slice(0, 15);
                const filename = `verdant-${this.captureFormat}-${timestamp}.webm`;
                this.downloadLink.href = this.downloadUrl;
                this.downloadLink.download = filename;
                this.downloadLink.hidden = false;
                this.downloadLink.setAttribute("aria-label", `Download actual ${this.captureSize.width} by ${this.captureSize.height} WebM clip, ${byteLabel(blob.size)}`);
                this.lastDownload = { filename, bytes: blob.size, mime, width: this.captureSize.width, height: this.captureSize.height, durationMs: Math.round(this.durationMs), framesComposited: this.framesComposited };
                this.status.textContent = `WebM ready · ${elapsedLabel(this.durationMs)} · ${byteLabel(blob.size)} · download below`;
                this.downloadLink.click();
            }
        } catch (error) {
            this.status.textContent = `Export failed: ${error.message || error}. Use system screen recording.`;
            this.fallback.hidden = false;
            this.onError(error);
        } finally {
            this.finalizing = false;
            this.renderControls(true);
            this.resolveStop?.(this.lastDownload);
            this.resolveStop = null;
            this.onChange();
        }
    }

    stopTracks() {
        for (const track of this.stream?.getTracks() || []) track.stop();
        this.stream = null;
    }

    renderControls(preserveStatus = false) {
        const active = this.recording || this.stopping;
        document.body.classList.toggle("is-recording", active);
        this.frameSelect.disabled = active;
        this.status.classList.toggle("is-recording", active);
        for (const button of this.buttons) {
            button.disabled = !this.supported || this.stopping;
            button.classList.toggle("is-recording", active);
            const text = button.querySelector("span");
            if (text) text.textContent = this.stopping ? "Finishing…" : active ? "Stop recording" : button.id === "recordBtn" ? "Record WebM" : "Record";
            button.setAttribute("aria-label", this.stopping ? "Finishing WebM recording" : active ? "Stop recording and download WebM" : `Record real canvas as ${this.format} WebM`);
        }
        if (!preserveStatus && !active && !this.lastDownload) {
            const size = FORMATS[this.format];
            this.status.textContent = this.supported ? `WebM supported · ${size.width} × ${size.height} · 30 fps capture` : "WebM capture unavailable · system screen recording supported";
        } else if (this.stopping) this.status.textContent = "Finishing actual WebM chunks…";
    }

    getDebugState() {
        const size = this.recording || this.stopping ? this.captureSize : FORMATS[this.format];
        return {
            supported: this.supported,
            recording: this.recording,
            stopping: this.stopping,
            clean: this.clean,
            overlay: this.overlay,
            format: this.format,
            width: size.width,
            height: size.height,
            requestedFps: FPS,
            mime: this.recorder?.mimeType || this.mime,
            framesComposited: this.framesComposited,
            chunks: this.chunks.length,
            bytes: this.bytes,
            durationMs: Math.round(this.recording && this.startedAt !== null ? performance.now() - this.startedAt : this.durationMs),
            downloadRequests: this.downloadRequests,
            lastDownload: this.lastDownload ? { ...this.lastDownload } : null,
            error: this.failure,
        };
    }

    dispose() {
        this.disposed = true;
        cancelAnimationFrame(this.animationFrame);
        if (this.recording) void this.stop();
        this.stopTracks();
        if (this.downloadUrl) URL.revokeObjectURL(this.downloadUrl);
    }
}
