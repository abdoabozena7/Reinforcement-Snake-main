import * as THREE from "./vendor/three.module.js";

const UP = new THREE.Vector3(0, 1, 0);
const HOME_AZIMUTH = 0.82;
const HOME_ELEVATION = 0.61;
const BODY_RADIUS = 0.62;
const EVENT_FOV = 38;
const CLEARANCE = 0.7;
const SAFE = { x: 0.84, top: 0.7, bottom: 0.58 };
const MODES = new Set(["game", "static", "film", "follow", "chase", "cine-chase", "cinema", "orbit", "free"]);
const clamp = THREE.MathUtils.clamp;
const cellKey = (point) => `${point.x},${point.y},${point.z}`;
const vector = (point) => new THREE.Vector3(point.x, point.y, point.z);
const wrap = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));
const ease = (value) => {
    const t = clamp(value, 0, 1);
    return t * t * (3 - 2 * t);
};
const directionAt = (azimuth, elevation) => new THREE.Vector3(
    Math.sin(azimuth) * Math.cos(elevation), Math.sin(elevation), Math.cos(azimuth) * Math.cos(elevation),
);
const screenBasis = (direction) => {
    const right = new THREE.Vector3().crossVectors(UP, direction).normalize();
    return { right, up: new THREE.Vector3().crossVectors(direction, right).normalize() };
};
const approach = (value, goal, speed, dt) => value + clamp(goal - value, -speed * dt, speed * dt);
const move = (point, goal, speed, dt) => {
    const distance = point.distanceTo(goal);
    if (distance > 0) point.lerp(goal, Math.min(1 - Math.exp(-dt * 3.2), speed * dt / distance));
};
const newRig = () => ({
    initialized: false, needsFit: true, phase: "composing", center: new THREE.Vector3(),
    goalCenter: new THREE.Vector3(), direction: directionAt(HOME_AZIMUTH, HOME_ELEVATION),
    goalDirection: directionAt(HOME_AZIMUTH, HOME_ELEVATION), azimuth: HOME_AZIMUTH,
    goalAzimuth: HOME_AZIMUTH, elevation: HOME_ELEVATION, goalElevation: HOME_ELEVATION,
    halfHeight: 6, goalHalfHeight: 6, distance: 30, goalDistance: 30,
    clock: 0, lastPlan: -Infinity, plans: 0, busyUntil: 0, turns: [], occluded: false,
});

export class CinematicDirector {
    constructor({ camera, gameplayCamera, cellToScene }) {
        this.camera = camera;
        this.gameplayCamera = gameplayCamera || new THREE.OrthographicCamera(-8, 8, 6, -6, 0.1, 300);
        this.cellToScene = cellToScene;
        this._activeCamera = this.gameplayCamera;
        this._mode = "game";
        this.board = null;
        this.renderState = null;
        this.presentation = null;
        this.dismissedContactId = null;
        this.eventRig = newRig();
        this.eventKey = null;
        this.contactShot = null;
        this.recapShot = null;
        this.eventFraming = false;
        this.smart = newRig();
        this.look = new THREE.Vector3();
        this.pose = { position: new THREE.Vector3(), target: new THREE.Vector3(), up: UP.clone(), fov: 48 };
        this.viewport = { width: 800, height: 600 };
        this.playback = { paused: false, sampled: false, reducedMotion: false };
        this.solids = new Set();
        this.solidCells = new Map();
        this.visibility = { framed: true, occluded: false, clearance: 1000 };
        this.cutaway = [];
        this.now = 0;
        this.intro = null;
        this.lastIntroShot = null;
        this.history = [];
        this.seenEvents = new Set();
        this.cooldowns = new Map();
        this.shot = { name: "follow", until: 0, priority: 0 };
        this.episodeKey = null;
        this.snap = true;
        this.overviewDirty = true;
        this.lastVisibility = -Infinity;
        this.lastEvent = null;
        this.setViewport(this.viewport.width, this.viewport.height);
    }

    get activeCamera() { return this._activeCamera; }
    get mode() { return this._mode; }
    set mode(mode) { this.setCameraMode(mode); }

    setCameraMode(mode) {
        const normalized = mode === "smart" ? "game" : mode === "overview" ? "static" : mode;
        if (!MODES.has(normalized)) throw new Error(`Unknown camera view: ${mode}`);
        if (normalized === this._mode) return;
        this._mode = normalized;
        this._activeCamera = normalized === "game" ? this.gameplayCamera : this.camera;
        this.snap = true;
        this.resetSmart();
        this.overviewDirty = true;
    }

    setBoard(board, playback = {}) {
        const previous = this.board;
        const key = `${board.seed}:${board.level}:${board.fullSurvival}:${board.gridSize}:${board.episode ?? ""}`;
        const newEpisode = key !== this.episodeKey || Boolean(previous && board.steps < previous.steps);
        const sampledChanged = Boolean(playback.sampled) !== this.playback.sampled;
        const reducedChanged = Boolean(playback.reducedMotion) !== this.playback.reducedMotion;
        this.board = board;
        this.playback = {
            paused: Boolean(playback.paused), sampled: Boolean(playback.sampled),
            reducedMotion: Boolean(playback.reducedMotion),
        };
        this.solidCells = new Map([
            ...(board.obstacles || []), ...(board.movingWalls || []).flatMap((wall) => wall.cells),
            ...(board.gates || []).filter((gate) => gate.state === "active").flatMap((gate) => gate.cells),
        ].map((point) => [cellKey(point), point]));
        this.solids = new Set(this.solidCells.keys());
        if (newEpisode) {
            this.episodeKey = key;
            this.seenEvents.clear();
            this.cooldowns.clear();
            this.presentation = null;
            this.dismissedContactId = null;
            this.eventKey = null;
            this.contactShot = null;
            this.recapShot = null;
            this.eventRig = newRig();
            this.lastEvent = null;
            this.intro = null;
            this.renderState = null;
            this.shot = { name: "follow", until: 0, priority: 0 };
            this.resetSmart();
            this.snap = true;
            this.overviewDirty = true;
        } else if (sampledChanged) {
            this.resetSmart();
            this.eventKey = null;
            this.contactShot = null;
            this.recapShot = null;
            if (this.playback.sampled) {
                this.presentation = null;
                this.lastEvent = null;
                this.dismissedContactId = board.contact?.id || null;
                this.overviewDirty = true;
            }
        }
        if (reducedChanged && this.playback.reducedMotion) {
            for (const rig of [this.smart, this.eventRig]) {
                rig.goalAzimuth = rig.azimuth;
                rig.goalElevation = rig.elevation;
            }
            this.smart.needsFit = true;
        }
        if (previous && !newEpisode && board.steps !== previous.steps && !this.playback.sampled) {
            const s = this.smart;
            if (previous.direction && board.direction && cellKey(previous.direction) !== cellKey(board.direction)) {
                s.turns.push(s.clock);
            }
            s.turns = s.turns.filter((time) => s.clock - time < 1.8);
            if (s.turns.length >= 3 || board.threat >= 0.5 || board.abilities?.dash?.active) {
                s.busyUntil = s.clock + 1.6;
            }
        }
        for (const event of board.events || []) {
            if (this.seenEvents.has(event.id)) continue;
            this.seenEvents.add(event.id);
            if (event.type === "death") this.offer("death", 100, 1.4);
            else if (event.type === "food" || event.type === "gold_food") this.offer("food", 40, 1);
            else if (event.type === "escape") this.offer("escape", 60, 1.3);
        }
        if (this.seenEvents.size > 128) this.seenEvents = new Set((board.events || []).map((event) => event.id));
        this.lastVisibility = -Infinity;
    }

    setViewport(width, height) {
        this.viewport = {
            width: Number.isFinite(width) ? Math.max(1, width) : 800,
            height: Number.isFinite(height) ? Math.max(1, height) : 600,
        };
        const aspect = this.viewport.width / this.viewport.height;
        this.camera.aspect = aspect;
        this.camera.near = 0.1;
        this.camera.far = Math.max(300, (this.board?.gridSize || 32) * 10);
        this.camera.updateProjectionMatrix();
        this.gameplayCamera.left = -this.smart.halfHeight * aspect;
        this.gameplayCamera.right = this.smart.halfHeight * aspect;
        this.gameplayCamera.top = this.smart.halfHeight;
        this.gameplayCamera.bottom = -this.smart.halfHeight;
        this.gameplayCamera.updateProjectionMatrix();
        this.smart.needsFit = true;
        this.overviewDirty = true;
        this.lastVisibility = -Infinity;
    }

    setPresentation(presentation) {
        if (!presentation) {
            this.presentation = null;
            this.dismissedContactId = this.board?.contact?.id || this.eventKey;
            this.eventKey = null;
            this.contactShot = null;
            this.recapShot = null;
            this.lastEvent = null;
            this.smart.needsFit = true;
            this.lastVisibility = -Infinity;
            return;
        }
        this.presentation = {
            phase: presentation.phase || "impact", progress: clamp(presentation.progress || 0, 0, 1),
            contact: presentation.contact || null,
        };
    }

    resetSmart() {
        this.smart.initialized = false;
        this.smart.needsFit = true;
        this.smart.turns = [];
        this.smart.busyUntil = 0;
        this.smart.lastPlan = -Infinity;
        this.lastVisibility = -Infinity;
    }

    player() { return { id: "player", type: "player", snake: this.board.snake, direction: this.board.direction }; }

    basis(actor) {
        const forward = vector(actor.direction || { x: 0, y: 0, z: 1 }).normalize();
        const horizontal = new THREE.Vector3(forward.x, 0, forward.z);
        if (horizontal.lengthSq() < 0.01) horizontal.set(0, 0, 1);
        horizontal.normalize();
        return { forward, side: new THREE.Vector3().crossVectors(horizontal, UP).normalize(), up: UP.clone() };
    }

    actorPoints(actor) {
        const rendered = actor.id === "player" ? this.renderState?.player
            : this.renderState?.enemies?.find((enemy) => enemy.id === actor.id)?.points;
        return rendered?.length ? rendered.map(vector) : (actor.snake || []).map((point) => this.cellToScene(point));
    }

    addActor(subjects, actor, points = this.actorPoints(actor)) {
        points.forEach((point, index) => subjects.push({
            point, radius: BODY_RADIUS, role: actor.type || "player", id: actor.id, head: index === 0,
        }));
    }

    smartSubjects({ context = true } = {}) {
        const subjects = [];
        const player = this.player();
        this.addActor(subjects, player);
        if (!subjects.length) return subjects;
        const head = subjects[0].point;
        const enemies = (this.board.enemies || []).map((actor) => ({ actor, points: this.actorPoints(actor) }))
            .filter(({ points }) => points.length)
            .map((item) => ({
                ...item, distance: Math.min(...item.points.map((point) => point.distanceTo(head))),
            })).sort((a, b) => a.distance - b.distance);
        const near = enemies.filter(({ distance }, index) => distance <= (index === 0 ? 13 : 7)).slice(0, 2);
        for (const { actor, points } of near) this.addActor(subjects, actor, points);
        if (context) {
            const forward = this.basis(player).forward;
            subjects.push({ point: head.clone().addScaledVector(forward, 3.2), radius: 1.1, role: "space" });
            const food = (this.board.foods || []).map((item) => this.cellToScene(item.position))
                .sort((a, b) => a.distanceToSquared(head) - b.distanceToSquared(head))[0];
            if (food && food.distanceTo(head) < 8) subjects.push({ point: food, radius: 0.65, role: "food" });
        }
        return subjects;
    }

    overviewSubjects() {
        const half = this.board.gridSize / 2;
        const subjects = [];
        for (const x of [-half, half]) for (const y of [-half, half]) for (const z of [-half, half]) {
            subjects.push({ point: new THREE.Vector3(x, y, z), radius: 0.6, role: "space" });
        }
        return subjects;
    }

    safeFrame() {
        const overlayPixels = this.viewport.width <= 320 ? 200 : this.viewport.width <= 520 ? 184 : 140;
        return {
            ...SAFE,
            bottom: this.eventFraming ? Math.min(SAFE.bottom, Math.max(0.06, 1 - overlayPixels * 2 / this.viewport.height)) : SAFE.bottom,
        };
    }

    compose(subjects, direction, minimum = 5.6, padding = 1.05) {
        const safe = this.safeFrame();
        const { right, up } = screenBasis(direction);
        const anchor = new THREE.Box3().setFromPoints(subjects.map(({ point }) => point)).getCenter(new THREE.Vector3());
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const { point, radius = BODY_RADIUS } of subjects) {
            const relative = point.clone().sub(anchor);
            const x = relative.dot(right), y = relative.dot(up), r = radius + padding;
            minX = Math.min(minX, x - r); maxX = Math.max(maxX, x + r);
            minY = Math.min(minY, y - r); maxY = Math.max(maxY, y + r);
        }
        const aspect = this.viewport.width / this.viewport.height;
        const halfHeight = Math.max(minimum, (maxX - minX) / (2 * aspect * safe.x), (maxY - minY) / (safe.top + safe.bottom));
        const center = anchor.addScaledVector(right, (minX + maxX) / 2)
            .addScaledVector(up, (minY + maxY) / 2 - halfHeight * (safe.top - safe.bottom) / 2);
        return { center, halfHeight, direction };
    }

    orthoFit(subjects, center, direction) {
        const safe = this.safeFrame();
        const { right, up } = screenBasis(direction);
        const aspect = this.viewport.width / this.viewport.height;
        let halfHeight = 0;
        for (const { point, radius = BODY_RADIUS } of subjects) {
            const relative = point.clone().sub(center), x = relative.dot(right), y = relative.dot(up);
            halfHeight = Math.max(halfHeight, (Math.abs(x) + radius) / (aspect * safe.x),
                (y + radius) / safe.top, (-y + radius) / safe.bottom);
        }
        return halfHeight;
    }

    fitSmartDistance(points, center, direction, fov = 48) {
        const safe = this.safeFrame();
        const { right, up } = screenBasis(direction);
        const tangent = Math.tan(THREE.MathUtils.degToRad(fov / 2));
        const aspect = this.viewport.width / this.viewport.height;
        let distance = 9;
        for (const item of points) {
            const point = item.point || item, radius = item.radius ?? BODY_RADIUS;
            const relative = point.clone().sub(center), x = relative.dot(right), y = relative.dot(up);
            distance = Math.max(distance, relative.dot(direction) + 6, relative.dot(direction) + radius + Math.max(
                (Math.abs(x) + radius) / (tangent * aspect * safe.x),
                (y + radius) / (tangent * safe.top), (-y + radius) / (tangent * safe.bottom),
            ));
        }
        return distance + 0.25;
    }

    visibilitySubjects(subjects) {
        const actors = new Map();
        for (const subject of subjects) {
            if (["space", "food"].includes(subject.role)) continue;
            const key = subject.id || subject.role;
            if (!actors.has(key)) actors.set(key, []);
            actors.get(key).push(subject);
        }
        return [...actors.values()].flatMap((points) => {
            if (points.length <= 9) return points;
            return Array.from({ length: 9 }, (_, i) => points[Math.round(i * (points.length - 1) / 8)]);
        });
    }

    rayCells(target, position, skip = 0.72) {
        if (!this.solids.size) return [];
        const delta = position.clone().sub(target), length = delta.length();
        if (length <= skip) return [];
        delta.divideScalar(length);
        const start = target.clone().addScaledVector(delta, skip).addScalar(this.board.gridSize / 2);
        const axes = ["x", "y", "z"];
        const cell = { x: Math.floor(start.x), y: Math.floor(start.y), z: Math.floor(start.z) };
        const step = {}, next = {}, stride = {};
        for (const axis of axes) {
            step[axis] = Math.sign(delta[axis]);
            stride[axis] = step[axis] ? Math.abs(1 / delta[axis]) : Infinity;
            next[axis] = step[axis] ? ((cell[axis] + (step[axis] > 0 ? 1 : 0)) - start[axis]) / delta[axis] : Infinity;
        }
        const hits = [];
        let distance = 0;
        for (let count = 0; count < this.board.gridSize * 6 + 12 && distance <= length - skip; count += 1) {
            const hit = this.solidCells.get(cellKey(cell));
            if (hit) hits.push(hit);
            const boundary = Math.min(next.x, next.y, next.z);
            if (!Number.isFinite(boundary)) break;
            for (const axis of axes) if (next[axis] <= boundary + 1e-8) {
                cell[axis] += step[axis];
                next[axis] += stride[axis];
            }
            distance = boundary;
            if (axes.some((axis) => cell[axis] < -1 || cell[axis] > this.board.gridSize)) break;
        }
        return hits;
    }

    clearView(target, position, skip = 0.72) { return this.rayCells(target, position, skip).length === 0; }

    cameraClearance(position) {
        let clearance = 1000;
        for (const cell of this.solidCells.values()) {
            const center = this.cellToScene(cell);
            const dx = Math.max(0, Math.abs(position.x - center.x) - 0.5);
            const dy = Math.max(0, Math.abs(position.y - center.y) - 0.5);
            const dz = Math.max(0, Math.abs(position.z - center.z) - 0.5);
            clearance = Math.min(clearance, Math.hypot(dx, dy, dz));
        }
        return clearance;
    }

    clearPosition(position) {
        const half = this.board.gridSize / 2;
        const cell = { x: Math.floor(position.x + half), y: Math.floor(position.y + half), z: Math.floor(position.z + half) };
        for (let x = cell.x - 1; x <= cell.x + 1; x += 1) {
            for (let y = cell.y - 1; y <= cell.y + 1; y += 1) for (let z = cell.z - 1; z <= cell.z + 1; z += 1) {
                const solid = this.solidCells.get(`${x},${y},${z}`);
                if (!solid) continue;
                const center = this.cellToScene(solid);
                const distance = Math.hypot(
                    Math.max(0, Math.abs(position.x - center.x) - 0.5),
                    Math.max(0, Math.abs(position.y - center.y) - 0.5),
                    Math.max(0, Math.abs(position.z - center.z) - 0.5),
                );
                if (distance < CLEARANCE) return false;
            }
        }
        return position.y > -half - 0.3;
    }

    outsideDistance(center) {
        return Math.max(this.board.gridSize * 1.15, center.length() + this.board.gridSize * Math.sqrt(3) / 2 + 3);
    }

    safePosition(target, preferred, _up = UP, skip = 0.72) {
        const displacement = preferred.clone().sub(target);
        const distance = Math.max(6, displacement.length());
        if (displacement.lengthSq() < 0.01) displacement.copy(directionAt(HOME_AZIMUTH, HOME_ELEVATION));
        displacement.normalize();
        let fallback = null;
        for (const angle of [0, 0.35, -0.35, 0.7, -0.7, 1.2, -1.2]) {
            const direction = displacement.clone().applyAxisAngle(UP, angle);
            direction.y = Math.max(0.24, direction.y);
            direction.normalize();
            const candidate = target.clone().addScaledVector(direction, distance);
            if (!this.clearPosition(candidate)) candidate.copy(target).addScaledVector(direction, this.outsideDistance(target));
            if (!fallback) fallback = candidate;
            if (this.clearPosition(candidate) && this.clearView(target, candidate, skip)) return candidate;
        }
        return fallback;
    }

    planSmart({ locked = false } = {}) {
        const s = this.smart, sampled = this.playback.sampled;
        const subjects = sampled ? this.overviewSubjects() : this.smartSubjects();
        const visible = this.visibilitySubjects(subjects);
        const busy = s.clock < s.busyUntil;
        const reduced = this.playback.reducedMotion;
        if (sampled) s.goalAzimuth = HOME_AZIMUTH;
        const angles = new Set([s.goalAzimuth]);
        if (!sampled && !reduced && !locked) {
            for (const offset of [-1.05, -0.7, -0.35, 0, 0.35, 0.7, 1.05]) angles.add(HOME_AZIMUTH + offset);
        }
        let best = null, current = null;
        for (const azimuth of angles) {
            const direction = directionAt(azimuth, HOME_ELEVATION);
            const frame = this.compose(subjects, direction, sampled ? 0 : 5.6);
            const position = frame.center.clone().addScaledVector(direction, this.outsideDistance(frame.center));
            const hidden = visible.filter(({ point }) => !this.clearView(point, position)).length;
            const head = subjects.find((subject) => subject.id === "player" && subject.head);
            const enemy = subjects.find((subject) => subject.id !== "player" && subject.id && subject.head);
            const separation = head && enemy ? Math.abs(enemy.point.clone().sub(head.point).dot(screenBasis(direction).right)) : 5;
            const overlap = enemy ? Math.max(0, Math.min(3, head.point.distanceTo(enemy.point) * 0.65) - separation) : 0;
            const score = hidden * 4 + frame.halfHeight * 0.16 + overlap * 0.75
                + Math.abs(wrap(azimuth - HOME_AZIMUTH)) * 0.12;
            const candidate = { ...frame, azimuth, hidden, score };
            if (azimuth === s.goalAzimuth) current = candidate;
            if (!best || score < best.score) best = candidate;
        }
        if (current && current.score - best.score < (busy ? 0.14 : 0.075)) best = current;
        s.goalAzimuth = best.azimuth;
        s.goalDirection.copy(best.direction);
        s.goalCenter.copy(best.center);
        s.goalHalfHeight = best.halfHeight;
        s.goalDistance = this.outsideDistance(best.center);
        s.lastPlan = s.clock;
        s.needsFit = false;
        s.plans += 1;
    }

    applySmartRig() {
        const s = this.smart;
        s.direction.copy(directionAt(s.azimuth, s.elevation));
        s.distance = this.outsideDistance(s.center);
        const camera = this.gameplayCamera, aspect = this.viewport.width / this.viewport.height;
        camera.position.copy(s.center).addScaledVector(s.direction, s.distance);
        camera.up.copy(UP);
        camera.near = 0.1;
        camera.far = Math.max(300, s.distance + this.board.gridSize * 3);
        camera.zoom = 1;
        camera.left = -s.halfHeight * aspect; camera.right = s.halfHeight * aspect;
        camera.top = s.halfHeight; camera.bottom = -s.halfHeight;
        camera.lookAt(s.center);
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld();
        this.look.copy(s.center);
        this._activeCamera = camera;
    }

    updateSmart(dt) {
        const s = this.smart, sampled = this.playback.sampled;
        const subjects = sampled ? this.overviewSubjects() : this.smartSubjects();
        if ((this.playback.paused || this.board.done) && s.initialized && !s.needsFit) {
            this._activeCamera = this.gameplayCamera;
            s.phase = this.board.done ? "episode end" : "paused";
            this.updateVisibility(subjects);
            return;
        }
        if (!s.initialized || s.needsFit) {
            this.planSmart();
            if (!s.initialized) {
                s.center.copy(s.goalCenter); s.azimuth = s.goalAzimuth;
                s.halfHeight = s.goalHalfHeight; s.initialized = true;
            }
        }
        if (this.playback.paused || sampled || this.board.done) {
            s.phase = sampled ? "training overview" : this.board.done ? "episode end" : "paused";
            s.halfHeight = Math.max(s.halfHeight, this.orthoFit(subjects, s.center, directionAt(s.azimuth, s.elevation)));
            this.applySmartRig();
            this.updateVisibility(subjects);
            return;
        }
        s.clock += dt;
        const busy = s.clock < s.busyUntil;
        if (s.clock - s.lastPlan >= (busy ? 0.65 : 0.9)) this.planSmart();
        const currentFrame = this.compose(subjects, directionAt(s.goalAzimuth, s.goalElevation));
        s.goalCenter.copy(currentFrame.center);
        s.goalHalfHeight = currentFrame.halfHeight;
        s.phase = busy ? "following action" : "gentle reframe";
        const reduced = this.playback.reducedMotion;
        move(s.center, s.goalCenter, reduced ? 3.5 : 6, dt);
        const turnSpeed = reduced ? 0 : busy ? 0.1 : 0.16;
        s.azimuth += clamp(wrap(s.goalAzimuth - s.azimuth), -turnSpeed * dt, turnSpeed * dt);
        s.halfHeight = approach(s.halfHeight, s.goalHalfHeight, reduced ? 1.4 : 2.8, dt);
        s.direction.copy(directionAt(s.azimuth, s.elevation));
        // Safety expansion is independent of smoothing: a real step must not crop the tail or a new threat.
        s.halfHeight = Math.max(s.halfHeight, this.orthoFit(subjects, s.center, s.direction));
        this.applySmartRig();
        this.updateVisibility(subjects);
        this.snap = false;
    }

    eventSubjects(contact) {
        const subjects = [];
        const actors = [contact?.victim, contact?.attacker].filter(Boolean);
        if (!actors.length) this.addActor(subjects, this.player());
        for (const actor of actors) {
            const id = actor.id || (actor === contact.victim ? "player" : contact.sourceId);
            const entity = { id, type: actor.type || (id === "player" ? "player" : "enemy"), snake: actor.before || [] };
            const rendered = id === "player" ? this.renderState?.player
                : this.renderState?.enemies?.find((enemy) => enemy.id === id)?.points;
            if (rendered?.length) this.addActor(subjects, entity, rendered.map(vector));
            for (const points of [actor.before, actor.attempted]) {
                if (points?.length) this.addActor(subjects, entity, points.map((point) => this.cellToScene(point)));
            }
            for (const point of [actor.from, actor.to]) if (point) {
                subjects.push({ point: this.cellToScene(point), radius: BODY_RADIUS, role: entity.type, id });
            }
        }
        if (contact?.position) subjects.push({
            point: this.cellToScene(contact.position), radius: 0.8, role: "contact", id: "contact",
        });
        return subjects;
    }

    effectivePresentation() {
        if (this.presentation) return this.presentation;
        const contact = this.board.contact || (this.board.events || []).find((event) => event.type === "death")?.contact;
        if (!this.playback.sampled && this.board.done && contact && contact.id !== this.dismissedContactId) {
            return { phase: "impact", progress: 0, contact };
        }
        return null;
    }

    contactBody(actor, fraction) {
        const before = actor.before?.length ? actor.before : [actor.from];
        const attempted = actor.attempted?.length ? actor.attempted : [actor.to || actor.from];
        return Array.from({ length: Math.max(before.length, attempted.length) }, (_, index) =>
            this.cellToScene(before[Math.min(index, before.length - 1)])
                .lerp(this.cellToScene(attempted[Math.min(index, attempted.length - 1)]), fraction));
    }

    incomingAxis(contact) {
        const attack = contact.attacker && ["head-on", "crossing", "body-strike"].includes(contact.kind);
        const source = attack ? contact.attacker : contact.victim;
        const axis = vector(source.to).sub(vector(source.from));
        if (axis.lengthSq() < 0.001) axis.copy(vector(contact.victim.to).sub(vector(contact.victim.from)));
        if (axis.lengthSq() < 0.001) axis.set(0, 0, 1);
        return { axis: axis.normalize(), attack: Boolean(attack) };
    }

    eventEnvelope(contact, subjects) {
        const envelope = [...subjects];
        const { axis, attack } = this.incomingAxis(contact);
        const recoil = axis.clone().multiplyScalar(attack ? 1.25 : -1.25);
        const body = this.contactBody(contact.victim, clamp(contact.timeFraction ?? 1, 0, 1));
        const bounds = new THREE.Box3().setFromPoints(body);
        bounds.min.add(new THREE.Vector3(Math.min(0, recoil.x) - 0.52,
            Math.min(0, recoil.y, contact.outcome === "death" ? -2.2 : 0) - 0.08, Math.min(0, recoil.z) - 0.52));
        bounds.max.add(new THREE.Vector3(Math.max(0, recoil.x) + 0.52,
            Math.max(0, recoil.y) + 0.08, Math.max(0, recoil.z) + 0.52));
        bounds.min.clampScalar(0.48 - this.board.gridSize / 2, this.board.gridSize / 2 - 0.48);
        bounds.max.clampScalar(0.48 - this.board.gridSize / 2, this.board.gridSize / 2 - 0.48);
        for (const x of [bounds.min.x, bounds.max.x]) for (const y of [bounds.min.y, bounds.max.y]) {
            for (const z of [bounds.min.z, bounds.max.z]) {
                envelope.push({ point: new THREE.Vector3(x, y, z), radius: BODY_RADIUS, role: "space" });
            }
        }
        return envelope;
    }

    fitEventFrame(subjects, direction, minimumDistance = 0) {
        const safe = this.safeFrame(), { right, up } = screenBasis(direction);
        const anchor = new THREE.Box3().setFromPoints(subjects.map(({ point }) => point)).getCenter(new THREE.Vector3());
        const tangent = Math.tan(THREE.MathUtils.degToRad(EVENT_FOV / 2));
        const horizontal = tangent * this.camera.aspect * safe.x;
        const top = tangent * safe.top, bottom = tangent * safe.bottom;
        const coordinates = subjects.map(({ point, radius = BODY_RADIUS }) => {
            const relative = point.clone().sub(anchor);
            return { x: relative.dot(right), y: relative.dot(up), z: relative.dot(direction), radius: radius + 0.1 };
        });
        const limits = (distance) => {
            let left = -Infinity, rightEdge = Infinity, lower = -Infinity, upper = Infinity;
            for (const { x, y, z, radius } of coordinates) {
                const depth = distance - z - radius;
                left = Math.max(left, x + radius - horizontal * depth);
                rightEdge = Math.min(rightEdge, x - radius + horizontal * depth);
                lower = Math.max(lower, y + radius - top * depth);
                upper = Math.min(upper, y - radius + bottom * depth);
            }
            return { left, right: rightEdge, lower, upper, fits: left <= rightEdge && lower <= upper };
        };
        let low = Math.max(minimumDistance, ...coordinates.map(({ z, radius }) => z + radius + 2));
        let high = low + 4;
        while (!limits(high).fits) high = high * 1.5 + 2;
        for (let iteration = 0; iteration < 22; iteration += 1) {
            const middle = (low + high) / 2;
            if (limits(middle).fits) high = middle; else low = middle;
        }
        const distance = high + 0.03, range = limits(distance);
        const center = anchor.addScaledVector(right, (range.left + range.right) / 2)
            .addScaledVector(up, (range.lower + range.upper) / 2);
        return {
            center, distance, direction: direction.clone(), position: center.clone().addScaledVector(direction, distance),
            viewport: `${this.viewport.width}:${this.viewport.height}`, expanded: false,
        };
    }

    planContactShot(contact, subjects) {
        const { axis } = this.incomingAxis(contact);
        const head = this.contactBody(contact.victim, clamp(contact.timeFraction ?? 1, 0, 1))[0];
        const target = this.cellToScene(contact.position), tangent = head.clone().sub(target);
        if (tangent.lengthSq() < 0.01) tangent.copy(vector(contact.victim.to).sub(vector(contact.victim.from)));
        const profile = new THREE.Vector3().crossVectors(axis, tangent);
        if (Math.hypot(profile.x, profile.z) < 0.1) profile.crossVectors(UP, axis);
        const prior = this.activeCamera.position.clone().sub(this.look).normalize();
        const priorAzimuth = Math.atan2(prior.x, prior.z);
        const azimuth = Math.hypot(profile.x, profile.z) > 0.1 ? Math.atan2(profile.x, profile.z) : priorAzimuth;
        const candidates = new Set([priorAzimuth]);
        for (const side of [azimuth, azimuth + Math.PI]) {
            for (const offset of [0, -0.18, 0.18]) candidates.add(side + offset);
        }
        const envelope = this.eventEnvelope(contact, subjects);
        const visible = this.visibilitySubjects(subjects);
        let best = null;
        for (const angle of candidates) for (const elevation of [axis.y * 0.1, 0, 0.18]) {
            const direction = directionAt(angle, elevation), frame = this.fitEventFrame(envelope, direction);
            const clear = this.clearPosition(frame.position);
            const hidden = visible.filter(({ point }) => !this.clearView(point, frame.position)).length;
            const headCover = tangent.lengthSq() > 0.01 ? Math.max(0, Math.abs(tangent.clone().normalize().dot(direction)) - 0.5) : 0;
            const foreshortening = Math.max(0, Math.abs(axis.dot(direction)) - 0.18);
            const score = (clear ? 0 : 1000) + hidden * 4 + foreshortening * foreshortening * 20 + headCover * 5
                + Math.abs(wrap(angle - priorAzimuth)) * 0.08 + Math.abs(elevation - axis.y * 0.1) * 0.15 + frame.distance * 0.006;
            if (!best || score < best.score) best = { frame, score };
        }
        let frame = best.frame;
        if (!this.clearPosition(frame.position)) {
            frame = this.fitEventFrame(envelope, frame.direction, this.outsideDistance(frame.center));
        }
        frame.envelope = envelope;
        return frame;
    }

    applyEventShot(shot, subjects) {
        const viewport = `${this.viewport.width}:${this.viewport.height}`;
        if (shot.viewport !== viewport) {
            Object.assign(shot, this.fitEventFrame(shot.envelope, shot.direction));
        }
        const camera = this.camera;
        const apply = () => {
            camera.position.copy(shot.position); camera.up.copy(UP); camera.fov = EVENT_FOV;
            camera.far = Math.max(300, shot.distance + this.board.gridSize * 3);
            camera.lookAt(shot.center); camera.updateProjectionMatrix(); camera.updateMatrixWorld();
            this.look.copy(shot.center);
            this._activeCamera = camera;
        };
        apply();
        if (!this.subjectsFramed(subjects) || !this.clearPosition(shot.position)) {
            const { right, up } = screenBasis(shot.direction), safe = this.safeFrame();
            const tangent = Math.tan(THREE.MathUtils.degToRad(EVENT_FOV / 2));
            let distance = shot.distance;
            for (const { point, radius = BODY_RADIUS } of subjects) {
                const relative = point.clone().sub(shot.center), x = relative.dot(right), y = relative.dot(up);
                distance = Math.max(distance, relative.dot(shot.direction) + radius + Math.max(
                    2, (Math.abs(x) + radius) / (tangent * this.camera.aspect * safe.x),
                    (y + radius) / (tangent * safe.top), (-y + radius) / (tangent * safe.bottom),
                ) + 0.03);
            }
            shot.distance = distance;
            shot.position.copy(shot.center).addScaledVector(shot.direction, distance);
            if (!this.clearPosition(shot.position)) {
                shot.distance = Math.max(distance, this.outsideDistance(shot.center));
                shot.position.copy(shot.center).addScaledVector(shot.direction, shot.distance);
            }
            shot.expanded = true;
            apply();
        }
        const rig = this.eventRig;
        rig.initialized = true;
        rig.center.copy(shot.center); rig.direction.copy(shot.direction);
        rig.azimuth = Math.atan2(shot.direction.x, shot.direction.z);
        rig.elevation = Math.asin(clamp(shot.direction.y, -1, 1));
        rig.distance = shot.distance;
        rig.halfHeight = shot.distance * Math.tan(THREE.MathUtils.degToRad(EVENT_FOV / 2));
        this.updateVisibility(subjects);
    }

    eventBodyCoverage(contact) {
        const subjects = [];
        for (const actor of [contact.victim, contact.attacker].filter(Boolean)) {
            this.addActor(subjects, { ...actor, snake: actor.before });
        }
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1);
        let left = Infinity, rightEdge = -Infinity, bottom = Infinity, top = -Infinity;
        for (const { point, radius } of subjects) for (const offset of [
            right.clone().multiplyScalar(radius), right.clone().multiplyScalar(-radius),
            up.clone().multiplyScalar(radius), up.clone().multiplyScalar(-radius),
        ]) {
            const projected = point.clone().add(offset).project(this.camera);
            left = Math.min(left, projected.x); rightEdge = Math.max(rightEdge, projected.x);
            bottom = Math.min(bottom, projected.y); top = Math.max(top, projected.y);
        }
        const safe = this.safeFrame();
        const widthFraction = (rightEdge - left) / (safe.x * 2), heightFraction = (top - bottom) / (safe.top + safe.bottom);
        return { widthFraction, heightFraction, occupiedFraction: Math.max(widthFraction, heightFraction), bounds: { left, right: rightEdge, bottom, top } };
    }

    updatePerspective(subjects, rig, dt, { azimuth, elevation = HOME_ELEVATION, minimum = 4.8, fov = 48, padding = 1.1 } = {}) {
        const direction = directionAt(azimuth ?? rig.azimuth, elevation);
        const frame = this.compose(subjects, direction, minimum, padding);
        if (!rig.initialized) {
            const source = this.smart.initialized ? this.smart : null;
            rig.center.copy(source?.center || frame.center);
            rig.azimuth = source?.azimuth ?? azimuth ?? HOME_AZIMUTH;
            rig.elevation = source?.elevation ?? elevation;
            rig.distance = source ? source.halfHeight / Math.tan(THREE.MathUtils.degToRad(fov / 2)) : 16;
            rig.initialized = true;
        }
        move(rig.center, frame.center, this.playback.reducedMotion ? 3 : 7, dt);
        const turnSpeed = this.playback.reducedMotion ? 0 : 0.16;
        rig.azimuth += clamp(wrap((azimuth ?? rig.azimuth) - rig.azimuth), -turnSpeed * dt, turnSpeed * dt);
        rig.elevation = approach(rig.elevation, elevation, this.playback.reducedMotion ? 0 : 0.09, dt);
        rig.direction.copy(directionAt(rig.azimuth, rig.elevation));
        const framingPoints = [...subjects, { point: frame.center, radius: minimum * 0.45 + padding, role: "space" }];
        rig.goalDistance = this.fitSmartDistance(framingPoints, rig.center, rig.direction, fov);
        rig.distance = Math.max(rig.goalDistance, approach(rig.distance, rig.goalDistance, 5, dt));
        const position = rig.center.clone().addScaledVector(rig.direction, rig.distance);
        if (!this.clearPosition(position)) {
            rig.distance = Math.max(rig.distance, this.outsideDistance(rig.center));
            position.copy(rig.center).addScaledVector(rig.direction, rig.distance);
        }
        const camera = this.camera;
        camera.position.copy(position); camera.up.copy(UP); camera.fov = fov;
        camera.far = Math.max(300, rig.distance + this.board.gridSize * 3);
        camera.lookAt(rig.center); camera.updateProjectionMatrix(); camera.updateMatrixWorld();
        rig.halfHeight = rig.distance * Math.tan(THREE.MathUtils.degToRad(fov / 2));
        this.look.copy(rig.center);
        this._activeCamera = camera;
        this.updateVisibility(subjects);
    }

    updateEvent(dt, presentation) {
        this.eventFraming = true;
        const contact = presentation.contact, key = contact?.id || "non-contact";
        const subjects = this.eventSubjects(contact);
        if (!contact) {
            this.updatePerspective(subjects, this.eventRig, dt, { minimum: 4.6 });
            this.lastEvent = { phase: presentation.phase, contactId: null, pairFramed: this.subjectsFramed(subjects), actors: 1 };
            return;
        }
        if (key !== this.eventKey) {
            this.eventKey = key;
            this.eventRig = newRig();
            this.contactShot = null;
            this.recapShot = null;
            this.lastVisibility = -Infinity;
        }
        if (!this.contactShot) this.contactShot = this.planContactShot(contact, subjects);
        const recap = presentation.phase === "recap";
        if (recap && !this.recapShot) {
            const envelope = [...this.contactShot.envelope, ...subjects];
            this.recapShot = { ...this.fitEventFrame(envelope, this.contactShot.direction), envelope };
        }
        // Establish one matched cut before the recorded approach; never fly into a bite.
        const shot = recap ? this.recapShot : this.contactShot;
        this.applyEventShot(shot, subjects);
        this.smart.phase = `contact ${presentation.phase}`;
        this.lastEvent = {
            phase: presentation.phase, contactId: contact.id, shot: recap ? "recorded-recap" : "contact-profile",
            established: true, safetyExpanded: shot.expanded, profileAlignment: Math.abs(this.incomingAxis(contact).axis.dot(shot.direction)),
            bodyCoverage: this.eventBodyCoverage(contact),
            pairFramed: this.subjectsFramed(subjects), actors: new Set(subjects.filter(({ role }) => role !== "contact").map(({ id }) => id)).size,
        };
    }

    updateStatic() {
        this._activeCamera = this.camera;
        if (!this.overviewDirty) return;
        const subjects = this.overviewSubjects(), direction = directionAt(HOME_AZIMUTH, HOME_ELEVATION);
        const frame = this.compose(subjects, direction, 0, 0.7);
        const distance = this.fitSmartDistance(subjects, frame.center, direction, 50);
        this.camera.position.copy(frame.center).addScaledVector(direction, distance);
        this.camera.up.copy(UP); this.camera.fov = 50;
        this.camera.lookAt(frame.center); this.camera.updateProjectionMatrix(); this.camera.updateMatrixWorld();
        this.look.copy(frame.center);
        this.overviewDirty = false;
        this.smart.phase = "static overview";
        this.updateVisibility(subjects, true);
    }

    updateFilm(dt = 0) {
        const s = this.smart, forward = this.basis(this.player()).forward;
        if (!s.initialized) {
            s.azimuth = HOME_AZIMUTH; s.goalAzimuth = HOME_AZIMUTH;
        }
        let azimuth = HOME_AZIMUTH, elevation = HOME_ELEVATION;
        if (["chase", "cine-chase", "follow"].includes(this.mode)) {
            if (Math.hypot(forward.x, forward.z) > 0.2) {
                s.goalAzimuth = Math.atan2(-forward.x, -forward.z) + (this.mode === "cine-chase" ? 0.28 : 0);
            }
            azimuth = s.goalAzimuth; elevation = 0.46;
        } else if (this.mode === "orbit") azimuth += Math.sin(s.clock * 0.06) * 0.55;
        else if (this.mode === "cinema") { azimuth += 0.3; elevation = 0.72; }
        if (this.playback.paused && s.initialized && !s.needsFit) {
            this._activeCamera = this.camera;
            s.phase = "paused";
            return;
        }
        s.clock += dt;
        this.updatePerspective(this.playback.sampled ? this.overviewSubjects() : this.smartSubjects(), s, dt, {
            azimuth, elevation, minimum: 5.4,
        });
        s.needsFit = false;
        s.phase = this.playback.sampled ? "training overview" : "bounded advanced";
    }

    offer(name, priority, duration) {
        if (this.intro || this.now < (this.cooldowns.get(name) || 0)) return;
        if (this.now < this.shot.until && priority <= this.shot.priority) return;
        this.shot = { name, priority, until: this.now + duration };
        this.cooldowns.set(name, this.now + duration + 2.5);
    }

    face(actor, drift = 0, returning = false) {
        const points = this.actorPoints(actor), head = points[0] || new THREE.Vector3();
        const { forward } = this.basis(actor);
        let azimuth = Math.hypot(forward.x, forward.z) > 0.2 ? Math.atan2(forward.x, forward.z) : HOME_AZIMUTH;
        azimuth += this.playback.reducedMotion ? 0.22 : 0.22 + drift * 0.18;
        if (returning && !this.playback.reducedMotion) azimuth += ease(drift) * 0.5;
        const target = head.clone().add(new THREE.Vector3(0, -0.15, 0));
        const preferred = target.clone().addScaledVector(directionAt(azimuth, returning ? 0.5 : 0.28), returning ? 11 : 7);
        this.pose.target.copy(target);
        this.pose.position.copy(this.safePosition(target, preferred));
        this.pose.up.copy(UP); this.pose.fov = returning ? 50 : 46;
    }

    foodPose() {
        const food = (this.board.foods || []).find((item) => item.type === "gold") || this.board.foods?.[0];
        if (!food) { this.face(this.player()); return; }
        const target = this.cellToScene(food.position);
        this.pose.target.copy(target);
        this.pose.position.copy(this.safePosition(target, target.clone().addScaledVector(directionAt(HOME_AZIMUTH, 0.6), 9)));
        this.pose.up.copy(UP); this.pose.fov = 48;
    }

    dangerPose() {
        const subjects = this.overviewSubjects(), direction = directionAt(HOME_AZIMUTH, 0.8);
        const frame = this.compose(subjects, direction, 0, 0.6);
        this.pose.target.copy(frame.center);
        this.pose.position.copy(frame.center).addScaledVector(direction, this.fitSmartDistance(subjects, frame.center, direction, 50));
        this.pose.up.copy(UP); this.pose.fov = 50;
    }

    playIntro(onComplete = () => {}) {
        if (!this.board) { onComplete(); return; }
        const shots = [{ name: "player", title: "Your snake", duration: 2.1, actor: this.player() }];
        for (const [type, title] of [["hunter", "The hunter"], ["rival", "The rival"]]) {
            const actor = (this.board.enemies || []).find((enemy) => enemy.type === type);
            if (actor) shots.push({ name: type, title, duration: 1.9, actor });
        }
        shots.push({ name: "danger", title: "The challenge", duration: 2 });
        if (this.board.foods?.length) shots.push({ name: "food", title: "Follow the golden glow", duration: 1.9 });
        shots.push({ name: "return", title: "Your move. The policy takes over.", duration: 2.7, actor: this.player() });
        this.presentation = null;
        this.intro = { shots, index: -1, age: 0, onComplete };
        this.lastIntroShot = null; this.history = []; this.snap = true;
    }

    cancelIntro() {
        this.intro = null; this.snap = true; this.resetSmart();
        this.overviewDirty = true;
    }

    updateIntro(dt = 0) {
        const intro = this.intro;
        intro.age += dt;
        let age = intro.age, index = 0;
        while (index < intro.shots.length - 1 && age >= intro.shots[index].duration) {
            age -= intro.shots[index].duration; index += 1;
        }
        const shot = intro.shots[index];
        if (index !== intro.index) {
            intro.index = index; this.lastIntroShot = shot.name; this.snap = true;
            this.history.push({ name: shot.name, time: this.now });
        }
        const progress = ease(age / shot.duration);
        if (shot.actor) this.face(shot.actor, progress, shot.name === "return");
        else if (shot.name === "food") this.foodPose();
        else this.dangerPose();
        const blend = this.snap ? 1 : 1 - Math.exp(-dt * 4);
        this.camera.position.lerp(this.pose.position, blend);
        this.look.lerp(this.pose.target, blend);
        this.camera.position.copy(this.safePosition(this.look, this.camera.position));
        this.camera.up.copy(UP); this.camera.fov = this.pose.fov;
        this.camera.lookAt(this.look); this.camera.updateProjectionMatrix(); this.camera.updateMatrixWorld();
        this._activeCamera = this.camera; this.snap = false;
        if (index === intro.shots.length - 1 && age >= shot.duration) {
            this.intro = null; this.resetSmart(); this.overviewDirty = true;
            intro.onComplete();
        }
    }

    subjectsFramed(subjects) {
        const safe = this.safeFrame();
        const camera = this.activeCamera;
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
        return subjects.every(({ point, radius = BODY_RADIUS }) => {
            for (const offset of [new THREE.Vector3(), right.clone().multiplyScalar(radius),
                right.clone().multiplyScalar(-radius), up.clone().multiplyScalar(radius), up.clone().multiplyScalar(-radius)]) {
                const p = point.clone().add(offset).project(camera);
                if (p.x < -safe.x - 1e-5 || p.x > safe.x + 1e-5 || p.y < -safe.bottom - 1e-5
                    || p.y > safe.top + 1e-5 || p.z < -1 || p.z > 1) return false;
            }
            return true;
        });
    }

    updateVisibility(subjects, force = false) {
        if (!force && this.now - this.lastVisibility < 0.25) return;
        const obstructions = new Map();
        for (const { point } of this.visibilitySubjects(subjects)) {
            for (const cell of this.rayCells(point, this.activeCamera.position).reverse()) {
                if (obstructions.size < 8) obstructions.set(cellKey(cell), { x: cell.x, y: cell.y, z: cell.z });
            }
        }
        this.cutaway = [...obstructions.values()];
        this.visibility = {
            framed: this.subjectsFramed(subjects), occluded: this.cutaway.length > 0,
            clearance: this.cameraClearance(this.activeCamera.position),
        };
        this.smart.occluded = this.visibility.occluded;
        this.lastVisibility = this.now;
    }

    update(dt, elapsedSeconds, renderState) {
        if (!this.board?.snake?.length) return false;
        this.now = Number.isFinite(elapsedSeconds) ? elapsedSeconds : this.now;
        if (this.now >= this.shot.until) this.shot = { name: "follow", until: 0, priority: 0 };
        this.renderState = renderState || null;
        dt = clamp(Number.isFinite(dt) ? dt : 0, 0, 0.05);
        this.eventFraming = false;
        if (this.intro) { this.updateIntro(dt); return true; }
        const presentation = this.effectivePresentation();
        if (presentation) { this.updateEvent(dt, presentation); return true; }
        this.lastEvent = null;
        if (this.mode === "game") this.updateSmart(dt);
        else if (this.mode === "static" || this.playback.sampled) this.updateStatic();
        else if (this.mode === "free") {
            this._activeCamera = this.camera;
            this.camera.up.copy(UP);
            return false;
        } else this.updateFilm(dt);
        return true;
    }

    debug() {
        const current = this.intro?.shots[Math.max(0, this.intro.index)];
        const direction = this.activeCamera.position.clone().sub(this.look).normalize();
        const halfHeight = this.activeCamera.isOrthographicCamera
            ? (this.activeCamera.top - this.activeCamera.bottom) / (2 * this.activeCamera.zoom)
            : this.activeCamera.position.distanceTo(this.look) * Math.tan(THREE.MathUtils.degToRad(this.activeCamera.fov / 2));
        return {
            active: Boolean(this.intro), shot: current?.name || null, title: current?.title || "",
            history: [...this.history], filmShot: this.shot.name, mode: this.mode,
            cameraType: this.activeCamera.isOrthographicCamera ? "orthographic" : "perspective",
            position: this.activeCamera.position.toArray(), target: this.look.toArray(),
            halfHeight, azimuth: Math.atan2(direction.x, direction.z), elevation: Math.asin(clamp(direction.y, -1, 1)),
            safeFrame: this.safeFrame(),
            event: this.lastEvent ? { ...this.lastEvent } : null,
            visibility: { ...this.visibility }, cutaway: this.cutaway.map((cell) => ({ ...cell })),
            smart: {
                phase: this.smart.phase, busy: this.smart.clock < this.smart.busyUntil,
                turns: this.smart.turns.length, plans: this.smart.plans, occluded: this.smart.occluded,
                halfHeight: this.smart.halfHeight, azimuth: this.smart.azimuth,
            },
        };
    }
}
