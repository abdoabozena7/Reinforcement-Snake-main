import * as THREE from "./vendor/three.module.js";
import { CinematicDirector } from "./director.js";
import { BlockSnakeCharacter } from "./character.js?v=readable-v6";
import { PoseInterpolator, describeObjectives, episodeKey } from "./presentation.js?v=readable-v6";

const UP = new THREE.Vector3(0, 1, 0);
const clamp = THREE.MathUtils.clamp;
const INTENT_COLORS = { player: 0xffd16b, rival: 0xc49bff, hunter: 0xff6475, patrol: 0xffc15d, ambush: 0x7ed6c3 };

export class Snake3DRenderer {
    constructor(canvas, { labelLayer = null, presentation = null, onPresentationFrame = () => {} } = {}) {
        this.canvas = canvas;
        this.labelLayer = labelLayer;
        this.roleLegend = labelLayer?.parentElement.querySelector(".arena-role-key") || null;
        this.presentation = presentation;
        this.onPresentationFrame = onPresentationFrame;
        this.labels = new Map();
        this.pools = new Map();
        this.characters = new Map();
        this.goalGuides = new Map();
        this.objectives = null;
        this.decision = {};
        this.intentVisuals = [];
        this.motion = new PoseInterpolator();
        this.gridSize = 32;
        this.cameraMode = "game";
        this.frames = 0;
        this.lastTime = 0;
        this.animationTime = 0;
        this.lastBoard = null;
        this.displayBoard = null;
        this.currentPose = null;
        this.currentPresentation = null;
        this.heldPresentation = null;
        this.localCues = [];
        this.seenEvents = new Set();
        this.playback = { paused: true, sampled: false, reducedMotion: false };
        this.motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
        this.reducedMotion = this.motionPreference.matches;
        this.onMotionPreference = (event) => {
            this.reducedMotion = event.matches;
            this.presentation?.setReducedMotion(event.matches);
            if (this.lastBoard) this.director.setBoard(this.lastBoard, { ...this.playback, reducedMotion: event.matches });
        };
        this.motionPreference.addEventListener("change", this.onMotionPreference);
        this.presentation?.setReducedMotion(this.reducedMotion);
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x080d12);
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 300);
        this.camera.position.set(28, 24, 35);
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.14;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
        this.boardGroup = new THREE.Group();
        this.gridGroup = new THREE.Group();
        this.boardGroup.add(this.gridGroup);
        this.scene.add(this.boardGroup);
        this.cutawayUniforms = {
            arenaCutCount: { value: 0 },
            arenaCutStart0: { value: new THREE.Vector3() },
            arenaCutStart1: { value: new THREE.Vector3() },
            arenaCutStart2: { value: new THREE.Vector3() },
            arenaCutEnd: { value: new THREE.Vector3() },
        };
        this.obstacleMaterial = this.cutawayMaterial(new THREE.MeshStandardMaterial({
            color: 0x294752, roughness: 0.83, metalness: 0.1,
        }));
        this.edgeMaterial = this.cutawayMaterial(new THREE.LineBasicMaterial({ color: 0x6d8d96, transparent: true, opacity: 0.22 }));
        this.wallMaterial = this.cutawayMaterial(new THREE.MeshStandardMaterial({ color: 0x72838a, roughness: 0.7 }));
        this.gateMaterials = {
            open: new THREE.MeshBasicMaterial({ color: 0x5ca7ac, wireframe: true, transparent: true, opacity: 0.13 }),
            warning: new THREE.MeshBasicMaterial({ color: 0xf2bc69, wireframe: true, transparent: true, opacity: 0.7 }),
            active: this.cutawayMaterial(new THREE.MeshStandardMaterial({
                color: 0xd6544b, emissive: 0xb83f30, emissiveIntensity: 0.65, roughness: 0.52,
            })),
        };
        this.previewMaterial = new THREE.MeshBasicMaterial({ color: 0xf2bc69, wireframe: true, transparent: true, opacity: 0.65 });
        this.foodMaterial = new THREE.MeshStandardMaterial({
            color: 0xf7c35c, emissive: 0xe89925, emissiveIntensity: 0.5, roughness: 0.3,
        });
        this.goldMaterial = new THREE.MeshStandardMaterial({
            color: 0xffdf8d, emissive: 0xffb52e, emissiveIntensity: 0.7, roughness: 0.24,
        });
        this.cellGeometry = new THREE.BoxGeometry(1, 1, 1);
        this.foodGeometry = new THREE.SphereGeometry(0.3, 18, 12);
        this.foodRingGeometry = new THREE.TorusGeometry(0.42, 0.018, 5, 28);
        this.foodRingMaterial = new THREE.MeshBasicMaterial({ color: 0xf2bd62, transparent: true, opacity: 0.65 });
        this.obstacleSurface = new THREE.Mesh(new THREE.BufferGeometry(), this.obstacleMaterial);
        this.obstacleEdges = new THREE.LineSegments(new THREE.BufferGeometry(), this.edgeMaterial);
        this.boardGroup.add(this.obstacleSurface, this.obstacleEdges);
        this.chamberGuide = new THREE.LineSegments(new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({ color: 0x87a5a9, transparent: true, opacity: 0.17, depthWrite: false }));
        this.boardGroup.add(this.chamberGuide);
        this.chamberKey = "";
        this.obstacleKey = "";
        this.obstacleCount = 0;
        this.scene.add(new THREE.AmbientLight(0xdce9e8, 1.1));
        const key = new THREE.DirectionalLight(0xffeed9, 2.5);
        key.position.set(12, 35, 18);
        const fill = new THREE.DirectionalLight(0x81b3c2, 1.15);
        fill.position.set(-20, 10, -12);
        this.scene.add(key, fill);
        this.director = new CinematicDirector({ camera: this.camera, cellToScene: (cell) => this.boardToScene(cell) });
        if (this.director.setCameraMode) this.director.setCameraMode("game");
        else this.director.mode = "smart";
        this.zone = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
            new THREE.LineBasicMaterial({ color: 0xe3aa5e, transparent: true, opacity: 0.38 }),
        );
        this.zone.visible = false;
        this.boardGroup.add(this.zone);
        const stemGeometry = new THREE.BufferGeometry();
        stemGeometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
        this.heightStem = new THREE.LineSegments(stemGeometry,
            new THREE.LineBasicMaterial({ color: 0x81bfa9, transparent: true, opacity: 0.16, depthWrite: false }));
        this.scene.add(this.heightStem);
        this.effectMaterial = new THREE.MeshBasicMaterial({
            color: 0xffd08b, transparent: true, opacity: 0, depthTest: false, depthWrite: false,
        });
        this.effectRing = new THREE.Mesh(new THREE.TorusGeometry(0.65, 0.04, 6, 40), this.effectMaterial);
        this.effectRing.renderOrder = 20;
        this.effectRing.visible = false;
        this.scene.add(this.effectRing);
        const rays = [];
        for (let i = 0; i < 8; i += 1) {
            const a = i / 8 * Math.PI * 2;
            rays.push(Math.cos(a) * 0.85, Math.sin(a) * 0.85, 0, Math.cos(a) * 1.2, Math.sin(a) * 1.2, 0);
        }
        this.effectRays = new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(rays, 3)),
            new THREE.LineBasicMaterial({ color: 0xffd08b, transparent: true, opacity: 0, depthTest: false, depthWrite: false }),
        );
        this.effectRays.renderOrder = 20;
        this.effectRays.visible = false;
        this.scene.add(this.effectRays);
        const brackets = [];
        for (const side of [-1, 1]) {
            brackets.push(side * 0.56, -0.36, 0, side * 0.56, 0.36, 0,
                side * 0.56, 0.36, 0, side * 0.34, 0.36, 0,
                side * 0.56, -0.36, 0, side * 0.34, -0.36, 0);
        }
        this.contactBracket = new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(brackets, 3)),
            new THREE.LineBasicMaterial({ color: 0xfff2d5, transparent: true, opacity: 0, depthTest: false, depthWrite: false }),
        );
        this.contactBracket.renderOrder = 22;
        this.contactBracket.visible = false;
        this.contactFlash = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.18, 24),
            new THREE.MeshBasicMaterial({ color: 0xfff9df, transparent: true, opacity: 0, depthTest: false, depthWrite: false }));
        this.contactFlash.renderOrder = 23;
        this.contactFlash.visible = false;
        this.scene.add(this.contactBracket, this.contactFlash);
        this.cutaway = [];
        this.create3DGrid();
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(canvas);
        this.animate = this.animate.bind(this);
        this.resize();
        this.animation = requestAnimationFrame(this.animate);
    }

    get activeCamera() { return this.director.activeCamera || this.camera; }

    cutawayMaterial(material) {
        material.onBeforeCompile = (shader) => {
            Object.assign(shader.uniforms, this.cutawayUniforms);
            shader.vertexShader = shader.vertexShader.replace("#include <common>", "#include <common>\nvarying vec3 arenaWorldPosition;")
                .replace("#include <project_vertex>", "#include <project_vertex>\narenaWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;");
            shader.fragmentShader = shader.fragmentShader.replace("#include <common>", `#include <common>
varying vec3 arenaWorldPosition;
uniform float arenaCutCount;
uniform vec3 arenaCutStart0;
uniform vec3 arenaCutStart1;
uniform vec3 arenaCutStart2;
uniform vec3 arenaCutEnd;
bool arenaCut(vec3 start) {
    vec3 axis = arenaCutEnd - start;
    float t = dot(arenaWorldPosition - start, axis) / max(0.001, dot(axis, axis));
    return t > 0.025 && t < 0.98 && distance(arenaWorldPosition, start + axis * t) < 0.9;
}`)
                .replace("#include <color_fragment>", `#include <color_fragment>
if ((arenaCutCount > 0.5 && arenaCut(arenaCutStart0)) || (arenaCutCount > 1.5 && arenaCut(arenaCutStart1)) || (arenaCutCount > 2.5 && arenaCut(arenaCutStart2))) {
    if (mod(floor(gl_FragCoord.x / 3.0) + floor(gl_FragCoord.y / 3.0), 4.0) > 0.5) discard;
}`);
        };
        material.customProgramCacheKey = () => "readable-surface-cutaway-v1";
        return material;
    }

    create3DGrid() {
        for (const child of [...this.gridGroup.children]) {
            child.geometry?.dispose();
            child.material?.dispose();
            this.gridGroup.remove(child);
        }
        const n = this.gridSize;
        this.boardGroup.position.set(-n / 2, -n / 2, -n / 2);
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(n, n),
            new THREE.MeshStandardMaterial({ color: 0x0d1820, roughness: 1 }));
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(n / 2, -0.02, n / 2);
        this.gridGroup.add(floor);
        const lines = [];
        for (let i = 0; i <= n; i += 4) lines.push(i, 0.01, 0, i, 0.01, n, 0, 0.01, i, n, 0.01, i);
        this.gridLines = new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(lines, 3)),
            new THREE.LineBasicMaterial({ color: 0x69969b, transparent: true, opacity: 0.16 }),
        );
        this.gridGroup.add(this.gridLines);
        const landmarks = [];
        for (const x of [0, n]) for (const z of [0, n]) {
            landmarks.push(x, 0, z, x, n, z);
            for (let y = 4; y <= n; y += 4) landmarks.push(x, y, z, x + (x ? -0.65 : 0.65), y, z);
        }
        this.gridGroup.add(new THREE.LineSegments(
            new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(landmarks, 3)),
            new THREE.LineBasicMaterial({ color: 0x83a2a3, transparent: true, opacity: 0.25 }),
        ));
    }

    resize() {
        const width = Math.max(1, this.canvas.clientWidth);
        const height = Math.max(1, this.canvas.clientHeight);
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.director.setViewport(width, height);
    }

    boardToWorld(cell) { return new THREE.Vector3(cell.x + 0.5, cell.y + 0.5, cell.z + 0.5); }
    boardToScene(cell) { return this.boardToWorld(cell).add(this.boardGroup.position); }

    syncObstacles(cells) {
        const key = cells.map((p) => `${p.x},${p.y},${p.z}`).join(";");
        if (key === this.obstacleKey) return;
        this.obstacleKey = key;
        this.obstacleCount = cells.length;
        const occupied = new Set(cells.map((p) => `${p.x},${p.y},${p.z}`));
        const positions = [], normals = [];
        const faces = [
            [[1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]],
            [[-1, 0, 0], [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]],
            [[0, 1, 0], [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]]],
            [[0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
            [[0, 0, 1], [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]],
            [[0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]],
        ];
        for (const cell of cells) for (const [normal, corners] of faces) {
            if (occupied.has(`${cell.x + normal[0]},${cell.y + normal[1]},${cell.z + normal[2]}`)) continue;
            for (const index of [0, 1, 2, 0, 2, 3]) {
                positions.push(cell.x + corners[index][0], cell.y + corners[index][1], cell.z + corners[index][2]);
                normals.push(...normal);
            }
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
        geometry.computeBoundingSphere();
        this.obstacleSurface.geometry.dispose();
        this.obstacleEdges.geometry.dispose();
        this.obstacleSurface.geometry = geometry;
        this.obstacleEdges.geometry = new THREE.EdgesGeometry(geometry, 30);
    }

    syncCells(key, cells, geometry, material, food = false) {
        let pool = this.pools.get(key);
        if (!pool) { pool = []; this.pools.set(key, pool); }
        while (pool.length < cells.length) {
            const mesh = new THREE.Mesh(geometry, material);
            if (food) {
                const ring = new THREE.Mesh(this.foodRingGeometry, this.foodRingMaterial);
                ring.rotation.x = Math.PI / 2;
                mesh.add(ring);
            }
            this.boardGroup.add(mesh);
            pool.push(mesh);
        }
        while (pool.length > cells.length) this.boardGroup.remove(pool.pop());
        cells.forEach((cell, index) => {
            pool[index].position.copy(this.boardToWorld(cell));
            pool[index].material = material;
        });
    }

    syncWorld(board) {
        if (this.displayBoard === board) return;
        this.displayBoard = board;
        const chamber = board.arena?.centralChamber;
        const chamberKey = chamber ? JSON.stringify(chamber) : "";
        if (chamberKey !== this.chamberKey) {
            this.chamberKey = chamberKey;
            const vertices = [];
            if (chamber) for (const x of [chamber.min.x, chamber.max.x + 1]) {
                for (const y of [chamber.min.y, chamber.max.y + 1]) for (const z of [chamber.min.z, chamber.max.z + 1]) {
                    vertices.push(x, y, z, x + (x === chamber.min.x ? 0.85 : -0.85), y, z,
                        x, y, z, x, y + (y === chamber.min.y ? 0.85 : -0.85), z,
                        x, y, z, x, y, z + (z === chamber.min.z ? 0.85 : -0.85));
                }
            }
            this.chamberGuide.geometry.dispose();
            this.chamberGuide.geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
        }
        this.syncObstacles(board.obstacles || []);
        const active = new Set(["food", "gold", "wall", "wall-preview"]);
        for (const type of ["normal", "gold"]) {
            this.syncCells(type === "normal" ? "food" : "gold",
                (board.foods || []).filter((item) => item.type === type).map((item) => item.position),
                this.foodGeometry, type === "normal" ? this.foodMaterial : this.goldMaterial, true);
        }
        for (const gate of board.gates || []) {
            const key = `gate:${gate.id}`;
            active.add(key);
            this.syncCells(key, gate.cells, this.cellGeometry, this.gateMaterials[gate.state] || this.gateMaterials.open);
        }
        this.syncCells("wall", (board.movingWalls || []).flatMap((wall) => wall.cells), this.cellGeometry, this.wallMaterial);
        this.syncCells("wall-preview", (board.movingWalls || []).filter((wall) => wall.warning).flatMap((wall) => wall.nextCells),
            this.cellGeometry, this.previewMaterial);
        for (const [key, pool] of this.pools) if (!active.has(key)) {
            pool.forEach((mesh) => this.boardGroup.remove(mesh));
            this.pools.delete(key);
        }
        const inset = board.zone?.warning ? board.zone.nextInset : board.zone?.inset || 0;
        this.zone.visible = inset > 0;
        this.zone.position.setScalar(this.gridSize / 2);
        this.zone.scale.setScalar(Math.max(0.01, this.gridSize - inset * 2));
        this.zone.material.color.setHex(board.zone?.warning ? 0xe3aa5e : 0xe37c6c);
        const entities = [{ id: "player", type: "player", snake: board.snake, direction: board.direction }, ...(board.enemies || [])];
        for (const entity of entities) {
            let character = this.characters.get(entity.id);
            if (!character || character.group.userData.role !== entity.type) {
                character?.dispose();
                character = new BlockSnakeCharacter(entity.type);
                character.group.userData.role = entity.type;
                this.characters.set(entity.id, character);
                this.scene.add(character.group);
            }
            character.setState(entity, board, this.resetCharacters);
        }
        this.resetCharacters = false;
    }

    consumeEvents(board, sampled) {
        if (sampled) return;
        for (const event of board.events || []) {
            if (this.seenEvents.has(event.id)) continue;
            this.seenEvents.add(event.id);
            if (event.step !== board.steps || !event.position) continue;
            if (["food", "gold_food"].includes(event.type)) this.localCues.push({ position: event.position, kind: "food", age: 0, duration: 0.7 });
        }
        if (board.contact?.outcome === "blocked" && !this.seenEvents.has(board.contact.id)) {
            this.seenEvents.add(board.contact.id);
            this.localCues.push({ position: board.contact.position, kind: "shield", contact: board.contact, age: 0, duration: 0.75 });
        }
        this.localCues = this.localCues.slice(-4);
        if (this.seenEvents.size > 160) this.seenEvents = new Set((board.events || []).map((event) => event.id));
    }

    applyBoard(board, playback = {}) {
        if (playback.decision) this.decision = playback.decision;
        this.objectives = describeObjectives(board, this.decision);
        const reset = !this.lastBoard || episodeKey(board) !== episodeKey(this.lastBoard)
            || board.steps < this.lastBoard.steps || Boolean(playback.snap);
        if (reset) {
            this.seenEvents.clear();
            this.localCues.length = 0;
            this.heldPresentation = null;
            this.resetCharacters = true;
        }
        if (!board.done) this.heldPresentation = null;
        if (board.gridSize !== this.gridSize) {
            this.gridSize = board.gridSize;
            this.create3DGrid();
        }
        this.lastBoard = board;
        this.playback = { ...playback, reducedMotion: this.reducedMotion };
        this.motion.setBoard(board, playback);
        this.director.setBoard(board, this.playback);
        this.syncWorld(board);
        this.consumeEvents(board, playback.sampled);
        this.updateObjectiveCopy();
    }

    updateObjectiveCopy() {
        if (!this.roleLegend || !this.objectives) return;
        const { player, enemies, food } = this.objectives;
        const copy = {
            player: player.activity,
            hunter: enemies.find((actor) => actor.role === "hunter")?.activity,
            rival: enemies.find((actor) => actor.role === "rival")?.activity,
            food: food ? `+${food.value} · ${player.gridDistance} grid cells away` : "Waiting for food",
        };
        for (const [role, text] of Object.entries(copy)) {
            const label = this.roleLegend.querySelector(`[data-objective="${role}"]`);
            if (!label) continue;
            label.parentElement.hidden = text == null;
            label.textContent = text || "";
        }
    }

    setCameraMode(mode) {
        if (!["game", "smart", "film", "orbit", "chase", "cine-chase", "static", "overview", "cinema"].includes(mode)) throw new Error(`Unknown camera view: ${mode}`);
        this.cameraMode = mode === "smart" ? "game" : mode === "overview" ? "static" : mode;
        if (this.director.setCameraMode) this.director.setCameraMode(this.cameraMode);
        else { this.director.mode = this.cameraMode === "game" ? "smart" : this.cameraMode; this.director.snap = true; }
    }

    playIntro(onComplete) { this.motion.snap(); this.director.playIntro(onComplete); }
    cancelIntro() { this.director.cancelIntro(); }

    clearPresentation() {
        this.currentPresentation = null;
        this.heldPresentation = null;
        this.motion.snap();
        this.director.setPresentation?.(null);
        this.displayBoard = null;
        if (this.lastBoard) this.syncWorld(this.lastBoard);
    }

    renderActors(pose, frame, dt) {
        const board = frame?.board || this.lastBoard;
        if (!board || !pose) return { player: [], enemies: [] };
        this.syncWorld(board);
        const renderState = {
            player: pose.player.map((p) => this.boardToScene(p)),
            enemies: pose.enemies.map((enemy) => ({ ...enemy, points: enemy.points.map((p) => this.boardToScene(p)) })),
        };
        const shieldCue = this.localCues.find((cue) => cue.kind === "shield");
        const shieldBlock = shieldCue ? Math.max(0, 1 - shieldCue.age / shieldCue.duration) : 0;
        const visible = new Set(["player", ...pose.enemies.map((enemy) => enemy.id)]);
        for (const [id, character] of this.characters) {
            character.group.visible = visible.has(id);
            if (!character.group.visible) continue;
            const player = id === "player";
            const actor = player ? null : renderState.enemies.find((enemy) => enemy.id === id);
            const look = player ? board.lookTarget : frame?.contact?.attacker?.id === id ? frame.contact.position
                : actor?.type === "rival" ? board.foods?.[0]?.position : pose.player[0];
            const attack = frame?.attack && frame.contact?.attacker?.id === id ? frame.jawOpen : 0;
            character.setRenderPose(player ? renderState.player : actor.points, {
                direction: player ? pose.direction : actor.direction,
                lookTarget: look ? this.boardToScene(look) : null,
                snap: this.playback.sampled,
            });
            character.setPresentation({
                death: player ? frame?.death ?? (board.done && board.lastDeath !== "time_limit" ? 1 : 0) : 0,
                impact: player ? frame?.impact || 0 : 0,
                hitStop: player && Boolean(frame?.hitStop), hitSegment: player ? frame?.contact?.victim?.segmentIndex : null,
                attack, shieldBlock: player ? shieldBlock : 0, reducedMotion: this.reducedMotion,
            });
            character.update(dt, this.animationTime, { intro: Boolean(this.director.intro) });
        }
        const head = renderState.player[0];
        if (head) {
            const stem = this.heightStem.geometry.attributes.position;
            stem.setXYZ(0, head.x, -this.gridSize / 2 + 0.02, head.z);
            stem.setXYZ(1, head.x, head.y - 0.35, head.z);
            stem.needsUpdate = true;
        }
        return renderState;
    }

    updateCutaway(camera, directorDebug) {
        this.cutaway = directorDebug.cutaway || [];
        const contact = this.currentPresentation?.contact;
        const points = contact?.position ? [this.boardToScene(contact.position), this.renderState?.player[0]] : [this.renderState?.player[0]];
        const attacker = this.currentPresentation?.contact?.attacker?.id;
        const other = attacker ? this.renderState?.enemies.find((enemy) => enemy.id === attacker)
            : this.renderState?.enemies.slice().sort((a, b) => a.points[0].distanceToSquared(points[0]) - b.points[0].distanceToSquared(points[0]))[0];
        if (other?.points[0]) points.push(other.points[0]);
        const valid = points.filter((p, index) => p && !points.slice(0, index).some((other) => other && p.distanceToSquared(other) < 0.16));
        this.cutawayUniforms.arenaCutCount.value = this.cutaway.length ? Math.min(3, valid.length) : 0;
        valid.slice(0, 3).forEach((p, index) => this.cutawayUniforms[`arenaCutStart${index}`].value.copy(p));
        this.cutawayUniforms.arenaCutEnd.value.copy(camera.position);
    }

    updateGoalGuides(camera) {
        const enabled = !this.currentPresentation && !this.director.intro && !this.playback.sampled && !this.lastBoard.done;
        for (const guide of this.goalGuides.values()) guide.group.visible = false;
        this.intentVisuals = [];
        if (!enabled || !this.objectives) return;
        for (const actor of [this.objectives.player, ...this.objectives.enemies]) {
            const rendered = actor.id === "player" ? this.renderState.player
                : this.renderState.enemies.find((item) => item.id === actor.id)?.points;
            if (!rendered?.length) continue;
            let guide = this.goalGuides.get(actor.id);
            if (!guide) {
                const color = INTENT_COLORS[actor.role];
                const group = new THREE.Group();
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
                geometry.setAttribute("lineDistance", new THREE.Float32BufferAttribute([0, 1], 1));
                const line = new THREE.Line(geometry, new THREE.LineDashedMaterial({
                    color, dashSize: 0.32, gapSize: 0.24, transparent: true, opacity: 0.72, depthTest: false, depthWrite: false,
                }));
                const ring = new THREE.Mesh(new THREE.TorusGeometry(actor.role === "rival" ? 0.82 : 0.65, 0.025, 6, 40),
                    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, depthTest: false, depthWrite: false }));
                const arrow = new THREE.ArrowHelper(UP, new THREE.Vector3(), 0.85,
                    actor.role === "player" ? 0xb9f281 : color, 0.23, 0.18);
                for (const part of [arrow.line, arrow.cone]) {
                    part.material.depthTest = false;
                    part.material.depthWrite = false;
                }
                group.add(line, ring, arrow);
                group.renderOrder = 4;
                this.scene.add(group);
                guide = { group, line, ring, arrow };
                this.goalGuides.set(actor.id, guide);
            }
            guide.group.visible = true;
            const start = rendered[0];
            const target = actor.target && actor.role !== "patrol" ? this.boardToScene(actor.target) : null;
            guide.line.visible = guide.ring.visible = Boolean(target);
            if (target) {
                const positions = guide.line.geometry.attributes.position;
                positions.setXYZ(0, start.x, start.y, start.z);
                positions.setXYZ(1, target.x, target.y, target.z);
                positions.needsUpdate = true;
                guide.line.geometry.attributes.lineDistance.setX(1, start.distanceTo(target));
                guide.line.geometry.attributes.lineDistance.needsUpdate = true;
                guide.line.frustumCulled = false;
                guide.ring.position.copy(target);
                guide.ring.quaternion.copy(camera.quaternion);
            }
            const direction = actor.moveDirection ? new THREE.Vector3(actor.moveDirection.x, actor.moveDirection.y, actor.moveDirection.z)
                : actor.nextPosition ? this.boardToScene(actor.nextPosition).sub(this.boardToScene(actor.head)) : new THREE.Vector3();
            guide.arrow.visible = direction.lengthSq() > 0;
            if (guide.arrow.visible) {
                direction.normalize();
                guide.arrow.setDirection(direction);
                guide.arrow.position.copy(start).addScaledVector(direction, 0.58);
            }
            this.intentVisuals.push({
                id: actor.id, role: actor.role, source: actor.source, moveKind: actor.moveKind,
                target: actor.target, targetId: actor.targetId, linkVisible: guide.line.visible,
                from: new THREE.Vector3().fromBufferAttribute(guide.line.geometry.attributes.position, 0).toArray(),
                to: target ? new THREE.Vector3().fromBufferAttribute(guide.line.geometry.attributes.position, 1).toArray() : null,
                moveArrowVisible: guide.arrow.visible,
                moveDirection: guide.arrow.visible ? UP.clone().applyQuaternion(guide.arrow.quaternion).toArray() : null,
            });
        }
    }

    updateEffect(frame, camera) {
        const cue = this.localCues[this.localCues.length - 1];
        const contactVisible = frame?.contact && frame.impact > 0;
        this.contactBracket.visible = Boolean(contactVisible);
        this.contactFlash.visible = Boolean(contactVisible && frame.hitStop);
        if (contactVisible) {
            this.effectRing.visible = this.effectRays.visible = false;
            this.contactBracket.position.copy(this.boardToScene(frame.contact.position));
            this.contactFlash.position.copy(this.contactBracket.position);
            this.contactBracket.quaternion.copy(camera.quaternion);
            this.contactFlash.quaternion.copy(camera.quaternion);
            this.contactBracket.scale.setScalar(frame.hitStop || this.reducedMotion ? 1 : 1 + (1 - frame.impact) * 0.75);
            this.contactBracket.material.opacity = frame.impact;
            this.contactFlash.material.opacity = frame.hitStop ? 0.95 : 0;
            return;
        }
        const position = cue?.position;
        const strength = cue ? 1 - cue.age / cue.duration : 0;
        this.effectRing.visible = this.effectRays.visible = Boolean(position && strength > 0);
        if (!position) return;
        const color = cue?.kind === "shield" ? 0x9fe5ee : 0xffd16b;
        this.effectRing.position.copy(this.boardToScene(position));
        this.effectRays.position.copy(this.effectRing.position);
        this.effectRing.quaternion.copy(camera.quaternion);
        this.effectRays.quaternion.copy(camera.quaternion);
        const scale = this.reducedMotion ? 0.85 : 0.6 + (1 - strength) * 0.85;
        this.effectRing.scale.setScalar(scale);
        this.effectRays.scale.setScalar(scale);
        this.effectMaterial.opacity = strength * 0.95;
        this.effectMaterial.color.setHex(color);
        this.effectRays.material.opacity = strength;
        this.effectRays.material.color.setHex(color);
    }

    animate(time) {
        this.animation = requestAnimationFrame(this.animate);
        const elapsedDt = Math.min(0.25, Math.max(0, (time - (this.lastTime || time)) / 1000));
        const dt = Math.min(0.05, elapsedDt);
        this.lastTime = time;
        if (document.hidden || !this.lastBoard) return;
        const frame = this.presentation?.advance(elapsedDt);
        if (frame) this.heldPresentation = frame;
        const held = !frame && this.lastBoard.done && this.heldPresentation
            ? { ...this.heldPresentation, phase: "summary", replay: false, attack: false, impact: 0, hitStop: false, jawOpen: 0,
                eliminated: Boolean(this.heldPresentation.lethal),
                death: this.heldPresentation.lethal ? 1 : 0, defeat: this.heldPresentation.contact ? 1 : 0, progress: 1 } : null;
        this.currentPresentation = frame || held;
        const animating = !this.playback.paused || Boolean(frame || this.director.intro || this.motion.motion);
        if (animating) this.animationTime += elapsedDt;
        const pose = this.currentPresentation?.pose || this.motion.advance(elapsedDt);
        this.currentPose = pose;
        for (const cue of this.localCues) if (animating) cue.age += elapsedDt;
        this.localCues = this.localCues.filter((cue) => cue.age < cue.duration);
        this.renderState = this.renderActors(pose, this.currentPresentation, animating ? dt : 0);
        this.director.setPresentation?.(this.currentPresentation ? {
            phase: frame?.phase || "complete", progress: this.currentPresentation.progress, contact: this.currentPresentation.contact,
        } : null);
        this.director.update(dt, this.animationTime, this.renderState);
        const camera = this.activeCamera;
        const directorDebug = this.director.debug();
        this.updateCutaway(camera, directorDebug);
        this.updateEffect(this.currentPresentation, camera);
        this.updateGoalGuides(camera);
        if (!this.reducedMotion) for (const key of ["food", "gold"]) {
            for (const mesh of this.pools.get(key) || []) mesh.scale.setScalar(1 + Math.sin(this.animationTime * 4) * 0.055);
        }
        this.renderer.render(this.scene, camera);
        this.updateLabels(camera);
        this.onPresentationFrame(frame, { director: directorDebug, cutaway: this.cutaway.length > 0,
            shieldBlocked: this.localCues.some((cue) => cue.kind === "shield") });
        this.frames += 1;
    }

    updateLabels(camera = this.activeCamera) {
        if (!this.labelLayer || !this.currentPose) return;
        this.labelLayer.hidden = Boolean(this.director.intro);
        const actors = [{ id: "player", role: "player",
            name: this.playback.sampled ? "YOU · SAMPLED" : this.objectives?.player.label || "YOU", point: this.currentPose.player[0] },
            ...this.currentPose.enemies.filter((enemy) => this.currentPresentation?.contact
                ? enemy.id === this.currentPresentation.contact.attacker?.id : true)
                .map((enemy) => ({ id: enemy.id, role: enemy.type,
                    name: this.currentPresentation ? enemy.type.toUpperCase()
                        : this.objectives?.enemies.find((item) => item.id === enemy.id)?.label || enemy.type.toUpperCase(),
                    point: enemy.points[0] }))];
        const presentation = this.currentPresentation;
        if (presentation?.eliminated) actors[0].name = "YOU · ELIMINATED";
        else if (presentation?.replay) actors[0].name = "YOU · REPLAY";
        if (presentation?.contact && presentation.impact > 0.25) actors.unshift({
            id: "contact-hit", role: "contact",
            name: presentation.contact.victim.segmentIndex > 0 ? `BODY HIT · ${presentation.contact.victim.segmentIndex + 1}` : "HEAD CONTACT",
            point: presentation.contact.position,
        });
        const board = this.displayBoard;
        if (!this.currentPresentation && !this.playback.sampled) {
            const targets = new Set([this.objectives?.player.targetId,
                ...(this.objectives?.enemies || []).filter((item) => item.role === "rival").map((item) => item.targetId)]);
            for (const food of (board.foods || []).filter((item) => targets.has(item.id))) actors.push({
                id: `goal:${food.id}`, role: "food",
                name: `${food.type === "gold" ? "GOLD" : "GET FOOD"} +${food.value}`, point: food.position,
            });
            for (const enemy of this.objectives?.enemies || []) {
                if (enemy.target && /CUT OFF/.test(enemy.label)) actors.push({
                    id: `goal:${enemy.id}`, role: "hunter", name: "CUT-OFF POINT", point: enemy.target,
                });
            }
        }
        const active = new Set(actors.map((actor) => actor.id));
        const boxes = [];
        const bottomInset = this.currentPresentation ? Math.min(170, this.canvas.clientHeight * 0.34)
            : this.roleLegend && !this.roleLegend.hidden ? this.roleLegend.offsetHeight + 20 : 16;
        for (const actor of actors) {
            let label = this.labels.get(actor.id);
            if (!label) {
                label = document.createElement("span");
                label.className = `actor-label actor-${actor.role}`;
                this.labelLayer.append(label);
                this.labels.set(actor.id, label);
            }
            const projected = this.boardToScene(actor.point).project(camera);
            let x = (projected.x * 0.5 + 0.5) * this.canvas.clientWidth;
            let y = (-projected.y * 0.5 + 0.5) * this.canvas.clientHeight - 18;
            const outside = projected.z < -1 || projected.z > 1 || Math.abs(projected.x) > 0.92 || Math.abs(projected.y) > 0.84
                || y > this.canvas.clientHeight - bottomInset - 10;
            const offscreen = ["food", "hunter", "rival", "patrol", "ambush"].includes(actor.role) && outside;
            const caption = offscreen ? `${actor.name} · OFF VIEW` : actor.name;
            const width = Math.min(this.canvas.clientWidth - 16, Math.max(38, caption.length * 6.6 + 12));
            if (offscreen) {
                x = clamp(x, width / 2 + 4, this.canvas.clientWidth - width / 2 - 4);
                y = clamp(y, 78, this.canvas.clientHeight - bottomInset - 10);
            }
            const anchorX = x;
            x = clamp(x, width / 2 + 4, this.canvas.clientWidth - width / 2 - 4);
            label.style.setProperty("--label-anchor", `${(anchorX - x).toFixed(2)}px`);
            label.classList.toggle("is-offscreen", offscreen);
            label.classList.toggle("is-eliminated", actor.id === "player" && Boolean(presentation?.eliminated));
            label.classList.toggle("is-replay", Boolean(presentation?.replay));
            label.textContent = caption;
            let lift = 0;
            const overlapsAt = (height) => boxes.some((box) => Math.abs(x - box.x) < (width + box.width) / 2 + 3
                && Math.abs(height - box.y) < 21);
            while (lift < 48 && overlapsAt(y - lift)) lift += 24;
            y -= lift;
            label.style.setProperty("--label-stem", `${11 + lift}px`);
            const overlaps = overlapsAt(y);
            label.hidden = !offscreen && (projected.z < -1 || projected.z > 1) || x < width / 2 || x > this.canvas.clientWidth - width / 2
                || y < 44 || y > this.canvas.clientHeight - bottomInset || overlaps;
            if (!label.hidden) boxes.push({ x, y, width });
            label.style.transform = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px) translate(-50%, -100%)`;
        }
        for (const [id, label] of this.labels) if (!active.has(id)) { label.remove(); this.labels.delete(id); }
    }

    debug() {
        const camera = this.activeCamera;
        const corners = [];
        const half = this.gridSize / 2;
        for (const x of [-half, half]) for (const y of [-half, half]) for (const z of [-half, half]) corners.push(new THREE.Vector3(x, y, z).project(camera).toArray());
        const head = this.renderState?.player[0] || new THREE.Vector3();
        const scenePositions = {
            player: this.renderState?.player.map((p) => p.toArray()) || [],
            enemies: this.renderState?.enemies.map((enemy) => ({ id: enemy.id, type: enemy.type, points: enemy.points.map((p) => p.toArray()) })) || [],
        };
        const p = this.currentPresentation;
        const participants = [{ id: "player", points: this.renderState?.player || [] },
            ...(this.renderState?.enemies || []).filter((enemy) => enemy.id === p?.contact?.attacker?.id)];
        const screenBounds = participants.filter((actor) => actor.points.length).map((actor) => {
            const projected = [];
            for (const point of actor.points) for (const x of [-0.48, 0.48]) for (const y of [-0.48, 0.48]) for (const z of [-0.48, 0.48]) {
                projected.push(point.clone().add(new THREE.Vector3(x, y, z)).project(camera));
            }
            const left = Math.min(...projected.map((point) => (point.x + 1) / 2 * this.canvas.clientWidth));
            const right = Math.max(...projected.map((point) => (point.x + 1) / 2 * this.canvas.clientWidth));
            const top = Math.min(...projected.map((point) => (1 - point.y) / 2 * this.canvas.clientHeight));
            const bottom = Math.max(...projected.map((point) => (1 - point.y) / 2 * this.canvas.clientHeight));
            return { id: actor.id, left, right, top, bottom, width: right - left, height: bottom - top };
        });
        const player = this.characters.get("player");
        return {
            frames: this.frames, gridSize: this.gridSize, cameraMode: this.cameraMode,
            actorStyle: "grid-blocks",
            actorStyles: Object.fromEntries([...this.characters].filter(([, actor]) => actor.group.visible)
                .map(([id, actor]) => [id, actor.group.userData.style])),
            objectiveView: { ...this.objectives, guides: this.intentVisuals },
            outlinedActors: [...this.characters].filter(([, actor]) => actor.group.visible && actor.group.userData.outlined)
                .map(([id]) => id),
            cameraType: camera.isOrthographicCamera ? "orthographic" : "perspective",
            cameraPosition: camera.position.toArray(), cameraUp: camera.up.toArray(), cameraQuaternion: camera.quaternion.toArray(),
            cameraFov: camera.fov ?? null, cameraZoom: camera.zoom, arenaCorners: corners,
            playerPositions: (this.currentPose?.player || []).map((cell) => this.boardToWorld(cell).toArray()),
            renderedActors: scenePositions, truePose: p?.truthPose || this.motion.pose,
            renderedCells: { ...Object.fromEntries([...this.pools].map(([key, meshes]) => [key, meshes.length])),
                player: this.currentPose?.player.length || 0,
                ...Object.fromEntries((this.currentPose?.enemies || []).map((enemy) => [`enemy:${enemy.id}`, enemy.points.length])),
                obstacles: this.obstacleCount },
            calls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles,
            cinematic: this.director.debug(), interpolation: this.motion.debug(), foodColor: this.foodMaterial.color.getHexString(),
            playerScreen: head.clone().project(camera).toArray(),
            approximateHeadPixels: camera.isOrthographicCamera
                ? this.canvas.clientHeight * 0.9 / (camera.top - camera.bottom) * camera.zoom
                : this.canvas.clientHeight * 0.9 / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.max(0.1, camera.position.distanceTo(head))),
            cutaway: { active: this.cutaway.length > 0, cells: this.cutaway, collisionUnchanged: true },
            presentation: {
                ...(this.presentation?.debug() || { active: false, phase: "idle", contactId: null, bufferSize: 0 }),
                renderedPhase: p?.phase ?? null, replay: Boolean(p?.replay), renderedActors: scenePositions,
                contactPosition: p?.contact?.position ? this.boardToScene(p.contact.position).toArray() : null,
                contactBoardPosition: p?.contact?.position || null,
                renderedBoardActors: this.currentPose, truePoseBoard: p?.truthPose || this.motion.pose,
                timeFraction: p?.contact?.timeFraction ?? null, victimSegmentIndex: p?.contact?.victim?.segmentIndex ?? null,
                attackerId: p?.contact?.attacker?.id ?? null, impact: p?.impact || 0, defeat: p?.defeat || 0,
                attackerJawOpen: p?.contact?.attacker?.id ? this.characters.get(p.contact.attacker.id)?._presentation?.attack || 0 : 0,
                hitStop: Boolean(p?.hitStop), recoil: p?.recoil || 0, eliminated: Boolean(p?.eliminated), death: p?.death || 0,
                participantScreenBounds: screenBounds, screenBoundsSource: "rendered body cell envelopes; no particles, labels or unrelated actors",
                playerSkinColor: player?._skinMaterial.color.toArray() || null,
                playerBodyColor: player?._bodyGeometry ? Array.from(player._bodyGeometry.attributes.color.array.slice(0, 3)) : null,
                playerDeadEyes: Boolean(player?._crosses?.visible),
            },
        };
    }

    dispose() {
        cancelAnimationFrame(this.animation);
        this.resizeObserver.disconnect();
        this.motionPreference.removeEventListener("change", this.onMotionPreference);
        for (const character of this.characters.values()) character.dispose();
        this.characters.clear();
        this.labels.forEach((label) => label.remove());
        const geometries = new Set([this.cellGeometry, this.foodGeometry, this.foodRingGeometry]);
        const materials = new Set([this.obstacleMaterial, this.edgeMaterial, this.wallMaterial, this.previewMaterial,
            this.foodMaterial, this.goldMaterial, this.foodRingMaterial, ...Object.values(this.gateMaterials)]);
        this.scene.traverse((object) => {
            if (object.geometry) geometries.add(object.geometry);
            if (object.material) materials.add(object.material);
        });
        geometries.forEach((geometry) => geometry.dispose());
        materials.forEach((material) => material.dispose());
        this.renderer.dispose();
    }
}
