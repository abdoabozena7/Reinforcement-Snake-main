import * as THREE from "./vendor/three.module.js";
import { ReferenceGameClient } from "./reference-client.js?v=reference-hud-4";
import { ReferenceWorld } from "./reference-world.js?v=reference-hud-4";
import { ReferenceOverlay } from "./reference-overlay.js?v=reference-hud-4";
import { ReferenceCameras } from "./reference-cameras.js?v=reference-hud-4";

const gridSize = 32;
const cellSize = 1;
let apple = null;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 1, 144);
let angle = 0;
const radius = 50;
const cameraHeight = 30;

const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById("gameCanvas"),
  antialias: true,
});
renderer.setSize(window.innerWidth, window.innerHeight);

function gridToWorld(cell) {
    return new THREE.Vector3(
        cell.x - 16 + 0.5,  // Center at (16,16,16)
        cell.y - 16 + 0.5,
        cell.z - 16 + 0.5
    );
}

function create3DGrid() {
    const gridGroup = new THREE.Group();
    const gridColor = 0x00ffff;
    const gridSize = 32;
    const cellSize = 1;

    // Fixed grid line creation
    const createAxisLines = (axis) => {
        const lines = [];
        for(let i = 0; i <= gridSize; i++) {
            // Correct buffer size calculation
            const positions = new Float32Array(24); // 8 points × 3 components
            
            switch(axis) {
                case 'x': // YZ planes
                    positions.set([
                        i, 0, 0,    i, gridSize, 0,
                        i, 0, 0,    i, 0, gridSize,
                        i, gridSize, 0, i, gridSize, gridSize,
                        i, 0, gridSize, i, gridSize, gridSize
                    ]);
                    break;
                case 'y': // XZ planes
                    positions.set([
                        0, i, 0,    gridSize, i, 0,
                        0, i, 0,    0, i, gridSize,
                        gridSize, i, 0, gridSize, i, gridSize,
                        0, i, gridSize, gridSize, i, gridSize
                    ]);
                    break;
                case 'z': // XY planes
                    positions.set([
                        0, 0, i,    gridSize, 0, i,
                        0, 0, i,    0, gridSize, i,
                        gridSize, 0, i, gridSize, gridSize, i,
                        0, gridSize, i, gridSize, gridSize, i
                    ]);
                    break;
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const line = new THREE.LineSegments(
                geometry,
                new THREE.LineBasicMaterial({
                    color: gridColor,
                    transparent: true,
                    opacity: 0.1,
                    depthWrite: false
                })
            );
            lines.push(line);
        }
        return lines;
    };

    // Add grid lines and bounding box
    gridGroup.add(...createAxisLines('x'));
    gridGroup.add(...createAxisLines('y'));
    gridGroup.add(...createAxisLines('z'));

    // Bounding box (keep existing code)
    // ... (same bounding box code as before)

    // Position grid to match snake coordinates
    gridGroup.position.set(
        -gridSize/2 + 0.5,
        -gridSize/2 + 0.5,
        -gridSize/2 + 0.5
    );
    scene.add(gridGroup);
    return gridGroup;
}

const gridGuide = create3DGrid();

const ambientLight = new THREE.AmbientLight(0x0fffff, 0.5);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0x0fffff, 1.2);
directionalLight.position.set(0, 100, 0);
scene.add(directionalLight);

const snakes = [];

function updateSnakeMesh(snake) {
    while (snake.meshes.length) scene.remove(snake.meshes.pop());
    
    const bodyGeo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
    const outlineGeo = new THREE.BoxGeometry(1.2, 1.2, 1.2);
    const outlineMat = new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.BackSide });
    const bodyMat = new THREE.MeshStandardMaterial({
      color: snake.color,
      emissive: snake.color,
      emissiveIntensity: 1.5,
      metalness: 0.2,
      roughness: 0.1,
    });

    snake.segments.forEach(segment => {
      const pos = gridToWorld(segment);
      
      const outline = new THREE.Mesh(outlineGeo, outlineMat);
      outline.position.set(pos.x, pos.y, pos.z);
      scene.add(outline);
      snake.meshes.push(outline);
      
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      body.position.set(pos.x, pos.y, pos.z);
      scene.add(body);
      snake.meshes.push(body);
    });
}

function removeApple() {
    if (apple) {
      scene.remove(apple.mesh);  // Remove from scene
      apple.mesh.geometry.dispose();  // Clean up resources
      apple.mesh.material.dispose();
      apple = null;
    }
}

function placeApple(applePos) {
    removeApple();
    const geometry = new THREE.SphereGeometry(0.4, 16, 16);
    const material = new THREE.MeshStandardMaterial({
      color: 0x8099ff,
      emissive: 0x8099ff,
      emissiveIntensity: 1.0,
      metalness: 0.8,
      roughness: 0.1,
    });
    
    const appleMesh = new THREE.Mesh(geometry, material);
    const pos = gridToWorld(applePos);
    appleMesh.position.set(pos.x, pos.y, pos.z);
    scene.add(appleMesh);
    apple = { pos: applePos, mesh: appleMesh };
}

function updateCamera() {
    angle += 0.002;
    camera.position.x = Math.cos(angle) * radius;
    camera.position.z = Math.sin(angle) * radius;
    camera.position.y = cameraHeight;
    camera.lookAt(0, 5, 0);
}

function animate(now = performance.now()) {
    requestAnimationFrame(animate);
    updateCamera();
    world.update(now);
    const player = actorMap.get("player");
    if (player?.meshes[1]) player.meshes[1].material.emissiveIntensity = 2 + world.growthPulse * 2;
    activeCamera = views.update(camera, client.state, clock.getDelta());
    renderer.render(scene, activeCamera);
}

const world = new ReferenceWorld(scene, gridToWorld, gridGuide);
const views = new ReferenceCameras(gridToWorld);
const clock = new THREE.Clock();
let activeCamera = camera;
const actorMap = new Map();

function disposeActor(actor) {
    const geometries = new Set(actor.meshes.map((mesh) => mesh.geometry));
    const materials = new Set(actor.meshes.map((mesh) => mesh.material));
    actor.meshes.forEach((mesh) => scene.remove(mesh));
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
}

function applyState(state, { advanced = false } = {}) {
    if (state.board.gridSize !== gridSize) throw new Error("The unchanged reference view requires the current 32-cell arena.");
    const entities = [{ id: "player", type: "player", snake: state.board.snake, resources: state.currentScore },
        ...state.board.enemies];
    const active = new Set(entities.map((entity) => entity.id));
    for (const [id, actor] of actorMap) if (!active.has(id)) {
        disposeActor(actor);
        actorMap.delete(id);
    }
    snakes.length = 0;
    for (const entity of entities) {
        let actor = actorMap.get(entity.id);
        if (!actor) {
            actor = { id: entity.id, type: entity.type,
                color: entity.type === "patrol" ? world.materials.obstacle.color.getHex()
                    : entity.id === "player" ? 0x870734 : 0xef473a, meshes: [] };
            actorMap.set(entity.id, actor);
        }
        const key = JSON.stringify(entity.snake);
        actor.segments = entity.snake;
        actor.score = entity.resources;
        if (actor.poseKey !== key) {
            if (actor.meshes.length === entity.snake.length * 2) {
                entity.snake.forEach((point, index) => {
                    const position = gridToWorld(point);
                    actor.meshes[index * 2].position.copy(position);
                    actor.meshes[index * 2 + 1].position.copy(position);
                });
            } else {
                disposeActor(actor);
                updateSnakeMesh(actor);
            }
            actor.poseKey = key;
        }
        if (actor.id === "player" && actor.meshes[1]) actor.meshes[1].material.emissiveIntensity = 2;
        snakes.push(actor);
    }
    const food = state.board.foods.find((item) => item.type === "normal");
    if (!food) removeApple();
    else if (!apple || JSON.stringify(apple.pos) !== JSON.stringify(food.position)) placeApple(food.position);
    if (apple) {
        apple.mesh.material.emissiveIntensity = 1.8;
        apple.mesh.scale.setScalar(1.5);
    }
    world.setBoard(state.board, { advanced, episode: state.episodeKey });
    overlay.syncControls();
    overlay.drawNeurons();
}

const client = new ReferenceGameClient({
    onState: applyState,
    onStatus: () => overlay.syncControls(),
    onError: () => overlay.syncControls(),
});
const overlay = new ReferenceOverlay(renderer.domElement, client);
overlay.camera.addEventListener("change", () => views.setMode(overlay.camera.value));
scene.onAfterRender = (_renderer, _scene, renderedCamera) => {
    const names = { orbit: "Original Orbit", tail: "Tail to head", food: "Food focus" };
    const status = `${names[views.mode]}${views.notice ? ` | ${views.notice}` : ""}`;
    if (overlay.viewStatus.textContent !== status) overlay.viewStatus.textContent = status;
    overlay.update(renderedCamera, snakes);
};
function resizeArena() {
    const height = Math.max(140, window.innerHeight - overlay.hud.getBoundingClientRect().height);
    renderer.setSize(window.innerWidth, height);
    camera.aspect = window.innerWidth / height;
    camera.updateProjectionMatrix();
    views.resize(window.innerWidth, height);
    document.documentElement.style.setProperty("--arena-height", `${height}px`);
}
const resizeObserver = new ResizeObserver(resizeArena);
resizeObserver.observe(overlay.hud);
window.addEventListener("resize", resizeArena);
resizeArena();

document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.repeat
        || ["INPUT", "SELECT", "BUTTON", "TEXTAREA"].includes(event.target.tagName)) return;
    const key = event.key.toLowerCase();
    if (event.code === "Space") { event.preventDefault(); void client.togglePause(); }
    else if (key === "r" && event.shiftKey) void overlay.reset();
    else if (key === "r") void client.ready();
    else if (key === "s") void client.save();
    else if (key === "l") void client.load();
    else if (key === "n") void client.configure({ newEpisode: true });
    else if (key === "t" || key === "e") void client.configure({ mode: key === "t" ? "train" : "eval" });
    else if (key === "f") void client.configure({ level: client.state?.fullSurvival ? 4 : 6,
        fullSurvival: !client.state?.fullSurvival, autoCurriculum: false });
    else if (key === "[") void client.configure({ level: Math.max(1, (client.state?.level || 4) - 1),
        fullSurvival: false, autoCurriculum: false });
    else if (key === "]") void client.configure({ level: Math.min(6, (client.state?.level || 4) + 1),
        fullSurvival: false, autoCurriculum: false });
    else if (key === "+" || key === "=" || key === "-") client.changeSpeed(key === "-" ? -1 : 1);
    else if (key === "0") client.setSpeed("fast");
    else if (["1", "2", "4", "6", "8"].includes(key)) client.setSpeed(
        { "1": "rate-1", "2": "slow", "4": "rate-4", "6": "rate-6", "8": "realtime" }[key]);
    else if (key === "c") void client.reconnect();
});

document.addEventListener("visibilitychange", () => client.pacer.suspend());
window.addEventListener("pagehide", () => { resizeObserver.disconnect(); client.dispose(); }, { once: true });
window.__snakeDebug = () => ({
    state: client.state, lifecycle: client.debug(),
    renderer: {
        reference: true, angle, cameraPosition: activeCamera.position.toArray(), cameraQuaternion: activeCamera.quaternion.toArray(),
        cameraFov: activeCamera.fov, cameraNear: activeCamera.near, cameraFar: activeCamera.far,
        cameraView: views.debug(), neurons: overlay.neuralDebug(),
        background: scene.background.getHexString(), gridSize, world: world.debug(),
        labels: overlay.debug(), obstacleColor: world.materials.obstacle.color.getHex(),
        actors: snakes.map((actor) => ({ id: actor.id, type: actor.type, color: actor.color, cells: actor.segments,
            meshes: actor.meshes.map((mesh) => ({ position: mesh.position.toArray(),
                size: mesh.geometry.parameters.width, color: mesh.material.color.getHex() })) })),
    },
});

void client.start();
animate();
