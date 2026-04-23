import * as THREE from "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.172.0/three.module.js";

const canvas = document.getElementById("gameCanvas");
const networkCanvas = document.getElementById("networkCanvas");
const networkCtx = networkCanvas.getContext("2d");
const gamePanel = document.getElementById("gamePanel");
const currentScoreCard = document.getElementById("currentScoreCard");
const avgScoreCard = document.getElementById("avgScoreCard");
const cameraControl = document.getElementById("cameraControl");
const cameraMenuToggle = document.getElementById("cameraMenuToggle");
const cameraMenu = document.getElementById("cameraMenu");
const cameraOptions = Array.from(document.querySelectorAll(".camera-option"));
const slowModeToggle = document.getElementById("slowModeToggle");
let lastState = null;
let slowModeEnabled = false;
let tickDelayMultiplier = 1;

function flashGreen(element) {
    if (!element) {
        return;
    }

    element.classList.remove("flash-green");
    void element.offsetWidth;
    element.classList.add("flash-green");
}

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function resizeNetworkCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = networkCanvas.clientWidth;
    const height = networkCanvas.clientHeight;
    networkCanvas.width = Math.floor(width * ratio);
    networkCanvas.height = Math.floor(height * ratio);
    networkCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function normalizeActivation(value, layerIndex, maxValue) {
    if (layerIndex === 0) {
        return Math.max(0, Math.min(1, Number(value)));
    }

    if (layerIndex === 3) {
        if (maxValue === 0) {
            return 0.5;
        }
        return Math.max(0, Math.min(1, (Number(value) / maxValue + 1) / 2));
    }

    if (maxValue === 0) {
        return 0;
    }

    return Math.max(0, Math.min(1, Number(value) / maxValue));
}

function makeLayerPositions(layer, x, top, height) {
    const count = layer.length;
    if (count === 1) {
        return [{ x, y: top + height / 2 }];
    }

    const gap = height / (count - 1);
    return layer.map((_, index) => ({
        x,
        y: top + gap * index
    }));
}

function drawNetwork(network) {
    if (!network) {
        return;
    }

    const width = networkCanvas.clientWidth;
    const height = networkCanvas.clientHeight;
    networkCtx.clearRect(0, 0, width, height);

    const paddingX = 72;
    const paddingTop = 42;
    const paddingBottom = 34;
    const usableWidth = width - paddingX * 2;
    const usableHeight = height - paddingTop - paddingBottom;
    const layerGap = usableWidth / (network.layers.length - 1);
    const time = performance.now() / 700;

    const maxPerLayer = network.layers.map((layer, layerIndex) => {
        if (layerIndex === 3) {
            return Math.max(0.001, ...layer.map((value) => Math.abs(value)));
        }
        return Math.max(0.001, ...layer.map((value) => Number(value)));
    });

    const positions = network.layers.map((layer, index) =>
        makeLayerPositions(layer, paddingX + layerGap * index, paddingTop, usableHeight)
    );

    const layerTitles = ["Input", "Hidden A", "Hidden B", "Output"];
    networkCtx.font = "12px Trebuchet MS";
    networkCtx.textAlign = "center";

    layerTitles.forEach((title, index) => {
        const x = paddingX + layerGap * index;
        networkCtx.fillStyle = "#9ca3af";
        networkCtx.fillText(title, x, 20);
    });

    network.weights.forEach((matrix, matrixIndex) => {
        const fromLayer = network.layers[matrixIndex];
        const toLayer = network.layers[matrixIndex + 1];
        const sourcePositions = positions[matrixIndex];
        const targetPositions = positions[matrixIndex + 1];

        for (let sourceIndex = 0; sourceIndex < fromLayer.length; sourceIndex += 1) {
            for (let targetIndex = 0; targetIndex < toLayer.length; targetIndex += 1) {
                const weight = Number(matrix[sourceIndex][targetIndex]);
                const magnitude = Math.min(1, Math.abs(weight) / 0.6);
                if (magnitude < 0.03) {
                    continue;
                }

                const fromActivation = normalizeActivation(
                    fromLayer[sourceIndex],
                    matrixIndex,
                    maxPerLayer[matrixIndex]
                );
                const toActivation = normalizeActivation(
                    toLayer[targetIndex],
                    matrixIndex + 1,
                    maxPerLayer[matrixIndex + 1]
                );
                const pulse = 0.65 + Math.sin(time + sourceIndex * 0.15 + targetIndex * 0.1) * 0.08;
                const opacity = Math.max(0.03, magnitude * (0.18 + (fromActivation + toActivation) * 0.35) * pulse);

                networkCtx.beginPath();
                networkCtx.moveTo(sourcePositions[sourceIndex].x, sourcePositions[sourceIndex].y);
                networkCtx.lineTo(targetPositions[targetIndex].x, targetPositions[targetIndex].y);
                networkCtx.strokeStyle = weight >= 0
                    ? `rgba(0, 230, 118, ${opacity})`
                    : `rgba(255, 176, 77, ${opacity})`;
                networkCtx.lineWidth = 0.4 + magnitude * 1.2;
                networkCtx.stroke();
            }
        }
    });

    network.layers.forEach((layer, layerIndex) => {
        const layerPositions = positions[layerIndex];
        const maxValue = maxPerLayer[layerIndex];

        layer.forEach((value, neuronIndex) => {
            const activation = normalizeActivation(value, layerIndex, maxValue);
            const pulse = 0.92 + Math.sin(time * 1.8 + layerIndex + neuronIndex * 0.25) * 0.08;
            const radius = layerIndex === 0 || layerIndex === 3
                ? 7 + activation * 4
                : 3.2 + activation * 3.8;
            const glow = 8 + activation * 24;
            const position = layerPositions[neuronIndex];
            const fill = layerIndex === 3
                ? (Number(value) >= 0 ? "#6cecff" : "#ffb04d")
                : "#6cecff";

            networkCtx.beginPath();
            networkCtx.arc(position.x, position.y, radius * pulse, 0, Math.PI * 2);
            networkCtx.fillStyle = `rgba(108, 236, 255, ${0.16 + activation * 0.2})`;
            networkCtx.shadowBlur = glow;
            networkCtx.shadowColor = fill;
            networkCtx.fill();

            networkCtx.beginPath();
            networkCtx.arc(position.x, position.y, Math.max(2.2, radius - 1.2), 0, Math.PI * 2);
            networkCtx.fillStyle = fill;
            networkCtx.fill();
            networkCtx.shadowBlur = 0;
        });
    });

    networkCtx.font = "11px Trebuchet MS";
    networkCtx.textAlign = "right";
    networkCtx.fillStyle = "#9ca3af";
    network.inputLabels.forEach((label, index) => {
        const position = positions[0][index];
        networkCtx.fillText(label, position.x - 14, position.y + 4);
    });

    networkCtx.textAlign = "left";
    network.outputLabels.forEach((label, index) => {
        const outputValue = Number(network.layers[3][index]).toFixed(2);
        const position = positions[3][index];
        networkCtx.fillStyle = "#d8d8d8";
        networkCtx.fillText(`${label} ${outputValue}`, position.x + 14, position.y + 4);
    });
}

class Snake3DRenderer {
    constructor({ canvasElement }) {
        this.canvas = canvasElement;
        this.gridSize = 32;
        this.cellSize = 1;
        this.angle = 0;
        this.radius = this.gridSize * 1.9;
        this.cameraHeight = this.gridSize * 1.25;
        this.center = new THREE.Vector3(0, 0, 0);
        this.segmentMeshes = [];
        this.obstacleMeshes = [];
        this.foodMesh = null;
        this.deathPulse = 0;
        this.lastBoard = null;
        this.cameraMode = "orbit";
        this.cameraTargetPosition = new THREE.Vector3(0, this.cameraHeight, this.radius);
        this.cameraTargetLookAt = new THREE.Vector3();
        this.cameraLookAt = new THREE.Vector3();
        this.cameraUpTarget = new THREE.Vector3(0, 1, 0);
        this.cameraFovTarget = 60;
        this.cinemaBlend = 0;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        this.camera = new THREE.PerspectiveCamera(60, 1, 1, 144);
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            preserveDrawingBuffer: true
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.camera.position.copy(this.cameraTargetPosition);
        this.camera.fov = this.cameraFovTarget;
        this.camera.updateProjectionMatrix();
        this.camera.lookAt(this.cameraLookAt);

        this.boardGroup = new THREE.Group();
        this.boardGroup.position.set(-this.gridSize / 2, -this.gridSize / 2, -this.gridSize / 2);
        this.scene.add(this.boardGroup);

        this.segmentGeometry = new THREE.BoxGeometry(0.9, 0.9, 0.9);
        this.segmentOutlineGeometry = new THREE.BoxGeometry(1.16, 1.16, 1.16);
        this.segmentMaterial = new THREE.MeshStandardMaterial({
            color: 0xef473a,
            emissive: 0xef473a,
            emissiveIntensity: 1.4,
            metalness: 0.22,
            roughness: 0.12
        });
        this.headMaterial = new THREE.MeshStandardMaterial({
            color: 0x870734,
            emissive: 0x870734,
            emissiveIntensity: 1.2,
            metalness: 0.14,
            roughness: 0.18
        });
        this.outlineMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.BackSide
        });
        this.obstacleGeometry = new THREE.BoxGeometry(0.92, 0.92, 0.92);
        this.obstacleMaterial = new THREE.MeshStandardMaterial({
            color: 0x2be7ff,
            emissive: 0x2be7ff,
            emissiveIntensity: 0.5,
            metalness: 0.35,
            roughness: 0.18
        });
        this.foodGeometry = new THREE.SphereGeometry(0.34, 20, 20);
        this.foodMaterial = new THREE.MeshStandardMaterial({
            color: 0x32ff72,
            emissive: 0x24ff66,
            emissiveIntensity: 2.2,
            metalness: 0.35,
            roughness: 0.08
        });

        this.handleResize = this.handleResize.bind(this);
        this.animate = this.animate.bind(this);

        this.setupScene();
    }

    setupScene() {
        this.create3DGrid();

        const ambientLight = new THREE.AmbientLight(0x0fffff, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0x0fffff, 1.2);
        directionalLight.position.set(0, 100, 0);
        this.scene.add(directionalLight);

        const fillLight = new THREE.DirectionalLight(0xff65d4, 0.55);
        fillLight.position.set(-20, 15, -12);
        this.scene.add(fillLight);
    }

    create3DGrid() {
        const gridColor = 0x00ffff;
        const createAxisLines = (axis) => {
            const lines = [];
            for (let i = 0; i <= this.gridSize; i += 1) {
                const positions = new Float32Array(24);

                if (axis === "x") {
                    positions.set([
                        i, 0, 0, i, this.gridSize, 0,
                        i, 0, 0, i, 0, this.gridSize,
                        i, this.gridSize, 0, i, this.gridSize, this.gridSize,
                        i, 0, this.gridSize, i, this.gridSize, this.gridSize
                    ]);
                } else if (axis === "y") {
                    positions.set([
                        0, i, 0, this.gridSize, i, 0,
                        0, i, 0, 0, i, this.gridSize,
                        this.gridSize, i, 0, this.gridSize, i, this.gridSize,
                        0, i, this.gridSize, this.gridSize, i, this.gridSize
                    ]);
                } else {
                    positions.set([
                        0, 0, i, this.gridSize, 0, i,
                        0, 0, i, 0, this.gridSize, i,
                        this.gridSize, 0, i, this.gridSize, this.gridSize, i,
                        0, this.gridSize, i, this.gridSize, this.gridSize, i
                    ]);
                }

                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
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

        this.boardGroup.add(...createAxisLines("x"));
        this.boardGroup.add(...createAxisLines("y"));
        this.boardGroup.add(...createAxisLines("z"));

        const box = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(this.gridSize, this.gridSize, this.gridSize)),
            new THREE.LineBasicMaterial({
                color: gridColor,
                transparent: true,
                opacity: 0.28
            })
        );
        box.position.set(this.gridSize / 2, this.gridSize / 2, this.gridSize / 2);
        this.boardGroup.add(box);

        const coreGlow = new THREE.Mesh(
            new THREE.BoxGeometry(this.gridSize * 0.88, this.gridSize * 0.88, this.gridSize * 0.88),
            new THREE.MeshBasicMaterial({
                color: 0x4f173e,
                transparent: true,
                opacity: 0.06,
                wireframe: true
            })
        );
        coreGlow.position.set(this.gridSize / 2, this.gridSize / 2, this.gridSize / 2);
        this.boardGroup.add(coreGlow);
    }

    handleResize() {
        const width = Math.max(1, this.canvas.clientWidth);
        const height = Math.max(1, this.canvas.clientHeight);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

        if (
            this.canvas.width !== Math.floor(width * pixelRatio)
            || this.canvas.height !== Math.floor(height * pixelRatio)
        ) {
            this.renderer.setPixelRatio(pixelRatio);
            this.renderer.setSize(width, height, false);
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
        }
    }

    boardToWorld(cell) {
        return new THREE.Vector3(cell.x + 0.5, cell.y + 0.5, cell.z + 0.5);
    }

    boardToScene(cell) {
        const world = this.boardToWorld(cell);
        return world.add(this.boardGroup.position);
    }

    createOutlinedMesh(geometry, material) {
        const outline = new THREE.Mesh(geometry === this.segmentGeometry ? this.segmentOutlineGeometry : geometry, this.outlineMaterial);
        const body = new THREE.Mesh(geometry, material);
        const group = new THREE.Group();
        if (geometry === this.segmentGeometry) {
            group.add(outline);
        }
        group.add(body);
        group.userData = { body };
        return group;
    }

    syncSnake(board) {
        const snake = board?.snake || [];

        while (this.segmentMeshes.length < snake.length) {
            const segmentGroup = this.createOutlinedMesh(this.segmentGeometry, this.segmentMaterial.clone());
            this.boardGroup.add(segmentGroup);
            this.segmentMeshes.push(segmentGroup);
        }

        while (this.segmentMeshes.length > snake.length) {
            const segmentGroup = this.segmentMeshes.pop();
            this.boardGroup.remove(segmentGroup);
        }

        snake.forEach((segment, index) => {
            const group = this.segmentMeshes[index];
            const position = this.boardToWorld(segment);
            group.position.copy(position);
            const body = group.userData.body;
            if (index === 0) {
                body.material.color.copy(this.headMaterial.color);
                body.material.emissive.copy(this.headMaterial.emissive);
                body.material.emissiveIntensity = this.headMaterial.emissiveIntensity;
            } else {
                body.material.color.copy(this.segmentMaterial.color);
                body.material.emissive.copy(this.segmentMaterial.emissive);
                body.material.emissiveIntensity = this.segmentMaterial.emissiveIntensity;
            }
        });
    }

    syncObstacles(board) {
        const obstacles = board?.obstacles || [];

        while (this.obstacleMeshes.length < obstacles.length) {
            const mesh = new THREE.Mesh(this.obstacleGeometry, this.obstacleMaterial.clone());
            this.boardGroup.add(mesh);
            this.obstacleMeshes.push(mesh);
        }

        while (this.obstacleMeshes.length > obstacles.length) {
            const mesh = this.obstacleMeshes.pop();
            this.boardGroup.remove(mesh);
        }

        obstacles.forEach((obstacle, index) => {
            const mesh = this.obstacleMeshes[index];
            const position = this.boardToWorld(obstacle);
            mesh.position.copy(position);
        });
    }

    syncFood(board) {
        if (!board?.food) {
            return;
        }

        if (!this.foodMesh) {
            this.foodMesh = new THREE.Mesh(this.foodGeometry, this.foodMaterial.clone());
            this.boardGroup.add(this.foodMesh);
        }

        const position = this.boardToWorld(board.food);
        this.foodMesh.position.copy(position);
    }

    applyBoard(board) {
        if (!board) {
            return;
        }

        if (typeof board.gridSize === "number" && board.gridSize !== this.gridSize) {
            this.gridSize = board.gridSize;
            this.radius = this.gridSize * 1.9;
            this.cameraHeight = this.gridSize * 1.25;
        }

        this.lastBoard = board;
        this.syncSnake(board);
        this.syncObstacles(board);
        this.syncFood(board);

        if (board.lastDeath) {
            this.deathPulse = 1;
        }
    }

    updateCamera() {
        const distance = this.gridSize * 1.72;
        const board = this.lastBoard;
        const head = board?.snake?.[0] ? this.boardToScene(board.snake[0]) : new THREE.Vector3();
        const secondSegment = board?.snake?.[1] ? this.boardToScene(board.snake[1]) : null;
        const direction = board?.direction || { x: 0, y: 0, z: -1 };
        const forward = secondSegment
            ? new THREE.Vector3().subVectors(head, secondSegment)
            : new THREE.Vector3(direction.x || 0, direction.y || 0, direction.z || -1);
        if (forward.lengthSq() === 0) {
            forward.set(0, 0, -1);
        }
        forward.normalize();

        const worldUp = new THREE.Vector3(0, 1, 0);
        const altUp = new THREE.Vector3(0, 0, 1);
        const referenceUp = Math.abs(forward.dot(worldUp)) > 0.9 ? altUp : worldUp;
        let side = new THREE.Vector3().crossVectors(forward, referenceUp);
        if (side.lengthSq() < 0.001) {
            side = new THREE.Vector3(1, 0, 0);
        }
        side.normalize();
        const cameraUp = new THREE.Vector3().crossVectors(side, forward).normalize();
        const focusPoint = new THREE.Vector3().copy(head).addScaledVector(forward, Math.max(2.8, this.gridSize * 0.12));
        if (this.cameraMode === "orbit") {
            this.angle += 0.00085;
            this.cameraTargetPosition.set(
                Math.cos(this.angle) * (this.gridSize * 1.34),
                this.gridSize * 0.98,
                Math.sin(this.angle) * (this.gridSize * 1.34)
            );
            this.cameraTargetLookAt.copy(this.center);
            this.cameraUpTarget.set(0, 1, 0);
            this.cameraFovTarget = 54;
        } else if (this.cameraMode === "chase") {
            this.angle += 0.00045;
            this.cameraTargetPosition.copy(focusPoint)
                .addScaledVector(forward, -15.2)
                .addScaledVector(cameraUp, 3.9)
                .addScaledVector(side, 0.35);
            this.cameraTargetLookAt.copy(focusPoint)
                .addScaledVector(forward, 4.9)
                .addScaledVector(cameraUp, 0.36);
            this.cameraUpTarget.copy(cameraUp);
            this.cameraFovTarget = 66;
        } else if (this.cameraMode === "cine-chase") {
            this.angle += 0.00025;
            const drift = Math.sin(this.angle * 2.6) * 0.9;
            this.cameraTargetPosition.copy(focusPoint)
                .addScaledVector(forward, -20.5)
                .addScaledVector(cameraUp, 6.2)
                .addScaledVector(side, 4.2 + drift);
            this.cameraTargetLookAt.copy(focusPoint)
                .addScaledVector(forward, 6.2)
                .addScaledVector(cameraUp, 0.75)
                .addScaledVector(side, drift * 0.18);
            this.cameraUpTarget.copy(cameraUp);
            this.cameraFovTarget = 48;
        } else if (this.cameraMode === "static") {
            this.cameraTargetPosition.set(
                this.gridSize * 0.08,
                this.gridSize * 1.02,
                this.gridSize * 1.6
            );
            this.cameraTargetLookAt.copy(this.center);
            this.cameraUpTarget.set(0, 1, 0);
            this.cameraFovTarget = 52;
        } else if (this.cameraMode === "cinema") {
            this.cinemaBlend += 0.0012;
            const anchorA = new THREE.Vector3(
                -this.gridSize * 1.08,
                this.gridSize * 0.94,
                this.gridSize * 1.18
            );
            const anchorB = new THREE.Vector3(
                this.gridSize * 1.14,
                this.gridSize * 0.82,
                -this.gridSize * 1.06
            );
            const blend = (Math.sin(this.cinemaBlend) + 1) / 2;
            this.cameraTargetPosition.copy(anchorA).lerp(anchorB, blend);
            this.cameraTargetLookAt.copy(this.center).lerp(focusPoint, 0.16);
            this.cameraUpTarget.set(0, 1, 0);
            this.cameraFovTarget = 50;
        } else {
            this.cameraMode = "orbit";
            this.angle += 0.00085;
            this.cameraTargetPosition.set(
                Math.cos(this.angle) * (this.gridSize * 1.34),
                this.gridSize * 0.98,
                Math.sin(this.angle) * (this.gridSize * 1.34)
            );
            this.cameraTargetLookAt.copy(this.center);
            this.cameraUpTarget.set(0, 1, 0);
            this.cameraFovTarget = 54;
        }

        const smoothing = this.cameraMode === "cine-chase"
            ? 0.045
            : this.cameraMode === "cinema"
                ? 0.055
                : this.cameraMode === "static"
                    ? 0.07
                    : 0.11;
        this.camera.position.lerp(this.cameraTargetPosition, smoothing);
        this.cameraLookAt.lerp(this.cameraTargetLookAt, Math.min(0.1, smoothing));
        this.camera.up.lerp(this.cameraUpTarget, Math.min(0.12, smoothing + 0.02)).normalize();
        if (Math.abs(this.camera.fov - this.cameraFovTarget) > 0.08) {
            this.camera.fov += (this.cameraFovTarget - this.camera.fov) * 0.08;
            this.camera.updateProjectionMatrix();
        }
        this.camera.lookAt(this.cameraLookAt);
    }

    setCameraMode(mode) {
        if (["orbit", "chase", "cine-chase", "static", "cinema"].includes(mode)) {
            this.cameraMode = mode;
            return;
        }
        this.cameraMode = "orbit";
    }

    animateFood(time) {
        if (!this.foodMesh) {
            return;
        }
        const baseY = this.lastBoard?.food ? this.boardToWorld(this.lastBoard.food).y : 0.55;
        this.foodMesh.position.y = baseY + Math.sin(time / 260) * 0.08;
        const pulse = 1 + Math.sin(time / 180) * 0.12;
        this.foodMesh.scale.setScalar(pulse);
        this.foodMesh.material.emissiveIntensity = 2.2 + (Math.sin(time / 140) + 1) * 0.55;
    }

    animateDeathPulse() {
        if (this.deathPulse <= 0) {
            return;
        }
        this.deathPulse = Math.max(0, this.deathPulse - 0.045);
        const intensity = 0.28 + this.deathPulse * 0.95;
        this.obstacleMeshes.forEach((mesh) => {
            mesh.material.emissiveIntensity = intensity * 0.55;
        });
        this.segmentMeshes.forEach((group, index) => {
            const body = group.userData.body;
            body.material.emissiveIntensity = index === 0 ? 1.2 + this.deathPulse : 1.4 + this.deathPulse * 0.65;
        });
    }

    start() {
        this.handleResize();
        window.addEventListener("resize", this.handleResize);
        requestAnimationFrame(this.animate);
    }

    animate(time) {
        requestAnimationFrame(this.animate);
        this.handleResize();
        this.updateCamera();
        this.animateFood(time);
        this.animateDeathPulse();
        this.renderer.render(this.scene, this.camera);
    }
}

const game3D = new Snake3DRenderer({ canvasElement: canvas });

function setCameraMenuOpen(isOpen) {
    cameraControl.classList.toggle("is-open", isOpen);
    cameraMenuToggle.setAttribute("aria-expanded", String(isOpen));
    cameraMenu.setAttribute("aria-hidden", String(!isOpen));
}

function setCameraView(mode) {
    game3D.setCameraMode(mode);
    cameraOptions.forEach((option) => {
        option.classList.toggle("is-active", option.dataset.cameraView === mode);
    });
    setCameraMenuOpen(false);
}

function updateUI(state) {
    const previousState = lastState;
    lastState = state;
    document.getElementById("episodes").innerText = state.episode;
    document.getElementById("highScore").innerText = state.highScore;
    document.getElementById("epsilon").innerText = Number(state.epsilon).toFixed(3);
    document.getElementById("currentScore").innerText = state.currentScore;
    document.getElementById("avgScore").innerText = Number(state.avgScore).toFixed(2);
    document.getElementById("bufferSize").innerText = state.bufferSize;
    drawNetwork(state.network);
    game3D.applyBoard(state.board);

    if (previousState) {
        if (Number(state.currentScore) > Number(previousState.currentScore)) {
            flashGreen(gamePanel);
            flashGreen(currentScoreCard);
        }

        if (Number(state.avgScore) > Number(previousState.avgScore)) {
            flashGreen(avgScoreCard);
        }
    }
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options
    });

    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }

    return response.json();
}

async function handleWeightsAction(url) {
    const state = await requestJson(url, { method: "POST" });
    updateUI(state);
    return state;
}

function flashButtonLabel(button, label) {
    const defaultLabel = button.dataset.defaultLabel || button.textContent;
    button.dataset.defaultLabel = defaultLabel;
    button.textContent = label;
    button.disabled = true;
    setTimeout(() => {
        button.textContent = defaultLabel;
        button.disabled = false;
    }, 900);
}

let loopActive = true;

async function loop() {
    while (loopActive) {
        try {
            const state = await requestJson("/api/tick", { method: "POST" });
            updateUI(state);
            await delay((state.delayMs || 36) * tickDelayMultiplier);
        } catch (error) {
            console.error(error);
            await delay(1200);
        }
    }
}

document.getElementById("resetBtn").addEventListener("click", async () => {
    if (!window.confirm("This will erase all training progress. Continue?")) {
        return;
    }

    const state = await requestJson("/api/reset", { method: "POST" });
    updateUI(state);
});

document.getElementById("saveWeightsBtn").addEventListener("click", async () => {
    const button = document.getElementById("saveWeightsBtn");
    await handleWeightsAction("/api/save-weights");
    flashButtonLabel(button, "Saved");
});

document.getElementById("loadWeightsBtn").addEventListener("click", async () => {
    const button = document.getElementById("loadWeightsBtn");
    const state = await handleWeightsAction("/api/load-weights");
    flashButtonLabel(button, state.logMessage.startsWith("Loaded") ? "Loaded" : "No Save");
});

cameraMenuToggle.addEventListener("click", () => {
    setCameraMenuOpen(!cameraControl.classList.contains("is-open"));
});

cameraOptions.forEach((option) => {
    option.addEventListener("click", () => {
        setCameraView(option.dataset.cameraView);
    });
});

slowModeToggle.addEventListener("click", () => {
    slowModeEnabled = !slowModeEnabled;
    tickDelayMultiplier = slowModeEnabled ? 6 : 1;
    slowModeToggle.classList.toggle("is-active", slowModeEnabled);
    slowModeToggle.setAttribute("aria-pressed", String(slowModeEnabled));
    slowModeToggle.textContent = slowModeEnabled ? "Slow x6" : "Slow";
});

document.addEventListener("click", (event) => {
    if (!cameraMenu.contains(event.target) && !cameraMenuToggle.contains(event.target)) {
        setCameraMenuOpen(false);
    }
});

window.addEventListener("resize", () => {
    resizeNetworkCanvas();
    if (lastState) {
        drawNetwork(lastState.network);
    }
});

async function boot() {
    game3D.start();
    resizeNetworkCanvas();

    try {
        const state = await requestJson("/api/state");
        updateUI(state);
    } catch (error) {
        console.error(error);
    }

    loop();
}

boot().catch((error) => {
    console.error(error);
});
