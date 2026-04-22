const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const networkCanvas = document.getElementById("networkCanvas");
const networkCtx = networkCanvas.getContext("2d");
const networkInfo = document.getElementById("networkInfo");
let lastState = null;

function resizeNetworkCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = networkCanvas.clientWidth;
    const height = networkCanvas.clientHeight;
    networkCanvas.width = Math.floor(width * ratio);
    networkCanvas.height = Math.floor(height * ratio);
    networkCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function renderBoard(board) {
    if (!board) {
        return;
    }

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const blockSize = board.blockSize;
    ctx.fillStyle = "#ff5252";
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#ff5252";
    ctx.fillRect(
        board.food.x * blockSize,
        board.food.y * blockSize,
        blockSize - 2,
        blockSize - 2
    );

    ctx.shadowBlur = 0;
    board.snake.forEach((part, index) => {
        ctx.fillStyle = index === 0 ? "#00e676" : "#00a451";
        ctx.fillRect(
            part.x * blockSize,
            part.y * blockSize,
            blockSize - 2,
            blockSize - 2
        );
    });
}

async function playDeathEffect(board) {
    if (!board || !board.lastDeath || !board.deathPoint) {
        return;
    }

    const frames = 8;
    const totalDuration = 250;
    const isWall = board.lastDeath === "wall";
    const color = isWall ? "#ff9800" : "#00e676";

    for (let i = 0; i < frames; i += 1) {
        const alpha = 1 - i / frames;
        renderBoard(board);
        ctx.save();
        ctx.shadowBlur = 25;
        ctx.shadowColor = color;

        if (isWall) {
            ctx.strokeStyle = `rgba(255, 152, 0, ${alpha})`;
            ctx.lineWidth = 6;
            ctx.strokeRect(3, 3, canvas.width - 6, canvas.height - 6);

            const px = Math.max(0, Math.min(board.gridSize - 1, board.deathPoint.x)) * board.blockSize;
            const py = Math.max(0, Math.min(board.gridSize - 1, board.deathPoint.y)) * board.blockSize;
            ctx.fillStyle = `rgba(255, 152, 0, ${alpha * 0.6})`;
            ctx.fillRect(px - 4, py - 4, board.blockSize + 6, board.blockSize + 6);
        } else {
            ctx.fillStyle = `rgba(0, 230, 118, ${alpha * 0.5})`;
            board.snake.forEach((part) => {
                ctx.fillRect(
                    part.x * board.blockSize - 3,
                    part.y * board.blockSize - 3,
                    board.blockSize + 4,
                    board.blockSize + 4
                );
            });

            ctx.strokeStyle = `rgba(0, 230, 118, ${alpha})`;
            ctx.lineWidth = 3;
            ctx.strokeRect(
                board.deathPoint.x * board.blockSize - 2,
                board.deathPoint.y * board.blockSize - 2,
                board.blockSize + 2,
                board.blockSize + 2
            );
        }

        ctx.restore();
        await new Promise((resolve) => setTimeout(resolve, totalDuration / frames));
    }
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
    networkCtx.font = "12px Segoe UI";
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
                    : `rgba(255, 152, 0, ${opacity})`;
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

    networkCtx.font = "11px Segoe UI";
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

function updateUI(state) {
    lastState = state;
    document.getElementById("episodes").innerText = state.episode;
    document.getElementById("highScore").innerText = state.highScore;
    document.getElementById("epsilon").innerText = Number(state.epsilon).toFixed(3);
    document.getElementById("currentScore").innerText = state.currentScore;
    document.getElementById("avgScore").innerText = Number(state.avgScore).toFixed(2);
    document.getElementById("bufferSize").innerText = state.bufferSize;
    document.getElementById("gameLog").innerText = state.logMessage;
    networkInfo.innerText = `Episode ${state.episode} | Buffer ${state.bufferSize} | Epsilon ${Number(state.epsilon).toFixed(3)}`;
    renderBoard(state.board);
    drawNetwork(state.network);
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

let loopActive = true;

async function loop() {
    while (loopActive) {
        const state = await requestJson("/api/tick", { method: "POST" });
        updateUI(state);

        if (state.episodeEnded) {
            await playDeathEffect(state.board);
        }

        await new Promise((resolve) => setTimeout(resolve, state.delayMs || 36));
    }
}

document.getElementById("resetBtn").addEventListener("click", async () => {
    if (!window.confirm("This will erase all training progress. Continue?")) {
        return;
    }

    const state = await requestJson("/api/reset", { method: "POST" });
    updateUI(state);
});

const langTitles = {
    en: "How does the model learn? (Deep Q-Network)",
    ar: "كيف يتعلم النموذج؟ (Deep Q-Network)"
};

function setLanguage(lang) {
    document.querySelectorAll(".lang-content").forEach((element) => {
        element.classList.toggle("active", element.id === `lang-${lang}`);
    });
    document.querySelectorAll(".lang-toggle button").forEach((button) => {
        button.classList.toggle("active", button.dataset.lang === lang);
    });
    document.getElementById("explanationTitle").innerText = langTitles[lang];
    localStorage.setItem("snake-explanation-lang", lang);
}

document.querySelectorAll(".lang-toggle button").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.lang));
});

window.addEventListener("resize", () => {
    resizeNetworkCanvas();
    if (lastState) {
        drawNetwork(lastState.network);
    }
});

async function boot() {
    resizeNetworkCanvas();
    const state = await requestJson("/api/state");
    updateUI(state);
    setLanguage(localStorage.getItem("snake-explanation-lang") || "en");
    loop();
}

boot().catch((error) => {
    document.getElementById("gameLog").innerText = `Failed to connect to API: ${error.message}`;
});
