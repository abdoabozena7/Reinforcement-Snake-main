function activation(value, layer, maximum) {
    return Math.max(0, Math.min(1, layer === 3 ? (value / maximum + 1) / 2 : value / maximum));
}

export function drawPolicyNetwork(canvas, network, compact = false) {
    const ctx = canvas.getContext("2d");
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!network) {
        ctx.fillStyle = "#bfffff";
        ctx.font = "11px Trebuchet MS";
        ctx.fillText("Waiting for Python", 10, 20);
        return;
    }
    const left = compact ? 14 : Math.min(138, width * 0.27);
    const right = compact ? 14 : Math.min(130, width * 0.27);
    const gap = (width - left - right) / (network.layers.length - 1);
    const maxima = network.layers.map((layer) => Math.max(0.001, ...layer.map(Math.abs)));
    const positions = network.layers.map((layer, column) => layer.map((_, row) => ({
        x: left + gap * column,
        y: (compact ? 12 : 42) + (height - (compact ? 24 : 76)) * row / Math.max(1, layer.length - 1),
    })));
    if (!compact) {
        ctx.textAlign = "center";
        ctx.font = "12px Trebuchet MS";
        ctx.fillStyle = "#a8d1d5";
        ["Input", "Hidden A", "Hidden B", "Output"].forEach((title, index) => ctx.fillText(title, left + gap * index, 20));
    }
    network.weights.forEach((matrix, layer) => matrix.forEach((row, source) => row.forEach((weight, target) => {
        const magnitude = Math.min(1, Math.abs(weight) / 0.6);
        if (magnitude < (compact ? 0.2 : 0.03)) return;
        const from = positions[layer][source], to = positions[layer + 1][target];
        const active = activation(network.layers[layer][source], layer, maxima[layer]);
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        const alpha = Math.max(0.03, magnitude * (0.18 + active * 0.5));
        ctx.strokeStyle = weight >= 0 ? `rgba(0,230,118,${alpha})` : `rgba(255,176,77,${alpha})`;
        ctx.lineWidth = compact ? 0.4 + magnitude * 0.4 : 0.4 + magnitude * 1.2;
        ctx.stroke();
    })));
    network.layers.forEach((layer, column) => layer.forEach((value, row) => {
        const position = positions[column][row];
        const active = activation(value, column, maxima[column]);
        const output = column === network.layers.length - 1;
        const action = output ? network.sampleIndices[column][row] : null;
        const masked = output && !network.actionMask[action];
        const selected = output && action === network.lastAction;
        const radius = compact ? 1.1 + active * 1.2 : (column === 0 || output ? 5 : 3.2) + active * 3.8;
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
        ctx.shadowBlur = masked ? 0 : compact ? 2 + active * 4 : 8 + active * 16;
        ctx.shadowColor = "#6cecff";
        ctx.fillStyle = masked ? "#465052" : value < 0 && output ? "#ffb04d" : "#6cecff";
        ctx.fill();
        ctx.shadowBlur = 0;
        if (selected) {
            ctx.strokeStyle = "#00e676";
            ctx.lineWidth = compact ? 1 : 2;
            ctx.stroke();
        }
    }));
    if (compact) return;
    ctx.font = "10px Trebuchet MS";
    function label(text, x, y, available, align) {
        ctx.textAlign = align;
        while (ctx.measureText(text).width > available && text.length > 4) text = text.slice(0, -4) + "...";
        ctx.fillText(text, x, y);
    }
    ctx.fillStyle = "#a8d1d5";
    network.inputLabels.forEach((text, index) => {
        const point = positions[0][index];
        label(text, point.x - 12, point.y + 4, left - 16, "right");
    });
    ctx.fillStyle = "#e9ffff";
    network.outputLabels.forEach((text, index) => {
        const point = positions[3][index];
        label(`${text} ${Number(network.layers[3][index]).toFixed(1)}`, point.x + 12, point.y + 4, right - 15, "left");
    });
}
