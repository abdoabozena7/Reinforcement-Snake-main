import * as THREE from "./vendor/three.module.js";

export class ReferenceWorld {
    constructor(scene, gridToWorld, grid = null) {
        this.scene = scene;
        this.gridToWorld = gridToWorld;
        this.pools = new Map();
        this.box = new THREE.BoxGeometry(1, 1, 1);
        this.sphere = new THREE.SphereGeometry(0.4, 16, 16);
        this.ring = new THREE.TorusGeometry(0.74, 0.035, 6, 40);
        this.gridMaterials = grid ? grid.children.map((line) => line.material) : [];
        this.pulseStarted = -Infinity;
        this.growthPulse = 0;
        this.lastFrame = null;
        this.materials = {
            obstacle: new THREE.MeshStandardMaterial({ color: 0x80858b, emissive: 0x303238, emissiveIntensity: 0.35, roughness: 0.48 }),
            wall: new THREE.MeshStandardMaterial({ color: 0x72838a, roughness: 0.7 }),
            preview: new THREE.MeshBasicMaterial({ color: 0xf2bc69, wireframe: true, transparent: true, opacity: 0.65 }),
            open: new THREE.MeshBasicMaterial({ color: 0x5ca7ac, wireframe: true, transparent: true, opacity: 0.13 }),
            warning: new THREE.MeshBasicMaterial({ color: 0xf2bc69, wireframe: true, transparent: true, opacity: 0.7 }),
            active: new THREE.MeshStandardMaterial({ color: 0xd6544b, emissive: 0xb83f30, emissiveIntensity: 0.65 }),
            gold: new THREE.MeshStandardMaterial({ color: 0xffdf8d, emissive: 0xffb52e, emissiveIntensity: 1.4, roughness: 0.24 }),
            foodRing: new THREE.MeshBasicMaterial({ color: 0x9dbcff }),
            goldRing: new THREE.MeshBasicMaterial({ color: 0xffd26b }),
        };
        const pixels = new Uint8Array(64 * 64 * 4);
        for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
            const radius = Math.hypot((x - 31.5) / 31.5, (y - 31.5) / 31.5);
            pixels.set([255, 255, 255, Math.round(255 * Math.max(0, 1 - radius) ** 2)], (y * 64 + x) * 4);
        }
        const glowTexture = new THREE.DataTexture(pixels, 64, 64, THREE.RGBAFormat);
        glowTexture.minFilter = glowTexture.magFilter = THREE.LinearFilter;
        glowTexture.needsUpdate = true;
        this.glowMaterials = Object.fromEntries([
            ["player", 0xff4b8a, 0.4], ["normal", 0x92baff, 0.82], ["gold", 0xffd26b, 0.94],
        ].map(([role, color, opacity]) => [role, new THREE.SpriteMaterial({
            map: glowTexture, color, opacity, transparent: true, blending: THREE.AdditiveBlending,
            depthWrite: false, toneMapped: false,
        })]));
        this.zone = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
            new THREE.LineBasicMaterial({ color: 0xe3aa5e, transparent: true, opacity: 0.38 }));
        this.zone.visible = false;
        scene.add(this.zone);
    }

    sync(key, cells, geometry, material, scale = 1) {
        let pool = this.pools.get(key);
        if (!pool) { pool = []; this.pools.set(key, pool); }
        while (pool.length < cells.length) {
            const mesh = geometry === null ? new THREE.Sprite(material) : new THREE.Mesh(geometry, material);
            mesh.scale.setScalar(scale);
            this.scene.add(mesh);
            pool.push(mesh);
        }
        while (pool.length > cells.length) this.scene.remove(pool.pop());
        cells.forEach((cell, index) => pool[index].position.copy(this.gridToWorld(cell)));
    }

    setBoard(board, { advanced = false, episode = "", now = performance.now() } = {}) {
        const identity = `${episode}:${board.seed}`;
        const length = board.snake?.length || 0;
        if (advanced && this.lastFrame?.identity === identity && board.steps > this.lastFrame.steps
            && length > this.lastFrame.length && (board.events || []).some((event) =>
                ["food", "gold_food"].includes(event.type) && event.step > this.lastFrame.steps)) {
            this.pulseStarted = now;
        } else if (this.lastFrame && (this.lastFrame.identity !== identity
            || !advanced && board.steps !== this.lastFrame.steps)) {
            this.pulseStarted = -Infinity;
        }
        this.lastFrame = { identity, steps: board.steps, length };
        const active = new Set(["obstacles", "walls", "wall-preview", "gold", "obstacle-warning",
            "player-glow", "normal-glow", "gold-glow", "normal-ring", "gold-ring"]);
        this.sync("obstacles", board.obstacles, this.box, this.materials.obstacle);
        this.sync("walls", board.movingWalls.flatMap((wall) => wall.cells), this.box, this.materials.wall);
        this.sync("wall-preview", board.movingWalls.filter((wall) => wall.warning).flatMap((wall) => wall.nextCells),
            this.box, this.materials.preview);
        this.sync("gold", board.foods.filter((food) => food.type === "gold").map((food) => food.position),
            this.sphere, this.materials.gold, 1.5);
        this.sync("obstacle-warning", board.obstacleChange?.addedCells || [], this.box, this.materials.preview);
        this.sync("player-glow", board.snake || [], null, this.glowMaterials.player, 2.8);
        for (const type of ["normal", "gold"]) {
            const cells = board.foods.filter((food) => food.type === type).map((food) => food.position);
            this.sync(`${type}-glow`, cells,
                null, this.glowMaterials[type], type === "gold" ? 3.6 : 3.2);
            this.sync(`${type}-ring`, cells, this.ring, type === "gold" ? this.materials.goldRing : this.materials.foodRing);
            for (const ring of this.pools.get(`${type}-ring`)) ring.rotation.x = Math.PI / 2;
        }
        for (const gate of board.gates) {
            const key = `gate:${gate.id}`;
            active.add(key);
            this.sync(key, gate.cells, this.box, this.materials[gate.state]);
            for (const mesh of this.pools.get(key)) mesh.material = this.materials[gate.state];
        }
        for (const [key, pool] of this.pools) if (!active.has(key)) {
            pool.forEach((mesh) => this.scene.remove(mesh));
            this.pools.delete(key);
        }
        const inset = board.zone.warning ? board.zone.nextInset : board.zone.inset;
        this.zone.visible = inset > 0;
        this.zone.scale.setScalar(Math.max(0.01, board.gridSize - inset * 2));
        this.zone.material.color.setHex(board.zone.warning ? 0xe3aa5e : 0xe37c6c);
    }

    update(now) {
        this.growthPulse = THREE.MathUtils.clamp(1 - (now - this.pulseStarted) / 650, 0, 1) ** 2;
        for (const material of this.gridMaterials) material.opacity = 0.1 + this.growthPulse * 0.4;
        this.glowMaterials.player.opacity = 0.4 + this.growthPulse * 0.4;
        for (const glow of this.pools.get("player-glow") || []) glow.scale.setScalar(2.8 + this.growthPulse * 0.6);
    }

    debug() {
        return {
            cells: Object.fromEntries([...this.pools].map(([key, pool]) => [key, pool.length])),
            positions: Object.fromEntries([...this.pools].map(([key, pool]) => [key, pool.map((mesh) => mesh.position.toArray())])),
            zone: { visible: this.zone.visible, extent: this.zone.scale.x },
            growthPulse: this.growthPulse,
            gridOpacity: this.gridMaterials[0]?.opacity ?? 0.1,
        };
    }
}
