import * as THREE from "./vendor/three.module.js";

export class ReferenceCameras {
    constructor(gridToWorld) {
        this.gridToWorld = gridToWorld;
        this.mode = "orbit";
        this.lastMode = null;
        this.side = new THREE.Vector3(1, 0, 0);
        this.tail = new THREE.PerspectiveCamera(60, 1, 0.1, 144);
        this.food = new THREE.PerspectiveCamera(60, 1, 0.1, 144);
        this.target = null;
        this.anchor = null;
        this.notice = "";
    }

    setMode(mode) {
        if (!["orbit", "tail", "food"].includes(mode)) throw new Error(`Unknown reference camera: ${mode}`);
        this.mode = mode;
    }

    resize(width, height) {
        for (const camera of [this.tail, this.food]) {
            camera.aspect = width / Math.max(1, height);
            camera.updateProjectionMatrix();
        }
    }

    update(orbit, state, dt = 1 / 60) {
        this.notice = "";
        this.target = this.anchor = null;
        if (this.mode === "orbit") { this.lastMode = "orbit"; return orbit; }
        const board = state?.board;
        if (!board?.snake?.length) {
            this.notice = "Waiting for YOU";
            return orbit;
        }
        const head = this.gridToWorld(board.snake[0]);
        const tail = this.gridToWorld(board.snake.at(-1));
        let camera, position, foodId = null;
        if (this.mode === "tail") {
            const forward = head.clone().sub(tail);
            if (forward.lengthSq() < 0.0001) forward.set(board.direction.x, board.direction.y, board.direction.z);
            forward.normalize();
            const horizontal = new THREE.Vector3(-forward.z, 0, forward.x);
            if (horizontal.lengthSq() > 0.01) this.side.copy(horizontal.normalize());
            position = tail.clone().addScaledVector(forward, -2).addScaledVector(this.side, 3);
            position.y += 3;
            this.anchor = tail;
            this.target = head;
            camera = this.tail;
        } else {
            const targetId = board.trainingReward?.objectiveAfter?.target?.id;
            const food = board.foods.find((item) => item.id === targetId)
                || board.foods.find((item) => item.type === "normal") || board.foods[0];
            if (!food) {
                this.notice = "Food camera: waiting for food";
                return orbit;
            }
            this.target = this.gridToWorld(food.position);
            foodId = food.id;
            const away = this.target.clone().sub(head);
            if (away.lengthSq() < 0.0001) {
                this.notice = "Food camera: waiting for the next food cell";
                return orbit;
            }
            away.normalize();
            const side = new THREE.Vector3(-away.z, 0, away.x);
            if (side.lengthSq() > 0.01) this.side.copy(side.normalize());
            position = this.target.clone().addScaledVector(away, 8).addScaledVector(this.side, 2);
            position.y += 2.5;
            this.anchor = this.target.clone();
            camera = this.food;
            this.notice = `${food.type === "gold" ? "Gold" : "Food"} in front, YOU behind`;
        }
        const changed = this.lastMode !== this.mode || this.lastEpisode !== state.episodeKey
            || this.mode === "food" && this.lastFood !== foodId;
        if (changed) camera.position.copy(position);
        else camera.position.lerp(position, 1 - Math.exp(-12 * Math.min(0.1, dt)));
        if (this.mode === "tail" && camera.position.distanceTo(tail) > 7) {
            camera.position.sub(tail).setLength(7).add(tail);
        }
        if (this.mode === "food" && camera.position.clone().sub(this.target).dot(head.clone().sub(this.target)) >= 0) {
            camera.position.copy(position);
        }
        camera.up.set(0, 1, 0);
        camera.lookAt(this.target);
        camera.updateMatrixWorld();
        this.lastMode = this.mode;
        this.lastEpisode = state.episodeKey;
        this.lastFood = foodId;
        return camera;
    }

    debug() {
        return { mode: this.mode, target: this.target?.toArray() || null,
            anchor: this.anchor?.toArray() || null, notice: this.notice };
    }
}
