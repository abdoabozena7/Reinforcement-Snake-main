import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "../static/vendor/three.module.js";
import { ReferenceCameras } from "../static/reference-cameras.js";

const convert = (p) => new THREE.Vector3(p.x - 15.5, p.y - 15.5, p.z - 15.5);
const orbit = new THREE.PerspectiveCamera(60, 1, 1, 144);
function aimedAt(camera, target) {
    const forward = camera.getWorldDirection(new THREE.Vector3());
    const expected = target.clone().sub(camera.position).normalize();
    assert(forward.dot(expected) > 0.999999, "The camera must aim at the actual target, not a velocity extrapolation.");
}

test("tail view stays near the real tail and targets the head through horizontal, vertical and bent poses", () => {
    const views = new ReferenceCameras(convert);
    views.setMode("tail");
    for (const snake of [
        [{ x: 18, y: 12, z: 12 }, { x: 12, y: 12, z: 12 }],
        [{ x: 12, y: 26, z: 12 }, { x: 12, y: 12, z: 12 }],
        [{ x: 12, y: 4, z: 12 }, { x: 12, y: 12, z: 12 }],
        [{ x: 18, y: 22, z: 14 }, { x: 12, y: 12, z: 12 }],
    ]) {
        const board = { snake, direction: { x: 1, y: 0, z: 0 }, foods: [] };
        const original = structuredClone(board);
        const camera = views.update(orbit, { board }, 1 / 60);
        aimedAt(camera, convert(snake[0]));
        assert(camera.position.distanceTo(convert(snake.at(-1))) <= 7.000001);
        assert(camera.quaternion.toArray().every(Number.isFinite));
        assert.deepEqual(board, original);
    }
});

test("food view holds food at the focal point with YOU beyond it, including vertical approaches", () => {
    for (const head of [{ x: 3, y: 15, z: 15 }, { x: 15, y: 3, z: 15 }, { x: 15, y: 27, z: 15 }]) {
        const views = new ReferenceCameras(convert);
        views.setMode("food");
        const food = { id: "food", type: "normal", position: { x: 15, y: 15, z: 15 } };
        const camera = views.update(orbit, { board: { snake: [head], foods: [food] } });
        const target = convert(food.position), player = convert(head);
        aimedAt(camera, target);
        assert(camera.position.clone().sub(target).dot(player.clone().sub(target)) < 0);
        assert(camera.position.distanceTo(player) > camera.position.distanceTo(target));
        const projected = target.clone().project(camera);
        assert(Math.abs(projected.x) < 0.000001 && Math.abs(projected.y) < 0.000001);
    }
});

test("switching cameras leaves Orbit intact and missing food is explicitly disclosed", () => {
    const views = new ReferenceCameras(convert);
    const before = orbit.matrixWorld.clone();
    assert.equal(views.update(orbit, null), orbit);
    views.setMode("food");
    assert.equal(views.update(orbit, { board: { snake: [{ x: 12, y: 12, z: 12 }], foods: [] } }), orbit);
    assert.match(views.notice, /waiting for food/);
    assert(orbit.matrixWorld.equals(before));
});
