import * as THREE from './vendor/three.module.js';

const MAX_SEGMENTS = 64;
const SUBDIVISIONS = 3;
const RADIAL_SIDES = 14;
const MAX_RINGS = MAX_SEGMENTS * SUBDIVISIONS + 4;
const UP = new THREE.Vector3(0, 1, 0);
const FRONT = new THREE.Vector3(0, 0, 1);
const IDENTITY = new THREE.Quaternion();
const TAU = Math.PI * 2;
const clamp = THREE.MathUtils.clamp;
const smooth = (a, b, x) => {
    const t = clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
};
const damp = (rate, dt) => 1 - Math.exp(-rate * dt);

const PALETTES = {
    player: {
        skin: 0x66c63b, muzzle: 0x83d64c, body: 0x58b935, belly: 0xdce5a4,
        detail: 0x3e902a, brow: 0x316b27, eye: 0xfffef2, iris: 0x3d8e60,
        accent: 0xa2de64, radius: 0.335,
    },
    hunter: {
        skin: 0xca4940, muzzle: 0xe36a52, body: 0xb83c36, belly: 0xe7b288,
        detail: 0x792f35, brow: 0x66252c, eye: 0xffefd4, iris: 0xe49e31,
        accent: 0x84323e, radius: 0.35,
    },
    rival: {
        skin: 0x8e68ca, muzzle: 0xae87dc, body: 0x7955b6, belly: 0xd6c6e7,
        detail: 0x553b8a, brow: 0x3b315a, eye: 0xf6f2ff, iris: 0x6bbec9,
        accent: 0xcc9ce8, radius: 0.30,
    },
    patrol: {
        skin: 0xd89430, muzzle: 0xf0b943, body: 0x48484c, belly: 0x787263,
        detail: 0x343942, brow: 0x343942, eye: 0xffd77d, iris: 0xffb636,
        accent: 0xffcd61, radius: 0.29,
    },
    ambusher: {
        skin: 0x497e70, muzzle: 0x649b78, body: 0x3a6e63, belly: 0xc3ca8e,
        detail: 0x304d51, brow: 0x253d3b, eye: 0xf5edbd, iris: 0xdcae4a,
        accent: 0x839d58, radius: 0.37,
    },
};

const geometryPool = new Map();
let resourceUsers = 0;

function sharedGeometry(key, build) {
    if (!geometryPool.has(key)) geometryPool.set(key, build());
    return geometryPool.get(key);
}

function sphere() {
    return sharedGeometry('sphere', () => new THREE.SphereGeometry(1, 24, 16));
}

function faceSphere() {
    return sharedGeometry('face-sphere', () => new THREE.SphereGeometry(1, 48, 32));
}

function detailSphere() {
    return sharedGeometry('detail-sphere', () => new THREE.SphereGeometry(1, 16, 10));
}

function cylinder() {
    return sharedGeometry('cylinder', () => new THREE.CylinderGeometry(1, 1, 1, 8));
}

function mergeParts(parts) {
    let vertexCount = 0;
    let indexCount = 0;
    for (const part of parts) {
        vertexCount += part.geometry.attributes.position.count;
        indexCount += part.geometry.index?.count ?? part.geometry.attributes.position.count;
    }
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const indices = new Uint32Array(indexCount);
    const transform = new THREE.Matrix4();
    const normalTransform = new THREE.Matrix3();
    const rotation = new THREE.Quaternion();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    const vector = new THREE.Vector3();
    let offset = 0;
    let indexOffset = 0;
    for (const part of parts) {
        position.fromArray(part.position ?? [0, 0, 0]);
        scale.fromArray(part.scale ?? [1, 1, 1]);
        rotation.setFromEuler(new THREE.Euler(...(part.rotation ?? [0, 0, 0])));
        transform.compose(position, rotation, scale);
        normalTransform.getNormalMatrix(transform);
        const source = part.geometry;
        for (let i = 0; i < source.attributes.position.count; i++) {
            vector.fromBufferAttribute(source.attributes.position, i).applyMatrix4(transform);
            vector.toArray(positions, (offset + i) * 3);
            vector.fromBufferAttribute(source.attributes.normal, i).applyMatrix3(normalTransform).normalize();
            vector.toArray(normals, (offset + i) * 3);
        }
        const count = source.index?.count ?? source.attributes.position.count;
        for (let i = 0; i < count; i++) {
            indices[indexOffset++] = offset + (source.index ? source.index.getX(i) : i);
        }
        offset += source.attributes.position.count;
    }
    const result = new THREE.BufferGeometry();
    result.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    result.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    result.setIndex(new THREE.BufferAttribute(indices, 1));
    result.computeBoundingSphere();
    return result;
}

function sculptedSkull(kind) {
    return sharedGeometry(`skull:${kind}`, () => {
        const geometry = new THREE.SphereGeometry(1, 60, 40);
        const positions = geometry.attributes.position;
        const point = new THREE.Vector3();
        for (let i = 0; i < positions.count; i++) {
            point.fromBufferAttribute(positions, i);
            const front = Math.max(0, point.z);
            const cheek = Math.exp(-(((point.z - 0.2) / 0.65) ** 2)) * (1 - point.y) * 0.035;
            const muzzle = front * front * Math.exp(-(((point.y + 0.22) / 0.62) ** 2));
            let width = 0.535 + cheek;
            if (kind === 'hunter') width *= 1 - front * 0.11;
            if (kind === 'ambusher') width *= 1.10;
            positions.setXYZ(i, point.x * width, point.y * 0.435 - muzzle * 0.025,
                point.z * 0.625 + muzzle * 0.085);
        }
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        return geometry;
    });
}

function skinGeometry(kind) {
    const socketZ = kind === 'player' ? 0.36 : 0.30;
    return sharedGeometry(`skin:${kind}`, () => mergeParts([
        { geometry: sculptedSkull(kind) },
        { geometry: faceSphere(), position: [-0.285, 0.23, socketZ], scale: [0.285, 0.327, 0.199] },
        { geometry: faceSphere(), position: [0.285, 0.23, socketZ], scale: [0.285, 0.327, 0.199] },
    ]));
}

function muzzleGeometry() {
    return sharedGeometry('muzzle', () => mergeParts([
        { geometry: faceSphere(), position: [0, -0.095, 0.48], scale: [0.39, 0.21, 0.27] },
        { geometry: faceSphere(), position: [-0.408, -0.065, 0.22], scale: [0.175, 0.185, 0.23] },
        { geometry: faceSphere(), position: [0.408, -0.065, 0.22], scale: [0.175, 0.185, 0.23] },
    ]));
}

function faceSurfaceZ(x, y) {
    return 0.48 + 0.27 * Math.sqrt(Math.max(0.01,
        1 - (x / 0.39) ** 2 - ((y + 0.095) / 0.21) ** 2)) + 0.008;
}

function frecklesGeometry() {
    return sharedGeometry('freckles', () => {
        const parts = [];
        for (const side of [-1, 1]) {
            parts.push({
                geometry: detailSphere(), position: [side * 0.115, -0.008, faceSurfaceZ(0.115, -0.008)],
                scale: [0.024, 0.014, 0.008],
            });
            for (let i = 0; i < 3; i++) {
                const x = side * (0.25 + i * 0.032);
                const y = -0.08 + (i % 2) * 0.025;
                parts.push({
                    geometry: detailSphere(), position: [x, y, faceSurfaceZ(x, y)],
                    scale: [0.0085, 0.0085, 0.005],
                });
            }
        }
        return mergeParts(parts);
    });
}

function leafGeometry() {
    return sharedGeometry('leaf', () => {
        const shape = new THREE.Shape();
        shape.moveTo(0, -0.5);
        shape.bezierCurveTo(-0.27, -0.22, -0.47, 0.13, 0, 0.5);
        shape.bezierCurveTo(0.47, 0.13, 0.27, -0.22, 0, -0.5);
        const geometry = new THREE.ShapeGeometry(shape, 7);
        geometry.rotateX(-Math.PI / 2);
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            positions.setY(i, -x * x * 0.10);
        }
        geometry.computeVertexNormals();
        return geometry;
    });
}

function foreheadLeafGeometry() {
    return sharedGeometry('forehead-leaf', () => {
        const source = leafGeometry();
        const points = source.attributes.position;
        let vertices = [];
        for (let i = 0; i < source.index.count; i++) {
            const index = source.index.getX(i);
            vertices.push(points.getX(index), points.getY(index), points.getZ(index));
        }
        // A projected boundary still has sunken interior chords on a curved head.
        for (let pass = 0; pass < 2; pass++) {
            const refined = [];
            for (let i = 0; i < vertices.length; i += 9) {
                const ax = vertices[i], ay = vertices[i + 1], az = vertices[i + 2];
                const bx = vertices[i + 3], by = vertices[i + 4], bz = vertices[i + 5];
                const cx = vertices[i + 6], cy = vertices[i + 7], cz = vertices[i + 8];
                const abx = (ax + bx) / 2, aby = (ay + by) / 2, abz = (az + bz) / 2;
                const bcx = (bx + cx) / 2, bcy = (by + cy) / 2, bcz = (bz + cz) / 2;
                const cax = (cx + ax) / 2, cay = (cy + ay) / 2, caz = (cz + az) / 2;
                refined.push(
                    ax, ay, az, abx, aby, abz, cax, cay, caz,
                    abx, aby, abz, bx, by, bz, bcx, bcy, bcz,
                    cax, cay, caz, bcx, bcy, bcz, cx, cy, cz,
                    abx, aby, abz, bcx, bcy, bcz, cax, cay, caz);
            }
            vertices = refined;
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.scale(0.16, 1, 0.35);
        geometry.translate(0, 0, -0.17);
        const positions = geometry.attributes.position;
        const normals = new Float32Array(positions.count * 3);
        const normal = new THREE.Vector3();
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const z = positions.getZ(i);
            const y = 0.435 * Math.sqrt(Math.max(0,
                1 - (x / 0.55) ** 2 - (z / 0.625) ** 2));
            positions.setY(i, y + 0.005);
            normal.set(x / (0.55 ** 2), y / (0.435 ** 2), z / (0.625 ** 2))
                .normalize().toArray(normals, i * 3);
        }
        geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        geometry.computeBoundingSphere();
        return geometry;
    });
}

function crestGeometry(small = false) {
    return sharedGeometry(small ? 'body-fin' : 'crest', () => {
        const shape = new THREE.Shape();
        if (small) {
            shape.moveTo(-0.27, 0);
            shape.bezierCurveTo(-0.14, 0.19, 0.06, 0.27, 0.26, 0);
        } else {
            shape.moveTo(-0.55, 0.29);
            shape.bezierCurveTo(-0.51, 0.66, -0.15, 0.79, 0.27, 0.34);
            shape.quadraticCurveTo(-0.04, 0.39, -0.55, 0.29);
        }
        shape.closePath();
        const geometry = new THREE.ExtrudeGeometry(shape, {
            depth: small ? 0.045 : 0.065, bevelEnabled: true,
            bevelSegments: 2, steps: 1, bevelSize: 0.018, bevelThickness: 0.012, curveSegments: 8,
        });
        geometry.rotateY(-Math.PI / 2);
        geometry.translate(small ? 0.0225 : 0.0325, 0, 0);
        return geometry;
    });
}

function shellGeometry() {
    return sharedGeometry('patrol-shell', () => {
        const shape = new THREE.Shape();
        shape.moveTo(-0.40, -0.19);
        shape.quadraticCurveTo(-0.59, -0.12, -0.52, 0.11);
        shape.quadraticCurveTo(-0.43, 0.37, -0.26, 0.39);
        shape.lineTo(0.26, 0.39);
        shape.quadraticCurveTo(0.43, 0.37, 0.52, 0.11);
        shape.quadraticCurveTo(0.59, -0.12, 0.40, -0.19);
        shape.closePath();
        const geometry = new THREE.ExtrudeGeometry(shape, {
            depth: 0.79, bevelEnabled: true, bevelSize: 0.055,
            bevelThickness: 0.055, bevelSegments: 3, steps: 1, curveSegments: 7,
        });
        geometry.translate(0, 0, -0.42);
        return geometry;
    });
}

function curveTube(key, points, radius, sides = 8) {
    return sharedGeometry(key, () => new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(points.map(point => new THREE.Vector3(...point))),
        20, radius, sides, false));
}

function eyePatch(key, width, height) {
    return sharedGeometry(key, () => {
        const rings = 8;
        const sides = 24;
        const positions = [];
        const indices = [];
        for (let ring = 0; ring <= rings; ring++) {
            for (let side = 0; side <= sides; side++) {
                const angle = side / sides * TAU;
                const x = Math.sin(ring / rings * width) * Math.cos(angle);
                const y = Math.sin(ring / rings * height) * Math.sin(angle);
                positions.push(x, y, Math.sqrt(Math.max(0, 1 - x * x - y * y)));
                if (ring < rings && side < sides) {
                    const a = ring * (sides + 1) + side;
                    const b = a + sides + 1;
                    indices.push(a, b, b + 1, a, b + 1, a + 1);
                }
            }
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setIndex(indices);
        geometry.computeBoundingSphere();
        return geometry;
    });
}

function tubeBuffer(rings, sides, colored = false) {
    const geometry = new THREE.BufferGeometry();
    const vertices = rings * (sides + 1);
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3)
        .setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3)
        .setUsage(THREE.DynamicDrawUsage));
    if (colored) {
        geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3)
            .setUsage(THREE.DynamicDrawUsage));
    }
    const indices = new Uint16Array((rings - 1) * sides * 6);
    let index = 0;
    for (let ring = 0; ring < rings - 1; ring++) {
        for (let side = 0; side < sides; side++) {
            const a = ring * (sides + 1) + side;
            const b = a + sides + 1;
            indices[index++] = a;
            indices[index++] = a + 1;
            indices[index++] = b;
            indices[index++] = a + 1;
            indices[index++] = b + 1;
            indices[index++] = b;
        }
    }
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 40);
    return geometry;
}

export class SnakeCharacter {
    constructor(kind = 'player', { headOnly = false } = {}) {
        this.kind = kind === 'ambush' ? 'ambusher' : (PALETTES[kind] ? kind : 'player');
        this.headOnly = headOnly;
        this.palette = PALETTES[this.kind];
        resourceUsers++;
        this.group = new THREE.Group();
        this.group.name = `character:${this.kind}`;
        this.group.userData.kind = this.kind;
        this.group.visible = false;
        this.head = new THREE.Group();
        this.head.name = 'head-anchor';
        this.head.scale.setScalar(headOnly ? 1 : 0.6);
        this.group.add(this.head);
        this.headPosition = new THREE.Vector3(0, 0.6, 0);
        this.forward = new THREE.Vector3(0, 0, 1);
        this._visual = new THREE.Group();
        this._visual.name = 'cosmetic-face-rig';
        this.head.add(this._visual);
        if (this.kind === 'rival') this._visual.scale.set(0.87, 0.83, 1.08);
        if (this.kind === 'hunter') this._visual.scale.set(1.02, 0.95, 1.08);
        if (this.kind === 'ambusher') this._visual.scale.set(1.05, 0.84, 0.99);
        this._restVisualScale = this._visual.scale.clone();

        this._materials = new Set();
        this._ownedGeometries = [];
        this._instances = [];
        this._points = Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector3());
        this._targets = Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector3());
        this._samples = Array.from({ length: MAX_RINGS }, () => new THREE.Vector3());
        this._mouthCenters = Array.from({ length: 21 }, () => new THREE.Vector3());
        this._part = new THREE.Object3D();
        this._matrix = new THREE.Matrix4();
        this._surfaceMatrix = new THREE.Matrix4();
        this._eyeMatrix = new THREE.Matrix4();
        this._ellipsoidMatrix = new THREE.Matrix4().makeScale(0.253, 0.305, 0.222);
        this._quaternion = new THREE.Quaternion();
        this._surfaceQuaternion = new THREE.Quaternion();
        this._targetQuaternion = new THREE.Quaternion();
        this._inverseQuaternion = new THREE.Quaternion();
        this._vector = new THREE.Vector3();
        this._tangent = new THREE.Vector3();
        this._normal = new THREE.Vector3();
        this._binormal = new THREE.Vector3();
        this._lookTarget = new THREE.Vector3();
        this._localLook = new THREE.Vector3();
        this._portraitLook = new THREE.Vector3(0, 0.2, 3);
        this._baseSkin = new THREE.Color(this.palette.skin);
        this._baseMuzzle = new THREE.Color(this.palette.muzzle);
        this._bodyColor = new THREE.Color(this.palette.body);
        this._bellyColor = new THREE.Color(this.palette.belly);
        this._color = new THREE.Color();
        this._deadTone = new THREE.Color(0x929fa4);
        this._baseBelly = new THREE.Color(this.palette.belly);
        this._baseDetail = new THREE.Color(this.palette.detail);
        this._baseAccent = new THREE.Color(this.palette.accent);
        this._hitTone = new THREE.Color(0xffefd0);
        this._lastBodyDeath = -1;
        this._lastBodyFlash = -1;
        this._eventIds = new Set();
        this._eventQueue = [];
        this._count = 0;
        this._ringCount = 0;
        this._headingYaw = 0;
        this._turnSign = 1;
        this._threat = 0;
        this._targetThreat = 0;
        this._death = 0;
        this._dead = false;
        this._alert = false;
        this._foodTime = 0;
        this._reliefTime = 0;
        this._dash = 0;
        this._shield = 0;
        this._gazeX = 0;
        this._gazeY = 0;
        this._elapsed = 0;
        this._movement = 0;
        this._lastEmotion = 'safe';
        this._externalPose = false;
        this._presentation = null;
        this._reducedMotion = false;
        this._disposed = false;

        this._skinMaterial = this._material(this.palette.skin, { clearcoat: 0.26 });
        this._muzzleMaterial = this._material(this.palette.muzzle, { clearcoat: 0.19 });
        this._bellyMaterial = this._material(this.palette.belly, { roughness: 0.57 });
        this._detailMaterial = this._material(this.palette.detail, { roughness: 0.62 });
        this._browMaterial = this._material(this.palette.brow, { roughness: 0.62 });
        this._accentMaterial = this._material(this.palette.accent, { roughness: 0.42 });
        this._bodyMaterial = this._material(0xffffff, { vertexColors: true, clearcoat: 0.18 });
        if (!headOnly) this._createBody();
        if (this.kind === 'patrol') this._createPatrol();
        else this._createFace();
        if (!headOnly) this._createAbilities();
    }

    _material(color, options = {}) {
        const material = new THREE.MeshPhysicalMaterial({
            color, roughness: 0.43, metalness: 0, clearcoatRoughness: 0.32,
            envMapIntensity: 0.65, ...options,
        });
        this._materials.add(material);
        return material;
    }

    _mesh(geometry, material, parent = this._visual, name = '') {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = name;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        parent.add(mesh);
        return mesh;
    }

    _instanced(geometry, material, count, parent = this._visual, name = '') {
        const mesh = new THREE.InstancedMesh(geometry, material, count);
        mesh.name = name;
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        mesh.frustumCulled = false;
        mesh.castShadow = false;
        mesh.receiveShadow = true;
        parent.add(mesh);
        this._instances.push(mesh);
        return mesh;
    }

    _put(mesh, index, x, y, z, sx, sy, sz, rotation = IDENTITY, parent = null) {
        this._part.position.set(x, y, z);
        this._part.scale.set(sx, sy, sz);
        this._part.quaternion.copy(rotation);
        this._part.updateMatrix();
        if (parent) {
            this._matrix.multiplyMatrices(parent, this._part.matrix);
            mesh.setMatrixAt(index, this._matrix);
        } else {
            mesh.setMatrixAt(index, this._part.matrix);
        }
    }

    _createBody() {
        this._bodyGeometry = tubeBuffer(MAX_RINGS, RADIAL_SIDES, true);
        this._ownedGeometries.push(this._bodyGeometry);
        this._body = this._mesh(this._bodyGeometry, this._bodyMaterial, this.group, 'continuous-tapered-body');
        this._body.frustumCulled = false;
        this._bodyGeometry.setDrawRange(0, 0);
        let detail = leafGeometry();
        let material = this._detailMaterial;
        if (this.kind === 'hunter' || this.kind === 'ambusher') {
            detail = sharedGeometry('spine', () => new THREE.ConeGeometry(1, 1, 5));
            material = this._accentMaterial;
        } else if (this.kind === 'rival') {
            detail = crestGeometry(true);
            material = this._accentMaterial;
        } else if (this.kind === 'patrol') {
            detail = shellGeometry();
            material = this._skinMaterial;
        }
        this._bodyDetails = this._instanced(detail, material, MAX_SEGMENTS, this.group, 'pooled-dorsal-details');
        this._bodyDetails.count = 0;
    }

    _createFace() {
        this._mesh(skinGeometry(this.kind), this._skinMaterial, this._visual, 'sculpted-skull-and-eye-sockets');
        this._upperMuzzle = this._mesh(muzzleGeometry(), this._muzzleMaterial, this._visual, 'muzzle-and-cheeks');
        this._jaw = this._mesh(faceSphere(), this._bellyMaterial, this._visual, 'cream-lower-lip');
        this._jaw.position.set(0, -0.245, 0.46);
        this._jaw.scale.set(0.365, 0.095, 0.23);
        this._mouthCavity = this._mesh(faceSphere(), this._material(0x301b22, { roughness: 0.8 }),
            this._visual, 'contact-only-open-mouth');
        this._mouthCavity.position.set(0, -0.205, 0.53);
        this._mouthCavity.scale.set(0.31, 0.18, 0.19);
        this._mouthCavity.visible = false;
        this._mesh(frecklesGeometry(), this._detailMaterial, this._visual, 'nostrils-and-cheek-freckles');

        this._whiteMaterial = this._material(this.palette.eye, { roughness: 0.26, clearcoat: 0.30 });
        this._irisMaterial = this._material(this.palette.iris, { roughness: 0.35 });
        this._pupilMaterial = this._material(0x12251b, { roughness: 0.23, clearcoat: 0.35 });
        this._highlightMaterial = this._material(0xffffff, {
            roughness: 0.18, emissive: 0xffffff, emissiveIntensity: 0.10,
        });
        this._whites = this._instanced(faceSphere(), this._whiteMaterial, 2, this._visual, 'eye-whites');
        const predator = this.kind === 'hunter' || this.kind === 'ambusher';
        this._irises = this._instanced(eyePatch(predator ? 'predator-iris' : 'iris', predator ? 0.355 : 0.441, 0.426),
            this._irisMaterial, 2, this._visual, 'surface-following-irises');
        this._pupils = this._instanced(eyePatch(predator ? 'slit-pupil' : 'pupil', predator ? 0.135 : 0.275, 0.31),
            this._pupilMaterial, 2, this._visual, 'pupils');
        this._highlights = this._instanced(detailSphere(), this._highlightMaterial, 4, this._visual, 'eye-catchlights');
        this._eyeRoots = [-1, 1].map(side => {
            const root = new THREE.Object3D();
            root.position.set(side * 0.285, 0.23, this.kind === 'player' ? 0.46 : 0.34);
            root.rotation.y = side * 0.11;
            return root;
        });
        const brow = curveTube('brow', [
            [-0.21, -0.01, 0], [-0.10, 0.043, 0.025],
            [0.08, 0.05, 0.022], [0.21, -0.005, -0.005],
        ], 0.031);
        this._brows = this._instanced(brow, this._browMaterial, 2, this._visual, 'expressive-brows');
        this._crosses = this._instanced(cylinder(), this._pupilMaterial, 4, this._visual, 'terminal-eye-marks');
        this._crosses.visible = false;
        this._eyeMeshes = [this._whites, this._irises, this._pupils, this._highlights, this._brows, this._crosses];
        this._mouthGeometry = tubeBuffer(21, 6);
        this._ownedGeometries.push(this._mouthGeometry);
        this._mouth = this._mesh(this._mouthGeometry, this._browMaterial, this._visual, 'curved-expression-mouth');
        this._mouth.castShadow = false;
        this._mouth.frustumCulled = false;

        const tongueGeometry = sharedGeometry('forked-tongue', () => mergeParts([
            { geometry: curveTube('tongue-stem', [[0, 0, 0], [0, -0.015, 0.10], [0, -0.035, 0.19]], 0.016, 6) },
            { geometry: curveTube('tongue-left', [[0, -0.035, 0.18], [-0.032, -0.042, 0.235]], 0.010, 5) },
            { geometry: curveTube('tongue-right', [[0, -0.035, 0.18], [0.032, -0.042, 0.235]], 0.010, 5) },
        ]));
        this._tongue = this._mesh(tongueGeometry, this._material(0xde7885, { roughness: 0.44 }), this._visual, 'forked-tongue');
        this._tongue.position.set(0, -0.175, faceSurfaceZ(0, -0.175));
        this._tongue.visible = false;
        this._tongue.castShadow = false;

        if (this.kind === 'player') {
            this._mesh(foreheadLeafGeometry(), this._accentMaterial, this._visual, 'forehead-leaf-mark');
        } else if (this.kind === 'hunter' || this.kind === 'ambusher') {
            const fang = sharedGeometry('fang', () => new THREE.ConeGeometry(1, 1, 10));
            this._fangs = this._instanced(fang, this._bellyMaterial, 2, this._visual, 'predator-fangs');
            this._attackJaw = new THREE.Group();
            this._attackJaw.name = 'hinged-bite-jaw';
            this._attackJaw.position.set(0, -0.12, -0.05);
            this._attackJaw.visible = false;
            this._visual.add(this._attackJaw);
            const lowerJaw = this._mesh(faceSphere(), this._bellyMaterial, this._attackJaw, 'bite-lower-jaw');
            lowerJaw.position.set(0, -0.12, 0.45);
            lowerJaw.scale.set(0.365, 0.11, 0.32);
            const mouthFloor = this._mesh(faceSphere(), this._material(0x65454d, { roughness: 0.72 }),
                this._attackJaw, 'bite-inner-mouth');
            mouthFloor.position.set(0, -0.025, 0.43);
            mouthFloor.scale.set(0.31, 0.045, 0.28);
            this._lowerFangs = this._instanced(fang, this._bellyMaterial, 2, this._attackJaw, 'bite-lower-fangs');
            for (let i = 0; i < 2; i++) this._put(this._lowerFangs, i, (i ? 1 : -1) * 0.245, 0.02, 0.59,
                0.046, 0.15, 0.052);
            this._lowerFangs.instanceMatrix.needsUpdate = true;
            this._quaternion.setFromAxisAngle(FRONT, Math.PI);
            for (let i = 0; i < 2; i++) {
                this._put(this._fangs, i, (i ? 1 : -1) * 0.255, -0.224, 0.635,
                    0.058, 0.235, 0.071, this._quaternion);
            }
            this._fangs.instanceMatrix.needsUpdate = true;
            const spines = this._instanced(
                sharedGeometry('spine', () => new THREE.ConeGeometry(1, 1, 5)),
                this._accentMaterial, 3, this._visual, 'predator-crown-spines');
            for (let i = 0; i < 3; i++) {
                this._put(spines, i, 0, 0.42 + i * 0.006, -0.39 + i * 0.19,
                    0.105 - i * 0.012, 0.32 - i * 0.055, 0.13);
            }
            spines.instanceMatrix.needsUpdate = true;
        } else if (this.kind === 'rival') {
            this._mesh(crestGeometry(), this._accentMaterial, this._visual, 'swept-rival-crest');
            this._mesh(curveTube('rival-visor-rim', [
                [-0.49, 0.36, 0.35], [-0.31, 0.49, 0.47], [0, 0.47, 0.46],
                [0.31, 0.49, 0.47], [0.49, 0.36, 0.35],
            ], 0.021), this._browMaterial, this._visual, 'sleek-visor-rim');
        }
    }

    _createPatrol() {
        this._mesh(shellGeometry(), this._skinMaterial, this._visual, 'armored-patrol-carapace');
        const underside = this._mesh(sphere(), this._detailMaterial, this._visual, 'mechanical-undercarriage');
        underside.scale.set(0.47, 0.15, 0.43);
        underside.position.set(0, -0.14, 0);
        const visor = this._mesh(sphere(), this._browMaterial, this._visual, 'patrol-sensor-visor');
        visor.position.set(0, 0.12, 0.433);
        visor.scale.set(0.435, 0.13, 0.073);
        this._opticMaterial = this._material(this.palette.accent, {
            emissive: this.palette.accent, emissiveIntensity: 0.45, metalness: 0.18, roughness: 0.30,
        });
        const optics = this._instanced(sphere(), this._opticMaterial, 3, this._visual, 'three-patrol-sensors');
        for (let i = 0; i < 3; i++) {
            this._put(optics, i, (i - 1) * 0.24, 0.13, 0.503, 0.085, 0.033, 0.018);
        }
        optics.instanceMatrix.needsUpdate = true;
        const seam = this._mesh(sphere(), this._accentMaterial, this._visual, 'carapace-center-stripe');
        seam.position.set(0, 0.43, -0.02);
        seam.scale.set(0.032, 0.015, 0.34);
        this._legs = this._instanced(cylinder(), this._detailMaterial, 12, this._visual, 'six-articulated-patrol-legs');
        this._feet = this._instanced(sphere(), this._muzzleMaterial, 6, this._visual, 'patrol-feet');
        const bolts = this._instanced(sphere(), this._bellyMaterial, 6, this._visual, 'shell-rivets');
        for (let i = 0; i < 6; i++) {
            this._put(bolts, i, i < 3 ? -0.45 : 0.45, 0.19,
                ((i % 3) - 1) * 0.23, 0.025, 0.025, 0.025);
        }
        bolts.instanceMatrix.needsUpdate = true;
    }

    _createAbilities() {
        const shieldMaterial = this._material(0x8cddd0, {
            transparent: true, opacity: 0.075, depthWrite: false,
            roughness: 0.24, emissive: 0x65bcb7, emissiveIntensity: 0.17,
        });
        this._shieldBubble = this._mesh(sphere(), shieldMaterial, this.head, 'active-shield');
        this._shieldBubble.scale.set(0.81, 0.73, 0.96);
        this._shieldBubble.position.z = 0.06;
        this._shieldBubble.castShadow = false;
        this._shieldBubble.receiveShadow = false;
        this._shieldBubble.visible = false;
        this._shieldRing = this._mesh(sharedGeometry('shield-ring', () => new THREE.TorusGeometry(0.76, 0.012, 6, 48)),
            this._material(0xa6f4d8, {
                emissive: 0x76d1bf, emissiveIntensity: 0.45, transparent: true, opacity: 0.7,
            }), this.head, 'shield-charge-ring');
        this._shieldRing.rotation.x = Math.PI / 2;
        this._shieldRing.position.y = -0.20;
        this._shieldRing.castShadow = false;
        this._shieldRing.visible = false;
    }

    setState(entity, board, snap = false) {
        if (this._disposed) return;
        board = board ?? entity ?? {};
        entity = entity ?? board;
        const snake = entity.snake;
        if (!Array.isArray(snake) || snake.length === 0) {
            this.group.visible = false;
            return;
        }
        this.group.visible = true;
        const count = Math.min(MAX_SEGMENTS, snake.length);
        const half = ((board.gridSize ?? 22) - 1) / 2;
        const first = this._count === 0;
        const oldCount = this._count;
        for (let i = 0; i < count; i++) {
            this._targets[i].set(snake[i].x - half, snake[i].y - half, snake[i].z - half);
        }
        // Batched training samples are snapshots, not a trajectory to invent between.
        const discontinuity = !first && this._points[0].distanceToSquared(this._targets[0]) > 12.25;
        const reset = first || snap || discontinuity;
        for (let i = 0; i < count; i++) {
            if (reset) this._points[i].copy(this._targets[i]);
            else if (i >= oldCount) this._points[i].copy(this._points[Math.max(0, oldCount - 1)]);
        }
        this._count = count;
        const direction = entity.direction ?? board.direction;
        if (direction && (direction.x || direction.y || direction.z)) {
            this._vector.set(direction.x, direction.y, direction.z).normalize();
            this._targetQuaternion.setFromUnitVectors(FRONT, this._vector);
        }
        if (reset) {
            this.head.quaternion.copy(this._targetQuaternion);
            this._foodTime = 0;
            this._reliefTime = 0;
            this._eventIds.clear();
            this._eventQueue.length = 0;
        }
        const player = this.kind === 'player';
        const emotion = entity.emotion ?? (player ? board.emotion : null) ?? 'safe';
        const threat = entity.threat ?? (player ? board.threat : null);
        this._targetThreat = Number.isFinite(threat) ? clamp(threat, 0, 1) : 0;
        if (!Number.isFinite(threat)) {
            if (['fear', 'frightened', 'scared', 'terrified'].includes(emotion)) this._targetThreat = 0.9;
            else if (emotion === 'alert' || emotion === 'nervous') this._targetThreat = 0.45;
        }
        this._dead = Boolean(player ? (entity.done ?? board.done) && board.lastDeath !== 'time_limit' : (entity.done ?? entity.dead));
        this._alert = Boolean(entity.alert);
        if (reset) this._death = this._dead ? 1 : 0;
        if (!reset && emotion !== this._lastEmotion) {
            if (['happy', 'joy', 'food'].includes(emotion)) this._foodTime = 1.05;
            if (['relief', 'relieved', 'escape'].includes(emotion)) this._reliefTime = 1.35;
        }
        this._lastEmotion = emotion;
        const events = board.events ?? [];
        for (const event of events) {
            const key = String(event.id ?? `${board.seed}:${event.step ?? board.steps}:${event.type}`);
            if (this._eventIds.has(key)) continue;
            this._eventIds.add(key);
            this._eventQueue.push(key);
            if (this._eventQueue.length > 128) this._eventIds.delete(this._eventQueue.shift());
            const owner = event.entityId ?? event.actorId ?? event.entity;
            const ownEvent = owner == null ? player : (owner === entity.id || (player && owner === 'player'));
            if (reset || !ownEvent) continue;
            if (['food', 'gold', 'gold_food', 'food_collected'].includes(event.type)) this._foodTime = 1.05;
            if (['escape', 'enemy_escape'].includes(event.type)) this._reliefTime = 1.35;
        }
        const abilities = entity.abilities ?? (player ? board.abilities : null);
        this._dash = abilities?.dash?.active ? 1 : 0;
        this._shield = abilities?.shield?.active ? 1 : 0;
        let look = entity.lookTarget ?? (player ? board.lookTarget : null);
        if (!look && (this.kind === 'hunter' || this.kind === 'ambusher')) look = board.snake?.[0];
        if (!look && this.kind === 'rival') look = board.foods?.[0]?.position ?? board.food;
        if (!look && player) {
            let nearest = Infinity;
            for (const enemy of board.enemies ?? []) {
                const point = enemy.snake?.[0];
                if (!point) continue;
                const distance = Math.abs(point.x - snake[0].x) + Math.abs(point.y - snake[0].y) + Math.abs(point.z - snake[0].z);
                if (distance < nearest && distance <= 7) {
                    nearest = distance;
                    look = point;
                }
            }
            if (!Number.isFinite(threat) && nearest < Infinity) this._targetThreat = clamp((7 - nearest) / 6, 0, 1);
            look ??= board.foods?.[0]?.position ?? board.food;
        }
        if (look && Number.isFinite(look.x) && Number.isFinite(look.z)) {
            this._lookTarget.set(look.x - half, look.y - half, look.z - half);
        } else {
            this._lookTarget.copy(this._targets[0]);
            this._vector.set(0, 0, 3).applyQuaternion(this._targetQuaternion);
            this._lookTarget.add(this._vector);
        }
        if (reset) {
            this._threat = this._targetThreat;
            this.headPosition.copy(this._points[0]);
            this.head.position.copy(this.headPosition);
            this.forward.copy(FRONT).applyQuaternion(this.head.quaternion);
            this.update(0, this._elapsed);
        }
    }

    setRenderPose(points, { direction, lookTarget, snap = false } = {}) {
        if (this._disposed || this.headOnly || !points?.length) return;
        this._externalPose = true;
        this.group.visible = true;
        const count = Math.min(points.length, MAX_SEGMENTS);
        const old = this.headPosition;
        const first = points[0];
        this._movement = Math.min(1, Math.hypot(first.x - old.x, first.y - old.y, first.z - old.z) * 12);
        this._count = count;
        for (let i = 0; i < count; i++) {
            this._points[i].set(points[i].x, points[i].y, points[i].z);
            this._targets[i].copy(this._points[i]);
        }
        if (direction && (direction.x || direction.y || direction.z)) {
            this._vector.set(direction.x, direction.y, direction.z).normalize();
            this._targetQuaternion.setFromUnitVectors(FRONT, this._vector);
        }
        if (lookTarget) this._lookTarget.set(lookTarget.x, lookTarget.y, lookTarget.z);
        this.headPosition.copy(this._points[0]);
        this.head.position.copy(this.headPosition);
        if (snap) this.head.quaternion.copy(this._targetQuaternion);
    }

    setPresentation(presentation = null) {
        this._presentation = presentation;
        this._reducedMotion = Boolean(presentation?.reducedMotion);
    }

    setPortraitState({ emotion = 'safe', threat = 0, alert = false, dead = false, look = [0, 0.2, 3], reset = false } = {}) {
        if (!this.headOnly) throw new Error('Portrait state requires a head-only character.');
        this.group.visible = true;
        this._targetThreat = clamp(threat, 0, 1);
        this._alert = alert;
        this._dead = dead;
        if (emotion !== this._lastEmotion) {
            if (emotion === 'happy') this._foodTime = 1.05;
            if (emotion === 'relieved') this._reliefTime = 1.35;
        }
        this._lastEmotion = emotion;
        this._portraitLook.fromArray(look);
        if (reset) {
            this._threat = this._targetThreat;
            this._death = dead ? 1 : 0;
            this._foodTime = emotion === 'happy' ? 1.05 : 0;
            this._reliefTime = emotion === 'relieved' ? 1.35 : 0;
        }
    }

    update(dt, elapsed, { intro = false, introPhase = 0 } = {}) {
        if (this._disposed || !this.group.visible || (!this.headOnly && !this._count)) return;
        dt = Number.isFinite(dt) ? clamp(dt, 0, 0.1) : 0;
        this._elapsed = Number.isFinite(elapsed) ? elapsed : this._elapsed + dt;
        elapsed = this._elapsed;
        this._foodTime = Math.max(0, this._foodTime - dt);
        this._reliefTime = Math.max(0, this._reliefTime - dt);
        this._threat += (this._targetThreat - this._threat) * damp(9, dt);
        const deathTarget = this._presentation?.death ?? (this._dead ? 1 : 0);
        this._death = this._presentation ? deathTarget : this._death + (deathTarget - this._death) * damp(8, dt);
        if (!this.headOnly) {
            if (!this._externalPose) {
                const travel = Math.min(1, this._points[0].distanceTo(this._targets[0]) * 5);
                this._movement += (travel - this._movement) * damp(8, dt);
                const blend = damp(this._dash ? 25 : 19, dt);
                for (let i = 0; i < this._count; i++) this._points[i].lerp(this._targets[i], blend);
            }
            this.headPosition.copy(this._points[0]);
            this.head.position.copy(this.headPosition);
            this.head.quaternion.slerp(this._targetQuaternion, damp(17, dt));
            this.forward.copy(FRONT).applyQuaternion(this.head.quaternion);
        }

        const death = this._death;
        const joy = smooth(0, 0.24, this._foodTime) * Math.min(1, this._foodTime / 0.65) * (1 - death);
        const relief = smooth(0, 0.28, this._reliefTime) * Math.min(1, this._reliefTime / 0.95) * (1 - death);
        const fear = this.kind === 'player'
            ? clamp(Math.max(intro ? 0.83 : 0, this._threat) * (1 - joy * 0.55 - relief * 0.35), 0, 1) * (1 - death)
            : 0;
        const anger = this.kind === 'hunter' || this.kind === 'ambusher' ? (this._alert ? 1 : 0.72) : 0;
        const motion = this._reducedMotion ? 0 : 1 - death;
        const attack = this._attackJaw ? clamp(this._presentation?.attack ?? 0, 0, 1) : 0;
        const impact = clamp(this._presentation?.impact ?? 0, 0, 1);
        const hitStop = this._presentation?.hitStop ? 1 : 0;
        this._visual.position.set(
            (Math.sin(elapsed * 29) + Math.sin(elapsed * 43) * 0.4) * fear * 0.006 * (1 - death) * motion,
            (Math.sin(elapsed * 2.8) * 0.008 + joy * Math.sin(elapsed * 12) * 0.035) * motion - death * 0.115,
            0);
        this._visual.rotation.set(
            fear * 0.035 + death * 0.48 - attack * 0.12,
            Math.sin(elapsed * (intro ? 2.0 : 7.0) + introPhase * 0.7) * fear * (intro ? 0.045 : 0.012) * motion,
            death * 1.4 + joy * Math.sin(elapsed * 10) * 0.045 * motion + impact * 0.3 * motion);
        this._visual.scale.copy(this._restVisualScale);
        this._visual.scale.y *= 1 - death * 0.08;
        this._skinMaterial.color.copy(this._baseSkin).lerp(this._deadTone, death * 0.94).lerp(this._hitTone, hitStop * 0.6);
        this._muzzleMaterial.color.copy(this._baseMuzzle).lerp(this._deadTone, death * 0.94);
        this._bellyMaterial.color.copy(this._baseBelly).lerp(this._deadTone, death * 0.85);
        this._detailMaterial.color.copy(this._baseDetail).lerp(this._deadTone, death * 0.9);
        this._accentMaterial.color.copy(this._baseAccent).lerp(this._deadTone, death * 0.92);
        this._bodyMaterial.color.setScalar(1 - death * 0.12);
        this._bodyMaterial.emissive.setHex(hitStop ? 0xe7dcb2 : this._dash && death < 0.1 ? this.palette.body : 0x000000);
        this._bodyMaterial.emissiveIntensity = hitStop ? 0.3 : this._dash && death < 0.1 ? 0.12 : 0;
        if (!this.headOnly) this._updateBody(elapsed);

        if (this.kind === 'patrol') {
            this._updatePatrol(elapsed, death);
        } else {
            if (this.headOnly) this._localLook.copy(this._portraitLook);
            else {
                this._inverseQuaternion.copy(this.head.quaternion).invert();
                this._localLook.copy(this._lookTarget).sub(this.headPosition).applyQuaternion(this._inverseQuaternion);
            }
            let gazeX = clamp(this._localLook.x / (Math.abs(this._localLook.z) + 1.0), -0.68, 0.68);
            let gazeY = clamp((this._localLook.y - 0.20) / (Math.abs(this._localLook.z) + 1.0), -0.35, 0.35);
            if (intro) {
                gazeX = Math.sin(elapsed * 2.15 + introPhase * 0.8) * 0.64;
                gazeY = 0.03 + Math.sin(elapsed * 1.35) * 0.11;
            } else {
                gazeX += fear * Math.sin(elapsed * 16.0) * 0.025;
            }
            this._gazeX += (gazeX - this._gazeX) * damp(12, dt);
            this._gazeY += (gazeY - this._gazeY) * damp(12, dt);
            const blinkClock = (elapsed + (this.kind === 'player' ? 0 : 1.7)) % 5.3;
            const blink = Math.max(0, 1 - Math.abs(blinkClock - 4.8) / 0.105) * (1 - fear * 0.6) * (1 - death);
            const open = (0.95 + fear * 0.17 - joy * 0.22 - relief * 0.10 - anger * 0.23)
                * (1 - blink * 0.90) * (1 - death * 0.25);
            this._updateEyes(open, fear, joy, relief, anger, death);
            this._updateMouth(fear, joy, relief, anger, death);
            this._jaw.position.y = -0.245 - death * 0.12 - attack * 0.23 - joy * Math.abs(Math.sin(elapsed * 18)) * 0.025;
            this._jaw.rotation.x = 0;
            if (this._attackJaw) {
                this._attackJaw.visible = attack > 0.001;
                this._attackJaw.rotation.x = attack * 1.05;
                this._jaw.visible = !this._attackJaw.visible;
                this._upperMuzzle.scale.y = 1 - attack * 0.5;
                this._upperMuzzle.position.y = attack * 0.12;
                this._quaternion.setFromAxisAngle(FRONT, Math.PI);
                for (let i = 0; i < 2; i++) this._put(this._fangs, i, (i ? 1 : -1) * 0.255,
                    -0.224 + attack * 0.1, 0.635, 0.058, 0.235 + attack * 0.13, 0.071, this._quaternion);
                this._fangs.instanceMatrix.needsUpdate = true;
            }
            this._mouthCavity.visible = attack > 0.08;
            this._mouthCavity.position.y = -0.15 - attack * 0.08;
            this._mouthCavity.scale.set(0.32, Math.max(0.015, attack * 0.3), 0.29);
            this._mouth.visible = attack < 0.35;
            const flickClock = elapsed % (intro ? 3.2 : 5.8);
            const flickStart = intro ? 2.85 : 5.45;
            const flick = flickClock > flickStart && motion ? Math.sin((flickClock - flickStart) / 0.35 * Math.PI) : 0;
            const tongue = death > 0.25 ? death * 0.95
                : Math.max(flick * 0.85, joy * (0.5 + 0.5 * Math.sin(elapsed * 23)) * 0.75);
            this._tongue.visible = tongue > 0.02;
            this._tongue.scale.z = Math.max(0.01, tongue);
            this._tongue.rotation.x = death * 0.7;
        }
        if (!this.headOnly) {
            const blocked = this._presentation?.shieldBlock ?? 0;
            this._shieldBubble.visible = (this._shield > 0 || blocked > 0) && death < 0.1;
            this._shieldRing.visible = this._shieldBubble.visible;
            if (this._shieldBubble.visible) {
                this._shieldBubble.material.opacity = 0.065 + Math.sin(elapsed * 7) * 0.015 + blocked * 0.24;
                this._shieldRing.rotation.z = elapsed * 0.3;
            }
        }
    }

    _samplePath(span, t, out) {
        const last = this._count - 1;
        if (last === 2) {
            const u = (span + t) * 0.5;
            const v = 1 - u;
            out.copy(this._points[0]).multiplyScalar(v * v)
                .addScaledVector(this._points[1], 2 * v * u)
                .addScaledVector(this._points[2], u * u);
            return;
        }
        const p0 = this._points[Math.max(0, span - 1)];
        const p1 = this._points[span];
        const p2 = this._points[Math.min(last, span + 1)];
        const p3 = this._points[Math.min(last, span + 2)];
        const t2 = t * t;
        const t3 = t2 * t;
        const a = 2 * t3 - 3 * t2 + 1;
        const b = (t3 - 2 * t2 + t) * 0.42;
        const c = -2 * t3 + 3 * t2;
        const d = (t3 - t2) * 0.42;
        out.set(
            a * p1.x + b * (p2.x - p0.x) + c * p2.x + d * (p3.x - p1.x),
            a * p1.y + b * (p2.y - p0.y) + c * p2.y + d * (p3.y - p1.y),
            a * p1.z + b * (p2.z - p0.z) + c * p2.z + d * (p3.z - p1.z));
        out.x = clamp(out.x, Math.min(p1.x, p2.x) - 0.065, Math.max(p1.x, p2.x) + 0.065);
        out.y = clamp(out.y, Math.min(p1.y, p2.y) - 0.065, Math.max(p1.y, p2.y) + 0.065);
        out.z = clamp(out.z, Math.min(p1.z, p2.z) - 0.065, Math.max(p1.z, p2.z) + 0.065);
    }

    _radius(t) {
        return this.palette.radius * (1 - t * 0.16) * (1 - smooth(0.55, 1, t));
    }

    _updateBody(elapsed) {
        const spanSamples = Math.max(1, this._count - 1) * SUBDIVISIONS;
        const rings = spanSamples + 3;
        if (this._count > 1) {
            for (let i = 0; i <= spanSamples; i++) {
                const value = i / SUBDIVISIONS;
                this._samplePath(Math.min(this._count - 2, Math.floor(value)),
                    i === spanSamples ? 1 : value % 1, this._samples[i]);
            }
            this._tangent.copy(this._points[this._count - 1]).sub(this._points[this._count - 2]);
            if (this._tangent.lengthSq() < 0.00001) this._tangent.copy(this.forward).negate();
            this._tangent.normalize();
            this._samples[spanSamples + 1].copy(this._points[this._count - 1]).addScaledVector(this._tangent, 0.19);
            this._samples[spanSamples + 2].copy(this._points[this._count - 1]).addScaledVector(this._tangent, 0.34);
        } else {
            for (let i = 0; i < rings; i++) {
                this._samples[i].copy(this._points[0]).addScaledVector(this.forward, -0.08 - i * 0.115);
            }
        }
        const positions = this._bodyGeometry.attributes.position.array;
        const normals = this._bodyGeometry.attributes.normal.array;
        const colors = this._bodyGeometry.attributes.color.array;
        const flash = this._presentation?.hitStop ? 1 : this._presentation?.impact ?? 0;
        const recolor = rings !== this._ringCount || Math.abs(this._death - this._lastBodyDeath) > 0.02
            || Math.abs(flash - this._lastBodyFlash) > 0.04;
        if (recolor) { this._lastBodyDeath = this._death; this._lastBodyFlash = flash; }
        this._ringCount = rings;
        for (let i = 0; i < rings; i++) {
            const t = i / (rings - 1);
            const radius = this._radius(t);
            const center = this._samples[i];
            this._tangent.copy(this._samples[Math.min(rings - 1, i + 1)])
                .sub(this._samples[Math.max(0, i - 1)]);
            if (this._tangent.lengthSq() < 0.000001) this._tangent.copy(this.forward).negate();
            this._tangent.normalize();
            this._binormal.crossVectors(this._tangent, UP).normalize();
            if (this._binormal.lengthSq() < 0.1) this._binormal.set(1, 0, 0);
            this._normal.crossVectors(this._binormal, this._tangent).normalize();
            const previous = Math.max(0, i - 1);
            const next = Math.min(rings - 1, i + 1);
            const distance = this._samples[previous].distanceTo(this._samples[next]);
            const radiusSlope = clamp((this._radius(next / (rings - 1)) - this._radius(previous / (rings - 1)))
                / Math.max(0.001, distance), -1.5, 1.5);
            const breath = this._reducedMotion ? 0 : Math.sin(elapsed * 2.8 - t * 2.4) * 0.004 * (1 - t) * (1 - this._death);
            const flatness = 0.94 * (1 - this._death * 0.33);
            for (let side = 0; side <= RADIAL_SIDES; side++) {
                const angle = side / RADIAL_SIDES * TAU;
                const sine = Math.sin(angle);
                const cosine = Math.cos(angle);
                const index = (i * (RADIAL_SIDES + 1) + side) * 3;
                positions[index] = center.x + this._binormal.x * sine * radius + this._normal.x * cosine * radius * flatness;
                positions[index + 1] = center.y + breath + this._binormal.y * sine * radius + this._normal.y * cosine * radius * flatness;
                positions[index + 2] = center.z + this._binormal.z * sine * radius + this._normal.z * cosine * radius * flatness;
                this._vector.copy(this._binormal).multiplyScalar(sine)
                    .addScaledVector(this._normal, cosine / flatness)
                    .addScaledVector(this._tangent, -radiusSlope).normalize();
                this._vector.toArray(normals, index);
                if (recolor) {
                    const tone = 0.97 + cosine * 0.045 + Math.cos(i * 0.58) * 0.025;
                    this._color.copy(this._bodyColor).multiplyScalar(tone)
                        .lerp(this._bellyColor, smooth(-0.10, -0.69, cosine))
                        .lerp(this._deadTone, this._death * 0.94);
                    const hit = this._presentation?.hitSegment ?? -20;
                    const band = Math.exp(-(((i / SUBDIVISIONS - hit) / 0.65) ** 2));
                    this._color.lerp(this._hitTone, band * flash * 0.8 * (1 - this._death));
                    this._color.toArray(colors, index);
                }
            }
        }
        this._bodyGeometry.setDrawRange(0, (rings - 1) * RADIAL_SIDES * 6);
        this._bodyGeometry.attributes.position.needsUpdate = true;
        this._bodyGeometry.attributes.normal.needsUpdate = true;
        if (recolor) this._bodyGeometry.attributes.color.needsUpdate = true;

        const details = Math.max(0, this._count - 2);
        this._bodyDetails.count = details;
        for (let i = 0; i < details; i++) {
            const sample = (i + 1) * SUBDIVISIONS;
            const point = this._samples[sample];
            const radius = this._radius(sample / (rings - 1));
            const taper = radius / this.palette.radius;
            this._tangent.copy(this._samples[Math.max(0, sample - 1)])
                .sub(this._samples[Math.min(rings - 1, sample + 1)]);
            if (this._tangent.lengthSq() < 0.00001) this._tangent.copy(this.forward);
            this._quaternion.setFromUnitVectors(FRONT, this._tangent.normalize());
            if (this.kind === 'patrol') {
                this._put(this._bodyDetails, i, point.x, point.y + 0.015, point.z,
                    taper * 0.67, taper * 0.62, 0.72, this._quaternion);
            } else if (this.kind === 'hunter' || this.kind === 'ambusher') {
                this._put(this._bodyDetails, i, point.x, point.y + radius * 0.94 + 0.055 * taper, point.z,
                    0.085 * taper, 0.14 * taper, 0.11 * taper, this._quaternion);
            } else if (this.kind === 'rival') {
                this._put(this._bodyDetails, i, point.x, point.y + radius * 0.93, point.z,
                    taper, taper * 0.6, 0.95, this._quaternion);
            } else {
                this._put(this._bodyDetails, i, point.x, point.y + radius * 0.94 + 0.005, point.z,
                    taper * 0.30, taper, 0.43, this._quaternion);
            }
        }
        this._bodyDetails.instanceMatrix.needsUpdate = true;
    }

    _updateEyes(open, fear, joy, relief, anger, death) {
        for (let i = 0; i < 2; i++) {
            const side = i ? 1 : -1;
            const root = this._eyeRoots[i];
            root.scale.y = Math.max(0.08, open * (this.kind === 'rival' ? (i ? 0.76 : 0.94) : 1));
            root.updateMatrix();
            this._put(this._whites, i, 0, 0, 0, 0.253, 0.305, 0.222, IDENTITY, root.matrix);
            const px = clamp(this._gazeX, -0.7, 0.7) * 0.19;
            const py = clamp(this._gazeY, -0.4, 0.4) * 0.16;
            const pz = 0.222 * Math.sqrt(Math.max(0.1, 1 - (px / 0.253) ** 2 - (py / 0.305) ** 2));
            this._vector.set(px / 0.253, py / 0.305, pz / 0.222).normalize();
            this._surfaceQuaternion.setFromUnitVectors(FRONT, this._vector);
            this._surfaceMatrix.makeRotationFromQuaternion(this._surfaceQuaternion);
            this._eyeMatrix.multiplyMatrices(root.matrix, this._ellipsoidMatrix).multiply(this._surfaceMatrix);
            // Curved patches share the white's ellipsoid, including at the limits of gaze.
            this._put(this._irises, i, 0, 0, 0, 1.009, 1.009, 1.009, IDENTITY, this._eyeMatrix);
            this._put(this._pupils, i, 0, 0, 0, 1.021, 1.021, 1.021, IDENTITY, this._eyeMatrix);
            this._put(this._highlights, i * 2, -0.10, 0.13, 1.025, 0.080, 0.083, 0.035, IDENTITY, this._eyeMatrix);
            this._put(this._highlights, i * 2 + 1, 0.105, -0.08, 1.030, 0.035, 0.036, 0.026, IDENTITY, this._eyeMatrix);
            const browAngle = -side * (fear * 0.30 + joy * 0.09 + relief * 0.04 - anger * 0.38 - death * 0.12)
                + (this.kind === 'rival' ? (i ? -0.22 : 0.06) : 0);
            this._quaternion.setFromAxisAngle(FRONT, browAngle);
            this._put(this._brows, i, side * 0.285,
                0.538 + fear * 0.035 - anger * 0.06 + death * 0.015, 0.472,
                1, 1 + anger * 0.22, 1, this._quaternion);
            for (let stroke = 0; stroke < 2; stroke++) {
                this._quaternion.setFromAxisAngle(FRONT, stroke ? Math.PI / 4 : -Math.PI / 4);
                this._put(this._crosses, i * 2 + stroke, 0, 0, 0.244,
                    0.025, 0.26, 0.025, this._quaternion, root.matrix);
            }
        }
        this._irises.visible = death < 0.65;
        this._pupils.visible = death < 0.65;
        this._highlights.visible = death < 0.65;
        this._crosses.visible = death >= 0.65;
        for (const mesh of this._eyeMeshes) {
            mesh.instanceMatrix.needsUpdate = true;
        }
    }

    _updateMouth(fear, joy, relief, anger, death) {
        const curve = joy * 0.085 + relief * 0.052 - fear * 0.041 - anger * 0.045 - death * 0.025
            + (this.kind === 'rival' ? 0.035 : 0);
        const width = 0.36 + joy * 0.055 - fear * 0.035;
        for (let i = 0; i < 21; i++) {
            const t = i / 20 * 2 - 1;
            const x = t * width * 0.5;
            const y = -0.175 + curve * (t * t - 0.45) + (this.kind === 'rival' ? t * 0.055 : 0);
            this._mouthCenters[i].set(x, y, faceSurfaceZ(x, y));
        }
        const positions = this._mouthGeometry.attributes.position.array;
        const normals = this._mouthGeometry.attributes.normal.array;
        for (let i = 0; i < 21; i++) {
            const center = this._mouthCenters[i];
            this._tangent.copy(this._mouthCenters[Math.min(20, i + 1)])
                .sub(this._mouthCenters[Math.max(0, i - 1)]).normalize();
            this._binormal.crossVectors(this._tangent, UP).normalize();
            this._normal.crossVectors(this._binormal, this._tangent).normalize();
            const radius = 0.0125 * (0.78 + Math.sin(i / 20 * Math.PI) * 0.22);
            for (let side = 0; side <= 6; side++) {
                const angle = side / 6 * TAU;
                this._vector.copy(this._normal).multiplyScalar(Math.cos(angle))
                    .addScaledVector(this._binormal, Math.sin(angle));
                const index = (i * 7 + side) * 3;
                this._vector.toArray(normals, index);
                positions[index] = center.x + this._vector.x * radius;
                positions[index + 1] = center.y + this._vector.y * radius;
                positions[index + 2] = center.z + this._vector.z * radius;
            }
        }
        this._mouthGeometry.attributes.position.needsUpdate = true;
        this._mouthGeometry.attributes.normal.needsUpdate = true;
    }

    _rod(mesh, index, ax, ay, az, bx, by, bz, radius) {
        this._vector.set(bx - ax, by - ay, bz - az);
        const length = this._vector.length();
        this._vector.normalize();
        this._quaternion.setFromUnitVectors(UP, this._vector);
        this._put(mesh, index, (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2,
            radius, length, radius, this._quaternion);
    }

    _updatePatrol(elapsed, death) {
        for (let i = 0; i < 6; i++) {
            const side = i < 3 ? -1 : 1;
            const row = i % 3;
            const z = (row - 1) * 0.285;
            const gait = Math.sin(elapsed * 10 + row * 2.2 + (side < 0 ? Math.PI : 0)) * this._movement;
            const kneeY = -0.21 + gait * 0.055 + death * 0.09;
            const footY = -0.45 + Math.max(0, gait) * 0.075 + death * 0.16;
            const footZ = z + gait * 0.09;
            this._rod(this._legs, i * 2, side * 0.43, -0.06, z,
                side * 0.66, kneeY, z + 0.045, 0.041);
            this._rod(this._legs, i * 2 + 1, side * 0.66, kneeY, z + 0.045,
                side * 0.73, footY, footZ, 0.032);
            this._put(this._feet, i, side * 0.73, footY, footZ, 0.075, 0.033, 0.10);
        }
        this._legs.instanceMatrix.needsUpdate = true;
        this._feet.instanceMatrix.needsUpdate = true;
        this._opticMaterial.emissiveIntensity = (1 - death) * (this._alert ? 0.65 : 0.36);
    }

    dispose() {
        if (this._disposed) return;
        this._disposed = true;
        this.group.removeFromParent();
        this.group.clear();
        for (const mesh of this._instances) mesh.dispose();
        for (const geometry of this._ownedGeometries) geometry.dispose();
        for (const material of this._materials) material.dispose();
        this._eventIds.clear();
        this._eventQueue.length = 0;
        resourceUsers--;
        if (resourceUsers === 0) {
            for (const geometry of geometryPool.values()) geometry.dispose();
            geometryPool.clear();
        }
    }
}

function blockBodyBuffer(template) {
    const vertices = template.attributes.position.count;
    const indices = template.index.count;
    const geometry = new THREE.BufferGeometry();
    const normals = new Float32Array(MAX_SEGMENTS * vertices * 3);
    const index = new Uint16Array(MAX_SEGMENTS * indices);
    for (let cell = 0; cell < MAX_SEGMENTS; cell++) {
        normals.set(template.attributes.normal.array, cell * vertices * 3);
        for (let i = 0; i < indices; i++) index[cell * indices + i] = cell * vertices + template.index.array[i];
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(normals.length), 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(normals.length), 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setDrawRange(0, 0);
    return geometry;
}

export class BlockSnakeCharacter {
    constructor(kind = 'player') {
        this.kind = kind === 'ambush' ? 'ambusher' : (PALETTES[kind] ? kind : 'player');
        this.palette = PALETTES[this.kind];
        this.group = new THREE.Group();
        this.group.name = `block-character:${this.kind}`;
        this.group.userData.kind = this.kind;
        this.group.userData.style = 'grid-blocks';
        this.group.visible = false;
        this.head = new THREE.Group();
        this.group.add(this.head);
        this.headPosition = new THREE.Vector3();
        this.forward = FRONT.clone();
        this._direction = FRONT.clone();
        this._lookTarget = new THREE.Vector3();
        this._localLook = new THREE.Vector3();
        this._inverseQuaternion = new THREE.Quaternion();
        this._points = Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector3());
        this._count = 0;
        this._dead = false;
        this._shield = false;
        this._dash = false;
        this._threat = 0;
        this._presentation = null;
        this._disposed = false;
        this._materials = new Set();
        this._predator = this.kind === 'hunter' || this.kind === 'ambusher';
        this._headColor = new THREE.Color(this.palette.skin).multiplyScalar(0.68);
        this._bodyColor = new THREE.Color(this.palette.body);
        this._deadTone = new THREE.Color(0x929fa4);
        this._hitTone = new THREE.Color(0xffefd0);
        this._shade = new THREE.Color();
        resourceUsers++;
        this._unitBox = sharedGeometry('block-unit', () => new THREE.BoxGeometry(1, 1, 1));
        this._cellBox = sharedGeometry('block-cell', () => new THREE.BoxGeometry(0.9, 0.9, 0.9));
        this._outlineMaterial = new THREE.LineBasicMaterial({ color: 0x020605 });
        this._materials.add(this._outlineMaterial);
        this._edgeTemplate = sharedGeometry('block-cell-edges', () => new THREE.EdgesGeometry(this._cellBox));
        this._edgeGeometry = new THREE.BufferGeometry();
        this._edgeGeometry.setAttribute('position', new THREE.BufferAttribute(
            new Float32Array(MAX_SEGMENTS * this._edgeTemplate.attributes.position.array.length), 3).setUsage(THREE.DynamicDrawUsage));
        this._edgeGeometry.setDrawRange(0, 0);
        this._edges = new THREE.LineSegments(this._edgeGeometry, this._outlineMaterial);
        this._edges.frustumCulled = false;
        this._edges.name = 'individual-black-cell-boundaries';
        this.group.add(this._edges);
        this.group.userData.outlined = true;
        this._skinMaterial = this._material(this.palette.skin);
        this._bodyMaterial = this._material(0xffffff, { vertexColors: true });
        this._bodyGeometry = blockBodyBuffer(this._cellBox);
        this._body = new THREE.Mesh(this._bodyGeometry, this._bodyMaterial);
        this._body.name = 'cell-aligned-block-body';
        this._body.frustumCulled = false;
        this.group.add(this._body);
        this._box(this.head, this._skinMaterial, [0, 0.15, 0], [0.9, 0.6, 0.9], 'block-head');
        this._jawPivot = new THREE.Group();
        this._jawPivot.position.set(0, -0.15, -0.3);
        this.head.add(this._jawPivot);
        this._box(this._jawPivot, this._skinMaterial, [0, -0.15, 0.3], [0.9, 0.3, 0.9], 'block-jaw');
        const eye = this._material(this.palette.eye);
        const dark = this._material(0x10231f);
        const brow = this._material(this.palette.brow);
        this._pupils = [];
        this._brows = [];
        this._crosses = new THREE.Group();
        this._crosses.visible = false;
        this.head.add(this._crosses);
        for (const side of [-1, 1]) {
            const height = this.kind === 'rival' && side > 0 ? 0.15 : this._predator ? 0.17 : 0.21;
            this._box(this.head, eye, [side * 0.19, 0.15, 0.46], [0.21, height, 0.018], 'square-eye');
            this._pupils.push(this._box(this.head, dark, [side * 0.19, 0.15, 0.478],
                [0.073, 0.095, 0.012], 'square-pupil'));
            this._brows.push(this._box(this.head, brow, [side * 0.19, 0.30, 0.47],
                [0.24, 0.036, 0.014], 'block-brow'));
            for (const sign of [-1, 1]) {
                const bar = this._box(this._crosses, dark, [side * 0.19, 0.15, 0.481],
                    [0.024, 0.165, 0.012], 'dead-eye-cross');
                bar.rotation.z = sign * Math.PI / 4;
            }
            if (this._predator) this._box(this.head, eye, [side * 0.24, -0.20, 0.468],
                [0.05, 0.13, 0.033], 'block-fang');
        }
        this._mouth = this._box(this.head, dark, [0, -0.15, 0.466], [0.32, 0.024, 0.018], 'block-mouth');
        if (this.kind === 'player') this._box(this.head, this._material(this.palette.accent),
            [0, 0.37, 0.462], [0.14, 0.04, 0.014], 'player-mark');
        const shieldMaterial = new THREE.MeshBasicMaterial({ color: 0x94e8d5, transparent: true, opacity: 0.08, depthWrite: false });
        this._materials.add(shieldMaterial);
        this._shieldBubble = new THREE.Mesh(sharedGeometry('block-shield', () => new THREE.SphereGeometry(0.68, 18, 12)), shieldMaterial);
        this._shieldBubble.visible = false;
        this.head.add(this._shieldBubble);
    }

    _material(color, options = {}) {
        const material = new THREE.MeshStandardMaterial({
            color, roughness: 0.18, metalness: 0.18,
            polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, ...options,
        });
        this._materials.add(material);
        return material;
    }

    _box(parent, material, position, size, name) {
        const mesh = new THREE.Mesh(this._unitBox, material);
        mesh.name = name;
        mesh.position.fromArray(position);
        mesh.scale.fromArray(size);
        if (material === this._skinMaterial) {
            mesh.add(new THREE.LineSegments(sharedGeometry('block-unit-edges', () => new THREE.EdgesGeometry(this._unitBox)),
                this._outlineMaterial));
        }
        parent.add(mesh);
        return mesh;
    }

    setState(entity, board) {
        if (this._disposed) return;
        if (!entity.snake?.length) {
            this.group.visible = false;
            this._count = 0;
            this._bodyGeometry.setDrawRange(0, 0);
            this._edgeGeometry.setDrawRange(0, 0);
            return;
        }
        const half = (board.gridSize - 1) / 2;
        this._count = Math.min(MAX_SEGMENTS, entity.snake.length);
        for (let i = 0; i < this._count; i++) {
            const point = entity.snake[i];
            this._points[i].set(point.x - half, point.y - half, point.z - half);
        }
        this.group.visible = true;
        const player = this.kind === 'player';
        this._dead = Boolean(player ? board.done && board.lastDeath !== 'time_limit' : entity.dead || entity.done);
        this._threat = clamp(entity.threat ?? (player ? board.threat : 0) ?? 0, 0, 1);
        const abilities = entity.abilities ?? (player ? board.abilities : null);
        this._shield = Boolean(abilities?.shield?.active);
        this._dash = Boolean(abilities?.dash?.active);
        this._setDirection(entity.direction ?? board.direction);
        this.headPosition.copy(this._points[0]);
        this.head.position.copy(this.headPosition);
    }

    _setDirection(direction) {
        if (direction && (direction.x || direction.y || direction.z)) {
            this._direction.set(direction.x, direction.y, direction.z).normalize();
        }
    }

    setRenderPose(points, { direction, lookTarget } = {}) {
        if (this._disposed || !points.length) return;
        this._count = Math.min(MAX_SEGMENTS, points.length);
        for (let i = 0; i < this._count; i++) this._points[i].set(points[i].x, points[i].y, points[i].z);
        this._setDirection(direction);
        if (lookTarget) this._lookTarget.set(lookTarget.x, lookTarget.y, lookTarget.z);
        this.headPosition.copy(this._points[0]);
        this.head.position.copy(this.headPosition);
    }

    setPresentation(presentation = null) { this._presentation = presentation; }

    update() {
        if (this._disposed || !this.group.visible || !this._count) return;
        const death = clamp(this._presentation?.death ?? (this._dead ? 1 : 0), 0, 1);
        const attack = this._predator ? clamp(this._presentation?.attack ?? 0, 0, 1) : 0;
        const flash = this._presentation?.hitStop ? 1 : this._presentation?.impact ?? 0;
        const hit = this._presentation?.hitSegment ?? -1;
        this.head.position.copy(this._points[0]);
        this.headPosition.copy(this._points[0]);
        this.head.quaternion.setFromUnitVectors(FRONT, this._direction);
        this.forward.copy(this._direction);
        this._skinMaterial.color.copy(this._headColor).lerp(this._deadTone, death * 0.96)
            .lerp(this._hitTone, hit === 0 ? flash * 0.8 * (1 - death) : 0);
        this._skinMaterial.emissive.setHex(this.kind === 'player' ? this.palette.skin : 0x000000);
        this._skinMaterial.emissiveIntensity = death < 0.1 ? this.kind === 'player' ? 0.16 : 0 : 0;
        this._bodyMaterial.emissive.setHex(this.palette.body);
        this._bodyMaterial.emissiveIntensity = death < 0.1 ? this._dash ? 0.7 : this.kind === 'player' ? 0.4 : 0 : 0;
        const source = this._cellBox.attributes.position.array;
        const positions = this._bodyGeometry.attributes.position.array;
        const colors = this._bodyGeometry.attributes.color.array;
        const edgeSource = this._edgeTemplate.attributes.position.array;
        const edgePositions = this._edgeGeometry.attributes.position.array;
        for (let cell = 1; cell < this._count; cell++) {
            const center = this._points[cell];
            const offset = (cell - 1) * source.length;
            this._shade.copy(this._bodyColor).lerp(this._deadTone, death * 0.96)
                .lerp(this._hitTone, cell === hit ? flash * 0.8 * (1 - death) : 0);
            for (let i = 0; i < source.length; i += 3) {
                positions[offset + i] = center.x + source[i];
                positions[offset + i + 1] = center.y + source[i + 1];
                positions[offset + i + 2] = center.z + source[i + 2];
                this._shade.toArray(colors, offset + i);
            }
            const edgeOffset = (cell - 1) * edgeSource.length;
            for (let i = 0; i < edgeSource.length; i += 3) {
                edgePositions[edgeOffset + i] = center.x + edgeSource[i];
                edgePositions[edgeOffset + i + 1] = center.y + edgeSource[i + 1];
                edgePositions[edgeOffset + i + 2] = center.z + edgeSource[i + 2];
            }
        }
        this._bodyGeometry.setDrawRange(0, Math.max(0, this._count - 1) * this._cellBox.index.count);
        this._bodyGeometry.attributes.position.needsUpdate = true;
        this._bodyGeometry.attributes.color.needsUpdate = true;
        this._edgeGeometry.setDrawRange(0, Math.max(0, this._count - 1) * this._edgeTemplate.attributes.position.count);
        this._edgeGeometry.attributes.position.needsUpdate = true;
        this._jawPivot.rotation.x = attack * 0.72;
        this._mouth.scale.y = 0.024 + attack * 0.23 + this._threat * 0.035 * (1 - death);
        this._crosses.visible = death > 0.45;
        this._inverseQuaternion.copy(this.head.quaternion).invert();
        this._localLook.copy(this._lookTarget).sub(this.headPosition).applyQuaternion(this._inverseQuaternion);
        const divisor = Math.abs(this._localLook.z) + 1;
        for (let i = 0; i < this._pupils.length; i++) {
            const side = i ? 1 : -1;
            this._pupils[i].visible = !this._crosses.visible;
            this._pupils[i].position.x = side * 0.19 + clamp(this._localLook.x / divisor, -0.04, 0.04);
            this._pupils[i].position.y = 0.15 + clamp(this._localLook.y / divisor, -0.04, 0.04);
            this._brows[i].rotation.z = this._predator ? side * 0.28
                : this.kind === 'rival' ? i ? -0.22 : -0.08 : -side * this._threat * 0.12 * (1 - death);
        }
        const blocked = this._presentation?.shieldBlock ?? 0;
        this._shieldBubble.visible = (this._shield || blocked > 0) && death < 0.1;
        this._shieldBubble.material.opacity = 0.08 + blocked * 0.24;
    }

    dispose() {
        if (this._disposed) return;
        this._disposed = true;
        this.group.removeFromParent();
        this.group.clear();
        this._bodyGeometry.dispose();
        this._edgeGeometry.dispose();
        for (const material of this._materials) material.dispose();
        resourceUsers--;
        if (resourceUsers === 0) {
            for (const geometry of geometryPool.values()) geometry.dispose();
            geometryPool.clear();
        }
    }
}
