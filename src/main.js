import * as THREE from 'three';

const CORRIDOR_HALF_WIDTH = 2.15;
const CORRIDOR_HEIGHT = 3.45;
/** Long run so slow camera drift stays inside geometry */
const CORRIDOR_LENGTH = 720;

/** Camera height as fraction of tunnel height — centered-ish */
const EYE_HEIGHT = CORRIDOR_HEIGHT * 0.45;
const CAMERA_START_Z = 3.2;
/** Mouse Y: top of viewport (low clientY) = fast forward, bottom = slow/stop */
const FWD_SPEED_AT_TOP = 2.0;
const FWD_SPEED_AT_BOTTOM = 0;

/** World-space grid density (cells per unit) — tuned for Tetris-ish scale */
const GRID_SCALE = 1.8;

const scene = new THREE.Scene();
const BACKGROUND = new THREE.Color(0x010104);
scene.background = BACKGROUND.clone();
scene.fog = null;

const clock = new THREE.Clock();

const camera = new THREE.PerspectiveCamera(
  62,
  window.innerWidth / window.innerHeight,
  0.05,
  900
);
camera.position.set(0, EYE_HEIGHT, CAMERA_START_Z);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
document.body.appendChild(renderer.domElement);

const lookDist = 28;

/** Neutral gray grid — slightly subdued so wires feel faint but legible */
const GRID_LINE_GRAY = new THREE.Color(0xaaaeb9);
const GRID_SURFACE_FILL = new THREE.Color(0x1a1f29);
const GRID_SURFACE_TINT = new THREE.Color(0x262c38);

function shaderUniforms(face, lineCol, surfaceFill, surfaceTint) {
  return {
    uFace: { value: face },
    uAccent: { value: new THREE.Vector3(surfaceTint.r, surfaceTint.g, surfaceTint.b) },
    uFillTint: { value: new THREE.Vector3(surfaceFill.r, surfaceFill.g, surfaceFill.b) },
    uLineColor: { value: new THREE.Vector3(lineCol.r, lineCol.g, lineCol.b) },
    /** How strongly grid lines tint over base (still gray, readable) */
    uLineStrength: { value: 0.7 },
    /** Extra additive lift only on wires so faint screens still resolve */
    uLineBoost: { value: 0.12 },
    uGridScale: { value: GRID_SCALE },
    uFogColor: { value: new THREE.Vector3(BACKGROUND.r, BACKGROUND.g, BACKGROUND.b) },
    uFogDensity: { value: 0.023 },
    uCameraWorldPos: { value: new THREE.Vector3() },
  };
}

function makeGridMaterial(faceIndex) {
  const lineCol = GRID_LINE_GRAY.clone();
  const fill = GRID_SURFACE_FILL.clone();
  const tint = GRID_SURFACE_TINT.clone();
  return new THREE.ShaderMaterial({
    uniforms: shaderUniforms(faceIndex, lineCol, fill, tint),
    vertexShader: /* glsl */ `
      varying vec3 vWorldPosition;
      varying float vFogDist;
      uniform vec3 uCameraWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPosition = wp.xyz;
        vFogDist = length(wp.xyz - uCameraWorldPos);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      varying vec3 vWorldPosition;
      varying float vFogDist;
      uniform vec3 uAccent;
      uniform vec3 uFillTint;
      uniform vec3 uLineColor;
      uniform float uLineStrength;
      uniform float uLineBoost;
      uniform float uGridScale;
      uniform int uFace;
      uniform vec3 uFogColor;
      uniform float uFogDensity;

      float gridLine(vec2 p) {
        vec2 coord = p * uGridScale;
        vec2 fw = vec2(length(vec2(dFdx(coord.x), dFdy(coord.x))),
                       length(vec2(dFdx(coord.y), dFdy(coord.y))));
        vec2 gv = fract(coord - 0.5) - 0.5;
        vec2 gAbs = abs(gv);
        vec2 line = smoothstep(fw * 0.55, fw * 1.12, gAbs);
        float m = 1.0 - min(line.x, line.y);
        return m;
      }

      void main() {
        vec2 gc;
        if (uFace == 0 || uFace == 1) {
          gc = vWorldPosition.xz;
        } else if (uFace == 2) {
          gc = vec2(vWorldPosition.z, vWorldPosition.y);
        } else {
          gc = vec2(-vWorldPosition.z, vWorldPosition.y);
        }

        float g = gridLine(gc);
        vec3 fill = uFillTint * 0.56;
        vec3 base = mix(fill, uAccent * 0.45, 0.2);
        vec3 lineMix = mix(base, uLineColor, g * uLineStrength);
        vec3 col = lineMix + uLineColor * g * uLineBoost;
        col += uAccent * g * (uLineStrength * 0.04);
        float fogF = 1.0 - exp(-vFogDist * uFogDensity);
        col = mix(col, uFogColor, fogF);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    side: THREE.DoubleSide,
  });
}

/** Same neutral grid on floor, ceiling, and both walls (face selector only differs). */
const planeMat = {
  floor: makeGridMaterial(0),
  ceiling: makeGridMaterial(1),
  left: makeGridMaterial(2),
  right: makeGridMaterial(3),
};

const gridMaterials = [
  planeMat.floor,
  planeMat.ceiling,
  planeMat.left,
  planeMat.right,
];

function addCorridor() {
  const len = CORRIDOR_LENGTH;
  const w = CORRIDOR_HALF_WIDTH * 2 + 0.04;
  const h = CORRIDOR_HEIGHT;

  const floorGeo = new THREE.PlaneGeometry(w, len);
  const floor = new THREE.Mesh(floorGeo, planeMat.floor);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, -len / 2);
  scene.add(floor);

  const ceiling = new THREE.Mesh(floorGeo.clone(), planeMat.ceiling);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, h, -len / 2);
  scene.add(ceiling);

  const wallGeo = new THREE.PlaneGeometry(len, h);
  const leftWall = new THREE.Mesh(wallGeo, planeMat.left);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-CORRIDOR_HALF_WIDTH, h / 2, -len / 2);
  scene.add(leftWall);

  const rightWall = new THREE.Mesh(wallGeo.clone(), planeMat.right);
  rightWall.rotation.y = -Math.PI / 2;
  rightWall.position.set(CORRIDOR_HALF_WIDTH, h / 2, -len / 2);
  scene.add(rightWall);
}

/** Glowing grid-aligned pads (classic synth-floor palette vs wall). */
function addParallaxGlowBlocks() {
  const eps = 0.034;
  const cell = 1 / GRID_SCALE;

  function neon(hex) {
    const m = new THREE.MeshBasicMaterial({ color: hex });
    m.fog = false;
    m.toneMapped = false;
    return m;
  }

  const floorC = neon(0x38e8ff);
  const ceilC = neon(0xffb04d);
  const leftC = neon(0x50ff96);
  const rightC = neon(0xff5a72);

  function flatFloor(cx, cz, gx, gz) {
    const hx = (gx * cell * 0.91) / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(gx * cell * 0.91, eps, gz * cell * 0.91),
      floorC
    );
    mesh.position.set(cxn, eps * 2, cz);
    scene.add(mesh);
  }

  function flatCeiling(cx, cz, gx, gz) {
    const hx = (gx * cell * 0.91) / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(gx * cell * 0.91, eps, gz * cell * 0.91),
      ceilC
    );
    mesh.position.set(cxn, CORRIDOR_HEIGHT - eps * 2, cz);
    scene.add(mesh);
  }

  function flatLeft(cy, cz, gy, gz) {
    const hy = (gy * cell * 0.91) / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(eps, gy * cell * 0.91, gz * cell * 0.91),
      leftC
    );
    mesh.position.set(-CORRIDOR_HALF_WIDTH + eps * 2, cyn, cz);
    scene.add(mesh);
  }

  function flatRight(cy, cz, gy, gz) {
    const hy = (gy * cell * 0.91) / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(eps, gy * cell * 0.91, gz * cell * 0.91),
      rightC
    );
    mesh.position.set(CORRIDOR_HALF_WIDTH - eps * 2, cyn, cz);
    scene.add(mesh);
  }

  /** Rows along -Z — mixed faces so parallax keeps cross-passing cues */
  const pack = [
    () => flatFloor(-0.4, -16, 3, 5),
    () => flatCeiling(1.05, -22, 4, 2),
    () => flatLeft(CORRIDOR_HEIGHT * 0.45, -30, 4, 6),
    () => flatRight(CORRIDOR_HEIGHT * 0.62, -26, 2, 5),
    () => flatFloor(0.7, -38, 5, 2),
    () => flatLeft(CORRIDOR_HEIGHT * 0.52, -44, 6, 3),
    () => flatCeiling(-1.05, -50, 2, 4),
    () => flatRight(CORRIDOR_HEIGHT * 0.42, -48, 5, 2),
    () => flatFloor(-1.05, -58, 4, 3),
    () => flatCeiling(0.15, -64, 3, 3),
    () => flatLeft(CORRIDOR_HEIGHT * 0.72, -70, 2, 9),
    () => flatRight(CORRIDOR_HEIGHT * 0.76, -66, 3, 4),
    () => flatFloor(-0.95, -78, 6, 1),
    () => flatCeiling(1.85, -84, 3, 2),
    () => flatLeft(CORRIDOR_HEIGHT * 0.54, -90, 3, 3),
    () => flatRight(CORRIDOR_HEIGHT * 0.62, -96, 2, 6),
    () => flatFloor(0.2, -102, 2, 6),
    () => flatCeiling(-2.03, -108, 5, 1),
    () => flatLeft(CORRIDOR_HEIGHT * 0.82, -114, 2, 4),
    () => flatFloor(1.05, -120, 2, 2),
    () => flatRight(CORRIDOR_HEIGHT * 0.74, -118, 4, 3),
    () => flatCeiling(0.45, -128, 2, 3),
    () => flatLeft(CORRIDOR_HEIGHT * 0.5, -136, 5, 2),
    () => flatFloor(-1.46, -132, 3, 2),
    () => flatRight(CORRIDOR_HEIGHT * 0.92, -144, 2, 2),
    () => flatCeiling(-0.92, -150, 4, 2),
    () => flatLeft(CORRIDOR_HEIGHT * 0.6, -158, 2, 5),
    () => flatFloor(0.5, -168, 4, 2),
    () => flatRight(CORRIDOR_HEIGHT * 0.46, -172, 3, 3),
    () => flatCeiling(1.38, -180, 2, 2),
    () => flatFloor(-0.92, -188, 2, 3),
    () => flatRight(CORRIDOR_HEIGHT * 0.94, -196, 2, 4),
    () => flatCeiling(-0.72, -204, 3, 2),
    () => flatRight(CORRIDOR_HEIGHT * 0.6, -214, 2, 5),
    () => flatLeft(CORRIDOR_HEIGHT * 0.73, -222, 3, 2),
    () => flatFloor(0.3, -230, 2, 2),
    () => flatCeiling(0.78, -248, 2, 6),
    () => flatCeiling(-0.78, -256, 2, 2),
    () => flatLeft(CORRIDOR_HEIGHT * 0.6, -260, 2, 2),
    () => flatLeft(CORRIDOR_HEIGHT * 0.92, -268, 2, 5),
    () => flatCeiling(0.1, -284, 2, 4),
    () => flatFloor(0.1, -320, 3, 2),
    () => flatRight(CORRIDOR_HEIGHT * 0.74, -340, 2, 3),
  ];

  /** L-like pairs on orthogonal faces */
  const Lpairs = [
    () => {
      flatLeft(CORRIDOR_HEIGHT * 0.5, -188, 2, 3);
      flatLeft(CORRIDOR_HEIGHT * 0.5 + cell * 2, -188 - cell, 2, 2);
    },
    () => {
      flatRight(CORRIDOR_HEIGHT * 0.45, -320, 2, 4);
      flatRight(CORRIDOR_HEIGHT * 0.45 + cell * 2, -320 - cell * 2, 3, 2);
    },
    () => {
      flatFloor(-0.5, -360, 3, 2);
      flatFloor(-0.5 + cell * 2, -360 - cell, 2, 2);
    },
  ];

  for (const fn of pack) fn();
  for (const fn of Lpairs) fn();
}

addCorridor();
addParallaxGlowBlocks();

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onResize);

/** 0 = top of viewport, 1 = bottom — interpolated into forward speed each frame */
let pointerYNormalized = 0.4;

function onPointerMove(event) {
  const h = window.innerHeight || 1;
  pointerYNormalized = THREE.MathUtils.clamp(event.clientY / h, 0, 1);
}

window.addEventListener('pointermove', onPointerMove, { passive: true });

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();

  const fwdSpeed = THREE.MathUtils.lerp(
    FWD_SPEED_AT_TOP,
    FWD_SPEED_AT_BOTTOM,
    pointerYNormalized
  );
  camera.position.z -= fwdSpeed * dt;
  if (camera.position.z < -CORRIDOR_LENGTH + 40) {
    camera.position.z = CAMERA_START_Z;
  }

  camera.position.x = 0;
  camera.position.y = EYE_HEIGHT;
  camera.lookAt(0, EYE_HEIGHT, camera.position.z - lookDist);

  const cam = camera.position;
  for (const m of gridMaterials) {
    m.uniforms.uCameraWorldPos.value.set(cam.x, cam.y, cam.z);
  }

  renderer.render(scene, camera);
}

animate();
