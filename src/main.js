import * as THREE from 'three';
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from '@mediapipe/tasks-vision';

/**
 * Mirror pixels before inference so landmarks match what you see in the preview
 * (raw webcam is not mirrored; CSS-only flip made skeletons look “wrong”).
 */
const SELFIE_MIRROR_FOR_INFERENCE = true;

/** MediaPipe hand landmarks (right hand only for controls) */
const WRIST = 0;
const INDEX_FINGER_MCP = 5;
const PINKY_MCP = 17;
const MIDDLE_FINGER_TIP = 12;

/** Exponential smoothing on knuckle angle — higher = less jitter, slower palm follow */
const RIGHT_KNUCKLE_ANGLE_SMOOTH_ALPHA = 0.92;
/** No roll within ±this rad of calibrated neutral */
const RIGHT_ROLL_DEAD_ZONE_RAD = THREE.MathUtils.degToRad(15);
/** Full roll reached at this knuckle-line tilt beyond dead zone (smaller = snappier). */
const RIGHT_ROLL_MAX_TILT_RAD = THREE.MathUtils.degToRad(34);
/**
 * Proximity via normalized wrist→middle-tip span: small = hand far = slow,
 * large = hand close = fast (“pushing” the camera).
 */
const HAND_SPAN_SPEED_MIN = 0.105;
/** Larger max ⇒ need a bigger span (closer) to read as “fast”; small pullback drops speed harder */
const HAND_SPAN_SPEED_MAX = 0.34;
/** Stronger ease: only very “in” span reaches high proximity */
const PROXIMITY_CLOSE_EASE_GAMMA = 2.38;

const CORRIDOR_HALF_WIDTH = 2.15;
const CORRIDOR_HEIGHT = 3.45;
/** Long run — larger value lets the glide loop restart farther down-tunnel */
const CORRIDOR_LENGTH = 1680;

/** Camera height as fraction of tunnel height — centered-ish */
const EYE_HEIGHT = CORRIDOR_HEIGHT * 0.45;
const CAMERA_START_Z = 3.2;
/**
 * Hand mode: palm close to camera (high proximity) = faster forward; palm farther away = slower (down to BOTTOM).
 * Mouse: top of screen = fast forward, bottom = slow/stop.
 */
const FWD_SPEED_AT_TOP = 8.75;
const FWD_SPEED_AT_BOTTOM = 0;
/**
 * blendSlow = 1 - proximity^p. Higher p ⇒ mid/back-off stays crawl/stop unless palm is quite close.
 */
const HAND_SPEED_PROXIMITY_CURVE_POWER = 3.45;
/** Hall roll eases toward palm (higher = snappier; ~4–6 feels heavy, ~8 lighter). */
const CORRIDOR_ROLL_EASE_LAMBDA = 5.25;

/** Max corridor bank angle when palm tilt is at full deflection. */
const CORRIDOR_ROLL_MAX = THREE.MathUtils.degToRad(74);

/** World-space grid density (cells per unit) — tuned for Tetris-ish scale */
const GRID_SCALE = 1.8;

function angleDeltaShortest(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function circularMeanAngles(angles) {
  if (!angles.length) return 0;
  let sx = 0;
  let sy = 0;
  for (const ang of angles) {
    sx += Math.cos(ang);
    sy += Math.sin(ang);
  }
  return Math.atan2(sy / angles.length, sx / angles.length);
}

/** Index MCP → pinky MCP knuckle line angle (radians). */
function knuckleLineAngleFromLm(lm) {
  const idx = lm[INDEX_FINGER_MCP];
  const pky = lm[PINKY_MCP];
  if (!idx || !pky) return null;
  return Math.atan2(pky.y - idx.y, pky.x - idx.x);
}

/** Larger = hand closer to camera (normalized image coords). */
function rightHandProximitySpan(lm) {
  const w = lm[WRIST];
  const tip = lm[MIDDLE_FINGER_TIP];
  if (!w || !tip) return null;
  return Math.hypot(tip.x - w.x, tip.y - w.y);
}

const scene = new THREE.Scene();
const BACKGROUND = new THREE.Color(0x010104);
scene.background = BACKGROUND.clone();
scene.fog = null;

const corridorRoot = new THREE.Group();
corridorRoot.position.set(
  0,
  CORRIDOR_HEIGHT / 2,
  -CORRIDOR_LENGTH / 2
);
scene.add(corridorRoot);

const clock = new THREE.Clock();

const camera = new THREE.PerspectiveCamera(
  62,
  window.innerWidth / window.innerHeight,
  0.05,
  3200
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

/** Achromatic grid — soft, understated lines matching lifted cell tones */
const GRID_LINE_LIGHT = new THREE.Color(0xa7abb4);
const GRID_CELL_BASE = new THREE.Color(0x1e222a);
const GRID_CELL_LIFT = new THREE.Color(0x2a3038);

function shaderUniforms(face, lineCol, cellBase, cellLift) {
  return {
    uFace: { value: face },
    uCellBase: { value: new THREE.Vector3(cellBase.r, cellBase.g, cellBase.b) },
    uCellLift: { value: new THREE.Vector3(cellLift.r, cellLift.g, cellLift.b) },
    uLineColor: { value: new THREE.Vector3(lineCol.r, lineCol.g, lineCol.b) },
    uLineMix: { value: 0.38 },
    uLineBoost: { value: 0.035 },
    uGridScale: { value: GRID_SCALE },
    uFogColor: { value: new THREE.Vector3(BACKGROUND.r, BACKGROUND.g, BACKGROUND.b) },
    uFogDensity: { value: 0.031 },
    uCameraWorldPos: { value: new THREE.Vector3() },
  };
}

function makeGridMaterial(faceIndex) {
  const lineCol = GRID_LINE_LIGHT.clone();
  const base = GRID_CELL_BASE.clone();
  const lift = GRID_CELL_LIFT.clone();
  return new THREE.ShaderMaterial({
    uniforms: shaderUniforms(faceIndex, lineCol, base, lift),
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
      uniform vec3 uCellBase;
      uniform vec3 uCellLift;
      uniform vec3 uLineColor;
      uniform float uLineMix;
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
        vec2 line = smoothstep(fw * 0.72, fw * 1.22, gAbs);
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
        vec3 fill = uCellBase * 0.5;
        vec3 base = mix(fill, uCellLift * 0.48, 0.18);
        vec3 col = mix(base, uLineColor, g * uLineMix);
        col += uLineColor * g * uLineBoost;
        float fogF = 1.0 - exp(-vFogDist * uFogDensity);
        col = mix(col, uFogColor, fogF);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    side: THREE.DoubleSide,
  });
}

/** Same neutral light-gray grid on floor, ceiling, and both walls. */
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

function addCorridor(parent) {
  const len = CORRIDOR_LENGTH;
  const w = CORRIDOR_HALF_WIDTH * 2 + 0.04;
  const h = CORRIDOR_HEIGHT;

  const floorGeo = new THREE.PlaneGeometry(w, len);
  const floor = new THREE.Mesh(floorGeo, planeMat.floor);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, -h / 2, 0);
  parent.add(floor);

  const ceiling = new THREE.Mesh(floorGeo.clone(), planeMat.ceiling);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, h / 2, 0);
  parent.add(ceiling);

  const wallGeo = new THREE.PlaneGeometry(len, h);
  const leftWall = new THREE.Mesh(wallGeo, planeMat.left);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-CORRIDOR_HALF_WIDTH, 0, 0);
  parent.add(leftWall);

  const rightWall = new THREE.Mesh(wallGeo.clone(), planeMat.right);
  rightWall.rotation.y = -Math.PI / 2;
  rightWall.position.set(CORRIDOR_HALF_WIDTH, 0, 0);
  parent.add(rightWall);
}

/** Grid-aligned pads — light gray only (parity with corridor grid). */
function addParallaxGlowBlocks(parent) {
  const eps = 0.034;
  const cell = 1 / GRID_SCALE;
  const len = CORRIDOR_LENGTH;
  const h = CORRIDOR_HEIGHT;
  const midZ = -len / 2;

  function matteGray(hex) {
    const m = new THREE.MeshBasicMaterial({ color: hex });
    m.fog = false;
    m.toneMapped = false;
    return m;
  }

  const slabA = matteGray(0xe2e6ee);
  const slabB = matteGray(0xd6dae2);
  const slabC = matteGray(0xcaced8);

  function flatFloor(cx, cz, gx, gz) {
    const hx = (gx * cell * 0.91) / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(gx * cell * 0.91, eps, gz * cell * 0.91),
      slabA
    );
    mesh.position.set(cxn, eps * 2 - h / 2, cz - midZ);
    parent.add(mesh);
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
      slabB
    );
    mesh.position.set(cxn, h / 2 - eps * 2, cz - midZ);
    parent.add(mesh);
  }

  function flatLeft(cy, cz, gy, gz) {
    const hy = (gy * cell * 0.91) / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(eps, gy * cell * 0.91, gz * cell * 0.91),
      slabC
    );
    mesh.position.set(-CORRIDOR_HALF_WIDTH + eps * 2, cyn - h / 2, cz - midZ);
    parent.add(mesh);
  }

  function flatRight(cy, cz, gy, gz) {
    const hy = (gy * cell * 0.91) / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(eps, gy * cell * 0.91, gz * cell * 0.91),
      slabC
    );
    mesh.position.set(CORRIDOR_HALF_WIDTH - eps * 2, cyn - h / 2, cz - midZ);
    parent.add(mesh);
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

  /** Extra gray slabs spaced deeper so the longer tunnel stays populated */
  const deepenStep = Math.min(520, CORRIDOR_LENGTH * 0.3);
  for (let seg = 1; seg * deepenStep < CORRIDOR_LENGTH * 0.9; seg += 1) {
    const dz = -seg * deepenStep;
    flatFloor(-0.35, dz - 48, 2, 3);
    flatCeiling(0.72, dz - 120, 2, 5);
    flatLeft(CORRIDOR_HEIGHT * 0.55, dz - 180, 2, 4);
    flatRight(CORRIDOR_HEIGHT * 0.76, dz - 90, 2, 2);
    flatFloor(0.6, dz - 220, 4, 1);
    flatCeiling(-0.55, dz - 268, 3, 2);
  }

  for (const fn of pack) fn();
  for (const fn of Lpairs) fn();
}

/** White gallery panels — only on corridor floor / ceiling / left / right (no mid-air floats). */
function addWhiteGalleryPlanes(parent) {
  const len = CORRIDOR_LENGTH;
  const h = CORRIDOR_HEIGHT;
  const midZ = -len / 2;
  const lift = 0.03;

  const white = new THREE.MeshBasicMaterial({
    color: 0xf6f8fc,
    side: THREE.DoubleSide,
    fog: false,
    toneMapped: false,
  });

  function whiteFloor(cx, cz, wx, dz) {
    const geo = new THREE.PlaneGeometry(wx, dz);
    const mesh = new THREE.Mesh(geo, white);
    mesh.rotation.x = -Math.PI / 2;
    const hx = wx / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    mesh.position.set(cxn, -h / 2 + lift, cz - midZ);
    mesh.renderOrder = 1;
    parent.add(mesh);
  }

  function whiteCeiling(cx, cz, wx, dz) {
    const geo = new THREE.PlaneGeometry(wx, dz);
    const mesh = new THREE.Mesh(geo, white);
    mesh.rotation.x = Math.PI / 2;
    const hx = wx / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    mesh.position.set(cxn, h / 2 - lift, cz - midZ);
    mesh.renderOrder = 1;
    parent.add(mesh);
  }

  function whiteLeft(cy, cz, dy, dz) {
    const hy = dy / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(dz, dy), white);
    mesh.rotation.y = Math.PI / 2;
    mesh.position.set(-CORRIDOR_HALF_WIDTH + lift, cyn - h / 2, cz - midZ);
    mesh.renderOrder = 1;
    parent.add(mesh);
  }

  function whiteRight(cy, cz, dy, dz) {
    const hy = dy / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(dz, dy), white);
    mesh.rotation.y = -Math.PI / 2;
    mesh.position.set(CORRIDOR_HALF_WIDTH - lift, cyn - h / 2, cz - midZ);
    mesh.renderOrder = 1;
    parent.add(mesh);
  }

  /** Hand-authored patches along −Z; same spirit as gray slabs, larger “frames”. */
  const whitePack = [
    () => whiteFloor(-0.55, -24, 1.15, 1.8),
    () => whiteCeiling(0.65, -30, 1.4, 1.1),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.38, -36, 1.25, 2.1),
    () => whiteRight(CORRIDOR_HEIGHT * 0.65, -32, 1.4, 1.6),
    () => whiteFloor(0.45, -48, 1.8, 1.0),
    () => whiteCeiling(-0.72, -54, 1.0, 2.2),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.72, -60, 0.95, 2.6),
    () => whiteRight(CORRIDOR_HEIGHT * 0.48, -58, 1.2, 1.9),
    () => whiteFloor(-0.25, -72, 1.3, 1.45),
    () => whiteCeiling(0.35, -78, 1.55, 1.35),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.52, -86, 1.6, 1.4),
    () => whiteRight(CORRIDOR_HEIGHT * 0.78, -82, 1.0, 2.4),
    () => whiteFloor(0.85, -96, 1.1, 2.0),
    () => whiteCeiling(-0.45, -102, 1.7, 1.0),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.42, -110, 1.35, 1.8),
    () => whiteRight(CORRIDOR_HEIGHT * 0.58, -106, 1.5, 1.5),
    () => whiteFloor(-0.95, -120, 1.6, 1.2),
    () => whiteCeiling(0.92, -128, 1.2, 1.7),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.66, -136, 1.1, 2.2),
    () => whiteRight(CORRIDOR_HEIGHT * 0.44, -132, 1.25, 1.75),
    () => whiteFloor(0.15, -148, 2.0, 0.85),
    () => whiteCeiling(-0.88, -154, 1.45, 1.25),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.86, -162, 0.9, 2.3),
    () => whiteRight(CORRIDOR_HEIGHT * 0.52, -158, 1.55, 1.35),
    () => whiteFloor(-0.35, -172, 1.25, 1.55),
    () => whiteCeiling(0.28, -180, 1.9, 1.05),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.48, -188, 1.45, 1.6),
    () => whiteRight(CORRIDOR_HEIGHT * 0.7, -184, 1.15, 2.0),
    () => whiteFloor(0.62, -198, 1.35, 1.4),
    () => whiteCeiling(-0.58, -206, 1.2, 1.9),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.58, -214, 1.3, 1.7),
    () => whiteRight(CORRIDOR_HEIGHT * 0.88, -210, 0.95, 2.5),
    () => whiteFloor(-0.72, -226, 1.5, 1.1),
    () => whiteCeiling(0.78, -234, 1.35, 1.45),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.74, -242, 1.2, 1.85),
    () => whiteRight(CORRIDOR_HEIGHT * 0.4, -238, 1.65, 1.2),
    () => whiteFloor(0.35, -252, 1.1, 1.85),
    () => whiteCeiling(-0.22, -260, 1.65, 1.15),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.62, -268, 1.4, 1.45),
    () => whiteRight(CORRIDOR_HEIGHT * 0.62, -264, 1.4, 1.45),
    () => whiteFloor(-0.15, -280, 1.8, 1.0),
    () => whiteCeiling(0.48, -288, 1.1, 1.75),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.5, -296, 1.6, 1.3),
    () => whiteRight(CORRIDOR_HEIGHT * 0.76, -292, 1.05, 2.15),
    () => whiteFloor(0.92, -308, 1.2, 1.5),
    () => whiteCeiling(-0.72, -316, 1.5, 1.2),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.8, -324, 1.0, 2.0),
    () => whiteRight(CORRIDOR_HEIGHT * 0.46, -320, 1.45, 1.45),
    () => whiteFloor(-0.48, -336, 1.4, 1.35),
    () => whiteCeiling(0.15, -344, 1.8, 1.0),
    () => whiteLeft(CORRIDOR_HEIGHT * 0.54, -352, 1.25, 1.65),
    () => whiteRight(CORRIDOR_HEIGHT * 0.7, -348, 1.25, 1.65),
  ];

  const deepenStep = Math.min(520, CORRIDOR_LENGTH * 0.3);
  for (let seg = 1; seg * deepenStep < CORRIDOR_LENGTH * 0.92; seg += 1) {
    const d = -seg * deepenStep;
    whiteFloor(-0.62, d - 14, 1.05, 1.35);
    whiteFloor(0.58, d - 92, 1.35, 1.1);
    whiteCeiling(0.55, d - 38, 1.25, 1.2);
    whiteCeiling(-0.68, d - 118, 1.15, 1.45);
    whiteLeft(CORRIDOR_HEIGHT * 0.45, d - 62, 1.2, 1.6);
    whiteLeft(CORRIDOR_HEIGHT * 0.78, d - 142, 1.0, 1.8);
    whiteRight(CORRIDOR_HEIGHT * 0.55, d - 52, 1.35, 1.45);
    whiteRight(CORRIDOR_HEIGHT * 0.82, d - 128, 0.95, 1.9);
  }

  for (const fn of whitePack) fn();
}

addCorridor(corridorRoot);
addParallaxGlowBlocks(corridorRoot);
addWhiteGalleryPlanes(corridorRoot);

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onResize);

/** Mouse Y fallback when right hand not visible: top = fast. */
let mouseYNormalized = 0.4;

/** Right-hand calibration & roll ('loading' | 'awaiting_flat' | 'done' | 'skipped') */
let rightHandCalibrationState = 'loading';
let neutralKnuckleAngleRad = 0;
let calHoldSeconds = 0;
const calSampleAngles = [];
let lastVideoFrameTimeMs = null;

let handHasRightKnuckle = false;
let handHasRightProximity = false;
let rightKnuckleAngleSmoothed = 0;
let rightKnuckleAngleSmoothedInit = false;

/** Proximity 0 = far/slow, 1 = close/fast — from wrist→middle-tip span */
let handTargetProximity01 = 0;
let handSmoothedProximity01 = 0.35;

let corridorRollSmoothed = 0;

function onPointerMove(event) {
  const h = window.innerHeight || 1;
  mouseYNormalized = THREE.MathUtils.clamp(event.clientY / h, 0, 1);
}

window.addEventListener('pointermove', onPointerMove, { passive: true });

const handVideo = document.createElement('video');
handVideo.setAttribute('playsinline', '');
handVideo.muted = true;
Object.assign(handVideo.style, {
  position: 'fixed',
  left: '-9999px',
  width: '2px',
  height: '2px',
  opacity: '0',
  pointerEvents: 'none',
});
document.body.appendChild(handVideo);

const handPreviewWrap = document.createElement('div');
handPreviewWrap.id = 'hand-tracking-preview';
const handPreviewCanvas = document.createElement('canvas');
handPreviewCanvas.id = 'hand-preview-overlay';
handPreviewWrap.appendChild(handPreviewCanvas);
document.body.appendChild(handPreviewWrap);

let handPreviewDrawingUtils = null;

function syncHandPreviewCanvas() {
  const vw = handVideo.videoWidth;
  const vh = handVideo.videoHeight;
  if (!vw || !vh) return false;
  if (
    handPreviewCanvas.width !== vw ||
    handPreviewCanvas.height !== vh
  ) {
    handPreviewCanvas.width = vw;
    handPreviewCanvas.height = vh;
    handPreviewDrawingUtils = new DrawingUtils(
      handPreviewCanvas.getContext('2d')
    );
  } else if (!handPreviewDrawingUtils) {
    handPreviewDrawingUtils = new DrawingUtils(
      handPreviewCanvas.getContext('2d')
    );
  }
  return true;
}

function clearHandPreviewCanvas() {
  if (!handPreviewCanvas.width || !handPreviewCanvas.height) return;
  const ctx = handPreviewCanvas.getContext('2d');
  ctx.clearRect(0, 0, handPreviewCanvas.width, handPreviewCanvas.height);
}

/** Mirrored selfie frame on canvas, then landmarks (must match detectForVideo input). */
function drawMirroredVideoFrameOntoPreviewCanvas() {
  const ctx = handPreviewCanvas.getContext('2d');
  const w = handPreviewCanvas.width;
  const h = handPreviewCanvas.height;
  if (!SELFIE_MIRROR_FOR_INFERENCE) {
    ctx.drawImage(handVideo, 0, 0, w, h);
    return;
  }
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-w, 0);
  ctx.drawImage(handVideo, 0, 0, w, h);
  ctx.restore();
}

function drawHandSkeletonOnPreview(result, activeLandmarks) {
  if (!handPreviewDrawingUtils || !activeLandmarks) return;
  const lineOpts = {
    color: 'rgba(255, 160, 210, 0.92)',
    lineWidth: 2,
  };
  const pointOpts = {
    color: 'rgba(255, 235, 248, 0.96)',
    lineWidth: 1,
    radius: 2.75,
  };
  for (let i = 0; i < result.landmarks.length; i++) {
    if (result.landmarks[i] !== activeLandmarks) continue;
    handPreviewDrawingUtils.drawConnectors(
      result.landmarks[i],
      HandLandmarker.HAND_CONNECTIONS,
      lineOpts
    );
    handPreviewDrawingUtils.drawLandmarks(result.landmarks[i], pointOpts);
  }
}

const calStyle = document.createElement('style');
calStyle.textContent = `
#hand-calibration {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  transition: opacity 0.45s ease;
}
#hand-calibration .cal-card {
  pointer-events: auto;
  max-width: 26rem;
  margin: 1rem;
  padding: 1.35rem 1.5rem;
  background: rgba(16, 18, 24, 0.92);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  color: #e8eaf0;
  font: 15px/1.45 system-ui, sans-serif;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.55);
}
#hand-calibration .cal-title { font-weight: 600; margin-bottom: 0.65rem; }
#hand-calibration .cal-bar-wrap {
  margin-top: 1rem;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}
#hand-calibration .cal-bar {
  height: 100%;
  width: 0%;
  border-radius: 3px;
  background: linear-gradient(90deg, #6b8cff, #a78bfa);
  transition: width 0.08s linear;
}
#hand-tracking-preview {
  position: fixed;
  top: 12px;
  left: 12px;
  z-index: 9000;
  width: min(280px, 34vw);
  aspect-ratio: 4 / 3;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.16);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.5);
  background: #050508;
}
#hand-tracking-preview canvas {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: fill;
  pointer-events: none;
}
`;
document.head.appendChild(calStyle);

const calOverlay = document.createElement('div');
calOverlay.id = 'hand-calibration';
calOverlay.innerHTML = `
  <div class="cal-card">
    <div class="cal-title">Right hand calibration</div>
    <p class="cal-msg">Loading hand tracking…</p>
    <div class="cal-bar-wrap"><div class="cal-bar"></div></div>
  </div>
`;
document.body.appendChild(calOverlay);
const calMsgEl = calOverlay.querySelector('.cal-msg');
const calBarEl = calOverlay.querySelector('.cal-bar');

function setCalibrationBar(p01) {
  calBarEl.style.width = `${THREE.MathUtils.clamp(p01, 0, 1) * 100}%`;
}

function hideCalibrationOverlay() {
  calOverlay.style.opacity = '0';
  setTimeout(() => {
    calOverlay.style.display = 'none';
  }, 500);
}

let handLandmarker = null;

function onHandLandmarkerVideoFrame(now) {
  const tMs = typeof now === 'number' ? now : performance.now();
  let dtFrame = 0;
  if (lastVideoFrameTimeMs != null) {
    dtFrame = Math.min(0.1, (tMs - lastVideoFrameTimeMs) / 1000);
  }
  lastVideoFrameTimeMs = tMs;

  handHasRightKnuckle = false;
  handHasRightProximity = false;

  if (
    !handLandmarker ||
    handVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    clearHandPreviewCanvas();
    handVideo.requestVideoFrameCallback(onHandLandmarkerVideoFrame);
    return;
  }

  if (!syncHandPreviewCanvas()) {
    handVideo.requestVideoFrameCallback(onHandLandmarkerVideoFrame);
    return;
  }

  drawMirroredVideoFrameOntoPreviewCanvas();
  const result = handLandmarker.detectForVideo(handPreviewCanvas, tMs);

  let rightLm = null;
  for (let i = 0; i < result.landmarks.length; i++) {
    const handed = result.handedness[i];
    const label = (
      handed?.[0]?.categoryName ??
      handed?.[0]?.displayName ??
      ''
    ).toLowerCase();
    if (label.includes('right')) {
      rightLm = result.landmarks[i];
      break;
    }
  }
  if (!rightLm && result.landmarks.length === 1) {
    rightLm = result.landmarks[0];
  }

  drawHandSkeletonOnPreview(result, rightLm);

  if (rightLm) {
    const span = rightHandProximitySpan(rightLm);
    if (span != null) {
      handHasRightProximity = true;
      const linearProx = THREE.MathUtils.clamp(
        (span - HAND_SPAN_SPEED_MIN) /
          (HAND_SPAN_SPEED_MAX - HAND_SPAN_SPEED_MIN),
        0,
        1
      );
      handTargetProximity01 =
        1 - Math.pow(1 - linearProx, PROXIMITY_CLOSE_EASE_GAMMA);
    }

    const ang = knuckleLineAngleFromLm(rightLm);
    if (ang !== null) {
      handHasRightKnuckle = true;

      if (rightHandCalibrationState === 'awaiting_flat') {
        calHoldSeconds += dtFrame;
        calSampleAngles.push(ang);
        setCalibrationBar(Math.min(1, calHoldSeconds));
        if (calHoldSeconds >= 1) {
          neutralKnuckleAngleRad = circularMeanAngles(calSampleAngles);
          rightHandCalibrationState = 'done';
          rightKnuckleAngleSmoothed = neutralKnuckleAngleRad;
          rightKnuckleAngleSmoothedInit = true;
          calMsgEl.textContent = 'Calibrated. Enjoy!';
          setCalibrationBar(1);
          hideCalibrationOverlay();
        }
      } else if (rightHandCalibrationState === 'done') {
        if (!rightKnuckleAngleSmoothedInit) {
          rightKnuckleAngleSmoothed = ang;
          rightKnuckleAngleSmoothedInit = true;
        } else {
          rightKnuckleAngleSmoothed =
            RIGHT_KNUCKLE_ANGLE_SMOOTH_ALPHA * rightKnuckleAngleSmoothed +
            (1 - RIGHT_KNUCKLE_ANGLE_SMOOTH_ALPHA) * ang;
        }
      }
    }
  } else if (rightHandCalibrationState === 'awaiting_flat') {
    calHoldSeconds = 0;
    calSampleAngles.length = 0;
    setCalibrationBar(0);
  }

  if (rightHandCalibrationState === 'done' && !handHasRightKnuckle) {
    rightKnuckleAngleSmoothedInit = false;
  }

  handVideo.requestVideoFrameCallback(onHandLandmarkerVideoFrame);
}

async function initHandTracking() {
  try {
    const wasmPath =
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
    const fileset = await FilesetResolver.forVisionTasks(wasmPath);
    handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    });
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 1280, min: 640 },
        height: { ideal: 720, min: 480 },
        frameRate: { ideal: 30, max: 60 },
      },
    });
    handVideo.srcObject = stream;
    await handVideo.play();
    rightHandCalibrationState = 'awaiting_flat';
    calMsgEl.textContent =
      'Hold your RIGHT hand flat, palm toward the camera, steady for 1 second.';
    handVideo.requestVideoFrameCallback(onHandLandmarkerVideoFrame);
  } catch (err) {
    console.warn('Hand tracking unavailable, using mouse for speed only:', err);
    handLandmarker = null;
    rightHandCalibrationState = 'skipped';
    handPreviewWrap.style.display = 'none';
    calMsgEl.textContent =
      'Camera or hand tracking unavailable. Corridor roll is disabled.';
    setTimeout(hideCalibrationOverlay, 3200);
  }
}

void initHandTracking();

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();

  /** Forward speed: right-hand proximity (close = fast), else mouse Y. */
  if (handHasRightProximity && rightHandCalibrationState === 'done') {
    const snapP = Math.min(1, dt * 78);
    handSmoothedProximity01 = THREE.MathUtils.lerp(
      handSmoothedProximity01,
      handTargetProximity01,
      snapP
    );
  } else {
    handSmoothedProximity01 = THREE.MathUtils.lerp(
      handSmoothedProximity01,
      0.15,
      Math.min(1, dt * 12)
    );
  }

  const proximityForSpeed =
    rightHandCalibrationState === 'done' && handHasRightProximity
      ? handSmoothedProximity01
      : null;

  const fwdSpeed =
    proximityForSpeed != null
      ? THREE.MathUtils.lerp(
          FWD_SPEED_AT_TOP,
          FWD_SPEED_AT_BOTTOM,
          1 -
            Math.pow(
              THREE.MathUtils.clamp(proximityForSpeed, 0, 1),
              HAND_SPEED_PROXIMITY_CURVE_POWER
            )
        )
      : THREE.MathUtils.lerp(
          FWD_SPEED_AT_TOP,
          FWD_SPEED_AT_BOTTOM,
          mouseYNormalized
        );
  camera.position.z -= fwdSpeed * dt;
  if (camera.position.z < -CORRIDOR_LENGTH + 40) {
    camera.position.z = CAMERA_START_Z;
  }

  let rollTarget = 0;
  if (rightHandCalibrationState === 'done' && handHasRightKnuckle) {
    let d = angleDeltaShortest(
      rightKnuckleAngleSmoothed,
      neutralKnuckleAngleRad
    );
    const dz = RIGHT_ROLL_DEAD_ZONE_RAD;
    if (Math.abs(d) <= dz) d = 0;
    else d = Math.sign(d) * (Math.abs(d) - dz);
    /** Negative: knuckle-line angle delta was inverted vs desired palm-left / hall-left */
    rollTarget =
      -CORRIDOR_ROLL_MAX *
      THREE.MathUtils.clamp(d / RIGHT_ROLL_MAX_TILT_RAD, -1, 1);
  }

  const rollEaseT = 1 - Math.exp(-CORRIDOR_ROLL_EASE_LAMBDA * dt);
  corridorRollSmoothed = THREE.MathUtils.lerp(
    corridorRollSmoothed,
    rollTarget,
    Math.min(1, rollEaseT)
  );
  corridorRoot.rotation.z = corridorRollSmoothed;

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
