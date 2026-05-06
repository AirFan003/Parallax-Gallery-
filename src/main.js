import * as THREE from 'three';
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from '@mediapipe/tasks-vision';

/** MediaPipe hand landmarks */
const INDEX_FINGER_MCP = 5;
const PINKY_MCP = 17;
const INDEX_FINGER_TIP = 8;

/** Exponential smoothing on knuckle-line angle: smoothed = α·smoothed + (1−α)·raw */
const LEFT_KNUCKLE_ANGLE_SMOOTH_ALPHA = 0.85;
/** No roll response within ±this many radians of calibrated neutral */
const LEFT_ROLL_DEAD_ZONE_RAD = THREE.MathUtils.degToRad(15);
/** Knuckle tilt beyond dead zone maps to full hall roll at this delta (radians) */
const LEFT_ROLL_MAX_TILT_RAD = THREE.MathUtils.degToRad(48);

const CORRIDOR_HALF_WIDTH = 2.15;
const CORRIDOR_HEIGHT = 3.45;
/** Long run — larger value lets the glide loop restart farther down-tunnel */
const CORRIDOR_LENGTH = 1680;

/** Camera height as fraction of tunnel height — centered-ish */
const EYE_HEIGHT = CORRIDOR_HEIGHT * 0.45;
const CAMERA_START_Z = 3.2;
/**
 * Forward speed from vertical control (mouse Y, or right index fingertip Y when tracked).
 * Top = fast forward; bottom = stop (no reverse motion).
 */
const FWD_SPEED_AT_TOP = 3.85;
const FWD_SPEED_AT_BOTTOM = 0;
/** Max tilt magnitude (radians) when knuckle line drives roll (after calibration). */
const CORRIDOR_ROLL_MAX = THREE.MathUtils.degToRad(52);

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

/** Index MCP → pinky MCP knuckle line angle in image space (radians). */
function leftKnuckleLineAngleFromLm(lm) {
  const idx = lm[INDEX_FINGER_MCP];
  const pky = lm[PINKY_MCP];
  if (!idx || !pky) return null;
  return Math.atan2(pky.y - idx.y, pky.x - idx.x);
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

addCorridor(corridorRoot);
addParallaxGlowBlocks(corridorRoot);

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onResize);

/** 0 = top of viewport, 1 = bottom — from mouse; used when hand is not driving */
let mouseYNormalized = 0.4;
/** Latest target from hand (updated on each video frame, not every render frame) */
let handTargetYNormalized = 0.4;
/** Display-follow value — quick lerp in animate so motion stays smooth between camera frames */
let handSmoothedYNormalized = 0.4;
/** True when the last video-frame pass saw a right index tip */
let handHasRightIndex = false;

/** Left-hand knuckle line: calibration & runtime ('loading' | 'awaiting_flat' | 'done' | 'skipped') */
let leftHandCalibrationState = 'loading';
let neutralKnuckleAngleRad = 0;
let calHoldSeconds = 0;
const calSampleAngles = [];
let lastVideoFrameTimeMs = null;

let handHasLeftKnuckle = false;
let leftKnuckleAngleSmoothed = 0;
let leftKnuckleAngleSmoothedInit = false;

let corridorRollSmoothed = 0;

function onPointerMove(event) {
  const h = window.innerHeight || 1;
  mouseYNormalized = THREE.MathUtils.clamp(event.clientY / h, 0, 1);
}

window.addEventListener('pointermove', onPointerMove, { passive: true });

const handPreviewWrap = document.createElement('div');
handPreviewWrap.id = 'hand-tracking-preview';

const handVideo = document.createElement('video');
handVideo.setAttribute('playsinline', '');
handVideo.muted = true;
handPreviewWrap.appendChild(handVideo);

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
  }
  return true;
}

function clearHandPreviewCanvas() {
  if (!handPreviewCanvas.width || !handPreviewCanvas.height) return;
  const ctx = handPreviewCanvas.getContext('2d');
  ctx.clearRect(0, 0, handPreviewCanvas.width, handPreviewCanvas.height);
}

function drawHandPreviewOverlay(result) {
  if (!syncHandPreviewCanvas() || !handPreviewDrawingUtils) return;
  const ctx = handPreviewCanvas.getContext('2d');
  ctx.clearRect(0, 0, handPreviewCanvas.width, handPreviewCanvas.height);
  for (let i = 0; i < result.landmarks.length; i++) {
    const lm = result.landmarks[i];
    const handed = result.handedness[i];
    const name = (handed?.[0]?.categoryName ?? '').toLowerCase();
    const isLeft = name.includes('left');
    const lineOpts = {
      color: isLeft
        ? 'rgba(96, 210, 255, 0.88)'
        : 'rgba(255, 130, 190, 0.88)',
      lineWidth: 2,
    };
    const pointOpts = {
      color: isLeft
        ? 'rgba(220, 250, 255, 0.96)'
        : 'rgba(255, 230, 245, 0.96)',
      lineWidth: 1,
      radius: 2.75,
    };
    handPreviewDrawingUtils.drawConnectors(
      lm,
      HandLandmarker.HAND_CONNECTIONS,
      lineOpts
    );
    handPreviewDrawingUtils.drawLandmarks(lm, pointOpts);
  }
}

const calStyle = document.createElement('style');
calStyle.textContent = `
#left-hand-calibration {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  transition: opacity 0.45s ease;
}
#left-hand-calibration .cal-card {
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
#left-hand-calibration .cal-title { font-weight: 600; margin-bottom: 0.65rem; }
#left-hand-calibration .cal-bar-wrap {
  margin-top: 1rem;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}
#left-hand-calibration .cal-bar {
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
  transform: scaleX(-1);
}
#hand-tracking-preview video,
#hand-tracking-preview canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: fill;
}
#hand-tracking-preview canvas {
  pointer-events: none;
}
`;
document.head.appendChild(calStyle);

const calOverlay = document.createElement('div');
calOverlay.id = 'left-hand-calibration';
calOverlay.innerHTML = `
  <div class="cal-card">
    <div class="cal-title">Left hand calibration</div>
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

  handHasRightIndex = false;
  handHasLeftKnuckle = false;

  if (
    !handLandmarker ||
    handVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    clearHandPreviewCanvas();
    handVideo.requestVideoFrameCallback(onHandLandmarkerVideoFrame);
    return;
  }

  const result = handLandmarker.detectForVideo(handVideo, performance.now());
  drawHandPreviewOverlay(result);
  let leftLm = null;

  for (let i = 0; i < result.landmarks.length; i++) {
    const handed = result.handedness[i];
    const label = (
      handed?.[0]?.categoryName ??
      handed?.[0]?.displayName ??
      ''
    ).toLowerCase();
    const lm = result.landmarks[i];

    if (label.includes('right')) {
      const tip = lm[INDEX_FINGER_TIP];
      if (tip) {
        handTargetYNormalized = THREE.MathUtils.clamp(tip.y, 0, 1);
        handHasRightIndex = true;
      }
    } else if (label.includes('left')) {
      leftLm = lm;
    }
  }

  if (leftLm) {
    const ang = leftKnuckleLineAngleFromLm(leftLm);
    if (ang !== null) {
      handHasLeftKnuckle = true;

      if (leftHandCalibrationState === 'awaiting_flat') {
        calHoldSeconds += dtFrame;
        calSampleAngles.push(ang);
        setCalibrationBar(Math.min(1, calHoldSeconds));
        if (calHoldSeconds >= 1) {
          neutralKnuckleAngleRad = circularMeanAngles(calSampleAngles);
          leftHandCalibrationState = 'done';
          leftKnuckleAngleSmoothed = neutralKnuckleAngleRad;
          leftKnuckleAngleSmoothedInit = true;
          calMsgEl.textContent = 'Calibrated. Enjoy!';
          setCalibrationBar(1);
          hideCalibrationOverlay();
        }
      } else if (leftHandCalibrationState === 'done') {
        if (!leftKnuckleAngleSmoothedInit) {
          leftKnuckleAngleSmoothed = ang;
          leftKnuckleAngleSmoothedInit = true;
        } else {
          leftKnuckleAngleSmoothed =
            LEFT_KNUCKLE_ANGLE_SMOOTH_ALPHA * leftKnuckleAngleSmoothed +
            (1 - LEFT_KNUCKLE_ANGLE_SMOOTH_ALPHA) * ang;
        }
      }
    }
  } else if (leftHandCalibrationState === 'awaiting_flat') {
    calHoldSeconds = 0;
    calSampleAngles.length = 0;
    setCalibrationBar(0);
  }

  if (leftHandCalibrationState === 'done' && !handHasLeftKnuckle) {
    leftKnuckleAngleSmoothedInit = false;
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
    });
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 480 },
        height: { ideal: 360 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    handVideo.srcObject = stream;
    await handVideo.play();
    leftHandCalibrationState = 'awaiting_flat';
    calMsgEl.textContent =
      'Hold your LEFT hand flat, palm toward the camera, steady for 1 second.';
    handVideo.requestVideoFrameCallback(onHandLandmarkerVideoFrame);
  } catch (err) {
    console.warn('Hand tracking unavailable, using mouse for speed only:', err);
    handLandmarker = null;
    leftHandCalibrationState = 'skipped';
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

  /** Follow fingertip quickly without stacking smoothing + duplicated per-frame inference */
  if (handHasRightIndex) {
    const snap = Math.min(1, dt * 52);
    handSmoothedYNormalized = THREE.MathUtils.lerp(
      handSmoothedYNormalized,
      handTargetYNormalized,
      snap
    );
  } else {
    handSmoothedYNormalized = mouseYNormalized;
  }

  const forwardYNormalized = handHasRightIndex
    ? handSmoothedYNormalized
    : mouseYNormalized;

  const fwdSpeed = THREE.MathUtils.lerp(
    FWD_SPEED_AT_TOP,
    FWD_SPEED_AT_BOTTOM,
    forwardYNormalized
  );
  camera.position.z -= fwdSpeed * dt;
  if (camera.position.z < -CORRIDOR_LENGTH + 40) {
    camera.position.z = CAMERA_START_Z;
  }

  /** Corridor roll: knuckle-line tilt vs calibrated neutral (dead zone + smoothing in video path). */
  let rollTarget = 0;
  if (leftHandCalibrationState === 'done' && handHasLeftKnuckle) {
    let d = angleDeltaShortest(
      leftKnuckleAngleSmoothed,
      neutralKnuckleAngleRad
    );
    const dz = LEFT_ROLL_DEAD_ZONE_RAD;
    if (Math.abs(d) <= dz) d = 0;
    else d = Math.sign(d) * (Math.abs(d) - dz);
    rollTarget =
      CORRIDOR_ROLL_MAX *
      THREE.MathUtils.clamp(d / LEFT_ROLL_MAX_TILT_RAD, -1, 1);
  }

  corridorRollSmoothed = THREE.MathUtils.lerp(
    corridorRollSmoothed,
    rollTarget,
    Math.min(1, dt * 20)
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
