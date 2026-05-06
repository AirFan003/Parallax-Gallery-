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
const INDEX_FINGER_PIP = 6;
const INDEX_FINGER_TIP = 8;
const MIDDLE_FINGER_MCP = 9;
const MIDDLE_FINGER_PIP = 10;
const RING_FINGER_PIP = 14;
const RING_FINGER_TIP = 16;
const PINKY_PIP = 18;
const PINKY_MCP = 17;
const MIDDLE_FINGER_TIP = 12;
const PINKY_TIP = 20;

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

function lmDist3(a, b) {
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(a.x - b.x, a.y - b.y, dz);
}

/** Low curl ≈ fist; uses 3D distances so palm-at-camera isn’t always “fist”. */
function rightHandFistCurlRatio(lm) {
  const tips = [
    INDEX_FINGER_TIP,
    MIDDLE_FINGER_TIP,
    RING_FINGER_TIP,
    PINKY_TIP,
  ];
  const pips = [
    INDEX_FINGER_PIP,
    MIDDLE_FINGER_PIP,
    RING_FINGER_PIP,
    PINKY_PIP,
  ];
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    const t = lm[tips[i]];
    const pip = lm[pips[i]];
    if (!t || !pip) return null;
    sum += lmDist3(t, pip);
  }
  const w = lm[WRIST];
  const midMcp = lm[MIDDLE_FINGER_MCP];
  if (!w || !midMcp) return null;
  const ref = lmDist3(midMcp, w);
  if (ref < 0.012) return null;
  return sum / (4 * ref);
}

/** Mean(tip→wrist) / (wrist→middle MCP); low when fingers are curled in (fist), higher when open. */
function rightHandTipSpreadToWristRatio(lm) {
  const w = lm[WRIST];
  const midMcp = lm[MIDDLE_FINGER_MCP];
  if (!w || !midMcp) return null;
  const ref = lmDist3(midMcp, w);
  if (ref < 0.012) return null;
  const tips = [
    INDEX_FINGER_TIP,
    MIDDLE_FINGER_TIP,
    RING_FINGER_TIP,
    PINKY_TIP,
  ];
  let sum = 0;
  for (const ix of tips) {
    const t = lm[ix];
    if (!t) return null;
    sum += lmDist3(t, w);
  }
  return sum / (4 * ref);
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

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  logarithmicDepthBuffer: true,
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
document.body.appendChild(renderer.domElement);

const textureLoader = new THREE.TextureLoader();

/** Many textured quads — keep sampling cost predictable. */
const GALLERY_TEX_MAX_ANISOTROPY = 1;

/** Dense strip of photo slots — tight spacing, many faces per step, large prints. */
const GALLERY_DENSE_Z_START = -0.28;
const GALLERY_RING_STEP = 5.25;
const GALLERY_DENSE_Z_EXTENT_FR = 0.94;
/** ~8 panels × rings; cap keeps frame time bounded; photos still cycle. */
const GALLERY_MAX_PANELS = 960;

/**
 * Gallery JPEGs (~576px long edge) — width/height match `/public/gallery/*.jpg` for plane aspect.
 * Panels cycle through this list along the dense layout (unique images reduce obvious repetition).
 */
const GALLERY_ITEMS = [
  { url: '/gallery/01.jpg', width: 432, height: 576 },
  { url: '/gallery/02.jpg', width: 432, height: 576 },
  { url: '/gallery/03.jpg', width: 576, height: 432 },
  { url: '/gallery/04.jpg', width: 432, height: 576 },
  { url: '/gallery/05.jpg', width: 381, height: 576 },
  { url: '/gallery/06.jpg', width: 576, height: 381 },
  { url: '/gallery/07.jpg', width: 381, height: 576 },
  { url: '/gallery/08.jpg', width: 576, height: 381 },
  { url: '/gallery/09.jpg', width: 432, height: 576 },
  { url: '/gallery/10.jpg', width: 324, height: 576 },
  { url: '/gallery/11.jpg', width: 432, height: 576 },
  { url: '/gallery/12.jpg', width: 432, height: 576 },
  { url: '/gallery/13.jpg', width: 432, height: 576 },
  { url: '/gallery/14.jpg', width: 576, height: 508 },
  { url: '/gallery/15.jpg', width: 432, height: 576 },
  { url: '/gallery/16.jpg', width: 432, height: 576 },
  { url: '/gallery/17.jpg', width: 432, height: 576 },
  { url: '/gallery/18.jpg', width: 432, height: 576 },
  { url: '/gallery/19.jpg', width: 432, height: 576 },
  { url: '/gallery/20.jpg', width: 432, height: 576 },
  { url: '/gallery/21.jpg', width: 432, height: 576 },
  { url: '/gallery/22.jpg', width: 432, height: 576 },
  { url: '/gallery/23.jpg', width: 432, height: 576 },
  { url: '/gallery/24.jpg', width: 576, height: 432 },
  { url: '/gallery/25.jpg', width: 576, height: 432 },
  { url: '/gallery/26.jpg', width: 432, height: 576 },
  { url: '/gallery/27.jpg', width: 432, height: 576 },
  { url: '/gallery/28.jpg', width: 576, height: 432 },
  { url: '/gallery/29.jpg', width: 384, height: 576 },
  { url: '/gallery/30.jpg', width: 384, height: 576 },
  { url: '/gallery/31.jpg', width: 576, height: 383 },
  { url: '/gallery/32.jpg', width: 576, height: 383 },
  { url: '/gallery/33.jpg', width: 432, height: 576 },
  { url: '/gallery/34.jpg', width: 576, height: 432 },
  { url: '/gallery/35.jpg', width: 576, height: 432 },
  { url: '/gallery/36.jpg', width: 576, height: 432 },
  { url: '/gallery/37.jpg', width: 576, height: 432 },
  { url: '/gallery/38.jpg', width: 432, height: 576 },
  { url: '/gallery/39.jpg', width: 432, height: 576 },
  { url: '/gallery/40.jpg', width: 576, height: 432 },
  { url: '/gallery/41.jpg', width: 432, height: 576 },
  { url: '/gallery/42.jpg', width: 432, height: 576 },
  { url: '/gallery/43.jpg', width: 432, height: 576 },
  { url: '/gallery/44.jpg', width: 432, height: 576 },
  { url: '/gallery/45.jpg', width: 432, height: 576 },
  { url: '/gallery/46.jpg', width: 432, height: 576 },
  { url: '/gallery/47.jpg', width: 432, height: 576 },
  { url: '/gallery/48.jpg', width: 432, height: 576 },
  { url: '/gallery/49.jpg', width: 432, height: 576 },
  { url: '/gallery/50.jpg', width: 432, height: 576 },
  { url: '/gallery/51.jpg', width: 383, height: 576 },
  { url: '/gallery/52.jpg', width: 432, height: 576 },
  { url: '/gallery/53.jpg', width: 432, height: 576 },
  { url: '/gallery/54.jpg', width: 432, height: 576 },
  { url: '/gallery/55.jpg', width: 432, height: 576 },
  { url: '/gallery/56.jpg', width: 432, height: 576 },
  { url: '/gallery/57.jpg', width: 432, height: 576 },
  { url: '/gallery/58.jpg', width: 432, height: 576 },
  { url: '/gallery/59.jpg', width: 432, height: 576 },
  { url: '/gallery/60.jpg', width: 432, height: 576 },
  { url: '/gallery/61.jpg', width: 432, height: 576 },
  { url: '/gallery/62.jpg', width: 432, height: 576 },
  { url: '/gallery/63.jpg', width: 432, height: 576 },
  { url: '/gallery/64.jpg', width: 432, height: 576 },
  { url: '/gallery/65.jpg', width: 432, height: 576 },
  { url: '/gallery/66.jpg', width: 576, height: 432 },
  { url: '/gallery/67.jpg', width: 576, height: 384 },
  { url: '/gallery/68.jpg', width: 432, height: 576 },
  { url: '/gallery/69.jpg', width: 383, height: 576 },
  { url: '/gallery/70.jpg', width: 576, height: 432 },
  { url: '/gallery/71.jpg', width: 432, height: 576 },
];

/** Longer edge along tunnel run (floor/ceiling Z or wall Z) — keeps panels readable. */
const MAX_GALLERY_PHOTO_ALONG_Z = 2.55;

/** Off the corridor shell — too shallow ⇒ z-fight with the grid under roll. */
const GALLERY_SURFACE_LIFT = 0.065;
/**
 * Tiny random offset along each surface’s outward normal (±). Monotonic per-slot bias
 * incorrectly stacked later panels toward the camera, so dark placeholders could sit
 * on top of loaded photos in overlapping layouts.
 */
const GALLERY_PANEL_NORMAL_JITTER = 0.0065;

function gallerySurfaceSeparation(slot) {
  const u = (Math.imul(slot + 1, 116129781) >>> 0) / 2 ** 32;
  return (u - 0.5) * 2 * GALLERY_PANEL_NORMAL_JITTER;
}

function configureGalleryPhotoTexture(tex) {
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = Math.min(
    GALLERY_TEX_MAX_ANISOTROPY,
    renderer.capabilities.getMaxAnisotropy()
  );
}

const gallerySharedMaterials = new WeakMap();

function getSharedGalleryMaterial(texture) {
  let m = gallerySharedMaterials.get(texture);
  if (!m) {
    configureGalleryPhotoTexture(texture);
    texture.repeat.set(1, 1);
    texture.offset.set(0, 0);
    m = new THREE.MeshBasicMaterial({
      map: texture,
      color: 0xffffff,
      side: THREE.DoubleSide,
      fog: false,
      toneMapped: true,
    });
    gallerySharedMaterials.set(texture, m);
  }
  return m;
}

/**
 * object-fit: cover via vertex UVs so many meshes can share one Texture (no per-mesh repeat).
 * Matches PlaneGeometry(1×1 segment) vertex / uv order from three.js.
 */
function applyCoverUVsToPlaneGeometry(geom, imageWidth, imageHeight, planeWidth, planeHeight) {
  const ia = imageWidth / imageHeight;
  const pa = planeWidth / planeHeight;
  let uMin = 0;
  let uMax = 1;
  let vMin = 0;
  let vMax = 1;
  if (ia > pa) {
    const rx = pa / ia;
    uMin = (1 - rx) / 2;
    uMax = uMin + rx;
  } else {
    const ry = ia / pa;
    vMin = (1 - ry) / 2;
    vMax = vMin + ry;
  }
  const uv = geom.attributes.uv;
  uv.setXY(0, uMin, vMax);
  uv.setXY(1, uMax, vMax);
  uv.setXY(2, uMin, vMin);
  uv.setXY(3, uMax, vMin);
  uv.needsUpdate = true;
}

/** Stream decals in front of the camera — avoid decoding/uploads for the whole tunnel at once. */
const GALLERY_STREAM_LOAD_DISTANCE = 118;
const GALLERY_STREAM_MAX_NEW_PER_FRAME = 3;
const GALLERY_STREAM_MAX_IN_FLIGHT = 6;

/** Fist “grab” — curl must stay below this (see `rightHandFistCurlRatio`) */
const FIST_CURL_GRAB = 0.58;
/** Open curl — release pop */
const FIST_CURL_RELEASE = 0.78;
/** Also require fingertips pulled in toward wrist (ratio); open palm stays above this */
const FIST_TIP_WRIST_RATIO_GRAB = 1.14;
/** Fingers extended — release pop */
const FIST_TIP_WRIST_RATIO_RELEASE = 1.38;
const GALLERY_FIST_FOCUS_COUNT = 3;
const GALLERY_FIST_IN_VIEW_DOT = 0.18;
const GALLERY_FOCUS_DISTANCE = 7.25;
const GALLERY_FOCUS_HORIZONTAL_SPREAD = 3.85;
const GALLERY_FOCUS_BLEND_LAMBDA = 7.2;
/** Only run fist curl ML math every N video frames */
const FIST_SAMPLE_EVERY_N_FRAMES = 5;
const FIST_GRAB_HOLD_MS = 300;
const FIST_PULL_COOLDOWN_MS = 2000;

const galleryTextureCache = new Map();
const galleryStreamPanels = [];

const galleryFocusPinned = [];
let galleryFistLatched = false;
let galleryFistLastPullMs = 0;
let galleryFocusBlend = 0;
let fistCurlLastSample = null;
let fistSpreadLastSample = null;
let fistHoldStartMs = null;
let fistMlFrameCounter = 0;

let galleryStreamLoadsInFlight = 0;
const galleryTextureLoadPromises = new Map();

const _gfFwd = new THREE.Vector3();
const _gfRight = new THREE.Vector3();
const _gfToCam = new THREE.Vector3();
const _gfWorldTarget = new THREE.Vector3();
const _gfLocalTarget = new THREE.Vector3();
const _gfWorldQuat = new THREE.Quaternion();
const _gfParentQuat = new THREE.Quaternion();
const _gfInvParentQuat = new THREE.Quaternion();
const _gfLocalQuat = new THREE.Quaternion();
const _gfPlaneNormal = new THREE.Vector3(0, 0, 1);
const _corridorLocalCam = new THREE.Vector3();
const _forwardLocal = new THREE.Vector3();

async function getOrLoadGalleryTexture(url) {
  const cached = galleryTextureCache.get(url);
  if (cached) return cached;
  let inflight = galleryTextureLoadPromises.get(url);
  if (!inflight) {
    inflight = textureLoader.loadAsync(url).then((tex) => {
      configureGalleryPhotoTexture(tex);
      tex.repeat.set(1, 1);
      tex.offset.set(0, 0);
      galleryTextureCache.set(url, tex);
      galleryTextureLoadPromises.delete(url);
      return tex;
    });
    galleryTextureLoadPromises.set(url, inflight);
  }
  return inflight;
}

function updateGalleryTextureStreaming(cam) {
  if (!galleryStreamPanels.length) return;

  _corridorLocalCam.copy(cam);
  corridorRoot.worldToLocal(_corridorLocalCam);

  const maxDSq = GALLERY_STREAM_LOAD_DISTANCE * GALLERY_STREAM_LOAD_DISTANCE;
  const pool = [];
  for (const p of galleryStreamPanels) {
    if (p.resolved || p.loading) continue;
    const dSq = p.mesh.position.distanceToSquared(_corridorLocalCam);
    if (dSq >= maxDSq) continue;
    pool.push({ p, d: Math.sqrt(dSq) });
  }
  if (!pool.length) return;
  pool.sort((a, b) => a.d - b.d);

  let started = 0;
  for (const { p } of pool) {
    if (started >= GALLERY_STREAM_MAX_NEW_PER_FRAME) break;
    if (galleryStreamLoadsInFlight >= GALLERY_STREAM_MAX_IN_FLIGHT) break;
    started += 1;
    p.loading = true;
    galleryStreamLoadsInFlight += 1;
    const { mesh, url, planeW, planeH } = p;
    (async () => {
      try {
        const tex = await getOrLoadGalleryTexture(url);
        if (mesh.parent && !p.resolved) {
          finalizeGalleryPhotoMesh(mesh, tex, planeW, planeH);
          p.resolved = true;
        }
      } catch (e) {
        console.warn('Gallery texture:', url, e);
      } finally {
        p.loading = false;
        galleryStreamLoadsInFlight -= 1;
      }
    })();
  }
}

function pickGalleryFocusPanels() {
  galleryFocusPinned.length = 0;
  if (!galleryStreamPanels.length) return;

  camera.getWorldDirection(_gfFwd);
  corridorRoot.getWorldQuaternion(_gfParentQuat);
  _gfInvParentQuat.copy(_gfParentQuat).invert();
  _forwardLocal.copy(_gfFwd).applyQuaternion(_gfInvParentQuat);

  _corridorLocalCam.copy(camera.position);
  corridorRoot.worldToLocal(_corridorLocalCam);

  const poolIn = [];
  const poolAll = [];

  for (const panel of galleryStreamPanels) {
    const m = panel.mesh;
    _gfToCam.subVectors(m.position, _corridorLocalCam);
    const dist = _gfToCam.length();
    if (dist < 1.05 || dist > 92) continue;
    _gfToCam.multiplyScalar(1 / Math.max(dist, 1e-6));
    const inView = _forwardLocal.dot(_gfToCam) >= GALLERY_FIST_IN_VIEW_DOT;
    const entry = { p: panel, d: dist };
    poolAll.push(entry);
    if (inView) poolIn.push(entry);
  }

  poolIn.sort((a, b) => a.d - b.d);
  poolAll.sort((a, b) => a.d - b.d);

  const picked = new Set();
  for (const { p } of poolIn) {
    if (galleryFocusPinned.length >= GALLERY_FIST_FOCUS_COUNT) break;
    if (!p || picked.has(p)) continue;
    galleryFocusPinned.push(p);
    picked.add(p);
  }
  for (const { p } of poolAll) {
    if (galleryFocusPinned.length >= GALLERY_FIST_FOCUS_COUNT) break;
    if (!p || picked.has(p)) continue;
    galleryFocusPinned.push(p);
    picked.add(p);
  }
}

function updateGalleryFocusPop(dt, cam) {
  const goal = galleryFistLatched ? 1 : 0;
  galleryFocusBlend = THREE.MathUtils.lerp(
    galleryFocusBlend,
    goal,
    1 - Math.exp(-GALLERY_FOCUS_BLEND_LAMBDA * dt)
  );
  if (!galleryFocusPinned.length) return;

  if (!galleryFistLatched && galleryFocusBlend < 0.02) {
    for (const p of galleryFocusPinned) {
      p.mesh.position.copy(p.restPos);
      p.mesh.quaternion.copy(p.restQuat);
      p.mesh.scale.setScalar(1);
      p.mesh.renderOrder = 1;
      p.mesh.visible = p.resolved;
    }
    galleryFocusPinned.length = 0;
    galleryFocusBlend = 0;
    return;
  }

  camera.getWorldDirection(_gfFwd);
  _gfRight.crossVectors(_gfFwd, camera.up).normalize();
  if (_gfRight.lengthSq() < 1e-8) _gfRight.set(1, 0, 0);

  corridorRoot.getWorldQuaternion(_gfParentQuat);

  const n = galleryFocusPinned.length;

  for (let i = 0; i < n; i++) {
    const p = galleryFocusPinned[i];
    const slot = i - (n - 1) * 0.5;
    _gfWorldTarget
      .copy(cam)
      .addScaledVector(_gfFwd, GALLERY_FOCUS_DISTANCE)
      .addScaledVector(_gfRight, slot * GALLERY_FOCUS_HORIZONTAL_SPREAD);

    corridorRoot.worldToLocal(_gfLocalTarget.copy(_gfWorldTarget));
    _gfToCam.copy(cam).sub(_gfWorldTarget).normalize();
    if (_gfToCam.lengthSq() < 1e-8) _gfToCam.copy(_gfFwd).negate();
    _gfWorldQuat.setFromUnitVectors(_gfPlaneNormal, _gfToCam);
    if (
      !Number.isFinite(_gfWorldQuat.x) ||
      !Number.isFinite(_gfWorldQuat.w)
    ) {
      _gfLocalQuat.copy(p.restQuat);
    } else {
      _gfLocalQuat.copy(_gfParentQuat).invert().multiply(_gfWorldQuat);
    }

    THREE.Quaternion.slerpQuaternions(
      p.restQuat,
      _gfLocalQuat,
      galleryFocusBlend,
      p.mesh.quaternion
    );
    if (!Number.isFinite(p.mesh.quaternion.w)) {
      p.mesh.quaternion.copy(p.restQuat);
    }
    p.mesh.position.lerpVectors(
      p.restPos,
      _gfLocalTarget,
      galleryFocusBlend
    );
    if (!Number.isFinite(p.mesh.position.x)) {
      p.mesh.position.copy(p.restPos);
    }
    const sc = THREE.MathUtils.lerp(1, 1.09, galleryFocusBlend);
    p.mesh.scale.setScalar(sc);
    p.mesh.renderOrder = 1 + Math.round(galleryFocusBlend * 4);
    p.mesh.visible = galleryFocusBlend > 0.04 || p.resolved;
  }
}

function floorCeilingSpansFromPhotoItem(item) {
  const aspect = item.width / item.height;
  if (aspect >= 1) {
    const wx = MAX_GALLERY_PHOTO_ALONG_Z;
    const dz = wx / aspect;
    return { wx, dz };
  }
  const dz = MAX_GALLERY_PHOTO_ALONG_Z;
  const wx = dz * aspect;
  return { wx, dz };
}

/** Wall PlaneGeometry(dz, dy): image width → tunnel Z (dz), height → Y (dy). */
function wallSpansFromPhotoItem(item) {
  const aspect = item.width / item.height;
  if (aspect >= 1) {
    const dz = MAX_GALLERY_PHOTO_ALONG_Z;
    const dy = dz / aspect;
    return { dz, dy };
  }
  const dy = MAX_GALLERY_PHOTO_ALONG_Z;
  const dz = dy * aspect;
  return { dz, dy };
}

function clampWallPhotoSpansToCorridor(cy, dy, dz) {
  const hy = dy * 0.5;
  const maxHalf = Math.min(cy - 0.02, CORRIDOR_HEIGHT - cy - 0.02);
  if (maxHalf <= 0.02) {
    return { dy: Math.min(dy, 0.35), dz: Math.min(dz, 0.35) };
  }
  if (hy <= maxHalf) return { dy, dz };
  const s = maxHalf / hy;
  return { dy: dy * s, dz: dz * s };
}

function finalizeGalleryPhotoMesh(mesh, texture, planeWidth, planeHeight) {
  const img = texture.image;
  const iw = Math.max(1, img.naturalWidth || img.width || 1);
  const ih = Math.max(1, img.naturalHeight || img.height || 1);
  applyCoverUVsToPlaneGeometry(mesh.geometry, iw, ih, planeWidth, planeHeight);
  mesh.material.dispose();
  const base = getSharedGalleryMaterial(texture);
  const m = base.clone();
  m.side = THREE.DoubleSide;
  m.polygonOffset = false;
  mesh.material = m;
  mesh.visible = true;
}

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

/**
 * Procedural layout: each ring adds many staggered floor / ceiling / wall photos so the tunnel
 * reads as mostly imagery. `wx`/`dz`/`dy` on specs are ignored — size comes from image aspect.
 */
function buildDenseGalleryLayoutSpecs() {
  const h = CORRIDOR_HEIGHT;
  const specs = [];
  const zEnd = -CORRIDOR_LENGTH * GALLERY_DENSE_Z_EXTENT_FR;
  const pad = 0.38;
  const lo = -CORRIDOR_HALF_WIDTH + pad;
  const hi = CORRIDOR_HALF_WIDTH - pad;

  let ring = 0;
  for (let cz = GALLERY_DENSE_Z_START; cz > zEnd; cz -= GALLERY_RING_STEP) {
    const bump = (ring % 7) * 0.07;
    const i = ring;
    const cxF0 = THREE.MathUtils.clamp(-0.7 + (i % 6) * 0.24 + bump, lo, hi);
    const cxF1 = THREE.MathUtils.clamp(0.55 - ((i * 2) % 5) * 0.22 + bump * 0.5, lo, hi);
    const cxC0 = THREE.MathUtils.clamp(0.42 - (i % 5) * 0.18, lo, hi);
    const cxC1 = THREE.MathUtils.clamp(-0.58 + ((i * 3) % 6) * 0.19, lo, hi);
    const yL0 = THREE.MathUtils.clamp(0.28 + (i % 9) * 0.056 + bump, 0.18, 0.88);
    const yL1 = THREE.MathUtils.clamp(0.42 + ((i * 2) % 8) * 0.058, 0.2, 0.88);
    const yR0 = THREE.MathUtils.clamp(0.46 + (i % 7) * 0.06, 0.22, 0.88);
    const yR1 = THREE.MathUtils.clamp(0.62 + ((i * 3) % 5) * 0.045, 0.28, 0.88);

    const ringBatch = [
      { kind: 'floor', cx: cxF0, cz, wx: 1, dz: 1 },
      { kind: 'floor', cx: cxF1, cz: cz - 0.75, wx: 1, dz: 1 },
      { kind: 'ceiling', cx: cxC0, cz: cz - 1.05, wx: 1, dz: 1 },
      { kind: 'ceiling', cx: cxC1, cz: cz - 1.78, wx: 1, dz: 1 },
      { kind: 'left', cy: h * yL0, cz: cz - 2.15, dy: 1, dz: 1 },
      { kind: 'left', cy: h * yL1, cz: cz - 2.92, dy: 1, dz: 1 },
      { kind: 'right', cy: h * yR0, cz: cz - 1.42, dy: 1, dz: 1 },
      { kind: 'right', cy: h * yR1, cz: cz - 2.58, dy: 1, dz: 1 },
    ];

    if (specs.length + ringBatch.length > GALLERY_MAX_PANELS) break;
    for (const s of ringBatch) specs.push(s);
    ring += 1;
  }
  return specs;
}

/**
 * Textured gallery — meshes use placeholders until `updateGalleryTextureStreaming` loads
 * each JPEG when the panel is near the camera (bounded decode/GPU upload per frame).
 */
function addPhotoGalleryPlanes(parent) {
  galleryStreamPanels.length = 0;

  const len = CORRIDOR_LENGTH;
  const h = CORRIDOR_HEIGHT;
  const midZ = -len / 2;
  const lift = GALLERY_SURFACE_LIFT;

  /** Throw-away material; finalizeGalleryPhotoMesh replaces it after load. */
  function placeholderMaterial() {
    return new THREE.MeshBasicMaterial({
      color: 0x1a1d24,
      toneMapped: false,
      fog: false,
      side: THREE.DoubleSide,
    });
  }

  function registerStreamPanel(mesh, url, planeW, planeH, depthSlot) {
    galleryStreamPanels.push({
      mesh,
      url,
      planeW,
      planeH,
      depthSlot,
      resolved: false,
      loading: false,
      restPos: new THREE.Vector3().copy(mesh.position),
      restQuat: new THREE.Quaternion().copy(mesh.quaternion),
    });
  }

  function addFloorPanel(cx, cz, wx, dz, url) {
    const depthSlot = galleryStreamPanels.length;
    const sep = gallerySurfaceSeparation(depthSlot);
    const geo = new THREE.PlaneGeometry(wx, dz);
    const mesh = new THREE.Mesh(geo, placeholderMaterial());
    mesh.rotation.x = -Math.PI / 2;
    const hx = wx / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    mesh.position.set(cxn, -h / 2 + lift + sep, cz - midZ);
    mesh.renderOrder = 1;
    mesh.visible = false;
    parent.add(mesh);
    registerStreamPanel(mesh, url, wx, dz, depthSlot);
  }

  function addCeilingPanel(cx, cz, wx, dz, url) {
    const depthSlot = galleryStreamPanels.length;
    const sep = gallerySurfaceSeparation(depthSlot);
    const geo = new THREE.PlaneGeometry(wx, dz);
    const mesh = new THREE.Mesh(geo, placeholderMaterial());
    mesh.rotation.x = Math.PI / 2;
    const hx = wx / 2;
    const cxn = THREE.MathUtils.clamp(
      cx,
      -CORRIDOR_HALF_WIDTH + hx + 0.02,
      CORRIDOR_HALF_WIDTH - hx - 0.02
    );
    mesh.position.set(cxn, h / 2 - lift + sep, cz - midZ);
    mesh.renderOrder = 1;
    mesh.visible = false;
    parent.add(mesh);
    registerStreamPanel(mesh, url, wx, dz, depthSlot);
  }

  function addLeftWallPanel(cy, cz, dy, dz, url) {
    const depthSlot = galleryStreamPanels.length;
    const sep = gallerySurfaceSeparation(depthSlot);
    const hy = dy / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(dz, dy), placeholderMaterial());
    mesh.rotation.y = Math.PI / 2;
    mesh.position.set(-CORRIDOR_HALF_WIDTH + lift + sep, cyn - h / 2, cz - midZ);
    mesh.renderOrder = 1;
    mesh.visible = false;
    parent.add(mesh);
    registerStreamPanel(mesh, url, dz, dy, depthSlot);
  }

  function addRightWallPanel(cy, cz, dy, dz, url) {
    const depthSlot = galleryStreamPanels.length;
    const sep = gallerySurfaceSeparation(depthSlot);
    const hy = dy / 2;
    const cyn = THREE.MathUtils.clamp(cy, hy + 0.02, CORRIDOR_HEIGHT - hy - 0.02);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(dz, dy), placeholderMaterial());
    mesh.rotation.y = -Math.PI / 2;
    mesh.position.set(CORRIDOR_HALF_WIDTH - lift + sep, cyn - h / 2, cz - midZ);
    mesh.renderOrder = 1;
    mesh.visible = false;
    parent.add(mesh);
    registerStreamPanel(mesh, url, dz, dy, depthSlot);
  }

  const specs = buildDenseGalleryLayoutSpecs();
  specs.sort((a, b) => b.cz - a.cz);

  if (!GALLERY_ITEMS.length) return;

  const nItem = GALLERY_ITEMS.length;
  const n = Math.min(specs.length, GALLERY_MAX_PANELS);

  for (let i = 0; i < n; i += 1) {
    const s = specs[i];
    const item = GALLERY_ITEMS[i % nItem];
    const url = item.url;
    if (s.kind === 'floor') {
      const { wx, dz } = floorCeilingSpansFromPhotoItem(item);
      addFloorPanel(s.cx, s.cz, wx, dz, url);
    } else if (s.kind === 'ceiling') {
      const { wx, dz } = floorCeilingSpansFromPhotoItem(item);
      addCeilingPanel(s.cx, s.cz, wx, dz, url);
    } else {
      let { dz: wdz, dy } = wallSpansFromPhotoItem(item);
      const fitted = clampWallPhotoSpansToCorridor(s.cy, dy, wdz);
      dy = fitted.dy;
      wdz = fitted.dz;
      if (s.kind === 'left') addLeftWallPanel(s.cy, s.cz, dy, wdz, url);
      else addRightWallPanel(s.cy, s.cz, dy, wdz, url);
    }
  }
}

addCorridor(corridorRoot);

async function startExperience() {
  addPhotoGalleryPlanes(corridorRoot);
  animate();
}

void startExperience();

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

  if (rightLm && rightHandCalibrationState === 'done') {
    fistMlFrameCounter += 1;
    if (fistMlFrameCounter % FIST_SAMPLE_EVERY_N_FRAMES === 0) {
      fistCurlLastSample = rightHandFistCurlRatio(rightLm);
      fistSpreadLastSample = rightHandTipSpreadToWristRatio(rightLm);
    }
    const curl = fistCurlLastSample;
    const spread = fistSpreadLastSample;
    if (
      curl != null &&
      spread != null &&
      Number.isFinite(curl) &&
      Number.isFinite(spread)
    ) {
      const looksOpen =
        curl > FIST_CURL_RELEASE || spread > FIST_TIP_WRIST_RATIO_RELEASE;
      const looksFist =
        curl < FIST_CURL_GRAB && spread < FIST_TIP_WRIST_RATIO_GRAB;
      if (looksOpen) {
        galleryFistLatched = false;
        fistHoldStartMs = null;
      } else if (looksFist) {
        if (fistHoldStartMs == null) fistHoldStartMs = tMs;
        if (
          tMs - fistHoldStartMs >= FIST_GRAB_HOLD_MS &&
          tMs - galleryFistLastPullMs >= FIST_PULL_COOLDOWN_MS
        ) {
          pickGalleryFocusPanels();
          fistHoldStartMs = null;
          if (galleryFocusPinned.length > 0) {
            galleryFistLastPullMs = tMs;
            galleryFistLatched = true;
          }
        }
      } else {
        fistHoldStartMs = null;
      }
    } else {
      fistHoldStartMs = null;
    }
  } else {
    fistMlFrameCounter = 0;
    fistCurlLastSample = null;
    fistSpreadLastSample = null;
    fistHoldStartMs = null;
    galleryFistLatched = false;
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

  camera.position.y = EYE_HEIGHT;
  camera.lookAt(0, EYE_HEIGHT, camera.position.z - lookDist);

  const cam = camera.position;
  for (const m of gridMaterials) {
    m.uniforms.uCameraWorldPos.value.set(cam.x, cam.y, cam.z);
  }

  updateGalleryTextureStreaming(cam);
  updateGalleryFocusPop(dt, cam);

  renderer.render(scene, camera);
}
