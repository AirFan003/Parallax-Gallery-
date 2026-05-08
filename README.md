# Parallax Gallery

A **first-person corridor** built with **[Three.js](https://threejs.org/)**—dense planes along the floor, ceiling, and walls, parallax as you move forward, plus optional **hand tracking** ([MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)) for speed, banking, and a **fist gesture** that pulls nearby panels into focus.

Use it as a starting point for photo tunnels, museum walkthroughs, or WebGL experiments.

---

## Features

- **Procedural tunnel** — grid shader on corridor surfaces; many gallery planes in a staggered layout.
- **Smooth glide** — forward motion along the hall with wrap-around when you reach the end.
- **Hand mode** (webcam, HTTPS or localhost) — palm **distance** controls speed; **knuckle-line tilt** banks the corridor; hold a **fist** to pick up to three panels in front of you and bring them closer.
- **Mouse fallback** — vertical pointer position sets speed when hands are not used.
- **Gray placeholders** — gallery slots are simple quads so the repo runs with no assets; easy to swap in textures or your own content in code.

---

## Requirements

- **Node.js**  
- A **modern desktop browser** with WebGL  
- For hand tracking: **Webcam** access; pages must be served from **localhost** or **HTTPS** (browser security for `getUserMedia`)

On first load, MediaPipe loads a small **hand landmark model** from Google’s CDN (see browser network tab).

---

## Quick start

```bash
git clone https://github.com/AirFan003/Parallax-Gallery-.git
cd parallax-gallery
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Allow the camera when prompted if you want hand controls.

**Production build:**

```bash
npm run build
npm run preview   # optional local test of dist/
```

Static output is in `dist/`; deploy to any static host (GitHub Pages, Netlify, Vercel, etc.). Use **HTTPS** if you need the webcam in production.

---

## Controls


| Input                   | Behavior                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **Mouse Y**             | Top of the **gallery** area = faster forward; bottom = slower / stop.                              |
| **Right hand (webcam)** | **Closer to camera** → faster glide. **Tilt knuckles** (vs calibrated “flat”) → corridor **roll**. |
| **Fist (held ~300 ms)** | Selects up to three visible panels and **animates them forward**; open hand releases.              |
| **Calibration**         | On startup, hold your **right** hand **flat** toward the camera for ~1 s when asked.               |


If the camera is denied or unavailable, the experience falls back to **mouse-only** speed; fist focus and banking are disabled.

---

## Customization

Most tuning lives in `**src/main.js`** at the top as `const` values:

- **Corridor** — `CORRIDOR_HALF_WIDTH`, `CORRIDOR_HEIGHT`, `CORRIDOR_LENGTH`, `GRID_SCALE`, colors under the grid shader setup.
- **Motion** — `FWD_SPEED_AT_TOP` / `FWD_SPEED_AT_BOTTOM`, `CAMERA_START_Z`, `EYE_HEIGHT`.
- **Gallery layout** — `GALLERY_RING_STEP`, `GALLERY_MAX_PANELS`, `MAX_GALLERY_PHOTO_ALONG_Z`, `PLACEHOLDER_GALLERY_ITEM` (aspect ratio for placeholder planes).
- **Hand / fist** — proximity span limits, `FIST_`* thresholds, `GALLERY_FOCUS_DISTANCE`, `GALLERY_FOCUS_HORIZONTAL_SPREAD`, `GALLERY_FOCUS_MOVEMENT_SCALE`.

The hand preview canvas is **hidden** (`display: none`) but still drives inference; skeleton drawing is optional if you want a debug view.

**Bringing back textured photos:** The previous version streamed images from `public/gallery/` via a Vite plugin. This fork uses gray placeholders to stay asset-free. To extend: replace `addPhotoGalleryPlanes` / materials with `TextureLoader` or your own asset pipeline, or reintroduce a folder scan if you prefer file-driven galleries.

---

## Project layout

```
├── index.html          # Full-screen #gallery-stage shell
├── src/main.js         # Scene, corridor, gallery, hand tracking, animation loop
├── vite.config.js      # Vite (no custom gallery plugin in current tree)
├── public/             # Static files (e.g. public/gallery/ for future assets)
└── package.json
```

---

## Privacy

Hand tracking runs **in the browser**. Video frames are processed for landmarks; nothing is sent to a server by this template beyond loading MediaPipe **model files** from their hosted URLs. Review `src/main.js` and MediaPipe’s terms if you ship a public product.

---

## Contributing

Issues and pull requests are welcome. If you add features, a short note in the README and focused diffs help others build on your work.

---

## License

Specify a license when you open the repository (for example **MIT**). Until then, all rights are reserved by default—add a `LICENSE` file so others know how they may use the code.