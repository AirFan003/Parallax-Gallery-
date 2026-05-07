import fs from 'node:fs';
import path from 'node:path';
import { imageSize } from 'image-size';
import { normalizePath } from 'vite';

const VIRTUAL_ID = '\0virtual:gallery-items';

const EXT_RE = /\.(jpe?g|png|webp)$/i;

function scanGallery(publicDirAbs) {
  const galleryDir = path.join(publicDirAbs, 'gallery');
  if (!fs.existsSync(galleryDir)) return [];

  const items = [];
  for (const name of fs.readdirSync(galleryDir)) {
    if (name.startsWith('.') || !EXT_RE.test(name)) continue;
    const full = path.join(galleryDir, name);
    if (!fs.statSync(full).isFile()) continue;
    try {
      const buf = fs.readFileSync(full);
      const dim = imageSize(buf);
      if (
        !dim ||
        typeof dim.width !== 'number' ||
        typeof dim.height !== 'number' ||
        dim.width < 1 ||
        dim.height < 1
      ) {
        continue;
      }
      items.push({
        url: `/gallery/${name}`,
        width: dim.width,
        height: dim.height,
      });
    } catch {
      /* unreadable or unsupported */
    }
  }

  items.sort((a, b) =>
    a.url.localeCompare(b.url, undefined, { numeric: true, sensitivity: 'base' })
  );
  return items;
}

/**
 * Builds `GALLERY_ITEMS` from every `.jpg` / `.jpeg` / `.png` / `.webp` in `public/gallery/`
 * (drop new files there and refresh the dev server or rebuild).
 */
export function galleryScanPlugin({ publicDir = 'public' } = {}) {
  const publicDirAbs = path.resolve(publicDir);
  const galleryDirAbs = path.join(publicDirAbs, 'gallery');

  return {
    name: 'gallery-scan',
    resolveId(id) {
      if (id === 'virtual:gallery-items') return VIRTUAL_ID;
    },
    load(id) {
      if (id !== VIRTUAL_ID) return null;
      const items = scanGallery(publicDirAbs);
      return `export default ${JSON.stringify(items)}`;
    },
    configureServer(server) {
      if (fs.existsSync(galleryDirAbs)) {
        server.watcher.add(galleryDirAbs);
      }
      const onFs = (file) => {
        if (!file) return;
        const nf = normalizePath(path.resolve(file));
        const root = normalizePath(galleryDirAbs);
        if (root && !nf.startsWith(root)) return;
        const mod = server.moduleGraph.getModuleById(VIRTUAL_ID);
        if (mod) server.moduleGraph.invalidateModule(mod);
      };
      server.watcher.on('add', onFs);
      server.watcher.on('unlink', onFs);
      server.watcher.on('change', onFs);
    },
  };
}
