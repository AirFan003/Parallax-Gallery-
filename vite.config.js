import { defineConfig } from 'vite';
import { galleryScanPlugin } from './gallery-scan-plugin.js';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  plugins: [galleryScanPlugin()],
});
