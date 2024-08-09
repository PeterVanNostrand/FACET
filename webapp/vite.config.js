import react from '@vitejs/plugin-react-swc';
import path from 'path';
import { defineConfig } from 'vite';
import config from "./config.json";

// https://vitejs.dev/config/
export default defineConfig({
  root: './frontend/',
  plugins: [react()],
  base: './',
  resolve: {
    alias: {
      '@src': path.resolve(__dirname, './frontend/src'),
      '@icons': path.resolve(__dirname, './frontend/icons'),
    },
  },
  server: {
    port: config.UI_PORT,
    fs: {
      cachedChecks: false
    }
  },
})
