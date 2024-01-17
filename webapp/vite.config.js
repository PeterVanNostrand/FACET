import react from '@vitejs/plugin-react-swc'
import { defineConfig } from 'vite'
import config from "./config.json"

// https://vitejs.dev/config/
export default defineConfig({
  root: './frontend',
  plugins: [react()],
  server: {
    port: config.UI_PORT,
  },
})
