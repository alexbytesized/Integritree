import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import process from 'node:process'

const apiTarget = process.env.INTEGRITREE_API_TARGET || 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': { target: apiTarget, ws: true } } },
  preview: { proxy: { '/api': { target: apiTarget, ws: true } } },
})
