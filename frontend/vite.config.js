// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  esbuild: {
    // Instruct esbuild to treat all .jsx files in /src as JSX
    loader: 'jsx',
    include: /src\/.*\.jsx$/,
    exclude: []
  },
  server: {
    port: 3000
  }
})
