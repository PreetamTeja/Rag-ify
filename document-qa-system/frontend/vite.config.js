import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(), 
    tailwindcss()
  ],
  server: {
    port: 3000,
    proxy: {
      '/upload': 'http://localhost:8000',
      '/query': 'http://localhost:8000', 
      '/agents': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})