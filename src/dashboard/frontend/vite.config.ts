import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/dashboard': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ledger': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/hdag': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/tic': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ml': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/zkml': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts'
  }
});
