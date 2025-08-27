import { defineConfig } from 'lynx'

export default defineConfig({
  // Application settings
  app: {
    name: 'Raga Detector',
    description: 'Indian Classical Raga Detection and Analysis',
    version: '1.0.0'
  },

  // Development server settings
  dev: {
    port: 3000,
    host: 'localhost',
    open: true
  },

  // Build settings
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: true
  },

  // API configuration
  api: {
    baseUrl: 'http://localhost:8000',
    timeout: 10000
  },

  // UI configuration
  ui: {
    theme: 'system', // 'light', 'dark', 'system'
    primaryColor: '#8B5CF6', // Purple for Indian classical theme
    accentColor: '#F59E0B'   // Amber accent
  },

  // Audio processing settings
  audio: {
    maxFileSize: 50 * 1024 * 1024, // 50MB
    supportedFormats: ['.wav', '.mp3', '.ogg', '.flac', '.m4a'],
    maxDuration: 300 // 5 minutes
  }
})
