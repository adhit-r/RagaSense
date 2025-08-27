import { defineConfig } from 'rspeedy';

export default defineConfig({
  // Project configuration
  name: 'ragasense',
  version: '1.0.0',
  
  // Platform configuration
  platforms: {
    web: {
      entry: 'src/App.tsx',
      output: 'dist/web',
      port: 3000
    },
    ios: {
      entry: 'src/App.tsx',
      output: 'dist/ios',
      simulator: true
    },
    android: {
      entry: 'src/App.tsx',
      output: 'dist/android',
      emulator: true
    }
  },
  
  // Build configuration
  build: {
    minify: true,
    sourcemap: true,
    target: 'es2020'
  },
  
  // Development configuration
  dev: {
    hot: true,
    open: true,
    port: 3000
  },
  
  // Styling configuration
  styling: {
    framework: 'tailwindcss',
    config: './tailwind.config.js'
  },
  
  // TypeScript configuration
  typescript: {
    strict: true,
    target: 'es2020',
    module: 'esnext'
  }
});
