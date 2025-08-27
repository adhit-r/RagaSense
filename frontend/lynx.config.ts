import { defineConfig } from 'lynx'

export default defineConfig({
  // Project configuration
  name: 'ragasense',
  version: '1.0.0',
  
  // Platform configuration
  platforms: {
    web: {
      entry: 'src/main.tsx',
      output: 'dist/web',
      port: 3000
    },
    ios: {
      entry: 'src/main.tsx',
      output: 'dist/ios',
      simulator: true
    },
    android: {
      entry: 'src/main.tsx',
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
  },
  
  // Convex integration
  convex: {
    enabled: true,
    schema: './convex/schema.ts',
    functions: './convex'
  }
})
