import { App } from '@lynx-js/core';
import { AudioService } from './services/audio';
import { RagaDetector } from './services/raga-detector';
import { AppRoot } from './components/AppRoot';

class RagaDetectApp extends App {
  async onLaunch() {
    // Initialize services
    this.audioService = new AudioService();
    this.ragaDetector = new RagaDetector();
    
    // Set up audio pipeline
    await this.audioService.initialize();
    
    // Start with the main view
    this.navigateTo(AppRoot);
  }
}

// Start the application
const app = new RagaDetectApp();
app.start();
