import { createApp } from 'lynx';
import RagaDetector from './components/RagaDetector';
import './styles/globals.css';

// Create the Lynx app
const app = createApp({
  root: RagaDetector,
  container: '#app'
});

// Start the app
app.mount();
