import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router } from 'react-router-dom';
import { AudioService } from '@shared/services/audio';
import { RagaDetector } from '@shared/services/raga-detector';
import { App } from './App';
import './index.css';

// Initialize services
const audioService = new AudioService();
const ragaDetector = new RagaDetector();

// Make services available to the app
const appContext = {
  audioService,
  ragaDetector,
  platform: 'web'
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Router>
      <App context={appContext} />
    </Router>
  </React.StrictMode>
);
