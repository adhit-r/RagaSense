import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { Navbar } from './components/Navbar';
import { Home } from './pages/Home';
import { RagaDetector as RagaDetectorPage } from './pages/RagaDetector';
import { RagaList } from './pages/RagaList';
import './App.css';

export function App({ context }) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initializeApp = async () => {
      try {
        await context.audioService.initialize();
        setIsInitialized(true);
      } catch (err) {
        console.error('Failed to initialize app:', err);
        setError('Failed to initialize audio. Please check your microphone permissions.');
      }
    };

    initializeApp();

    return () => {
      // Cleanup
      if (context.audioService && context.audioService.cleanup) {
        context.audioService.cleanup();
      }
    };
  }, [context]);

  if (error) {
    return (
      <div className="error-screen">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  if (!isInitialized) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Initializing audio...</p>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route 
              path="/detect" 
              element={
                <RagaDetectorPage 
                  audioService={context.audioService} 
                  ragaDetector={context.ragaDetector} 
                />
              } 
            />
            <Route path="/ragas" element={<RagaList ragaDetector={context.ragaDetector} />} />
          </Routes>
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;
