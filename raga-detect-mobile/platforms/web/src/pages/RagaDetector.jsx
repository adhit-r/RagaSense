import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, RotateCcw, Info, AlertTriangle } from 'lucide-react';
import { RagaResult } from '../components/RagaResult';
import { AudioVisualizer } from '../components/AudioVisualizer';
import { useAudio } from '../context/AudioContext';
import '../styles/RagaDetector.css';

export function RagaDetector() {
  const {
    isRecording,
    audioLevel,
    ragaResult,
    error,
    isAnalyzing,
    recordingTime,
    startRecording,
    stopRecording,
    resetDetection
  } = useAudio();
  
  const [showInstructions, setShowInstructions] = useState(true);
  const canvasRef = useRef(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [showError, setShowError] = useState(false);
  
  const toggleRecording = useCallback(async () => {
    try {
      setShowError(false);
      if (isRecording) {
        setIsProcessing(true);
        await stopRecording();
      } else {
        resetDetection();
        const started = await startRecording();
        if (!started) {
          setShowError(true);
          return;
        }
      }
    } catch (err) {
      console.error('Error in toggleRecording:', err);
      setShowError(true);
    } finally {
      if (!isRecording) {
        setIsProcessing(false);
      }
    }
  }, [isRecording, startRecording, stopRecording, resetDetection]);
  
  // Auto-hide error after 5 seconds
  useEffect(() => {
    if (error) {
      setShowError(true);
      const timer = setTimeout(() => setShowError(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording().catch(console.error);
      }
    };
  }, [isRecording, stopRecording]);
  
  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Space or Enter to toggle recording when not in an input field
      if ((e.code === 'Space' || e.code === 'Enter') && 
          e.target.tagName !== 'INPUT' && 
          e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        toggleRecording();
      }
      // Escape to stop recording
      if (e.code === 'Escape' && isRecording) {
        e.preventDefault();
        toggleRecording();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRecording, toggleRecording]);

  return (
    <div className="raga-detector">
      <h1>Raga Detection</h1>
      
      <div className="recording-container">
        {/* Error message */}
        {(error && showError) && (
          <div className="error-message">
            <AlertTriangle size={18} className="icon" />
            <p>{error}</p>
          </div>
        )}
        
        <div className="visualization-container">
          <AudioVisualizer 
            ref={canvasRef} 
            audioLevel={audioLevel} 
            isActive={isRecording || isAnalyzing}
          />
          
          {(isRecording || isAnalyzing) && (
            <div className="recording-time">
              {isRecording && <span className="pulse">●</span>}
              {formatTime(recordingTime)}
            </div>
          )}
        </div>
        
        {isProcessing ? (
          <div className="processing-message">
            <div className="spinner" />
            <p>Analyzing your recording...</p>
            <p className="hint">This may take a few moments</p>
          </div>
        ) : ragaResult ? (
          <div className="result-container">
            <RagaResult result={ragaResult} />
            <button 
              onClick={resetDetection}
              className="btn secondary"
            >
              <RotateCcw size={16} /> Try Again
            </button>
          </div>
        ) : (
          <div className="recording-actions">
            <button
              onClick={toggleRecording}
              disabled={isAnalyzing || isProcessing}
              className={`record-button ${isRecording ? 'recording' : ''}`}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            >
              <Mic size={24} />
            </button>
            <p className="recording-hint">
              <Info size={16} /> 
              {isRecording 
                ? 'Click to stop recording' 
                : 'Click to start recording or press Space/Enter'}
            </p>
          </div>
        )}
        
        {showInstructions && (
          <div className="instructions">
            <Info size={16} />
            <p>
              {isRecording 
                ? 'Sing or play a raga. Click the button or press Escape to stop.'
                : 'Click the record button or press Space/Enter to start detecting ragas.'}
            </p>
          </div>
        )}
        
        {showInstructions && !isRecording && !ragaResult && (
          <div className="instructions">
            <div className="instructions-header">
              <h3>How to use</h3>
              <button 
                onClick={() => setShowInstructions(false)}
                className="close-button"
                aria-label="Hide instructions"
              >
                ×
              </button>
            </div>
            <ol>
              <li>Click the record button to start capturing audio</li>
              <li>Sing or play a raga for at least 10 seconds</li>
              <li>Click stop when you're done</li>
              <li>View the detected raga and related information</li>
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}
