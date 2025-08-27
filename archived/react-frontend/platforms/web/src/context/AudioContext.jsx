import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { AudioService } from '../../src/shared/services/audio';
import { RagaDetector } from '../../src/shared/services/raga-detector';

// Mock raga data for development
const MOCK_RAGAS = [
  {
    name: 'Mohanam',
    aarohana: ['S', 'R2', 'G3', 'P', 'D2'],
    avarohana: ['D2', 'P', 'G3', 'R2', 'S'],
    mood: 'Joyful',
    time: 'Evening',
    similarRagas: [
      { name: 'Shuddha Saveri', confidence: 0.75 },
      { name: 'Bhoopali', confidence: 0.68 },
      { name: 'Deskar', confidence: 0.52 }
    ]
  },
  // Add more mock ragas as needed
];

const AudioContext = createContext();

export const AudioProvider = ({ children }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [ragaResult, setRagaResult] = useState(null);
  const [error, setError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);
  
  const audioService = useRef(null);
  const ragaDetector = useRef(new RagaDetector());
  const timerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const analyserRef = useRef(null);
  
  // Initialize audio service
  useEffect(() => {
    const init = async () => {
      try {
        const service = new AudioService({
          onAudioProcess: (data) => {
            setAudioLevel(data.volume);
            setPitch(data.pitch);
          }
        });
        await service.initialize();
        audioService.current = service;
        setIsInitialized(true);
      } catch (err) {
        console.error('Failed to initialize audio service:', err);
        setError('Failed to initialize audio. Please refresh the page and try again.');
      }
    };
    
    init();
    
    return () => {
      if (audioService.current) {
        audioService.current.cleanup();
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    if (!isInitialized) {
      setError('Audio service not initialized');
      return false;
    }
    
    try {
      setError(null);
      setRagaResult(null);
      setIsAnalyzing(false);
      
      const success = await audioService.current.startRecording();
      if (!success) {
        throw new Error('Failed to start recording');
      }
      
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
      return true;
    } catch (err) {
      console.error('Error starting recording:', err);
      const errorMessage = err.message.includes('permission') 
        ? 'Microphone access was denied. Please allow microphone access to use this feature.'
        : 'Failed to start recording. Please try again.';
      setError(errorMessage);
      return false;
    }
  }, [isInitialized]);

  const stopRecording = useCallback(async () => {
    if (!isRecording) return null;
    
    try {
      clearInterval(timerRef.current);
      setIsRecording(false);
      setIsAnalyzing(true);
      
      // Stop the recording and get the audio data
      const { audioBlob, audioBuffer } = await audioService.current.stopRecording();
      
      // Process the audio for raga detection
      const detectionResult = await ragaDetector.current.detectRaga(audioBuffer);
      
      // For now, use mock data if detection fails or in development
      const result = detectionResult || MOCK_RAGAS[0];
      
      setRagaResult({
        ...result,
        confidence: result.confidence || 0.87, // Default confidence if not provided
        matchingNotes: result.matchingNotes || result.aarohana, // Use aarohana if matchingNotes not available
      });
      
      setIsAnalyzing(false);
      return audioBlob;
    } catch (err) {
      console.error('Error stopping recording:', err);
      setError('Failed to process audio. Please try again.');
      setIsAnalyzing(false);
      return null;
    }
  }, [isRecording]);

  const resetDetection = useCallback(() => {
    setRagaResult(null);
    setError(null);
    setRecordingTime(0);
    setIsAnalyzing(false);
    
    // Stop any ongoing recording
    if (isRecording) {
      audioService.current?.stopRecording().catch(console.error);
      clearInterval(timerRef.current);
      setIsRecording(false);
    }
  }, [isRecording]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      resetDetection();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioService.current) {
        audioService.current.cleanup();
      }
    };
  }, [resetDetection]);

  const value = React.useMemo(() => ({
    isRecording,
    audioLevel,
    pitch,
    ragaResult,
    error,
    isAnalyzing,
    recordingTime,
    isInitialized,
    startRecording,
    stopRecording,
    resetDetection,
    // Don't expose internal services directly
  }), [
    isRecording,
    audioLevel,
    pitch,
    ragaResult,
    error,
    isAnalyzing,
    recordingTime,
    isInitialized,
    startRecording,
    stopRecording,
    resetDetection,
    // Remove from dependencies
  ]);

  return (
    <AudioContext.Provider value={value}>
      {children}
    </AudioContext.Provider>
  );
};

export const useAudio = () => {
  const context = useContext(AudioContext);
  
  if (context === undefined) {
    throw new Error('useAudio must be used within an AudioProvider');
  }
  
  return context;
};
