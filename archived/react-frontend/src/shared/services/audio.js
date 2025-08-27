import { Platform } from 'react-native';

export class AudioService {
  constructor({ onAudioProcess } = {}) {
    this.audioContext = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.sampleRate = 44100; // CD quality
    this.platform = Platform.OS;
    this.onAudioProcess = onAudioProcess || (() => {});
    this.analyser = null;
    this.scriptProcessor = null;
    this.microphone = null;
    this.audioStream = null;
  }

  async initialize() {
    try {
      if (this.platform === 'web') {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: this.sampleRate,
        });
      } else {
        // Native initialization
        const { AudioRecorder, AudioUtils } = require('react-native-audio');
        this.audioPath = `${AudioUtils.DocumentDirectoryPath}/raga_recording.wav`;
        
        const audioSet = {
          AudioEncoderAndroid: AudioEncoderAndroidType.AAC,
          AudioSourceAndroid: AudioSourceAndroidType.MIC,
          AudioIos: {
            Category: 'PlayAndRecord',
          },
          AudioIosCategory: 'PlayAndRecord',
        };
        
        await AudioRecorder.prepareRecordingAtPath(this.audioPath, audioSet);
      }
      return true;
    } catch (error) {
      console.error('Failed to initialize audio:', error);
      return false;
    }
  }

  async startRecording() {
    if (this.isRecording) return false;

    try {
      if (this.platform === 'web') {
        this.audioChunks = [];
        
        // Get audio stream from microphone
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: this.sampleRate
          } 
        });
        
        this.audioStream = stream;
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: this.sampleRate,
        });
        
        // Create media recorder for saving audio
        this.mediaRecorder = new MediaRecorder(stream);
        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            this.audioChunks.push(event.data);
          }
        };
        
        // Set up audio analysis
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 2048;
        
        // Create script processor for audio processing
        this.scriptProcessor = this.audioContext.createScriptProcessor(2048, 1, 1);
        
        // Connect audio nodes
        this.microphone = this.audioContext.createMediaStreamSource(stream);
        this.microphone.connect(this.analyser);
        this.analyser.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);
        
        // Process audio data
        this.scriptProcessor.onaudioprocess = () => {
          if (!this.analyser) return;
          
          const array = new Uint8Array(this.analyser.frequencyBinCount);
          this.analyser.getByteFrequencyData(array);
          
          // Calculate volume level (0-1)
          const volume = Math.sqrt(
            array.reduce((sum, value) => sum + value * value, 0) / array.length
          ) / 255;
          
          // Simple pitch detection (basic implementation)
          let maxVolume = 0;
          let maxIndex = 0;
          for (let i = 0; i < array.length; i++) {
            if (array[i] > maxVolume) {
              maxVolume = array[i];
              maxIndex = i;
            }
          }
          const pitch = maxIndex * (this.audioContext.sampleRate / 2) / array.length;
          
          // Call the audio process callback
          this.onAudioProcess({
            volume,
            pitch,
            frequencyData: array,
            sampleRate: this.audioContext.sampleRate
          });
        };
        
        // Start recording
        this.mediaRecorder.start();
      } else {
        // Native recording
        const { check, request, PERMISSIONS, RESULTS } = require('react-native-permissions');
        const permission = Platform.OS === 'ios' 
          ? PERMISSIONS.IOS.MICROPHONE 
          : PERMISSIONS.ANDROID.RECORD_AUDIO;

        const status = await check(permission);
        if (status !== RESULTS.GRANTED) {
          const requestStatus = await request(permission);
          if (requestStatus !== RESULTS.GRANTED) {
            throw new Error('Microphone permission not granted');
          }
        }

        await this.audioRecorder.startRecording();
      }

      this.isRecording = true;
      return true;
    } catch (error) {
      console.error('Error starting recording:', error);
      return false;
    }
  }

  async stopRecording() {
    if (!this.isRecording) return null;
    
    this.isRecording = false;

    try {
      if (this.platform === 'web') {
        return new Promise((resolve) => {
          if (!this.mediaRecorder) {
            resolve(null);
            return;
          }
          
          this.mediaRecorder.onstop = async () => {
            try {
              const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
              let audioBuffer = null;
              
              if (this.audioContext) {
                try {
                  audioBuffer = await this.blobToAudioBuffer(audioBlob);
                } catch (err) {
                  console.error('Error converting blob to audio buffer:', err);
                }
              }
              
              this.cleanup();
              resolve({ audioBlob, audioBuffer });
            } catch (err) {
              console.error('Error in mediaRecorder.onstop:', err);
              this.cleanup();
              resolve(null);
            }
          };
          
          try {
            this.mediaRecorder.stop();
            // Stop all tracks in the stream
            if (this.audioStream) {
              this.audioStream.getTracks().forEach(track => track.stop());
            }
          } catch (err) {
            console.error('Error stopping media recorder:', err);
            this.cleanup();
            resolve(null);
          }
        });
      } else {
        // Native stop recording
        const audioFile = await this.audioRecorder.stopRecording();
        return { audioPath: audioFile };
      }
  }

  async blobToAudioBuffer(blob) {
    if (this.platform !== 'web') {
      throw new Error('blobToAudioBuffer is only available on web');
    }
    const arrayBuffer = await blob.arrayBuffer();
    return this.audioContext.decodeAudioData(arrayBuffer);
  }
  
  cleanup() {
    // Disconnect and clean up audio nodes
    if (this.scriptProcessor) {
      try {
        this.scriptProcessor.disconnect();
      } catch (e) {
        console.warn('Error disconnecting script processor:', e);
      }
      this.scriptProcessor = null;
    }
    
    if (this.analyser) {
      try {
        this.analyser.disconnect();
      } catch (e) {
        console.warn('Error disconnecting analyser:', e);
      }
      this.analyser = null;
    }
    
    if (this.microphone) {
      try {
        this.microphone.disconnect();
      } catch (e) {
        console.warn('Error disconnecting microphone:', e);
      }
      this.microphone = null;
    }
    
    // Stop all tracks in the stream
    if (this.audioStream) {
      this.audioStream.getTracks().forEach(track => track.stop());
      this.audioStream = null;
    }
    
    // Close the audio context
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(console.error);
      this.audioContext = null;
    }
    
    // Clear media recorder
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
  }

  processAudio(audioBuffer) {
    // Platform-agnostic audio processing
    if (this.platform === 'web') {
      const leftChannel = audioBuffer.getChannelData(0);
      const rightChannel = audioBuffer.numberOfChannels > 1 
        ? audioBuffer.getChannelData(1) 
        : leftChannel;
      
      const monoData = new Float32Array(leftChannel.length);
      for (let i = 0; i < leftChannel.length; i++) {
        monoData[i] = (leftChannel[i] + rightChannel[i]) / 2;
      }

      return {
        sampleRate: audioBuffer.sampleRate,
        channelData: monoData,
        duration: audioBuffer.duration
      };
    } else {
      // Native audio processing
      // Implement native-specific processing if needed
      return {
        sampleRate: this.sampleRate,
        channelData: audioBuffer, // Assuming audioBuffer is already processed
        duration: audioBuffer.length / this.sampleRate
      };
    }
  }
}
