import { Audio } from '@lynx-js/audio';

export class AudioService {
  constructor() {
    this.audioContext = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.sampleRate = 44100; // CD quality
  }

  async initialize() {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate,
      });
      return true;
    } catch (error) {
      console.error('Failed to initialize audio context:', error);
      return false;
    }
  }

  async startRecording() {
    if (this.isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      return true;
    } catch (error) {
      console.error('Error starting recording:', error);
      return false;
    }
  }

  async stopRecording() {
    if (!this.isRecording) return null;

    return new Promise((resolve) => {
      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const audioBuffer = await this.blobToAudioBuffer(audioBlob);
        this.isRecording = false;
        resolve({ audioBlob, audioBuffer });
      };

      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
    });
  }

  async blobToAudioBuffer(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    return this.audioContext.decodeAudioData(arrayBuffer);
  }

  // Process audio data for raga detection
  processAudio(audioBuffer) {
    // Convert audio buffer to mono for analysis
    const leftChannel = audioBuffer.getChannelData(0);
    const rightChannel = audioBuffer.numberOfChannels > 1 
      ? audioBuffer.getChannelData(1) 
      : leftChannel;
    
    // Mix down to mono
    const monoData = new Float32Array(leftChannel.length);
    for (let i = 0; i < leftChannel.length; i++) {
      monoData[i] = (leftChannel[i] + rightChannel[i]) / 2;
    }

    return {
      sampleRate: audioBuffer.sampleRate,
      channelData: monoData,
      duration: audioBuffer.duration
    };
  }
}
