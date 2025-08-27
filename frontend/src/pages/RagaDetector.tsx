import React, { useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import toast from 'react-hot-toast';
import { Upload, Mic, Loader2, X, Play, Pause, RotateCcw } from 'lucide-react';
import { useAudioPlayer } from '../hooks/use-audio-player';
import { detectRaga, RagaDetectionResult } from '../api/ragas';

// Type for the prediction results
interface RagaPrediction {
  raga: string;
  probability: number;
  info?: {
    aroha?: string[];
    avaroha?: string[];
    time?: string;
    mood?: string;
  };
}

export function RagaDetector() {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<RagaDetectionResult | null>(null);
  const [progress, setProgress] = useState(0);
  // Audio player hook
  const { 
    audioUrl,
    isPlaying, 
    currentTime, 
    duration, 
    play, 
    pause, 
    reset,
    setAudioUrl,
    setCurrentTime
  } = useAudioPlayer();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.ogg'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (file) {
        setFile(file);
        setAudioUrl(URL.createObjectURL(file));
        setResults(null);
      }
    },
  });

  const analyzeAudio = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    setProgress(0);
    
    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return 90;
        return prev + 10;
      });
    }, 500);
    
    try {
      const formData = new FormData();
      formData.append('audio', file);
      
      const result = await detectRaga(formData, (progressEvent: ProgressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
        setProgress(progress > 90 ? 90 : progress);
      });
      
      setResults(result);
      setProgress(100);
      
      if (result.predictions && result.predictions.length > 0) {
        toast.success(`Detected raga: ${result.predictions[0].raga}`);
      } else {
        toast.success('Analysis complete, but no raga detected');
      }
    } catch (error) {
      console.error('Error analyzing audio:', error);
      toast.error("There was an error analyzing the audio file. Please try again.");
    } finally {
      clearInterval(interval);
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResults(null);
    setProgress(0);
    reset();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Raga Detector</h1>
          <p className="text-muted-foreground">Upload an audio file to detect the raga</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={handleReset} disabled={!file}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          <Button 
            onClick={analyzeAudio} 
            disabled={!file || isAnalyzing}
            className="w-full md:w-auto"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze Audio'
            )}
          </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Upload Audio</CardTitle>
            <CardDescription>
              Upload an audio file or record directly from your microphone
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50'
              }`}
            >
              <input {...getInputProps()} />
              {file ? (
                <div className="space-y-2">
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                    <Upload className="h-6 w-6 text-primary" />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {isDragActive
                      ? 'Drop the audio file here'
                      : 'Drag & drop an audio file here, or click to select'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Supports MP3, WAV, M4A, OGG (max 10MB)
                  </p>
                </div>
              )}
            </div>

            {file && (
              <div className="mt-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={isPlaying ? pause : play}
                      disabled={!audioUrl}
                    >
                      {isPlaying ? (
                        <Pause className="h-4 w-4" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      {formatTime(currentTime)} / {formatTime(duration)}
                    </span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {Math.round((currentTime / (duration || 1)) * 100)}% played
                  </div>
                </div>
                <Progress
                  value={(currentTime / (duration || 1)) * 100}
                  className="h-2"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const pos = (e.clientX - rect.left) / rect.width;
                    setCurrentTime(pos * duration);
                  }}
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>
              {results
                ? 'Here are the detected ragas'
                : 'Upload and analyze an audio file to see results'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isAnalyzing ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Analyzing audio...</span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
                <div className="grid grid-cols-3 gap-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-4 bg-muted rounded animate-pulse" />
                  ))}
                </div>
              </div>
            ) : results ? (
              <div className="space-y-4">
                {results.predictions.map((prediction: any, index: number) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg border ${
                      index === 0
                        ? 'bg-primary/5 border-primary/20'
                        : 'bg-muted/50 border-muted'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium">
                          {index + 1}. {prediction.raga}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          Confidence: {(prediction.probability * 100).toFixed(1)}%
                        </p>
                      </div>
                      {index === 0 && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                          Most Likely
                        </span>
                      )}
                    </div>
                    
                    {index === 0 && prediction.info && (
                      <div className="mt-3 pt-3 border-t border-border">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <p className="text-muted-foreground">Aroha</p>
                            <p>{prediction.info.aroha?.join(' ') || 'N/A'}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Avaroha</p>
                            <p>{prediction.info.avaroha?.join(' ') || 'N/A'}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Time</p>
                            <p>{prediction.info.time || 'N/A'}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Mood</p>
                            <p>{prediction.info.mood || 'N/A'}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="rounded-full bg-muted p-4 mb-4">
                  <Mic className="h-8 w-8 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-medium">No analysis results</h3>
                <p className="text-sm text-muted-foreground">
                  Upload and analyze an audio file to see the detected ragas
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
