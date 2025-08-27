import React, { useState, useRef, useCallback } from 'react';
import { View, Text, TouchableOpacity, TextInput, Image, ScrollView, StyleSheet } from 'react-lynx';

interface RagaPrediction {
  raga: string;
  probability: number;
  confidence: string;
}

interface DetectionResult {
  success: boolean;
  predicted_raga?: string;
  confidence?: number;
  top_predictions?: RagaPrediction[];
  supported_ragas?: string[];
  error?: string;
}

export default function RagaDetector() {
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    if (!file) return;

    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/m4a'];
    if (!allowedTypes.includes(file.type)) {
      setResult({
        success: false,
        error: 'Unsupported file type. Please upload WAV, MP3, OGG, FLAC, or M4A files.'
      });
      return;
    }

    setIsUploading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('audio', file);

      const response = await fetch('/api/ragas/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);
      } else {
        setResult({
          success: false,
          error: data.detail || 'Failed to detect raga'
        });
      }
    } catch (error) {
      setResult({
        success: false,
        error: 'Network error. Please try again.'
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (e: Event) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      handleFileUpload(target.files[0]);
    }
  };

  const startListening = () => {
    setIsListening(true);
    // TODO: Implement microphone recording
    setTimeout(() => {
      setIsListening(false);
      // Simulate detection result
      setResult({
        success: true,
        predicted_raga: 'Yaman',
        confidence: 0.85,
        top_predictions: [
          { raga: 'Yaman', probability: 0.85, confidence: 'High' },
          { raga: 'Bhairav', probability: 0.10, confidence: 'Low' },
          { raga: 'Kafi', probability: 0.05, confidence: 'Low' }
        ],
        supported_ragas: ['Yaman', 'Bhairav', 'Kafi']
      });
    }, 3000);
  };

  const resetDetection = () => {
    setResult(null);
    setIsUploading(false);
    setIsListening(false);
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.logoContainer}>
          <Text style={styles.logoText}>🎵</Text>
        </View>
        <Text style={styles.title}>Raga Recognition</Text>
        <Text style={styles.subtitle}>
          Discover the raga in Indian classical music. Upload an audio file or record live to identify the raga instantly.
        </Text>
      </View>

      {/* Main Content */}
      <View style={styles.mainContent}>
        {!result ? (
          <View style={styles.uploadInterface}>
            {/* Upload Area */}
            <TouchableOpacity 
              style={[styles.uploadArea, isUploading && styles.uploadAreaDisabled]}
              onPress={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              <View style={styles.uploadIcon}>
                {isUploading ? (
                  <Text style={styles.loadingText}>⏳</Text>
                ) : (
                  <Text style={styles.uploadText}>📁</Text>
                )}
              </View>
              
              <Text style={styles.uploadTitle}>
                {isUploading ? 'Analyzing...' : 'Upload Audio File'}
              </Text>
              <Text style={styles.uploadDescription}>
                {isUploading 
                  ? 'Processing your audio file to detect the raga'
                  : 'Tap to select an audio file'
                }
              </Text>
              {!isUploading && (
                <Text style={styles.uploadHint}>
                  Supports WAV, MP3, OGG, FLAC, M4A • Max 30 seconds
                </Text>
              )}
            </TouchableOpacity>

            {/* Divider */}
            <View style={styles.divider}>
              <Text style={styles.dividerText}>or</Text>
            </View>

            {/* Record Button */}
            <TouchableOpacity
              style={[styles.recordButton, isListening && styles.recordButtonActive]}
              onPress={startListening}
              disabled={isListening}
            >
              <Text style={styles.recordButtonText}>
                {isListening ? '🎤 Listening...' : '🎤 Record Live'}
              </Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.resultsInterface}>
            {result.success ? (
              <>
                {/* Success Result */}
                <View style={styles.successContainer}>
                  <View style={styles.successIcon}>
                    <Text style={styles.successText}>✅</Text>
                  </View>
                  
                  <Text style={styles.successTitle}>Raga Detected!</Text>
                  
                  <View style={styles.resultCard}>
                    <Text style={styles.resultRaga}>{result.predicted_raga}</Text>
                    <Text style={styles.resultConfidence}>
                      Confidence: {(result.confidence! * 100).toFixed(1)}%
                    </Text>
                  </View>
                </View>

                {/* Top Predictions */}
                {result.top_predictions && result.top_predictions.length > 0 && (
                  <View style={styles.predictionsContainer}>
                    <Text style={styles.predictionsTitle}>All Predictions</Text>
                    {result.top_predictions.map((prediction, index) => (
                      <View key={index} style={[styles.predictionItem, index === 0 && styles.predictionItemFirst]}>
                        <View style={styles.predictionRank}>
                          <Text style={styles.predictionRankText}>{index + 1}</Text>
                        </View>
                        <View style={styles.predictionInfo}>
                          <Text style={styles.predictionRaga}>{prediction.raga}</Text>
                          <Text style={styles.predictionConfidence}>{prediction.confidence} confidence</Text>
                        </View>
                        <Text style={styles.predictionProbability}>
                          {(prediction.probability * 100).toFixed(1)}%
                        </Text>
                      </View>
                    ))}
                  </View>
                )}

                {/* Supported Ragas */}
                {result.supported_ragas && (
                  <View style={styles.supportedContainer}>
                    <Text style={styles.supportedTitle}>Supported Ragas</Text>
                    <View style={styles.supportedTags}>
                      {result.supported_ragas.map((raga, index) => (
                        <View key={index} style={styles.supportedTag}>
                          <Text style={styles.supportedTagText}>{raga}</Text>
                        </View>
                      ))}
                    </View>
                  </View>
                )}
              </>
            ) : (
              /* Error Result */
              <View style={styles.errorContainer}>
                <View style={styles.errorIcon}>
                  <Text style={styles.errorText}>❌</Text>
                </View>
                
                <Text style={styles.errorTitle}>Detection Failed</Text>
                
                <View style={styles.errorMessage}>
                  <Text style={styles.errorMessageText}>{result.error}</Text>
                </View>
              </View>
            )}

            {/* Action Buttons */}
            <View style={styles.actionButtons}>
              <TouchableOpacity style={styles.tryAgainButton} onPress={resetDetection}>
                <Text style={styles.tryAgainButtonText}>🔄 Try Again</Text>
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.learnMoreButton}>
                <Text style={styles.learnMoreButtonText}>📚 Learn More</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>

      {/* Footer Info */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Powered by advanced machine learning • Currently supporting 3 ragas
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#8b5cf6',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  logoText: {
    fontSize: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    color: '#6b7280',
    textAlign: 'center',
    lineHeight: 24,
  },
  mainContent: {
    flex: 1,
    paddingHorizontal: 20,
  },
  uploadInterface: {
    gap: 32,
  },
  uploadArea: {
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#d1d5db',
    borderRadius: 16,
    padding: 48,
    alignItems: 'center',
    backgroundColor: '#ffffff',
  },
  uploadAreaDisabled: {
    opacity: 0.7,
  },
  uploadIcon: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: '#f3e8ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  uploadText: {
    fontSize: 48,
  },
  loadingText: {
    fontSize: 48,
  },
  uploadTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  uploadDescription: {
    fontSize: 16,
    color: '#6b7280',
    marginBottom: 16,
    textAlign: 'center',
  },
  uploadHint: {
    fontSize: 14,
    color: '#9ca3af',
  },
  divider: {
    alignItems: 'center',
  },
  dividerText: {
    backgroundColor: '#ffffff',
    paddingHorizontal: 16,
    color: '#9ca3af',
    fontSize: 14,
  },
  recordButton: {
    backgroundColor: '#8b5cf6',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 16,
    alignItems: 'center',
  },
  recordButtonActive: {
    backgroundColor: '#ef4444',
  },
  recordButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
  resultsInterface: {
    gap: 32,
  },
  successContainer: {
    alignItems: 'center',
  },
  successIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#10b981',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  successText: {
    fontSize: 40,
  },
  successTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  resultCard: {
    backgroundColor: '#8b5cf6',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 16,
    alignItems: 'center',
  },
  resultRaga: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  resultConfidence: {
    fontSize: 16,
    color: '#ffffff',
    opacity: 0.9,
  },
  predictionsContainer: {
    gap: 12,
  },
  predictionsTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 16,
  },
  predictionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e5e7eb',
    backgroundColor: '#f9fafb',
  },
  predictionItemFirst: {
    borderColor: '#e9d5ff',
    backgroundColor: '#faf5ff',
  },
  predictionRank: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#d1d5db',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  predictionRankText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#374151',
  },
  predictionInfo: {
    flex: 1,
  },
  predictionRaga: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  predictionConfidence: {
    fontSize: 14,
    color: '#6b7280',
  },
  predictionProbability: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
  },
  supportedContainer: {
    gap: 16,
  },
  supportedTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1f2937',
  },
  supportedTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  supportedTag: {
    backgroundColor: '#f3e8ff',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  supportedTagText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#7c3aed',
  },
  errorContainer: {
    alignItems: 'center',
  },
  errorIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#ef4444',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  errorText: {
    fontSize: 40,
  },
  errorTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  errorMessage: {
    backgroundColor: '#fef2f2',
    borderWidth: 1,
    borderColor: '#fecaca',
    borderRadius: 16,
    padding: 24,
    maxWidth: 320,
  },
  errorMessageText: {
    color: '#dc2626',
    textAlign: 'center',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
  },
  tryAgainButton: {
    backgroundColor: '#f3f4f6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
  },
  tryAgainButtonText: {
    color: '#374151',
    fontSize: 16,
    fontWeight: '600',
  },
  learnMoreButton: {
    backgroundColor: '#8b5cf6',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
  },
  learnMoreButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  footer: {
    paddingVertical: 24,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
  },
});
