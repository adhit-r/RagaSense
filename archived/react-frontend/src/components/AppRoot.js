import { Component } from '@lynx-js/core';
import { View, Text, Button, StyleSheet } from 'react-native';

export class AppRoot extends Component {
  static css = StyleSheet.create({
    container: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: '#121212',
    },
    title: {
      fontSize: 28,
      fontWeight: 'bold',
      color: '#FFFFFF',
      marginBottom: 40,
    },
    button: {
      backgroundColor: '#6200EE',
      padding: 15,
      borderRadius: 25,
      width: 200,
      margin: 10,
    },
    buttonText: {
      color: '#FFFFFF',
      textAlign: 'center',
      fontSize: 18,
    },
    resultContainer: {
      marginTop: 40,
      padding: 20,
      backgroundColor: '#1E1E1E',
      borderRadius: 10,
      width: '80%',
    },
    resultText: {
      color: '#FFFFFF',
      fontSize: 18,
      marginBottom: 10,
    },
  });

  state = {
    isRecording: false,
    isProcessing: false,
    result: null,
    error: null,
  };

  handleRecord = async () => {
    try {
      if (this.state.isRecording) {
        this.setState({ isProcessing: true });
        await this.props.audioService.stopRecording();
        this.analyzeAudio();
      } else {
        await this.props.audioService.startRecording();
      }
      this.setState({ isRecording: !this.state.isRecording });
    } catch (error) {
      console.error('Recording error:', error);
      this.setState({ error: 'Failed to access microphone' });
    }
  };

  analyzeAudio = async () => {
    try {
      const { audioBuffer } = await this.props.audioService.stopRecording();
      const processedAudio = this.props.audioService.processAudio(audioBuffer);
      
      const result = await this.props.ragaDetector.detectRaga(processedAudio);
      
      if (result.success) {
        this.setState({
          result: result.results[0], // Get top result
          isProcessing: false,
        });
      } else {
        throw new Error('Analysis failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      this.setState({
        error: 'Failed to analyze audio',
        isProcessing: false,
      });
    }
  };

  render() {
    const { isRecording, isProcessing, result, error } = this.state;
    const buttonText = isRecording ? 'Stop Recording' : 'Start Raga Detection';
    const buttonColor = isRecording ? '#FF3D00' : '#6200EE';

    return (
      <View style={AppRoot.styles.container}>
        <Text style={AppRoot.styles.title}>Raga Detect</Text>
        
        <View style={[AppRoot.styles.button, { backgroundColor: buttonColor }]}>
          <Button
            title={isProcessing ? 'Analyzing...' : buttonText}
            onPress={this.handleRecord}
            disabled={isProcessing}
            color="#FFFFFF"
          />
        </View>

        {error && (
          <Text style={{ color: '#FF5252', marginTop: 20 }}>{error}</Text>
        )}

        {result && (
          <View style={AppRoot.styles.resultContainer}>
            <Text style={[AppRoot.styles.resultText, { fontSize: 24, fontWeight: 'bold' }]}>
              Detected Raga: {result.raga}
            </Text>
            <Text style={AppRoot.styles.resultText}>
              Confidence: {Math.round(result.confidence * 100)}%
            </Text>
            <Text style={AppRoot.styles.resultText}>
              Mood: {result.mood}
            </Text>
            <Text style={AppRoot.styles.resultText}>
              Best time: {result.time}
            </Text>
            <Text style={[AppRoot.styles.resultText, { marginTop: 10 }]}>
              Matching notes: {result.matchingNotes.join(', ')}
            </Text>
          </View>
        )}
      </View>
    );
  }
}

// Connect to app services
export default (app) => ({
  audioService: app.audioService,
  ragaDetector: app.ragaDetector,
});
