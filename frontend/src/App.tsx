import React from 'react';
import { View, Text, StyleSheet } from 'react-lynx';
import RagaDetector from './components/RagaDetector';

export default function App() {
  return (
    <View style={styles.container}>
      <RagaDetector />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
});
