#!/usr/bin/env python3
"""
Proper Raga Detection with Deep Learning
Using CNN-LSTM and mel-spectrograms for real raga detection
"""

import os
import json
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available")
    TENSORFLOW_AVAILABLE = False

class ProperRagaDetector:
    """Proper raga detection using deep learning"""
    
    def __init__(self, sample_rate: int = 22050, duration: int = 30):
        self.sample_rate = sample_rate
        self.duration = duration
        self.model = None
        
    def extract_mel_spectrogram(self, audio_path: str, n_mels: int = 128, hop_length: int = 512) -> np.ndarray:
        """Extract mel-spectrogram from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=n_mels, 
                hop_length=hop_length,
                n_fft=2048
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"❌ Error extracting mel-spectrogram: {e}")
            return None
    
    def extract_chroma_sequence(self, audio_path: str, hop_length: int = 512) -> np.ndarray:
        """Extract chroma sequence over time"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract chroma features over time
            chroma = librosa.feature.chroma_stft(
                y=y, 
                sr=sr, 
                hop_length=hop_length,
                n_fft=2048
            )
            
            return chroma
            
        except Exception as e:
            print(f"❌ Error extracting chroma: {e}")
            return None
    
    def extract_mfcc_sequence(self, audio_path: str, n_mfcc: int = 13, hop_length: int = 512) -> np.ndarray:
        """Extract MFCC sequence over time"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract MFCC features over time
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=n_mfcc, 
                hop_length=hop_length,
                n_fft=2048
            )
            
            return mfcc
            
        except Exception as e:
            print(f"❌ Error extracting MFCC: {e}")
            return None
    
    def build_cnn_lstm_model(self, input_shape: Tuple, num_classes: int) -> keras.Model:
        """Build CNN-LSTM hybrid model for raga detection"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        # Input layer
        input_layer = layers.Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Reshape for LSTM
        # Convert spatial features to temporal sequence
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1] * tf.shape(x)[2]
        features = tf.shape(x)[3]
        
        x = layers.Reshape((time_steps, features))(x)
        
        # LSTM layers for temporal modeling
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers for classification
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=x)
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple, num_classes: int) -> keras.Model:
        """Build Transformer model for raga detection"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        # Input layer
        input_layer = layers.Input(shape=input_shape)
        
        # Reshape for transformer (sequence of feature vectors)
        x = layers.Reshape((input_shape[0] * input_shape[1], input_shape[2]))(input_layer)
        
        # Positional encoding
        pos_encoding = self.positional_encoding(x.shape[1], x.shape[2])
        x = x + pos_encoding
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=64
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(256, activation='relu')(x)
        ffn_output = layers.Dense(x.shape[2])(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=x)
        
        return model
    
    def positional_encoding(self, position, d_model):
        """Generate positional encoding for transformer"""
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        """Get angles for positional encoding"""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def prepare_proper_dataset(self, audio_files: List[Dict], output_dir: str = "proper_ml_data"):
        """Prepare dataset with proper audio features"""
        print("🔄 Preparing proper dataset with deep learning features...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        mel_specs = []
        chroma_seqs = []
        mfcc_seqs = []
        labels = []
        filenames = []
        
        for audio_file in audio_files:
            audio_path = audio_file.get('path')
            filename = audio_file.get('filename', '')
            
            if not audio_path or not os.path.exists(audio_path):
                continue
            
            print(f"Processing: {filename}")
            
            # Extract features
            mel_spec = self.extract_mel_spectrogram(audio_path)
            chroma_seq = self.extract_chroma_sequence(audio_path)
            mfcc_seq = self.extract_mfcc_sequence(audio_path)
            
            if mel_spec is not None and chroma_seq is not None and mfcc_seq is not None:
                mel_specs.append(mel_spec)
                chroma_seqs.append(chroma_seq)
                mfcc_seqs.append(mfcc_seq)
                labels.append(filename)  # For now, use filename as label
                filenames.append(filename)
        
        # Convert to numpy arrays
        mel_specs = np.array(mel_specs)
        chroma_seqs = np.array(chroma_seqs)
        mfcc_seqs = np.array(mfcc_seqs)
        
        # Save features
        np.save(output_path / "mel_spectrograms.npy", mel_specs)
        np.save(output_path / "chroma_sequences.npy", chroma_seqs)
        np.save(output_path / "mfcc_sequences.npy", mfcc_seqs)
        
        # Save metadata
        metadata = {
            "filenames": filenames,
            "labels": labels,
            "mel_specs_shape": mel_specs.shape,
            "chroma_seqs_shape": chroma_seqs.shape,
            "mfcc_seqs_shape": mfcc_seqs.shape
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Prepared dataset:")
        print(f"   Mel-spectrograms: {mel_specs.shape}")
        print(f"   Chroma sequences: {chroma_seqs.shape}")
        print(f"   MFCC sequences: {mfcc_seqs.shape}")
        print(f"   Samples: {len(filenames)}")
        
        return mel_specs, chroma_seqs, mfcc_seqs, labels
    
    def train_proper_model(self, mel_specs: np.ndarray, labels: List[str], model_type: str = "cnn_lstm"):
        """Train proper deep learning model"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        print(f"🔄 Training {model_type} model...")
        
        # Prepare data
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            mel_specs, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Add channel dimension for CNN
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {num_classes}")
        
        # Build model
        if model_type == "cnn_lstm":
            model = self.build_cnn_lstm_model(X_train.shape[1:], num_classes)
        elif model_type == "transformer":
            model = self.build_transformer_model(X_train.shape[1:], num_classes)
        else:
            print(f"❌ Unknown model type: {model_type}")
            return None
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_proper_model.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Test accuracy: {test_acc:.4f}")
        
        return model, history, label_encoder

def main():
    """Main function to demonstrate proper raga detection"""
    print("🎵 Proper Raga Detection with Deep Learning")
    print("="*50)
    
    # Initialize detector
    detector = ProperRagaDetector()
    
    # Load audio files
    with open("new_ml_data/metadata/our_audio_data.json", 'r') as f:
        audio_data = json.load(f)
    
    audio_files = audio_data.get('audio_files', [])
    
    if not audio_files:
        print("❌ No audio files found")
        return
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Prepare proper dataset
    print("\n" + "="*50)
    print("🔧 PREPARING PROPER DATASET")
    print("="*50)
    
    mel_specs, chroma_seqs, mfcc_seqs, labels = detector.prepare_proper_dataset(audio_files)
    
    if len(mel_specs) == 0:
        print("❌ No features extracted")
        return
    
    # Train proper model
    print("\n" + "="*50)
    print("🧠 TRAINING PROPER MODEL")
    print("="*50)
    
    model, history, label_encoder = detector.train_proper_model(mel_specs, labels, "cnn_lstm")
    
    if model is not None:
        print("\n" + "="*50)
        print("✅ PROPER MODEL TRAINED")
        print("="*50)
        
        print("🎯 This approach is much better because:")
        print("   ✅ Uses mel-spectrograms (time-frequency representation)")
        print("   ✅ CNN-LSTM captures both spatial and temporal patterns")
        print("   ✅ Processes audio as sequences, not static features")
        print("   ✅ Can learn raga-specific melodic patterns")
        print("   ✅ More suitable for audio classification")
        
        print("\n🚀 Next steps for real raga detection:")
        print("   1. Get properly labeled raga audio files")
        print("   2. Use longer audio segments (2-5 minutes)")
        print("   3. Include multiple artists and instruments")
        print("   4. Add raga-specific features (arohana/avarohana)")
        print("   5. Use attention mechanisms for key phrases")
        print("   6. Validate with expert musicians")

if __name__ == "__main__":
    main()



