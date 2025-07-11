import numpy as np
import librosa
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RagaClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_dim = 128  # Mel spectrogram features
        self.sample_rate = 22050
        self.duration = 30  # seconds per clip
        self.hop_length = 512
        self.n_mels = 128
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_mel_features(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram features from audio file."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Take mean across time to get feature vector
            features = np.mean(mel_spec_db, axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            raise
    
    def create_model(self, num_classes: int) -> keras.Model:
        """Create a neural network model for raga classification."""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.feature_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, audio_files: List[str], raga_labels: List[str], 
              validation_split: float = 0.2, epochs: int = 100) -> Dict:
        """Train the raga classifier model."""
        try:
            # Extract features from all audio files
            features = []
            for audio_file in audio_files:
                feat = self.extract_mel_features(audio_file)
                features.append(feat)
            
            X = np.array(features)
            y = np.array(raga_labels)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            y_categorical = keras.utils.to_categorical(y_encoded)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_categorical, test_size=validation_split, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Create and train model
            num_classes = len(self.label_encoder.classes_)
            self.model = self.create_model(num_classes)
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            return {
                'history': history.history,
                'num_classes': num_classes,
                'class_names': self.label_encoder.classes_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, audio_path: str) -> Dict:
        """Predict raga for a given audio file."""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Extract features
            features = self.extract_mel_features(audio_path)
            features_scaled = self.scaler.transform([features])
            
            # Get predictions
            predictions = self.model.predict(features_scaled)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_raga = self.label_encoder.classes_[predicted_class_idx]
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                top_predictions.append({
                    'raga': self.label_encoder.classes_[idx],
                    'confidence': float(predictions[0][idx])
                })
            
            return {
                'predicted_raga': predicted_raga,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'all_predictions': predictions[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error predicting raga: {str(e)}")
            raise
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing objects."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            self.model.save(f"{model_path}_model.h5")
            
            # Save preprocessing objects
            with open(f"{model_path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(f"{model_path}_label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing objects."""
        try:
            # Load model
            self.model = keras.models.load_model(f"{model_path}_model.h5")
            
            # Load preprocessing objects
            with open(f"{model_path}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f"{model_path}_label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_from_upload(self, uploaded_file_path: str) -> Dict:
        """Predict raga from uploaded audio file."""
        try:
            # Ensure the file is in a supported format
            supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            file_ext = os.path.splitext(uploaded_file_path)[1].lower()
            
            if file_ext not in supported_formats:
                raise ValueError(f"Unsupported audio format: {file_ext}")
            
            # Predict raga
            result = self.predict(uploaded_file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting from upload: {str(e)}")
            raise

# Utility function to create a simple model for testing
def create_simple_raga_classifier():
    """Create a simple raga classifier for testing purposes."""
    classifier = RagaClassifier()
    
    # Create a simple model with mock data for testing
    num_classes = 5  # Mock number of ragas
    classifier.model = classifier.create_model(num_classes)
    
    # Create mock preprocessing objects
    classifier.scaler = StandardScaler()
    classifier.label_encoder = LabelEncoder()
    classifier.label_encoder.classes_ = np.array(['Yaman', 'Bhairav', 'Khamaj', 'Kafi', 'Bhairavi'])
    
    return classifier 