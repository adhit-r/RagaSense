from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import tempfile
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='build', static_url_path='/')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Raga definitions with characteristics
RAGA_INFO = {
    'Yaman': {
        'aroha': ['Sa', 'Re', 'Ga', 'Ma#', 'Pa', 'Dha', 'Ni'],
        'avaroha': ['Sa', 'Ni', 'Dha', 'Pa', 'Ma#', 'Ga', 'Re', 'Sa'],
        'vadi': 'Ga',
        'samvadi': 'Ni',
        'time': 'Evening',
        'mood': 'Romantic, Devotional'
    },
    'Bhairav': {
        'aroha': ['Sa', 'Re♭', 'Ga', 'Ma', 'Pa', 'Dha♭', 'Ni'],
        'avaroha': ['Sa', 'Ni', 'Dha♭', 'Pa', 'Ma', 'Ga', 'Re♭', 'Sa'],
        'vadi': 'Dha♭',
        'samvadi': 'Re♭',
        'time': 'Early Morning',
        'mood': 'Serious, Devotional'
    },
    'Malkauns': {
        'aroha': ['Sa', 'Ga♭', 'Ma', 'Dha♭', 'Sa'],
        'avaroha': ['Sa', 'Dha♭', 'Ma', 'Ga♭', 'Sa'],
        'vadi': 'Ma',
        'samvadi': 'Sa',
        'time': 'Late Evening',
        'mood': 'Meditative, Peaceful'
    },
    'Darbari': {
        'aroha': ['Sa', 'Re', 'Ga♭', 'Ma', 'Pa', 'Dha♭', 'Ni♭'],
        'avaroha': ['Sa', 'Ni♭', 'Dha♭', 'Pa', 'Ma', 'Ga♭', 'Re', 'Sa'],
        'vadi': 'Re',
        'samvadi': 'Pa',
        'time': 'Late Night',
        'mood': 'Majestic, Royal'
    },
    'Bhoopali': {
        'aroha': ['Sa', 'Re', 'Ga', 'Pa', 'Dha'],
        'avaroha': ['Sa', 'Dha', 'Pa', 'Ga', 'Re', 'Sa'],
        'vadi': 'Ga',
        'samvadi': 'Dha',
        'time': 'Evening',
        'mood': 'Peaceful, Contemplative'
    }
}

class RagaClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def extract_features(self, audio_path, duration=30):
        """Extract comprehensive audio features for raga classification"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=duration)
            
            # Basic features
            features = []
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # 2. MFCC features (important for timbral characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 3. Chroma features (important for pitch/harmony analysis)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 4. Pitch and harmony features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # 5. Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Aggregate features
            features.extend([
                np.mean(spectral_centroids), np.var(spectral_centroids),
                np.mean(spectral_rolloff), np.var(spectral_rolloff),
                np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
                tempo
            ])
            
            # MFCC statistics
            for mfcc in mfccs:
                features.extend([np.mean(mfcc), np.var(mfcc)])
            
            # Chroma statistics
            for chroma_bin in chroma:
                features.extend([np.mean(chroma_bin), np.var(chroma_bin)])
            
            # Pitch statistics
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features.extend([
                    np.mean(pitch_values), np.var(pitch_values),
                    np.median(pitch_values), np.std(pitch_values)
                ])
            else:
                features.extend([0, 0, 0, 0])
                
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(67)  # Return zero features if extraction fails
    
    def create_model(self, input_shape):
        """Create CNN-LSTM hybrid model for raga classification"""
        model = Sequential([
            # Reshape for CNN
            tf.keras.layers.Reshape((input_shape, 1)),
            
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # LSTM layers for temporal patterns
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            
            # Dense layers for classification
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dense(len(RAGA_INFO), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for demonstration"""
        print("Generating synthetic training data...")
        
        X = []
        y = []
        raga_names = list(RAGA_INFO.keys())
        
        for i, raga in enumerate(raga_names):
            for _ in range(n_samples // len(raga_names)):
                # Generate synthetic features based on raga characteristics
                features = np.random.normal(0, 1, 67)
                
                # Add raga-specific patterns
                if raga == 'Yaman':
                    features[0:5] += np.random.normal(0.5, 0.2, 5)  # Higher spectral features
                elif raga == 'Bhairav':
                    features[5:10] += np.random.normal(-0.3, 0.2, 5)  # Lower spectral features
                elif raga == 'Malkauns':
                    features[10:15] += np.random.normal(0.2, 0.1, 5)  # Moderate features
                elif raga == 'Darbari':
                    features[15:20] += np.random.normal(-0.1, 0.3, 5)  # Variable features
                elif raga == 'Bhoopali':
                    features[20:25] += np.random.normal(0.3, 0.15, 5)  # Consistent features
                
                X.append(features)
                y.append(i)
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the raga classification model"""
        print("Training raga classification model...")
        
        # Generate synthetic data
        X, y = self.generate_synthetic_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self.create_model(X_train_scaled.shape[1])
        
        # Train with early stopping
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        self.is_trained = True
        return history
    
    def predict_raga(self, audio_path):
        """Predict raga from audio file"""
        if not self.is_trained:
            self.train_model()
        
        # Extract features
        features = self.extract_features(audio_path)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        predictions = self.model.predict(features_scaled, verbose=0)[0]
        
        # Get top predictions
        raga_names = list(RAGA_INFO.keys())
        results = []
        
        for i, prob in enumerate(predictions):
            results.append({
                'raga': raga_names[i],
                'confidence': float(prob * 100),
                'info': RAGA_INFO[raga_names[i]]
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results

# Initialize classifier
classifier = RagaClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the audio file
            results = classifier.predict_raga(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predictions': results,
                'filename': filename
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_trained': classifier.is_trained,
        'supported_ragas': list(RAGA_INFO.keys())
    })

if __name__ == '__main__':
    print("Starting Raga Detection Server...")
    print("Supported Ragas:", list(RAGA_INFO.keys()))
    app.run(debug=True, host='0.0.0.0', port=5000)