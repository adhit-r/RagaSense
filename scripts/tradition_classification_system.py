#!/usr/bin/env python3
"""
Tradition Classification System for RagaSense
============================================

This script implements the first phase of the hierarchical classification system:
1. Tradition Classification (Carnatic vs Hindustani)
2. Uses advanced audio features to distinguish between traditions
3. Implements cultural validation framework
4. Prepares for parent scale classification

Author: RagaSense AI Team
Date: 2024
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import librosa
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.feature_selection import SelectKBest, f_classif
    import joblib
    print("✅ All ML libraries imported successfully")
except ImportError as e:
    print(f"❌ Error importing ML libraries: {e}")
    print("Please install required packages: pip install -r ml/requirements_enhanced_training.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tradition_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TraditionFeatureExtractor:
    """Extract tradition-specific features for Carnatic vs Hindustani classification"""
    
    def __init__(self, sample_rate=22050, duration=30):
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_names = []
        
    def extract_tradition_features(self, audio_path: str) -> np.ndarray:
        """Extract features specifically designed to distinguish between traditions"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            features = []
            feature_names = []
            
            # 1. Ornamentation Features (Key differentiator)
            # Gamaka detection (Carnatic-specific)
            gamaka_features = self.detect_gamaka_patterns(y, sr)
            features.extend(gamaka_features)
            feature_names.extend(['gamaka_intensity', 'gamaka_frequency', 'gamaka_duration'])
            
            # Meend detection (Hindustani-specific)
            meend_features = self.detect_meend_patterns(y, sr)
            features.extend(meend_features)
            feature_names.extend(['meend_intensity', 'meend_frequency', 'meend_duration'])
            
            # 2. Microtonal Analysis
            # Shruti system analysis (22-shruti vs 12-note)
            shruti_features = self.analyze_shruti_system(y, sr)
            features.extend(shruti_features)
            feature_names.extend(['shruti_complexity', 'microtonal_variations', 'note_bending_intensity'])
            
            # 3. Rhythmic Patterns
            # Tala vs Taal differences
            tala_features = self.analyze_rhythmic_patterns(y, sr)
            features.extend(tala_features)
            feature_names.extend(['rhythm_complexity', 'beat_structure', 'tempo_variations'])
            
            # 4. Melodic Structure
            # Raga progression patterns
            melodic_features = self.analyze_melodic_structure(y, sr)
            features.extend(melodic_features)
            feature_names.extend(['melodic_progression', 'phrase_length', 'note_transitions'])
            
            # 5. Instrumental Characteristics
            # Timbre and playing style differences
            timbre_features = self.analyze_timbre_characteristics(y, sr)
            features.extend(timbre_features)
            feature_names.extend(['timbre_complexity', 'harmonic_content', 'attack_characteristics'])
            
            # 6. Performance Style
            # Alap vs Alapana differences
            performance_features = self.analyze_performance_style(y, sr)
            features.extend(performance_features)
            feature_names.extend(['alap_structure', 'improvisation_patterns', 'ornament_density'])
            
            # Store feature names
            self.feature_names = feature_names
            
            # Ensure all features are numeric
            features = [float(f) if not np.isnan(f) else 0.0 for f in features]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting tradition features from {audio_path}: {e}")
            return None
    
    def detect_gamaka_patterns(self, y, sr):
        """Detect Carnatic-specific gamaka patterns"""
        try:
            # Analyze pitch variations for gamaka characteristics
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Calculate gamaka intensity (pitch variation rate)
            pitch_variations = np.diff(pitches, axis=1)
            gamaka_intensity = np.mean(np.abs(pitch_variations))
            
            # Calculate gamaka frequency (how often variations occur)
            variation_threshold = 0.1
            gamaka_frequency = np.sum(np.abs(pitch_variations) > variation_threshold) / pitch_variations.size
            
            # Calculate gamaka duration (average length of variations)
            gamaka_duration = np.mean(np.sum(np.abs(pitch_variations) > variation_threshold, axis=1))
            
            return [gamaka_intensity, gamaka_frequency, gamaka_duration]
            
        except:
            return [0.0, 0.0, 0.0]
    
    def detect_meend_patterns(self, y, sr):
        """Detect Hindustani-specific meend (slide) patterns"""
        try:
            # Analyze pitch contours for sliding patterns
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Calculate meend intensity (smoothness of pitch changes)
            pitch_smoothness = np.mean(np.abs(np.diff(pitches, n=2, axis=1)))
            meend_intensity = 1.0 / (1.0 + pitch_smoothness)  # Higher = smoother
            
            # Calculate meend frequency (how often slides occur)
            slide_threshold = 0.05
            slides = np.sum(np.abs(np.diff(pitches, axis=1)) > slide_threshold)
            meend_frequency = slides / pitches.size
            
            # Calculate meend duration (average slide length)
            slide_lengths = []
            for i in range(pitches.shape[0]):
                slide_length = 0
                for j in range(pitches.shape[1] - 1):
                    if np.abs(pitches[i, j+1] - pitches[i, j]) > slide_threshold:
                        slide_length += 1
                    else:
                        if slide_length > 0:
                            slide_lengths.append(slide_length)
                            slide_length = 0
                if slide_length > 0:
                    slide_lengths.append(slide_length)
            
            meend_duration = np.mean(slide_lengths) if slide_lengths else 0.0
            
            return [meend_intensity, meend_frequency, meend_duration]
            
        except:
            return [0.0, 0.0, 0.0]
    
    def analyze_shruti_system(self, y, sr):
        """Analyze microtonal complexity (22-shruti vs 12-note system)"""
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Calculate microtonal complexity
            chroma_variance = np.var(chroma, axis=1)
            shruti_complexity = np.mean(chroma_variance)
            
            # Count microtonal variations
            microtonal_variations = np.sum(chroma_variance > np.mean(chroma_variance))
            
            # Analyze note bending
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            note_bending = np.mean(np.abs(np.diff(pitches, axis=1)))
            
            return [shruti_complexity, microtonal_variations, note_bending]
            
        except:
            return [0.0, 0.0, 0.0]
    
    def analyze_rhythmic_patterns(self, y, sr):
        """Analyze rhythmic complexity differences"""
        try:
            # Extract rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Calculate rhythm complexity
            rhythm_complexity = len(beats) / (len(y) / sr)  # Beats per second
            
            # Analyze beat structure
            beat_intervals = np.diff(beats)
            beat_structure = np.std(beat_intervals)
            
            # Analyze tempo variations
            tempo_variations = np.std(beat_intervals)
            
            return [rhythm_complexity, beat_structure, tempo_variations]
            
        except:
            return [0.0, 0.0, 0.0]
    
    def analyze_melodic_structure(self, y, sr):
        """Analyze melodic progression patterns"""
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Calculate melodic progression (how melody evolves over time)
            melodic_progression = np.mean(np.diff(mfcc, axis=1))
            
            # Calculate phrase length
            phrase_length = np.mean(np.sum(mfcc > np.mean(mfcc), axis=0))
            
            # Calculate note transitions
            note_transitions = np.mean(np.abs(np.diff(mfcc, axis=1)))
            
            return [melodic_progression, phrase_length, note_transitions]
            
        except:
            return [0.0, 0.0, 0.0]
    
    def analyze_timbre_characteristics(self, y, sr):
        """Analyze timbre and playing style differences"""
        try:
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Calculate timbre complexity
            timbre_complexity = np.std(spectral_centroids)
            
            # Calculate harmonic content
            harmonic_content = np.mean(spectral_rolloff)
            
            # Calculate attack characteristics
            attack_characteristics = np.mean(np.diff(spectral_centroids))
            
            return [timbre_complexity, harmonic_content, attack_characteristics]
            
        except:
            return [0.0, 0.0, 0.0]
    
    def analyze_performance_style(self, y, sr):
        """Analyze performance style differences"""
        try:
            # Extract features related to performance style
            
            # Alap structure analysis
            # Carnatic: More structured, Hindustani: More free-form
            energy = librosa.feature.rms(y=y)[0]
            energy_variations = np.std(energy)
            alap_structure = energy_variations
            
            # Improvisation patterns
            # Analyze how much the melody varies
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
            improvisation_patterns = np.std(mfcc)
            
            # Ornament density
            # Carnatic: Higher ornament density
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_variations = np.diff(pitches, axis=1)
            ornament_density = np.sum(np.abs(pitch_variations) > 0.1) / pitch_variations.size
            
            return [alap_structure, improvisation_patterns, ornament_density]
            
        except:
            return [0.0, 0.0, 0.0]

class TraditionClassifier:
    """Main tradition classification system"""
    
    def __init__(self):
        self.output_dir = Path("ml/tradition_classification")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = TraditionFeatureExtractor()
        self.models = {}
        self.results = {}
        
    def create_synthetic_tradition_dataset(self):
        """Create a synthetic dataset for tradition classification training"""
        logger.info("Creating synthetic tradition classification dataset...")
        
        # Generate synthetic features for Carnatic vs Hindustani
        carnatic_samples = []
        hindustani_samples = []
        
        # Carnatic characteristics: Higher gamaka, structured phrases, 22-shruti complexity
        for i in range(100):
            carnatic_features = [
                # Gamaka features (high)
                np.random.uniform(0.7, 1.0),  # gamaka_intensity
                np.random.uniform(0.6, 0.9),  # gamaka_frequency
                np.random.uniform(0.5, 0.8),  # gamaka_duration
                
                # Meend features (low)
                np.random.uniform(0.1, 0.4),  # meend_intensity
                np.random.uniform(0.1, 0.3),  # meend_frequency
                np.random.uniform(0.1, 0.3),  # meend_duration
                
                # Shruti features (high complexity)
                np.random.uniform(0.6, 1.0),  # shruti_complexity
                np.random.uniform(0.7, 1.0),  # microtonal_variations
                np.random.uniform(0.6, 0.9),  # note_bending_intensity
                
                # Rhythm features
                np.random.uniform(0.4, 0.7),  # rhythm_complexity
                np.random.uniform(0.3, 0.6),  # beat_structure
                np.random.uniform(0.2, 0.5),  # tempo_variations
                
                # Melodic features
                np.random.uniform(0.5, 0.8),  # melodic_progression
                np.random.uniform(0.6, 0.9),  # phrase_length
                np.random.uniform(0.5, 0.8),  # note_transitions
                
                # Timbre features
                np.random.uniform(0.4, 0.7),  # timbre_complexity
                np.random.uniform(0.5, 0.8),  # harmonic_content
                np.random.uniform(0.4, 0.7),  # attack_characteristics
                
                # Performance features
                np.random.uniform(0.6, 0.9),  # alap_structure
                np.random.uniform(0.5, 0.8),  # improvisation_patterns
                np.random.uniform(0.7, 1.0),  # ornament_density
            ]
            carnatic_samples.append(carnatic_features)
        
        # Hindustani characteristics: Higher meend, free-form alap, different ornamentation
        for i in range(100):
            hindustani_features = [
                # Gamaka features (low)
                np.random.uniform(0.1, 0.4),  # gamaka_intensity
                np.random.uniform(0.1, 0.3),  # gamaka_frequency
                np.random.uniform(0.1, 0.3),  # gamaka_duration
                
                # Meend features (high)
                np.random.uniform(0.7, 1.0),  # meend_intensity
                np.random.uniform(0.6, 0.9),  # meend_frequency
                np.random.uniform(0.5, 0.8),  # meend_duration
                
                # Shruti features (different complexity)
                np.random.uniform(0.3, 0.7),  # shruti_complexity
                np.random.uniform(0.4, 0.8),  # microtonal_variations
                np.random.uniform(0.3, 0.6),  # note_bending_intensity
                
                # Rhythm features
                np.random.uniform(0.6, 0.9),  # rhythm_complexity
                np.random.uniform(0.5, 0.8),  # beat_structure
                np.random.uniform(0.4, 0.7),  # tempo_variations
                
                # Melodic features
                np.random.uniform(0.3, 0.6),  # melodic_progression
                np.random.uniform(0.4, 0.7),  # phrase_length
                np.random.uniform(0.3, 0.6),  # note_transitions
                
                # Timbre features
                np.random.uniform(0.5, 0.8),  # timbre_complexity
                np.random.uniform(0.6, 0.9),  # harmonic_content
                np.random.uniform(0.5, 0.8),  # attack_characteristics
                
                # Performance features
                np.random.uniform(0.3, 0.6),  # alap_structure
                np.random.uniform(0.6, 0.9),  # improvisation_patterns
                np.random.uniform(0.4, 0.7),  # ornament_density
            ]
            hindustani_samples.append(hindustani_features)
        
        # Combine datasets
        X = np.array(carnatic_samples + hindustani_samples)
        y = ['Carnatic'] * 100 + ['Hindustani'] * 100
        
        logger.info(f"Created synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount([1 if label == 'Hindustani' else 0 for label in y])}")
        
        return X, y
    
    def train_tradition_classifiers(self, X, y):
        """Train multiple classifiers for tradition classification"""
        logger.info("Training tradition classification models...")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # Store results
                self.models[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Store preprocessing objects
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.feature_names = self.feature_extractor.feature_names
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def select_best_tradition_model(self):
        """Select the best performing tradition classification model"""
        if not self.models:
            raise ValueError("No models trained yet")
        
        # Find model with highest cross-validation score
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['cv_mean'])
        
        best_model = self.models[best_model_name]
        logger.info(f"Best tradition classification model: {best_model_name} with CV score: {best_model['cv_mean']:.4f}")
        
        return best_model_name, best_model
    
    def save_tradition_models(self, best_model_name):
        """Save the tradition classification models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best model
        best_model = self.models[best_model_name]['model']
        model_path = self.output_dir / f"tradition_classifier_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save preprocessing objects
        scaler_path = self.output_dir / "tradition_scaler.pkl"
        encoder_path = self.output_dir / "tradition_label_encoder.pkl"
        feature_names_path = self.output_dir / "tradition_feature_names.json"
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save latest model with fixed names
        latest_model_path = self.output_dir / "tradition_classifier.pkl"
        with open(latest_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save training results
        results_path = self.output_dir / "tradition_training_results.json"
        results = {
            'timestamp': timestamp,
            'best_model': best_model_name,
            'best_accuracy': self.models[best_model_name]['accuracy'],
            'best_cv_score': self.models[best_model_name]['cv_mean'],
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'all_models': {name: {
                'accuracy': model['accuracy'],
                'cv_mean': model['cv_mean'],
                'cv_std': model['cv_std']
            } for name, model in self.models.items()}
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Tradition classification models saved to {self.output_dir}")
        return model_path, results_path
    
    def run_tradition_classification(self):
        """Run the complete tradition classification pipeline"""
        logger.info("🚀 Starting Tradition Classification Pipeline")
        logger.info("=" * 50)
        
        try:
            # 1. Create synthetic dataset
            logger.info("Step 1: Creating synthetic tradition classification dataset...")
            X, y = self.create_synthetic_tradition_dataset()
            
            # 2. Train tradition classifiers
            logger.info("Step 2: Training tradition classification models...")
            X_train, X_test, y_train, y_test = self.train_tradition_classifiers(X, y)
            
            # 3. Select best model
            logger.info("Step 3: Selecting best tradition classification model...")
            best_model_name, best_model = self.select_best_tradition_model()
            
            # 4. Save models and results
            logger.info("Step 4: Saving tradition classification models...")
            model_path, results_path = self.save_tradition_models(best_model_name)
            
            # 5. Generate final report
            logger.info("Step 5: Generating final tradition classification report...")
            self.generate_tradition_report()
            
            logger.info("✅ Tradition Classification Pipeline completed successfully!")
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Accuracy: {best_model['accuracy']:.4f}")
            logger.info(f"Cross-validation: {best_model['cv_mean']:.4f} ± {best_model['cv_std']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Tradition classification pipeline failed: {e}")
            return False
    
    def generate_tradition_report(self):
        """Generate a comprehensive tradition classification report"""
        report_path = self.output_dir / "TRADITION_CLASSIFICATION_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Tradition Classification Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Accuracy | CV Score | CV Std |\n")
            f.write("|-------|----------|----------|--------|\n")
            
            for name, model in self.models.items():
                f.write(f"| {name} | {model['accuracy']:.4f} | {model['cv_mean']:.4f} | {model['cv_std']:.4f} |\n")
            
            f.write(f"\n## Feature Information\n\n")
            f.write(f"- **Total Features**: {len(self.feature_names)}\n")
            f.write(f"- **Feature Categories**:\n")
            f.write(f"  - Ornamentation (Gamaka vs Meend)\n")
            f.write(f"  - Microtonal Analysis (Shruti System)\n")
            f.write(f"  - Rhythmic Patterns (Tala vs Taal)\n")
            f.write(f"  - Melodic Structure (Raga Progression)\n")
            f.write(f"  - Timbre Characteristics\n")
            f.write(f"  - Performance Style (Alap vs Alapana)\n")
            
            f.write(f"\n## Cultural Validation Framework\n\n")
            f.write(f"### Key Differentiators:\n")
            f.write(f"1. **Gamaka Patterns**: Carnatic music uses more gamakas\n")
            f.write(f"2. **Meend Usage**: Hindustani music emphasizes meend (slides)\n")
            f.write(f"3. **Shruti Complexity**: Carnatic uses 22-shruti system\n")
            f.write(f"4. **Performance Structure**: Different alap/alapana approaches\n")
            f.write(f"5. **Ornamentation Density**: Carnatic has higher ornament density\n")
            
            f.write(f"\n## Next Steps\n\n")
            f.write(f"1. Integrate with real audio dataset\n")
            f.write(f"2. Implement parent scale classification\n")
            f.write(f"3. Add cultural expert validation\n")
            f.write(f"4. Deploy to production system\n")
        
        logger.info(f"Tradition classification report generated: {report_path}")

def main():
    """Main function to run the tradition classification pipeline"""
    print("🎵 Tradition Classification System for RagaSense")
    print("=" * 50)
    
    # Create and run tradition classification pipeline
    tradition_classifier = TraditionClassifier()
    success = tradition_classifier.run_tradition_classification()
    
    if success:
        print("\n✅ Tradition classification completed successfully!")
        print("📁 Check the 'ml/tradition_classification' directory for results")
        print("📊 View 'TRADITION_CLASSIFICATION_REPORT.md' for detailed analysis")
        print("\n🎯 Next Phase: Parent Scale Classification (Melakarta vs Thaat)")
    else:
        print("\n❌ Tradition classification failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
