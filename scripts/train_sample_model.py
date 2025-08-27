#!/usr/bin/env python3
"""
Train a sample raga classification model for testing.

This script trains a simple model on synthetic data to test the API.
In a production environment, replace this with your actual training pipeline.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime

# Configuration
MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Sample ragas and their features (synthetic data for testing)
RAGAS = ["Yaman", "Bhairav", "Kafi", "Bhairavi", "Kalyani", "Todi"]
NUM_FEATURES = 100
NUM_SAMPLES_PER_RAGA = 50


def generate_synthetic_data():
    """Generate synthetic training data for testing."""
    X = []
    y = []
    
    for raga in RAGAS:
        # Generate random features for each raga with some noise
        base_features = np.random.normal(size=NUM_FEATURES)
        
        for _ in range(NUM_SAMPLES_PER_RAGA):
            # Add some noise to make samples different
            features = base_features + np.random.normal(scale=0.1, size=NUM_FEATURES)
            X.append(features)
            y.append(raga)
    
    return np.array(X), np.array(y)


def create_model(input_shape, num_classes):
    """Create a simple neural network model."""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train():
    """Train a sample model and save it."""
    print("Generating synthetic data...")
    X, y = generate_synthetic_data()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    print("Training model...")
    model = create_model(X.shape[1], len(RAGAS))
    model.fit(
        X_scaled, y_encoded,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model and artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"raga_model_{timestamp}.h5")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    
    # Save the latest model with fixed names for the API
    model.save(os.path.join(MODEL_DIR, "raga_model.h5"))
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    # Create a symlink to the latest model for the API
    try:
        latest_model = os.path.join(MODEL_DIR, "raga_model.h5")
        if os.path.islink(latest_model):
            os.remove(latest_model)
        os.symlink(f"raga_model_{timestamp}.h5", latest_model)
        print(f"Created symlink to latest model: {latest_model}")
    except OSError as e:
        print(f"Warning: Could not create symlink: {e}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
